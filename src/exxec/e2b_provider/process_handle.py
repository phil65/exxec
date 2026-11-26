"""Process handle for e2b sandboxes.

- Pipe processes use `commands.run(background=True, stdin=True)`. e2b delivers
  stdout/stderr as decoded text through callbacks; chunks are re-encoded as UTF-8
  and queued for the single-owner `output()` stream.
- PTY processes use `pty.create()`, which always starts an interactive login shell.
  For an explicit command, the handle types `exec <command>` into that shell right
  after clearing the screen, so the program replaces the shell (its exit code and
  signals are the program's). The raw byte stream therefore starts with the shell's
  startup output followed by a screen clear.
- e2b's own kill is SIGKILL; other signals are delivered with `kill` run in the
  sandbox (process group first, PID as fallback; e2b processes are not group
  leaders, so in practice the PID).
- Signal deaths are reported as exit code -1 with e.g. "signal: terminated"; the
  signal is parsed from that text.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING

from exxec.process_handle import ProcessHandle, ProcessHandleError, signal_exit_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from e2b import AsyncSandbox
    from e2b.sandbox_async.commands.command_handle import AsyncCommandHandle

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


EOT = b"\x04"
_GO_SIGNAL_NAMES: dict[str, str] = {
    "hangup": "HUP",
    "interrupt": "INT",
    "quit": "QUIT",
    "killed": "KILL",
    "terminated": "TERM",
    "broken pipe": "PIPE",
    "aborted": "ABRT",
    "segmentation fault": "SEGV",
}
"""Go's `syscall.Signal.String()` names (as reported by e2b's envd) to signal names."""
_CLEAR_SCREEN = r"printf '\033[H\033[2J\033[3J'"


class E2BProcessHandle(ProcessHandle):
    """A process running in an e2b sandbox."""

    def __init__(
        self,
        sandbox: AsyncSandbox,
        handle: AsyncCommandHandle[None] | AsyncCommandHandle[str],
        output: asyncio.Queue[bytes | None],
        size: PtySize | None,
    ) -> None:
        """Wrap a started e2b command or PTY.

        Args:
            sandbox: The sandbox the process runs in
            handle: e2b command handle
            output: Queue the output callbacks feed
            size: Terminal size for PTY processes
        """
        super().__init__(pid=handle.pid, pty=size)
        self._sandbox = sandbox
        self._handle = handle
        self._output = output
        self._stdin_closed = False
        self._last_signal: SignalName | None = None
        self._exit_watcher = asyncio.create_task(self._watch_exit())

    @classmethod
    async def start(
        cls,
        sandbox: AsyncSandbox,
        command_line: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        size: PtySize | None,
        login_shell: bool = False,
    ) -> E2BProcessHandle:
        """Start a process.

        Args:
            sandbox: Running sandbox
            command_line: Shell command line (args already quoted, `exec` applied)
            cwd: Working directory
            env: Environment variables for the process
            size: Allocate a PTY of this size, None for pipes
            login_shell: For PTYs, keep the login shell itself instead of exec'ing
                `command_line` (which is then ignored)

        Returns:
            Handle for the started process
        """
        output: asyncio.Queue[bytes | None] = asyncio.Queue()
        if size is None:
            pipe_handle = await sandbox.commands.run(
                command_line,
                background=True,
                stdin=True,
                cwd=cwd,
                envs=env,
                on_stdout=lambda text: output.put_nowait(text.encode()),
                on_stderr=lambda text: output.put_nowait(text.encode()),
                timeout=0,
                retain_output=False,
            )
            return cls(sandbox, pipe_handle, output, None)
        from e2b.sandbox.commands.command_handle import PtySize as E2BPtySize

        pty_handle = await sandbox.pty.create(
            size=E2BPtySize(rows=size.rows, cols=size.cols),
            on_data=output.put_nowait,
            cwd=cwd,
            envs=env,
            timeout=0,
        )
        process = cls(sandbox, pty_handle, output, size)
        if not login_shell:
            await process.write(f"{_CLEAR_SCREEN}; {command_line}\r".encode())
        return process

    async def _watch_exit(self) -> None:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            await self._handle.wait()
        except CommandExitException:
            pass  # exit code is available on the handle
        except Exception:  # noqa: BLE001
            pass  # connection errors: reported as lost below
        finally:
            self._output.put_nowait(None)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        while (chunk := await self._output.get()) is not None:
            yield chunk

    async def write(self, data: bytes) -> None:
        if self._stdin_closed:
            msg = "stdin is closed"
            raise ProcessHandleError(msg)
        if self.is_pty:
            await self._sandbox.pty.send_stdin(self._handle.pid, data)
        else:
            await self._sandbox.commands.send_stdin(self._handle.pid, data)

    async def close_stdin(self) -> None:
        if self.is_pty:
            await self.write(EOT)
            return
        if not self._stdin_closed:
            self._stdin_closed = True
            await self._sandbox.commands.close_stdin(self._handle.pid)

    async def send_signal(self, sig: SignalName) -> None:
        if self._exit_watcher.done():
            return
        self._last_signal = sig
        if sig == "KILL":
            with contextlib.suppress(Exception):
                await self._handle.kill()
            return
        pid = self._handle.pid
        script = f"kill -s {sig} -- -{pid} 2>/dev/null || kill -s {sig} {pid} 2>/dev/null; true"
        await self._sandbox.commands.run(script)

    async def _wait(self) -> int:
        await asyncio.shield(self._exit_watcher)
        code = self._handle.exit_code
        if code is not None and code >= 0:
            return int(code)
        # Signal deaths are reported as -1 with the signal in the error text,
        # e.g. "signal: terminated" (Go's signal names).
        error = (self._handle.error or "").lower()
        if error.startswith("signal: "):
            name = _GO_SIGNAL_NAMES.get(error.removeprefix("signal: ").strip())
            if name is not None:
                return signal_exit_code(name)
        return signal_exit_code(self._last_signal or "KILL")

    async def _has_exited(self) -> bool:
        return self._exit_watcher.done()

    async def _resize(self, size: PtySize) -> None:
        from e2b.sandbox.commands.command_handle import PtySize as E2BPtySize

        await self._sandbox.pty.resize(self._handle.pid, E2BPtySize(rows=size.rows, cols=size.cols))

    async def _release(self) -> None:
        if not self._exit_watcher.done():
            with contextlib.suppress(Exception):
                await self._handle.disconnect()
            self._exit_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._exit_watcher
