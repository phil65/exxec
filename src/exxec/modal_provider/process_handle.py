"""Process handle for Modal sandboxes via `Sandbox.exec()`.

Modal's `ContainerProcess` streams stdin/stdout but has no kill, no PID and no PTY
resize. The handle fills those gaps with POSIX tools inside the container:

- the target is started through `sh`, which records its PID (and, for PTYs, its
  terminal device) in files before `exec`ing it, so the recorded PID is the
  target's;
- signals are delivered with `kill` via a helper exec;
- the initial PTY size is applied with `stty` before `exec`, and resizes run
  `stty -F <tty> rows R cols C` via a helper exec, which makes the kernel deliver
  SIGWINCH to the program.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from typing import TYPE_CHECKING
import uuid

from exxec.process_handle import (
    SIGNAL_NUMBERS,
    ProcessHandle,
    ProcessHandleError,
    normalize_returncode,
    signal_exit_code,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from modal import Sandbox
    from modal.container_process import ContainerProcess

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


EOT = b"\x04"
_PIDFILE_WAIT_SECONDS = 5.0


class ModalProcessHandle(ProcessHandle):
    """A process running in a Modal sandbox."""

    def __init__(
        self,
        sandbox: Sandbox,
        process: ContainerProcess[bytes],
        state_file: str,
        size: PtySize | None,
    ) -> None:
        """Wrap a started exec process.

        Args:
            sandbox: Sandbox the process runs in
            process: The started process (`text=False`)
            state_file: Path prefix of the PID (and TTY) files inside the sandbox
            size: Terminal size for PTY processes
        """
        super().__init__(pid=None, pty=size)
        self._sandbox = sandbox
        self._process = process
        self._state_file = state_file
        self._stdin_closed = False
        self._last_signal: SignalName | None = None

    @classmethod
    async def start(
        cls,
        sandbox: Sandbox,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        size: PtySize | None,
    ) -> ModalProcessHandle:
        """Start argv in the sandbox.

        Args:
            sandbox: Running sandbox
            argv: Program and arguments (use `["sh", "-c", line]` for shell lines)
            cwd: Working directory
            env: Environment variables for the process
            size: Allocate a PTY of this size, None for pipes

        Returns:
            Handle for the started process
        """
        from modal.stream_type import StreamType

        state_file = f"/tmp/.exxec-{uuid.uuid4().hex}"
        setup = 'echo $$ > "$0"'
        if size is not None:
            setup = f'stty rows {size.rows} cols {size.cols} 2>/dev/null; tty > "$0.tty"; {setup}'
        # `$0` is the state file; "$@" is the target argv, which replaces the shell.
        wrapped = ["sh", "-c", f'{setup} && exec "$@"', state_file, *argv]
        process = await sandbox.exec.aio(
            *wrapped,
            workdir=cwd,
            env=dict(env) if env else None,
            text=False,
            stdout=StreamType.PIPE,
            stderr=StreamType.STDOUT,
            pty=size is not None,
        )
        return cls(sandbox, process, state_file, size)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        async for chunk in self._process.stdout:
            if chunk:
                yield chunk

    async def write(self, data: bytes) -> None:
        if self._stdin_closed:
            msg = "stdin is closed"
            raise ProcessHandleError(msg)
        self._process.stdin.write(data)
        await self._process.stdin.drain.aio()

    async def close_stdin(self) -> None:
        if self.is_pty:
            await self.write(EOT)
            return
        if not self._stdin_closed:
            self._stdin_closed = True
            self._process.stdin.write_eof()
            await self._process.stdin.drain.aio()

    async def _run_helper(self, script: str) -> str:
        helper = await self._sandbox.exec.aio("sh", "-c", script, text=False)
        output = await helper.stdout.read.aio()
        await helper.wait.aio()
        return output.decode(errors="replace")

    async def _read_pid(self) -> int | None:
        if self.pid is not None:
            return self.pid
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _PIDFILE_WAIT_SECONDS
        while True:
            content = (await self._run_helper(f"cat {shlex.quote(self._state_file)}")).strip()
            if content.isdigit():
                self.pid = int(content)
                return self.pid
            if loop.time() >= deadline or await self._has_exited():
                return None
            await asyncio.sleep(0.1)

    async def send_signal(self, sig: SignalName) -> None:
        if await self._has_exited():
            return
        pid = await self._read_pid()
        if pid is None:
            return
        self._last_signal = sig
        signum = SIGNAL_NUMBERS[sig]
        await self._run_helper(
            f"kill -{signum} -- -{pid} 2>/dev/null || kill -{signum} {pid} 2>/dev/null; true"
        )

    async def _wait(self) -> int:
        code = await self._process.wait.aio()
        if code is not None and code >= 0:
            return int(code)
        if self._last_signal is not None:
            return signal_exit_code(self._last_signal)
        return normalize_returncode(code)

    async def _has_exited(self) -> bool:
        if self._returncode is not None:
            return True
        return await self._process.poll.aio() is not None

    async def _resize(self, size: PtySize) -> None:
        tty_file = shlex.quote(f"{self._state_file}.tty")
        await self._run_helper(f'stty -F "$(cat {tty_file})" rows {size.rows} cols {size.cols}')

    async def _release(self) -> None:
        with contextlib.suppress(Exception):
            path = shlex.quote(self._state_file)
            await self._run_helper(f"rm -f {path} {path}.tty")
