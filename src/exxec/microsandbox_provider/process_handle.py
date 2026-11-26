"""Process handle for microsandbox microVMs via `Sandbox.exec_stream()`.

microsandbox delivers a process' lifecycle as an event stream (started, stdout,
stderr, exited/failed). A pump task consumes it from the start, so the exit code is
known even if nobody reads the output; output chunks are queued for the
single-owner `output()` stream (stdout and stderr merged in arrival order).

Signal deaths are reported with exit code -1 and no signal number, so the handle
remembers the last signal it sent to report `128 + signal` like other providers.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING

from exxec.process_handle import (
    SIGNAL_NUMBERS,
    ProcessHandle,
    ProcessHandleError,
    signal_exit_code,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from microsandbox import ExecHandle, ExecSink, Sandbox

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


EOT = b"\x04"
FAILED_TO_START_EXIT_CODE = 127
"""Exit code reported when the guest could not start the program (shell convention)."""


class MicrosandboxProcessHandle(ProcessHandle):
    """A process running inside a microsandbox microVM."""

    def __init__(
        self, exec_handle: ExecHandle, stdin: ExecSink | None, size: PtySize | None
    ) -> None:
        """Wrap a started exec stream.

        Args:
            exec_handle: Handle returned by `Sandbox.exec_stream()`
            stdin: The handle's stdin sink (started with `Stdin.pipe()`)
            size: Terminal size when started with a TTY
        """
        super().__init__(pid=None, pty=size)
        self._exec = exec_handle
        self._stdin = stdin
        self._stdin_closed = stdin is None
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._last_signal: SignalName | None = None
        self._pump = asyncio.create_task(self._consume_events())

    @classmethod
    async def start(
        cls,
        sandbox: Sandbox,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        size: PtySize | None,
    ) -> MicrosandboxProcessHandle:
        """Start argv in the sandbox.

        Args:
            sandbox: Running sandbox
            argv: Program and arguments (use `["sh", "-c", line]` for shell lines)
            cwd: Working directory in the guest
            env: Environment variables for the process
            size: Allocate a TTY of this size, None for pipes

        Returns:
            Handle for the started process
        """
        from microsandbox import Stdin

        program, *args = argv
        exec_handle = await sandbox.exec_stream(
            program, args, cwd=cwd, env=env, stdin=Stdin.pipe(), tty=size is not None
        )
        handle = cls(exec_handle, exec_handle.take_stdin(), size)
        if size is not None:
            await handle._resize(size)
        return handle

    async def _consume_events(self) -> None:
        from microsandbox import ExecEventType

        exit_code: int | None = None
        try:
            while (event := await self._exec.recv()) is not None:
                match event.event_type:
                    case ExecEventType.STARTED:
                        self.pid = event.pid
                    case ExecEventType.STDOUT | ExecEventType.STDERR if event.data:
                        self._output.put_nowait(event.data)
                    case ExecEventType.EXITED:
                        exit_code = self._normalize(event.code)
                    case ExecEventType.FAILED:
                        exit_code = FAILED_TO_START_EXIT_CODE
                    case _:
                        pass
        finally:
            self._output.put_nowait(None)
            if not self._exit.done():
                self._exit.set_result(exit_code if exit_code is not None else self._normalize(-1))

    def _normalize(self, code: int | None) -> int:
        if code is not None and code >= 0:
            return code
        return signal_exit_code(self._last_signal or "KILL")

    async def _iter_output(self) -> AsyncIterator[bytes]:
        while (chunk := await self._output.get()) is not None:
            yield chunk

    async def write(self, data: bytes) -> None:
        if self._stdin is None or self._stdin_closed:
            msg = "stdin is closed"
            raise ProcessHandleError(msg)
        await self._stdin.write(data)

    async def close_stdin(self) -> None:
        if self.is_pty:
            await self.write(EOT)
            return
        if self._stdin is not None and not self._stdin_closed:
            self._stdin_closed = True
            await self._stdin.close()

    async def send_signal(self, sig: SignalName) -> None:
        if self._exit.done():
            return
        self._last_signal = sig
        if sig == "KILL":
            await self._exec.kill()
        else:
            await self._exec.signal(SIGNAL_NUMBERS[sig])

    async def _wait(self) -> int:
        return await asyncio.shield(self._exit)

    async def _has_exited(self) -> bool:
        return self._exit.done()

    async def _resize(self, size: PtySize) -> None:
        await self._exec.resize(size.rows, size.cols)

    async def _release(self) -> None:
        if self._stdin is not None and not self._stdin_closed:
            self._stdin_closed = True
            with contextlib.suppress(Exception):
                await self._stdin.close()
        if not self._pump.done():
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump
