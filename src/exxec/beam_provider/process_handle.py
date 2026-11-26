"""Process handle for Beam sandboxes.

Beam's process API is synchronous and pipe-only: no interactive stdin and no PTY.
All SDK calls run in worker threads; a reader thread iterates the combined log
stream and hands lines to the event loop, so output streams without blocking it.
Signals other than SIGKILL are delivered with `kill` via a second exec (Beam PIDs
are in-sandbox PIDs).
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING

from exxec.process_handle import (
    SIGNAL_NUMBERS,
    ProcessHandle,
    ProcessHandleError,
    normalize_returncode,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from beam import SandboxInstance, SandboxProcess  # type: ignore[import-untyped]

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


class BeamProcessHandle(ProcessHandle):
    """A process running in a Beam sandbox."""

    def __init__(self, instance: SandboxInstance, process: SandboxProcess) -> None:
        """Wrap a started Beam process.

        Args:
            instance: Sandbox the process runs in
            process: The started process
        """
        super().__init__(pid=process.pid, pty=None)
        self._instance = instance
        self._process = process
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self._reader = asyncio.create_task(asyncio.to_thread(self._read_logs, loop))

    @classmethod
    async def start(
        cls,
        instance: SandboxInstance,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> BeamProcessHandle:
        """Start argv in the sandbox.

        Args:
            instance: Running sandbox instance
            argv: Program and arguments (use `["sh", "-c", line]` for shell lines)
            cwd: Working directory (Beam defaults to /workspace)
            env: Environment variables for the process

        Returns:
            Handle for the started process
        """
        process = await asyncio.to_thread(
            instance.process.exec, *argv, cwd=cwd or "/workspace", env=env
        )
        return cls(instance, process)

    def _read_logs(self, loop: asyncio.AbstractEventLoop) -> None:
        """Runs in a worker thread: forward log lines until the stream ends."""
        try:
            for line in self._process.logs:
                loop.call_soon_threadsafe(self._output.put_nowait, line.encode())
        finally:
            loop.call_soon_threadsafe(self._output.put_nowait, None)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        while (chunk := await self._output.get()) is not None:
            yield chunk

    async def write(self, data: bytes) -> None:
        msg = "Beam processes do not support interactive stdin"
        raise ProcessHandleError(msg)

    async def close_stdin(self) -> None:
        # There is no stdin stream to close.
        return

    async def send_signal(self, sig: SignalName) -> None:
        if await self._has_exited():
            return
        if sig == "KILL":
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._process.kill)
            return
        signum = SIGNAL_NUMBERS[sig]
        killer = await asyncio.to_thread(
            self._instance.process.exec, "kill", f"-{signum}", str(self._process.pid)
        )
        await asyncio.to_thread(killer.wait)

    async def _wait(self) -> int:
        return normalize_returncode(await asyncio.to_thread(self._process.wait))

    async def _has_exited(self) -> bool:
        if self._returncode is not None:
            return True
        status: tuple[int, str] = await asyncio.to_thread(self._process.status)
        return status[1] != "running"

    async def _resize(self, size: PtySize) -> None:
        msg = "Beam processes have no terminal"
        raise ProcessHandleError(msg)

    async def _release(self) -> None:
        # The reader thread ends with the process' log stream; nothing to close.
        return
