"""Process handles for Vercel sandboxes.

- `VercelProcessHandle`: `Sandbox.create_command()` with piped stdin and stderr
  merged into stdout; signals via `Command.send_signal()` / `kill()`.
- `VercelInteractiveHandle`: `Sandbox.open_interactive()` for PTY processes. The
  interactive session owns an anyio task group that must unwind in the task that
  opened it, so each session lives in a dedicated owner task; closing the handle
  tells that task to exit the session. Interactive sessions have no signal API:
  INT is delivered as Ctrl-C through the terminal, other signals hang up the
  session (closing the terminal ends the process).
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from typing import TYPE_CHECKING

import anyio

from exxec.process_handle import (
    ProcessHandle,
    ProcessHandleError,
    normalize_returncode,
    signal_exit_code,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from vercel.sandbox import Sandbox
    from vercel.sandbox._internal.async_runtime import Command, InteractiveSession

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


READ_CHUNK_SIZE = 65536
CTRL_C = b"\x03"
EOT = b"\x04"


class VercelProcessHandle(ProcessHandle):
    """A pipe process created with `Sandbox.create_command()`."""

    def __init__(self, command: Command) -> None:
        """Wrap a started command (stdin piped, stderr merged into stdout)."""
        super().__init__(pid=None, pty=None)
        self._command = command
        self._stdin_closed = command.stdin is None

    @classmethod
    async def start(
        cls,
        sandbox: Sandbox,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> VercelProcessHandle:
        """Start argv in the sandbox.

        Args:
            sandbox: Running sandbox
            argv: Program and arguments (use `["sh", "-c", line]` for shell lines)
            cwd: Working directory
            env: Environment variables for the process

        Returns:
            Handle for the started process
        """
        program, *args = argv
        command = await sandbox.create_command(
            program,
            args,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return cls(command)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        stdout = self._command.stdout
        if stdout is None:
            return
        while True:
            try:
                yield await stdout.receive(READ_CHUNK_SIZE)
            except anyio.EndOfStream:
                return

    async def write(self, data: bytes) -> None:
        if self._command.stdin is None or self._stdin_closed:
            msg = "stdin is closed"
            raise ProcessHandleError(msg)
        await self._command.stdin.send(data)

    async def close_stdin(self) -> None:
        if self._command.stdin is not None and not self._stdin_closed:
            self._stdin_closed = True
            await self._command.stdin.aclose()

    async def send_signal(self, sig: SignalName) -> None:
        if self._command.returncode is not None:
            return
        if sig == "KILL":
            await self._command.kill()
        else:
            await self._command.send_signal(sig)

    async def _wait(self) -> int:
        return normalize_returncode(await self._command.wait())

    async def _has_exited(self) -> bool:
        return self._command.returncode is not None

    async def _resize(self, size: PtySize) -> None:
        msg = "Pipe processes have no terminal"
        raise ProcessHandleError(msg)

    async def _release(self) -> None:
        await self.close_stdin()
        if self._command.stdout is not None:
            with contextlib.suppress(Exception):
                await self._command.stdout.aclose()


class VercelInteractiveHandle(ProcessHandle):
    """A PTY process created with `Sandbox.open_interactive()`."""

    def __init__(
        self,
        session: InteractiveSession,
        owner: asyncio.Task[None],
        close_requested: asyncio.Event,
        size: PtySize,
    ) -> None:
        """Wrap an open interactive session (use `start()`)."""
        super().__init__(pid=None, pty=size)
        self._session = session
        self._owner = owner
        self._close_requested = close_requested
        self._hung_up = False
        self._last_signal: SignalName = "HUP"

    @classmethod
    async def start(
        cls,
        sandbox: Sandbox,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        size: PtySize,
    ) -> VercelInteractiveHandle:
        """Open an interactive session running argv on a PTY.

        Args:
            sandbox: Running sandbox
            argv: Program and arguments
            cwd: Working directory
            env: Environment variables for the process
            size: Terminal size

        Returns:
            Handle for the running process
        """
        program, *args = argv
        opened: asyncio.Future[InteractiveSession] = asyncio.get_running_loop().create_future()
        close_requested = asyncio.Event()

        async def own_session() -> None:
            operation = sandbox.open_interactive(
                program, args, cwd=cwd, env=env, cols=size.cols, rows=size.rows
            )
            try:
                async with operation as session:
                    opened.set_result(session)
                    await close_requested.wait()
            except BaseException as e:
                if not opened.done():
                    opened.set_exception(e)
                raise

        owner = asyncio.create_task(own_session())
        session = await opened
        return cls(session, owner, close_requested, size)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        while True:
            try:
                yield await self._session.stream.receive(READ_CHUNK_SIZE)
            except anyio.EndOfStream, anyio.ClosedResourceError:
                return

    async def write(self, data: bytes) -> None:
        if self._hung_up:
            msg = "Terminal is closed"
            raise ProcessHandleError(msg)
        await self._session.stream.send(data)

    async def close_stdin(self) -> None:
        # Terminals cannot be half-closed; EOT makes the line discipline report EOF.
        await self.write(EOT)

    async def send_signal(self, sig: SignalName) -> None:
        if self._session.returncode is not None or self._hung_up:
            return
        if sig == "INT":
            await self._session.stream.send(CTRL_C)
            return
        self._hung_up = True
        self._last_signal = sig
        self._close_requested.set()

    async def _wait(self) -> int:
        try:
            code = await self._session.wait()
        except anyio.BrokenResourceError:
            code = None
        if code is not None:
            return normalize_returncode(code)
        # The session ended without an exit status: it was hung up.
        return signal_exit_code(self._last_signal)

    async def _has_exited(self) -> bool:
        return self._session.returncode is not None or self._owner.done()

    async def _resize(self, size: PtySize) -> None:
        await self._session.resize(size.cols, size.rows)

    async def _release(self) -> None:
        self._close_requested.set()
        with contextlib.suppress(Exception):
            await self._owner
