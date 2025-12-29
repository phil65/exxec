"""Modal PTY manager using sandbox pty=True.

This module provides PTY support for Modal serverless sandbox environments
using Modal's native pty=True parameter.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from modal import Sandbox


@dataclass
class ModalPtySession:
    """Tracks a Modal PTY session."""

    info: PtyInfo
    process: object  # Modal ContainerProcess
    sandbox: Sandbox
    _output_buffer: list[bytes] = field(default_factory=list)
    _output_event: asyncio.Event = field(default_factory=asyncio.Event)
    _reader_task: asyncio.Task[None] | None = None


class ModalPtyManager(BasePtyManager):
    """PTY manager for Modal serverless sandbox execution.

    Uses Modal's sandbox.exec() with pty=True for interactive
    terminal sessions in serverless containers.

    Note: Modal's PTY API uses a simple pty=True flag. The older
    pty_info protobuf parameter is deprecated. Terminal resize
    is not currently supported through the public API.

    Key Modal features used:
        - sandbox.exec(..., pty=True)
        - process.stdin.write()
        - async for line in process.stdout
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize the Modal PTY manager.

        Args:
            sandbox: An active Modal Sandbox instance
        """
        super().__init__()
        self._sandbox = sandbox
        self._modal_sessions: dict[str, ModalPtySession] = {}

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session in the Modal sandbox.

        Args:
            size: Initial terminal size (stored but not applied - Modal doesn't
                  expose resize API publicly)
            command: Shell command (defaults to /bin/bash)
            args: Arguments for the command
            cwd: Working directory (prepended as cd command)
            env: Environment variables

        Returns:
            PtyInfo with session details
        """
        size = size or PtySize()
        command = command or "/bin/bash"
        args = args or []
        cwd = cwd or "/tmp"

        pty_id = self._generate_id()

        # Build command with optional cd
        cmd_parts = [command, *args]
        if cwd:
            # Prepend cd to change to working directory
            shell_cmd = f"cd {cwd} && {' '.join(cmd_parts)}"
            exec_cmd = "sh"
            exec_args = ["-c", shell_cmd]
        else:
            exec_cmd = command
            exec_args = args

        # Create process with PTY enabled
        process = await self._sandbox.exec.aio(
            exec_cmd,
            *exec_args,
            pty=True,
            env=env,  # type: ignore[arg-type]
        )

        # Modal doesn't expose PID directly
        info = PtyInfo(
            id=pty_id,
            pid=0,  # Modal doesn't expose PID
            command=command,
            args=args,
            cwd=cwd,
            size=size,  # Stored but can't be applied - Modal has no resize API
            status="running",
        )

        output_buffer: list[bytes] = []
        output_event = asyncio.Event()

        session = ModalPtySession(
            info=info,
            process=process,
            sandbox=self._sandbox,
            _output_buffer=output_buffer,
            _output_event=output_event,
        )
        self._sessions[pty_id] = info
        self._modal_sessions[pty_id] = session

        # Start background reader task
        session._reader_task = asyncio.create_task(self._read_output(session))

        return info

    async def _read_output(self, session: ModalPtySession) -> None:
        """Background task to read from Modal process stdout."""
        try:
            async for line in session.process.stdout:  # type: ignore[attr-defined]
                data = line.encode() if isinstance(line, str) else line
                session._output_buffer.append(data)
                session._output_event.set()
        except Exception:  # noqa: BLE001
            pass
        finally:
            session.info.status = "exited"
            # Try to get exit code
            with contextlib.suppress(Exception):
                session.info.exit_code = await session.process.wait.aio()  # type: ignore[attr-defined]

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Note: Modal's public API does not currently support terminal resize.
        This method stores the requested size but cannot apply it.

        Args:
            pty_id: The PTY session ID
            size: New terminal size (stored but not applied)

        Raises:
            KeyError: If PTY session not found
        """
        session = self._modal_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Store the size (we can't actually resize - Modal doesn't expose this)
        session.info.size = size
        # Note: Modal's PTY resize API is not publicly available
        # The deprecated pty_info protobuf had winsz_rows/winsz_cols but
        # the new pty=True API doesn't expose resize functionality

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._modal_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Modal's stdin expects string
        text = data.decode() if isinstance(data, bytes) else data
        session.process.stdin.write(text)  # type: ignore[attr-defined]

    async def read(self, pty_id: str, size: int = 4096) -> bytes:
        """Read data from a PTY's output buffer.

        Args:
            pty_id: The PTY session ID
            size: Maximum bytes to read

        Returns:
            Output data from the PTY

        Raises:
            KeyError: If PTY session not found
        """
        session = self._modal_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Wait for data if buffer is empty
        if not session._output_buffer:
            try:
                await asyncio.wait_for(session._output_event.wait(), timeout=0.1)
            except TimeoutError:
                return b""

        # Collect buffered data
        if session._output_buffer:
            data = b"".join(session._output_buffer)
            session._output_buffer.clear()
            session._output_event.clear()
            return data[:size] if len(data) > size else data

        return b""

    async def stream(self, pty_id: str) -> AsyncIterator[bytes]:
        """Stream output from a PTY session.

        Args:
            pty_id: The PTY session ID

        Yields:
            Chunks of output data as they become available

        Raises:
            KeyError: If PTY session not found
        """
        session = self._modal_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        while session.info.status == "running":
            # Wait for data
            try:
                await asyncio.wait_for(session._output_event.wait(), timeout=0.5)
            except TimeoutError:
                continue

            # Yield buffered data
            if session._output_buffer:
                data = b"".join(session._output_buffer)
                session._output_buffer.clear()
                session._output_event.clear()
                yield data

    async def kill(self, pty_id: str) -> bool:
        """Kill a PTY session.

        Args:
            pty_id: The PTY session ID

        Returns:
            True if killed successfully, False if not found
        """
        session = self._modal_sessions.get(pty_id)
        if not session:
            return False

        try:
            # Cancel reader task
            if session._reader_task:
                session._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await session._reader_task

            # Modal process doesn't have a direct kill method
            # The process will terminate when stdin is closed
            session.process.stdin.write_eof()  # type: ignore[attr-defined]

            session.info.status = "exited"
        except Exception:  # noqa: BLE001
            pass

        # Cleanup
        del self._modal_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        return self._sessions.get(pty_id)
