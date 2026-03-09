"""Daytona PTY manager using sandbox.process.create_pty_session.

This module provides PTY support for Daytona cloud sandbox environments
using Daytona's native PTY API with full resize and streaming support.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from daytona._async.sandbox import AsyncSandbox
    from daytona.handle.async_pty_handle import AsyncPtyHandle


@dataclass
class DaytonaPtySession:
    """Tracks a Daytona PTY session."""

    info: PtyInfo
    handle: AsyncPtyHandle
    sandbox: AsyncSandbox
    _output_buffer: list[bytes] = field(default_factory=list)
    _output_event: asyncio.Event = field(default_factory=asyncio.Event)


class DaytonaPtyManager(BasePtyManager):
    """PTY manager for Daytona cloud sandbox execution.

    Uses Daytona's process.create_pty_session API for interactive
    terminal sessions with full resize support.

    The on_data callback is used for streaming output, eliminating
    the need for background reader tasks.
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        """Initialize the Daytona PTY manager.

        Args:
            sandbox: An active Daytona AsyncSandbox instance
        """
        super().__init__()
        self._sandbox = sandbox
        self._daytona_sessions: dict[str, DaytonaPtySession] = {}

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session in the Daytona sandbox.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Command to run after shell starts (sent via stdin)
            args: Arguments for the command
            cwd: Working directory for the PTY session
            env: Environment variables for the PTY session

        Returns:
            PtyInfo with session details
        """
        from daytona.common.pty import PtySize as DaytonaPtySize

        size = size or PtySize()
        command = command or "/bin/bash"
        args = args or []
        cwd = cwd or "/home/daytona"

        pty_id = self._generate_id()

        # Create Daytona PtySize object
        daytona_size = DaytonaPtySize(cols=size.cols, rows=size.rows)

        # Daytona doesn't expose PID directly
        info = PtyInfo(
            id=pty_id,
            pid=0,  # Daytona doesn't expose PID
            command=command,
            args=args,
            cwd=cwd,
            size=size,
            status="running",
        )

        output_buffer: list[bytes] = []
        output_event = asyncio.Event()

        # Callback to handle PTY output - called by Daytona when data arrives
        def on_data(data: bytes) -> None:
            output_buffer.append(data)
            output_event.set()

        # Create PTY session using Daytona API with callback
        handle = await self._sandbox.process.create_pty_session(
            id=pty_id,
            on_data=on_data,
            cwd=cwd,
            envs=env,
            pty_size=daytona_size,
        )

        session = DaytonaPtySession(
            info=info,
            handle=handle,
            sandbox=self._sandbox,
            _output_buffer=output_buffer,
            _output_event=output_event,
        )
        self._sessions[pty_id] = info
        self._daytona_sessions[pty_id] = session

        # If command specified (not default shell), send it
        if command and command != "/bin/bash":
            full_cmd = f"{command} {' '.join(args)}\n" if args else f"{command}\n"
            await self.write(pty_id, full_cmd.encode())

        return info

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Args:
            pty_id: The PTY session ID
            size: New terminal size

        Raises:
            KeyError: If PTY session not found
        """
        from daytona.common.pty import PtySize as DaytonaPtySize

        session = self._daytona_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        daytona_size = DaytonaPtySize(cols=size.cols, rows=size.rows)
        await session.handle.resize(daytona_size)
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._daytona_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Daytona send_input expects string
        text = data.decode() if isinstance(data, bytes) else data
        await session.handle.send_input(text)

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
        session = self._daytona_sessions.get(pty_id)
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
        session = self._daytona_sessions.get(pty_id)
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
        session = self._daytona_sessions.get(pty_id)
        if not session:
            return False

        try:
            await session.handle.kill()
            session.info.status = "exited"

            # Try to get exit code
            with contextlib.suppress(Exception):
                result = await session.handle.wait()
                session.info.exit_code = result.exit_code
        except Exception:  # noqa: BLE001
            pass

        # Cleanup
        del self._daytona_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        return self._sessions.get(pty_id)
