"""E2B PTY manager using sandbox.pty.

This module provides PTY support for E2B cloud sandbox environments
using the E2B SDK's pty API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from e2b import AsyncSandbox  # type: ignore[import-untyped]


@dataclass
class E2BPtySession:
    """Tracks an E2B PTY session."""

    info: PtyInfo
    handle: object  # E2B Pty handle
    sandbox: AsyncSandbox
    _output_buffer: list[bytes] = field(default_factory=list)
    _output_event: asyncio.Event = field(default_factory=asyncio.Event)


class E2BPtyManager(BasePtyManager):
    """PTY manager for E2B cloud sandbox execution.

    Uses E2B's sandbox.pty API for interactive terminal sessions
    in cloud sandboxes.

    API Reference (E2B sandbox.pty):
        - sandbox.pty.create(rows, cols, on_data, cwd, envs, cmd, timeout)
        - handle.resize(rows, cols)
        - handle.send_stdin(data)
        - handle.kill()
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        """Initialize the E2B PTY manager.

        Args:
            sandbox: An active E2B AsyncSandbox instance
        """
        super().__init__()
        self._sandbox = sandbox
        self._e2b_sessions: dict[str, E2BPtySession] = {}

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session in the E2B sandbox.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Shell command (defaults to /bin/bash)
            args: Arguments for the command (will be joined with command)
            cwd: Working directory
            env: Environment variables

        Returns:
            PtyInfo with session details
        """
        size = size or PtySize()
        command = command or "/bin/bash"
        args = args or []
        cwd = cwd or "/home/user"

        # Build full command with args
        command if not args else f"{command} {' '.join(args)}"

        pty_id = self._generate_id()

        # Create output buffer and event for this session
        output_buffer: list[bytes] = []
        output_event = asyncio.Event()

        def on_data(data: bytes) -> None:
            """Callback for PTY output."""
            output_buffer.append(data)
            output_event.set()

        # Create PTY using E2B API
        # Import E2B's PtySize for the API call
        from e2b import PtySize as E2BPtySize

        e2b_size = E2BPtySize(rows=size.rows, cols=size.cols)
        handle = await self._sandbox.pty.create(
            size=e2b_size,
            on_data=on_data,
            cwd=cwd,
            envs=env,
        )

        # E2B doesn't expose PID directly, use a placeholder
        info = PtyInfo(
            id=pty_id,
            pid=0,  # E2B doesn't expose the actual PID
            command=command,
            args=args,
            cwd=cwd,
            size=size,
            status="running",
        )

        session = E2BPtySession(
            info=info,
            handle=handle,
            sandbox=self._sandbox,
            _output_buffer=output_buffer,
            _output_event=output_event,
        )
        self._sessions[pty_id] = info
        self._e2b_sessions[pty_id] = session

        return info

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Args:
            pty_id: The PTY session ID
            size: New terminal size

        Raises:
            KeyError: If PTY session not found
        """
        session = self._e2b_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        await session.handle.resize(rows=size.rows, cols=size.cols)  # type: ignore[attr-defined]
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._e2b_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        await session.handle.send_stdin(data)  # type: ignore[attr-defined]

    async def read(self, pty_id: str, size: int = 4096) -> bytes:
        """Read data from a PTY's output buffer.

        Args:
            pty_id: The PTY session ID
            size: Maximum bytes to read (ignored, returns all buffered data)

        Returns:
            Output data from the PTY

        Raises:
            KeyError: If PTY session not found
        """
        session = self._e2b_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Wait for data if buffer is empty
        if not session._output_buffer:
            try:
                await asyncio.wait_for(session._output_event.wait(), timeout=0.1)
            except TimeoutError:
                return b""

        # Collect all buffered data
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
        session = self._e2b_sessions.get(pty_id)
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
        session = self._e2b_sessions.get(pty_id)
        if not session:
            return False

        try:
            await session.handle.kill()  # type: ignore[attr-defined]
            session.info.status = "exited"
        except Exception:  # noqa: BLE001
            pass

        # Cleanup
        del self._e2b_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        return self._sessions.get(pty_id)
