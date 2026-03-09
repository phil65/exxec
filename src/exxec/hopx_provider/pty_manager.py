"""Hopx PTY manager using WebSocket-based terminal.

This module provides PTY support for Hopx cloud VM sandbox environments
using the Hopx SDK's WebSocket terminal API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hopx_ai import AsyncSandbox  # type: ignore[import-untyped]


@dataclass
class HopxPtySession:
    """Tracks a Hopx PTY session."""

    info: PtyInfo
    sandbox: AsyncSandbox
    ws: Any = None  # WebSocket connection
    _output_buffer: list[bytes] = field(default_factory=list)
    _output_event: asyncio.Event = field(default_factory=asyncio.Event)
    _reader_task: asyncio.Task[None] | None = None


class HopxPtyManager(BasePtyManager):
    """PTY manager for Hopx cloud VM sandbox execution.

    Uses Hopx's WebSocket terminal API for interactive terminal sessions
    in cloud VM sandboxes.
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        """Initialize the Hopx PTY manager.

        Args:
            sandbox: An active Hopx AsyncSandbox instance
        """
        super().__init__()
        self._sandbox = sandbox
        self._hopx_sessions: dict[str, HopxPtySession] = {}

    async def _read_loop(self, session: HopxPtySession) -> None:
        """Background task to read terminal output."""
        try:
            async for message in self._sandbox.terminal.iter_output(session.ws):
                msg_type = message.get("type")
                if msg_type == "output":
                    data = message.get("data", "")
                    if isinstance(data, str):
                        data = data.encode()
                    session._output_buffer.append(data)
                    session._output_event.set()
                elif msg_type == "exit":
                    session.info.status = "exited"
                    break
        except Exception:  # noqa: BLE001
            session.info.status = "exited"

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session in the Hopx sandbox.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Shell command (ignored, Hopx terminal uses default shell)
            args: Arguments for the command (ignored)
            cwd: Working directory (ignored, use sandbox default)
            env: Environment variables (ignored, use sandbox env)

        Returns:
            PtyInfo with session details
        """
        size = size or PtySize()
        pty_id = self._generate_id()

        ws = await self._sandbox.terminal.connect()

        default_size = PtySize()
        if size.rows != default_size.rows or size.cols != default_size.cols:
            await self._sandbox.terminal.resize(ws, cols=size.cols, rows=size.rows)

        info = PtyInfo(
            id=pty_id,
            pid=0,  # Hopx doesn't expose the actual PID
            command=command or "/bin/bash",
            args=args or [],
            cwd=cwd or "/workspace",
            size=size,
            status="running",
        )

        session = HopxPtySession(
            info=info,
            sandbox=self._sandbox,
            ws=ws,
        )
        self._sessions[pty_id] = info
        self._hopx_sessions[pty_id] = session

        # Start background reader
        session._reader_task = asyncio.create_task(self._read_loop(session))

        return info

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Args:
            pty_id: The PTY session ID
            size: New terminal size

        Raises:
            KeyError: If PTY session not found
        """
        session = self._hopx_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        await self._sandbox.terminal.resize(session.ws, cols=size.cols, rows=size.rows)
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._hopx_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        text = data.decode("utf-8", errors="replace")
        await self._sandbox.terminal.send_input(session.ws, text)

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
        session = self._hopx_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        if not session._output_buffer:
            try:
                await asyncio.wait_for(session._output_event.wait(), timeout=0.1)
            except TimeoutError:
                return b""

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
        session = self._hopx_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        while session.info.status == "running":
            try:
                await asyncio.wait_for(session._output_event.wait(), timeout=0.5)
            except TimeoutError:
                continue

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
        session = self._hopx_sessions.get(pty_id)
        if not session:
            return False

        try:
            if session._reader_task and not session._reader_task.done():
                session._reader_task.cancel()
            if session.ws:
                await session.ws.close()
            session.info.status = "exited"
        except Exception:  # noqa: BLE001
            pass

        del self._hopx_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        return self._sessions.get(pty_id)
