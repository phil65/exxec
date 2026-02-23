"""Sprites PTY manager using WebSocket-based TTY execution.

This module provides PTY support for Sprites cloud VM environments
using the Sprites SDK's WebSocket command execution with TTY mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sprites import Sprite  # type: ignore[import-untyped]


@dataclass
class SpritesPtySession:
    """Tracks a Sprites PTY session."""

    info: PtyInfo
    sprite: Sprite
    ws_cmd: Any = None  # WSCommand instance
    _output_buffer: list[bytes] = field(default_factory=list)
    _output_event: asyncio.Event = field(default_factory=asyncio.Event)
    _reader_task: asyncio.Task[None] | None = None


class SpritesPtyManager(BasePtyManager):
    """PTY manager for Sprites cloud VM execution.

    Uses Sprites' WebSocket command execution with TTY mode for
    interactive terminal sessions in cloud VMs.
    """

    def __init__(self, sprite: Sprite) -> None:
        """Initialize the Sprites PTY manager.

        Args:
            sprite: An active Sprite instance
        """
        super().__init__()
        self._sprite = sprite
        self._sprites_sessions: dict[str, SpritesPtySession] = {}

    async def _read_loop(self, session: SpritesPtySession) -> None:
        """Background task to wait for command completion and collect output."""
        try:
            ws_cmd = session.ws_cmd
            if ws_cmd is None:
                return
            await ws_cmd.wait()
            # Collect any remaining buffered output
            stdout = ws_cmd.get_stdout()
            if stdout:
                session._output_buffer.append(stdout)
                session._output_event.set()
            session.info.status = "exited"
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
        """Create a new PTY session in the Sprites VM.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Shell command (defaults to /bin/bash)
            args: Arguments for the command
            cwd: Working directory
            env: Environment variables

        Returns:
            PtyInfo with session details
        """
        from sprites.exec import Cmd
        from sprites.websocket import WSCommand

        size = size or PtySize()
        command = command or "/bin/bash"
        args = args or []
        pty_id = self._generate_id()

        # Build the full command args list
        full_args = [command, *args]

        # Create a Cmd with TTY mode
        cmd = Cmd(
            sprite=self._sprite,
            args=full_args,
            cwd=cwd,
            env=env,
            tty=True,
            tty_rows=size.rows,
            tty_cols=size.cols,
        )

        # Create WSCommand and start it
        ws_cmd = WSCommand(cmd)

        # Set up output capture
        output_buffer: list[bytes] = []
        output_event = asyncio.Event()

        def on_text_message(data: bytes) -> None:
            output_buffer.append(data)
            output_event.set()

        ws_cmd.text_message_handler = on_text_message
        await ws_cmd.start()

        info = PtyInfo(
            id=pty_id,
            pid=0,  # Sprites doesn't expose the actual PID
            command=command,
            args=args,
            cwd=cwd or "/",
            size=size,
            status="running",
        )

        session = SpritesPtySession(
            info=info,
            sprite=self._sprite,
            ws_cmd=ws_cmd,
            _output_buffer=output_buffer,
            _output_event=output_event,
        )
        self._sessions[pty_id] = info
        self._sprites_sessions[pty_id] = session

        # Start background reader to detect completion
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
        session = self._sprites_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        if session.ws_cmd:
            await session.ws_cmd.resize(cols=size.cols, rows=size.rows)
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._sprites_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        if session.ws_cmd:
            await session.ws_cmd._write_stdin(data)

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
        session = self._sprites_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Also check the WSCommand's internal stdout buffer
        if session.ws_cmd:
            ws_stdout = session.ws_cmd.get_stdout()
            if ws_stdout:
                session._output_buffer.append(ws_stdout)
                session.ws_cmd._stdout_buffer.clear()

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
        session = self._sprites_sessions.get(pty_id)
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
        session = self._sprites_sessions.get(pty_id)
        if not session:
            return False

        try:
            if session._reader_task and not session._reader_task.done():
                session._reader_task.cancel()
            if session.ws_cmd:
                await session.ws_cmd.close()
            session.info.status = "exited"
        except Exception:  # noqa: BLE001
            pass

        del self._sprites_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        return self._sessions.get(pty_id)
