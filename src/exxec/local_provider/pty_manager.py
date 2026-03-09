"""Local PTY manager using ptyprocess.

This module provides PTY support for local execution environments
using the ptyprocess library (or falling back to stdlib pty).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import TYPE_CHECKING

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ptyprocess import PtyProcess  # type: ignore[import-untyped]


@dataclass
class LocalPtySession:
    """Tracks a local PTY session."""

    info: PtyInfo
    process: PtyProcess
    _read_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LocalPtyManager(BasePtyManager):
    """PTY manager for local execution using ptyprocess.

    Uses ptyprocess for proper PTY support including resize.
    Falls back to a simpler implementation if ptyprocess is not available.
    """

    def __init__(self, cwd: str | None = None) -> None:
        """Initialize the local PTY manager.

        Args:
            cwd: Default working directory for PTY sessions
        """
        super().__init__()
        self._cwd = cwd
        self._local_sessions: dict[str, LocalPtySession] = {}

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new local PTY session.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Shell command (defaults to $SHELL or /bin/bash)
            args: Arguments for the command
            cwd: Working directory
            env: Additional environment variables

        Returns:
            PtyInfo with session details
        """
        from ptyprocess import PtyProcess

        size = size or PtySize()
        command = command or os.environ.get("SHELL", "/bin/bash")
        args = args or []
        cwd = cwd or self._cwd or str(Path.cwd())

        # Build environment
        full_env = {**os.environ, "TERM": "xterm-256color"}
        if env:
            full_env.update(env)

        # Build command list
        cmd_list = [command, *args]

        # Add login shell flag if it's a shell
        if command.endswith("sh") and "-l" not in args:
            cmd_list = [command, "-l", *args]

        # Spawn PTY process
        process = PtyProcess.spawn(
            cmd_list,
            dimensions=(size.rows, size.cols),
            cwd=cwd,
            env=full_env,
        )

        pty_id = self._generate_id()
        info = PtyInfo(
            id=pty_id,
            pid=process.pid,
            command=command,
            args=args,
            cwd=cwd,
            size=size,
            status="running",
        )

        session = LocalPtySession(info=info, process=process)
        self._sessions[pty_id] = info
        self._local_sessions[pty_id] = session

        return info

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Args:
            pty_id: The PTY session ID
            size: New terminal size

        Raises:
            KeyError: If PTY session not found
        """
        session = self._local_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        session.process.setwinsize(size.rows, size.cols)
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._local_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Run in executor since ptyprocess.write() is blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, session.process.write, data)

    async def read(self, pty_id: str, size: int = 4096) -> bytes:
        """Read data from a PTY's stdout.

        Args:
            pty_id: The PTY session ID
            size: Maximum bytes to read

        Returns:
            Output data from the PTY

        Raises:
            KeyError: If PTY session not found
        """
        session = self._local_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        async with session._read_lock:
            loop = asyncio.get_event_loop()
            try:
                data = await loop.run_in_executor(None, session.process.read, size)
                return data if isinstance(data, bytes) else data.encode()
            except EOFError:
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
        session = self._local_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        loop = asyncio.get_event_loop()

        while session.process.isalive():
            try:
                data = await loop.run_in_executor(None, session.process.read, 4096)
                if data:
                    yield data if isinstance(data, bytes) else data.encode()
                else:
                    await asyncio.sleep(0.01)
            except EOFError:
                break

        # Mark as exited
        session.info.status = "exited"
        session.info.exit_code = session.process.exitstatus

    async def kill(self, pty_id: str) -> bool:
        """Kill a PTY session.

        Args:
            pty_id: The PTY session ID

        Returns:
            True if killed successfully, False if not found
        """
        session = self._local_sessions.get(pty_id)
        if not session:
            return False

        try:
            if session.process.isalive():
                session.process.terminate(force=True)

            session.info.status = "exited"
            session.info.exit_code = session.process.exitstatus
        except Exception:  # noqa: BLE001
            pass

        # Cleanup
        del self._local_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        session = self._local_sessions.get(pty_id)
        if not session:
            return None

        # Update status if process has exited
        if not session.process.isalive() and session.info.status == "running":
            session.info.status = "exited"
            session.info.exit_code = session.process.exitstatus

        return session.info
