"""SSH PTY manager using asyncssh.

This module provides PTY support for remote SSH execution environments
using asyncssh's built-in PTY functionality.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from asyncssh import SSHClientConnection, SSHClientProcess


@dataclass
class SshPtySession:
    """Tracks an SSH PTY session."""

    info: PtyInfo
    process: SSHClientProcess[Any]
    connection: SSHClientConnection
    _read_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SshPtyManager(BasePtyManager):
    """PTY manager for SSH remote execution.

    Uses asyncssh for interactive terminal sessions over SSH.
    asyncssh provides native PTY support with term_type and term_size.

    Key asyncssh features used:
        - connection.create_process() with term_type and term_size
        - process.change_terminal_size(width, height)
        - process.stdin.write() / process.stdout.read()
    """

    def __init__(self, connection: SSHClientConnection, cwd: str | None = None) -> None:
        """Initialize the SSH PTY manager.

        Args:
            connection: An active asyncssh SSHClientConnection
            cwd: Default working directory for PTY sessions
        """
        super().__init__()
        self._connection = connection
        self._cwd = cwd
        self._ssh_sessions: dict[str, SshPtySession] = {}

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session over SSH.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Shell command (defaults to remote user's login shell)
            args: Arguments for the command
            cwd: Working directory (will cd into it if specified)
            env: Environment variables

        Returns:
            PtyInfo with session details
        """
        size = size or PtySize()
        args = args or []
        cwd = cwd or self._cwd

        # Build command - if no command specified, use login shell
        full_cmd = (f"{command} {' '.join(args)}" if args else command) if command else None

        # Prepend cd if cwd is specified
        if cwd and full_cmd:
            full_cmd = f"cd {cwd} && {full_cmd}"
        elif cwd:
            full_cmd = f"cd {cwd} && exec $SHELL -l"

        pty_id = self._generate_id()

        # Create SSH process with PTY
        # asyncssh uses (width, height) format for term_size
        process = await self._connection.create_process(
            full_cmd,
            term_type="xterm-256color",
            term_size=(size.cols, size.rows),
            env=env,
            encoding=None,  # Use bytes mode
        )

        # asyncssh doesn't expose remote PID easily, use placeholder
        info = PtyInfo(
            id=pty_id,
            pid=0,  # SSH doesn't expose remote PID
            command=command or "/bin/bash",
            args=args,
            cwd=cwd,
            size=size,
            status="running",
        )

        session = SshPtySession(
            info=info,
            process=process,
            connection=self._connection,
        )
        self._sessions[pty_id] = info
        self._ssh_sessions[pty_id] = session

        return info

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Args:
            pty_id: The PTY session ID
            size: New terminal size

        Raises:
            KeyError: If PTY session not found
        """
        session = self._ssh_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # asyncssh uses (width, height) format
        session.process.change_terminal_size(size.cols, size.rows)
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._ssh_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        session.process.stdin.write(data)
        await session.process.stdin.drain()

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
        session = self._ssh_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        async with session._read_lock:
            try:
                data = await asyncio.wait_for(
                    session.process.stdout.read(size),
                    timeout=0.1,
                )
                return data if isinstance(data, bytes) else data.encode()
            except TimeoutError:
                return b""
            except Exception:  # noqa: BLE001
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
        session = self._ssh_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        try:
            while True:
                data = await session.process.stdout.read(4096)
                if not data:
                    break
                yield data if isinstance(data, bytes) else data.encode()
        except Exception:  # noqa: BLE001
            pass

        # Update status
        session.info.status = "exited"
        session.info.exit_code = session.process.returncode

    async def kill(self, pty_id: str) -> bool:
        """Kill a PTY session.

        Args:
            pty_id: The PTY session ID

        Returns:
            True if killed successfully, False if not found
        """
        session = self._ssh_sessions.get(pty_id)
        if not session:
            return False

        try:
            session.process.terminate()
            await session.process.wait()
            session.info.status = "exited"
            session.info.exit_code = session.process.returncode
        except Exception:  # noqa: BLE001
            pass

        # Cleanup
        del self._ssh_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        session = self._ssh_sessions.get(pty_id)
        if not session:
            return None

        # Check if process has exited
        if session.process.returncode is not None and session.info.status == "running":
            session.info.status = "exited"
            session.info.exit_code = session.process.returncode

        return session.info
