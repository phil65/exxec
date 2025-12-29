"""PTY (Pseudo-Terminal) manager protocol and base classes.

This module provides the interface for managing interactive terminal sessions
across different execution environments (local, SSH, Docker, cloud sandboxes).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class PtySize:
    """Terminal size in rows and columns."""

    rows: int = 24
    cols: int = 80


@dataclass
class PtyInfo:
    """Information about a PTY session."""

    id: str
    """Unique identifier for this PTY session."""

    pid: int
    """Process ID of the PTY process."""

    command: str
    """Command that was used to start the PTY (e.g., '/bin/bash')."""

    args: list[str] = field(default_factory=list)
    """Arguments passed to the command."""

    cwd: str | None = None
    """Working directory of the PTY."""

    size: PtySize = field(default_factory=PtySize)
    """Current terminal size."""

    status: Literal["running", "exited"] = "running"
    """Current status of the PTY session."""

    exit_code: int | None = None
    """Exit code if the PTY has exited."""

    created_at: datetime = field(default_factory=datetime.now)
    """When the PTY session was created."""


@runtime_checkable
class PtyManagerProtocol(Protocol):
    """Protocol for managing PTY (pseudo-terminal) sessions.

    This protocol defines the interface that all PTY managers must implement,
    allowing interactive terminal sessions across different execution environments.

    Implementations:
        - LocalPtyManager: Uses ptyprocess for local PTY
        - SshPtyManager: Uses asyncssh for SSH-based PTY
        - DockerPtyManager: Uses Docker exec with tty=True
        - E2BPtyManager: Uses E2B sandbox.pty API
        - ModalPtyManager: Uses Modal's PTYInfo
        - DaytonaPtyManager: Uses Daytona's create_pty_session
    """

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Shell command to run (defaults to user's shell or /bin/bash)
            args: Arguments for the command
            cwd: Working directory
            env: Environment variables

        Returns:
            PtyInfo with session details including the PTY ID
        """
        ...

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Args:
            pty_id: The PTY session ID
            size: New terminal size
        """
        ...

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write (typically user input)
        """
        ...

    async def read(self, pty_id: str, size: int = 4096) -> bytes:
        """Read data from a PTY's stdout.

        Args:
            pty_id: The PTY session ID
            size: Maximum bytes to read

        Returns:
            Output data from the PTY
        """
        ...

    def stream(self, pty_id: str) -> AsyncIterator[bytes]:
        """Stream output from a PTY session.

        Args:
            pty_id: The PTY session ID

        Yields:
            Chunks of output data as they become available
        """
        ...

    async def kill(self, pty_id: str) -> bool:
        """Kill a PTY session.

        Args:
            pty_id: The PTY session ID

        Returns:
            True if killed successfully, False if not found
        """
        ...

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session.

        Args:
            pty_id: The PTY session ID

        Returns:
            PtyInfo if found, None otherwise
        """
        ...

    async def list_sessions(self) -> list[PtyInfo]:
        """List all active PTY sessions.

        Returns:
            List of PtyInfo for all active sessions
        """
        ...


class BasePtyManager:
    """Base class with common functionality for PTY managers.

    Subclasses should implement the abstract methods for their specific
    execution environment.
    """

    def __init__(self) -> None:
        """Initialize the PTY manager."""
        self._sessions: dict[str, PtyInfo] = {}

    def _generate_id(self) -> str:
        """Generate a unique PTY session ID."""
        import uuid

        return f"pty_{uuid.uuid4().hex[:12]}"

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        return self._sessions.get(pty_id)

    async def list_sessions(self) -> list[PtyInfo]:
        """List all active PTY sessions."""
        return list(self._sessions.values())

    @abstractmethod
    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session."""
        ...

    @abstractmethod
    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session."""
        ...

    @abstractmethod
    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin."""
        ...

    @abstractmethod
    async def read(self, pty_id: str, size: int = 4096) -> bytes:
        """Read data from a PTY's stdout."""
        ...

    @abstractmethod
    def stream(self, pty_id: str) -> AsyncIterator[bytes]:
        """Stream output from a PTY session."""
        ...

    @abstractmethod
    async def kill(self, pty_id: str) -> bool:
        """Kill a PTY session."""
        ...
