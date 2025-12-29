"""Docker PTY manager using Docker exec with TTY.

This module provides PTY support for Docker container execution
using the Docker SDK's exec_create/exec_start with tty=True.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from docker.models.containers import Container  # type: ignore[import-untyped]


@dataclass
class DockerPtySession:
    """Tracks a Docker PTY session."""

    info: PtyInfo
    exec_id: str
    container: Container
    socket: object  # Docker socket for the exec session
    _output_buffer: list[bytes] = field(default_factory=list)
    _output_event: asyncio.Event = field(default_factory=asyncio.Event)
    _reader_task: asyncio.Task[None] | None = None


class DockerPtyManager(BasePtyManager):
    """PTY manager for Docker container execution.

    Uses Docker's exec_create/exec_start API with tty=True for
    interactive terminal sessions inside containers.

    Key Docker API features used:
        - container.exec_run(tty=True, stdin=True, socket=True)
        - api_client.exec_resize(exec_id, height, width)
        - socket.sendall() / socket.recv()
    """

    def __init__(self, container: Container) -> None:
        """Initialize the Docker PTY manager.

        Args:
            container: A running Docker container instance
        """
        super().__init__()
        self._container = container
        self._docker_sessions: dict[str, DockerPtySession] = {}

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a new PTY session in the Docker container.

        Args:
            size: Initial terminal size (defaults to 24x80)
            command: Shell command (defaults to /bin/bash)
            args: Arguments for the command
            cwd: Working directory
            env: Environment variables

        Returns:
            PtyInfo with session details
        """
        import docker  # type: ignore[import-untyped]

        size = size or PtySize()
        command = command or "/bin/bash"
        args = args or []
        cwd = cwd or "/workspace"

        # Build full command with args
        cmd_list = [command, *args]

        # Prepend cd if cwd is specified
        if cwd:
            cmd_str = " ".join(cmd_list)
            full_cmd = f"cd {cwd} && {cmd_str}"
        else:
            full_cmd = " ".join(cmd_list)

        pty_id = self._generate_id()

        # Get Docker API client
        client = docker.from_env()
        api_client = client.api

        # Build environment list
        env_list = [f"{k}={v}" for k, v in (env or {}).items()]
        env_list.append("TERM=xterm-256color")

        # Create exec instance with TTY
        exec_instance = api_client.exec_create(
            self._container.id,
            cmd=["sh", "-c", full_cmd],
            tty=True,
            stdin=True,
            environment=env_list,
        )
        exec_id = exec_instance["Id"]

        # Start exec with socket for interactive I/O
        socket = api_client.exec_start(
            exec_id,
            tty=True,
            socket=True,
            demux=False,
        )

        # Resize to requested size
        api_client.exec_resize(exec_id, height=size.rows, width=size.cols)

        # Docker doesn't expose PID of exec'd process easily
        info = PtyInfo(
            id=pty_id,
            pid=0,  # Docker exec doesn't expose PID
            command=command,
            args=args,
            cwd=cwd,
            size=size,
            status="running",
        )

        output_buffer: list[bytes] = []
        output_event = asyncio.Event()

        session = DockerPtySession(
            info=info,
            exec_id=exec_id,
            container=self._container,
            socket=socket,
            _output_buffer=output_buffer,
            _output_event=output_event,
        )
        self._sessions[pty_id] = info
        self._docker_sessions[pty_id] = session

        # Start background reader task
        session._reader_task = asyncio.create_task(self._read_socket(session))

        return info

    async def _read_socket(self, session: DockerPtySession) -> None:
        """Background task to read from Docker socket."""
        loop = asyncio.get_event_loop()
        sock = session.socket._sock  # type: ignore[attr-defined]  # Get underlying socket

        try:
            while session.info.status == "running":
                try:
                    # Read in executor since socket is blocking
                    data = await loop.run_in_executor(None, sock.recv, 4096)
                    if data:
                        session._output_buffer.append(data)
                        session._output_event.set()
                    else:
                        # Socket closed
                        break
                except Exception:  # noqa: BLE001
                    break
        finally:
            session.info.status = "exited"

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a PTY session.

        Args:
            pty_id: The PTY session ID
            size: New terminal size

        Raises:
            KeyError: If PTY session not found
        """
        import docker

        session = self._docker_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        # Use Docker API to resize
        client = docker.from_env()
        client.api.exec_resize(session.exec_id, height=size.rows, width=size.cols)
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a PTY's stdin.

        Args:
            pty_id: The PTY session ID
            data: Data to write

        Raises:
            KeyError: If PTY session not found
        """
        session = self._docker_sessions.get(pty_id)
        if not session:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        loop = asyncio.get_event_loop()
        sock = session.socket._sock  # type: ignore[attr-defined]
        await loop.run_in_executor(None, sock.sendall, data)

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
        session = self._docker_sessions.get(pty_id)
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
        session = self._docker_sessions.get(pty_id)
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
        session = self._docker_sessions.get(pty_id)
        if not session:
            return False

        try:
            # Cancel reader task
            if session._reader_task:
                session._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await session._reader_task

            # Close the socket
            with contextlib.suppress(Exception):
                session.socket._sock.close()  # type: ignore[attr-defined]

            session.info.status = "exited"
        except Exception:  # noqa: BLE001
            pass

        # Cleanup
        del self._docker_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        session = self._docker_sessions.get(pty_id)
        if not session:
            return None

        # Check exec status to update exit code
        import docker

        try:
            client = docker.from_env()
            exec_info = client.api.exec_inspect(session.exec_id)
            if exec_info.get("ExitCode") is not None:
                session.info.status = "exited"
                session.info.exit_code = exec_info["ExitCode"]
        except Exception:  # noqa: BLE001
            pass

        return session.info
