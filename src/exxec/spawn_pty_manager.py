"""PTY manager built on `ExecutionEnvironment.spawn()`.

Works for every environment with `supports_pty`. Each session owns one
`ProcessHandle` and one `OutputBroadcaster`, so `stream()` may be called by several
consumers concurrently and `read()` does not steal bytes from streams.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import uuid

from exxec.output_broadcast import OutputBroadcaster
from exxec.pty_manager import PtyInfo, PtyManagerProtocol, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.base import ExecutionEnvironment
    from exxec.process_handle import ProcessHandle


DEFAULT_SHELL_COMMAND = 'exec "${SHELL:-$(command -v bash || command -v sh)}" -l'
"""Shell command line used when no command is given: the user's login shell."""


@dataclass
class _SpawnPtySession:
    info: PtyInfo
    handle: ProcessHandle
    broadcaster: OutputBroadcaster
    exit_watcher: asyncio.Task[None] | None = None
    read_stream: AsyncIterator[bytes] | None = None
    read_leftover: bytes = b""
    read_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SpawnPtyManager(PtyManagerProtocol):
    """PTY sessions for any environment implementing `spawn(pty=...)`."""

    def __init__(self, env: ExecutionEnvironment) -> None:
        """Initialize for an execution environment.

        Args:
            env: Environment with `supports_pty`
        """
        self.env = env
        self._sessions: dict[str, _SpawnPtySession] = {}

    def _get(self, pty_id: str) -> _SpawnPtySession:
        if (session := self._sessions.get(pty_id)) is None:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)
        return session

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Start a process attached to a new pseudo-terminal.

        Args:
            size: Terminal size (defaults to 24x80)
            command: Program to run (defaults to the user's login shell); a shell
                command line when no args are given
            args: Program arguments
            cwd: Working directory
            env: Additional environment variables

        Returns:
            Session information
        """
        size = size or PtySize()
        handle = await self.env.spawn(
            command or DEFAULT_SHELL_COMMAND, args or None, cwd=cwd, env=env, pty=size
        )
        pty_id = f"pty_{uuid.uuid4().hex[:12]}"
        info = PtyInfo(
            id=pty_id,
            pid=handle.pid or 0,
            command=command or "shell",
            args=args or [],
            cwd=cwd,
            size=size,
        )
        session = _SpawnPtySession(info=info, handle=handle, broadcaster=OutputBroadcaster(handle))
        session.exit_watcher = asyncio.create_task(self._watch_exit(session))
        self._sessions[pty_id] = session
        return info

    async def _watch_exit(self, session: _SpawnPtySession) -> None:
        await session.handle.wait()
        _sync_info(session)

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a session's terminal."""
        session = self._get(pty_id)
        await session.handle.resize(size)
        session.info.size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write input to a session's terminal."""
        await self._get(pty_id).handle.write(data)

    async def read(self, pty_id: str, size: int = 4096) -> bytes:
        """Read up to `size` bytes of output; b"" once the session's output ended.

        The first call replays output retained since the session started.
        """
        session = self._get(pty_id)
        async with session.read_lock:
            if not session.read_leftover:
                if session.read_stream is None:
                    session.read_stream = session.broadcaster.subscribe(replay=True)
                session.read_leftover = await anext(session.read_stream, b"")
            data, session.read_leftover = (
                session.read_leftover[:size],
                session.read_leftover[size:],
            )
            return data

    async def stream(self, pty_id: str) -> AsyncIterator[bytes]:
        """Stream output until the session ends, starting with retained output."""
        session = self._get(pty_id)
        async for chunk in session.broadcaster.subscribe(replay=True):
            yield chunk

    async def wait(self, pty_id: str) -> int:
        """Wait for the session's process to exit and return its exit code."""
        session = self._get(pty_id)
        exit_code = await session.handle.wait()
        _sync_info(session)
        return exit_code

    async def kill(self, pty_id: str) -> bool:
        """Terminate a session's process and drop the session.

        Returns:
            True if the session existed
        """
        session = self._sessions.pop(pty_id, None)
        if session is None:
            return False
        await session.handle.aclose()
        await session.broadcaster.aclose()
        if session.exit_watcher is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await session.exit_watcher
        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get session information, None if unknown."""
        session = self._sessions.get(pty_id)
        if session is None:
            return None
        _sync_info(session)
        return session.info

    async def list_sessions(self) -> list[PtyInfo]:
        """List all sessions."""
        for session in self._sessions.values():
            _sync_info(session)
        return [session.info for session in self._sessions.values()]

    async def cleanup(self) -> None:
        """Kill all sessions."""
        for pty_id in list(self._sessions):
            await self.kill(pty_id)


def _sync_info(session: _SpawnPtySession) -> None:
    """Copy exit status and late-known PID from the handle into the session info."""
    if session.handle.returncode is not None:
        session.info.status = "exited"
        session.info.exit_code = session.handle.returncode
    if session.info.pid == 0 and session.handle.pid is not None:
        session.info.pid = session.handle.pid
