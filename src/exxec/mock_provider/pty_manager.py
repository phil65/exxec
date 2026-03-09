"""Mock PTY manager for testing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from exxec.pty_manager import BasePtyManager, PtyInfo, PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class MockPtyManager(BasePtyManager):
    """Mock PTY manager for testing.

    Provides predictable behavior without actual PTY operations.
    Useful for unit tests and simulating PTY interactions.
    """

    def __init__(self) -> None:
        """Initialize the mock PTY manager."""
        super().__init__()
        self._input_history: dict[str, list[bytes]] = {}
        self._output_queues: dict[str, asyncio.Queue[bytes]] = {}

    async def create(
        self,
        size: PtySize | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> PtyInfo:
        """Create a mock PTY session."""
        size = size or PtySize()
        command = command or "/bin/bash"
        args = args or []
        cwd = cwd or "/home/user"

        pty_id = self._generate_id()

        info = PtyInfo(
            id=pty_id,
            pid=12345,  # Fake PID
            command=command,
            args=args,
            cwd=cwd,
            size=size,
            status="running",
        )

        self._sessions[pty_id] = info
        self._input_history[pty_id] = []
        self._output_queues[pty_id] = asyncio.Queue()

        # Send initial prompt
        await self._output_queues[pty_id].put(b"$ ")

        return info

    async def resize(self, pty_id: str, size: PtySize) -> None:
        """Resize a mock PTY session."""
        if pty_id not in self._sessions:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        self._sessions[pty_id].size = size

    async def write(self, pty_id: str, data: bytes) -> None:
        """Write data to a mock PTY."""
        if pty_id not in self._sessions:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        self._input_history[pty_id].append(data)

        # Echo input and simulate response
        text = data.decode() if isinstance(data, bytes) else data
        queue = self._output_queues[pty_id]

        # Echo the input
        await queue.put(data)

        # If it looks like a command (ends with newline), send a fake response
        if text.endswith("\n"):
            cmd = text.strip()
            if cmd:
                response = f"mock output for: {cmd}\n$ ".encode()
                await queue.put(response)

    async def read(self, pty_id: str, size: int = 4096) -> bytes:
        """Read data from a mock PTY."""
        if pty_id not in self._sessions:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        queue = self._output_queues[pty_id]

        try:
            data = await asyncio.wait_for(queue.get(), timeout=0.1)
            return data[:size] if len(data) > size else data
        except TimeoutError:
            return b""

    async def stream(self, pty_id: str) -> AsyncIterator[bytes]:
        """Stream output from a mock PTY session."""
        if pty_id not in self._sessions:
            msg = f"PTY session {pty_id} not found"
            raise KeyError(msg)

        queue = self._output_queues[pty_id]

        while self._sessions.get(pty_id, PtyInfo(id="", pid=0, command="")).status == "running":
            try:
                data = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield data
            except TimeoutError:
                continue

    async def kill(self, pty_id: str) -> bool:
        """Kill a mock PTY session."""
        if pty_id not in self._sessions:
            return False

        self._sessions[pty_id].status = "exited"
        self._sessions[pty_id].exit_code = 0

        del self._sessions[pty_id]
        del self._input_history[pty_id]
        del self._output_queues[pty_id]

        return True

    def get_input_history(self, pty_id: str) -> list[bytes]:
        """Get all input that was written to a PTY session.

        Useful for testing to verify what was sent.
        """
        return self._input_history.get(pty_id, [])
