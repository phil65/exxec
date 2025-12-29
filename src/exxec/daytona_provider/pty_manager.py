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

    from daytona._async.sandbox import AsyncSandbox  # type: ignore[import-untyped]


@dataclass
class DaytonaPtySession:
    """Tracks a Daytona PTY session."""

    info: PtyInfo
    handle: object  # Daytona PTY handle
    sandbox: AsyncSandbox
    _output_buffer: list[bytes] = field(default_factory=list)
    _output_event: asyncio.Event = field(default_factory=asyncio.Event)
    _reader_task: asyncio.Task[None] | None = None


class DaytonaPtyManager(BasePtyManager):
    """PTY manager for Daytona cloud sandbox execution.

    Uses Daytona's process.create_pty_session API for interactive
    terminal sessions with full resize support.

    Key Daytona API features used:
        - sandbox.process.create_pty_session(id, pty_size)
        - handle.send_input(data)
        - handle.resize(PtySize)
        - handle.kill()
        - handle.wait()
        - Iteration over handle for output streaming
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
            command: Not directly used - Daytona creates a shell session.
                     Send commands via send_input after creation.
            args: Not used for Daytona PTY
            cwd: Working directory - change via cd command after session start
            env: Not directly supported - set via export commands

        Returns:
            PtyInfo with session details
        """
        from daytona.common.pty import PtySize as DaytonaPtySize  # type: ignore[import-untyped]

        size = size or PtySize()
        command = command or "/bin/bash"
        args = args or []
        cwd = cwd or "/home/daytona"

        pty_id = self._generate_id()

        # Create Daytona PtySize object
        daytona_size = DaytonaPtySize(cols=size.cols, rows=size.rows)

        # Create PTY session using Daytona API
        handle = await self._sandbox.process.create_pty_session(  # pyright: ignore[reportAttributeAccessIssue]
            id=pty_id,
            pty_size=daytona_size,
        )

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

        session = DaytonaPtySession(
            info=info,
            handle=handle,
            sandbox=self._sandbox,
            _output_buffer=output_buffer,
            _output_event=output_event,
        )
        self._sessions[pty_id] = info
        self._daytona_sessions[pty_id] = session

        # Start background reader task
        session._reader_task = asyncio.create_task(self._read_output(session))

        # If cwd specified, send cd command
        if cwd:
            await self.write(pty_id, f"cd {cwd}\n".encode())

        # If command specified (not default shell), send it
        if command and command != "/bin/bash":
            full_cmd = f"{command} {' '.join(args)}\n" if args else f"{command}\n"
            await self.write(pty_id, full_cmd.encode())

        return info

    async def _read_output(self, session: DaytonaPtySession) -> None:
        """Background task to read from Daytona PTY handle."""
        loop = asyncio.get_event_loop()

        try:
            # Daytona handle supports iteration for output
            # Run in executor since it may block
            def read_sync() -> bytes | None:
                try:
                    for data in session.handle:  # type: ignore[attr-defined]
                        return data  # type: ignore[no-any-return]
                except StopIteration:
                    return None
                except Exception:  # noqa: BLE001
                    return None
                return None

            while session.info.status == "running":
                data = await loop.run_in_executor(None, read_sync)
                if data:
                    session._output_buffer.append(data)
                    session._output_event.set()
                else:
                    # End of stream
                    break
        except Exception:  # noqa: BLE001
            pass
        finally:
            session.info.status = "exited"
            # Try to get exit code
            try:
                result = await loop.run_in_executor(None, session.handle.wait)  # type: ignore[attr-defined]
                session.info.exit_code = result.exit_code
            except Exception:  # noqa: BLE001
                pass

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

        # Resize using Daytona API
        loop = asyncio.get_event_loop()
        daytona_size = DaytonaPtySize(cols=size.cols, rows=size.rows)
        await loop.run_in_executor(None, session.handle.resize, daytona_size)  # type: ignore[attr-defined]
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
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, session.handle.send_input, text)  # type: ignore[attr-defined]

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
            # Cancel reader task
            if session._reader_task:
                session._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await session._reader_task

            # Kill using Daytona API
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, session.handle.kill)  # type: ignore[attr-defined]

            session.info.status = "exited"
        except Exception:  # noqa: BLE001
            pass

        # Cleanup
        del self._daytona_sessions[pty_id]
        del self._sessions[pty_id]

        return True

    async def get_info(self, pty_id: str) -> PtyInfo | None:
        """Get information about a PTY session."""
        return self._sessions.get(pty_id)
