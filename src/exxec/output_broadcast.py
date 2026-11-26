"""Fan-out of a process handle's single-owner output stream.

`ProcessHandle.output()` may only be consumed once. `OutputBroadcaster` is that one
consumer: it reads continuously and delivers every chunk to any number of
subscribers, keeping a bounded backlog so late subscribers can replay recent output.
"""

import asyncio
import collections
import contextlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.process_handle import ProcessHandle


DEFAULT_BACKLOG_LIMIT = 1024 * 1024
"""Bytes of recent output retained for replay."""


class OutputBroadcaster:
    """Reads a process' output once and distributes it to subscribers."""

    def __init__(
        self, handle: ProcessHandle, *, backlog_limit: int = DEFAULT_BACKLOG_LIMIT
    ) -> None:
        """Start reading the handle's output.

        Args:
            handle: Process whose output to distribute (claims its output stream)
            backlog_limit: Maximum bytes of recent output retained for replay
        """
        self._backlog: collections.deque[bytes] = collections.deque()
        self._backlog_size = 0
        self._backlog_limit = backlog_limit
        self._subscribers: set[asyncio.Queue[bytes | None]] = set()
        self._finished = asyncio.Event()
        self._error: BaseException | None = None
        self._task = asyncio.create_task(self._pump(handle.output()))

    @property
    def finished(self) -> bool:
        """Whether the output stream reached EOF (or failed)."""
        return self._finished.is_set()

    @property
    def backlog(self) -> bytes:
        """Recent output, up to the backlog limit."""
        return b"".join(self._backlog)

    async def wait_finished(self) -> None:
        """Wait until the output stream reached EOF.

        Raises:
            BaseException: The error that terminated the output stream, if any
        """
        await self._finished.wait()
        if self._error is not None:
            raise self._error

    def _remember(self, chunk: bytes) -> None:
        self._backlog.append(chunk)
        self._backlog_size += len(chunk)
        while self._backlog_size > self._backlog_limit and len(self._backlog) > 1:
            self._backlog_size -= len(self._backlog.popleft())

    async def _pump(self, output: AsyncIterator[bytes]) -> None:
        try:
            async for chunk in output:
                self._remember(chunk)
                for queue in self._subscribers:
                    queue.put_nowait(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            # Stored and re-raised to waiters; subscribers simply see EOF.
            self._error = e
        finally:
            self._finished.set()
            for queue in self._subscribers:
                queue.put_nowait(None)

    async def subscribe(self, *, replay: bool = True) -> AsyncIterator[bytes]:
        """Iterate over output chunks until EOF.

        Args:
            replay: First yield the retained backlog (output produced before
                subscribing, up to the backlog limit)

        Yields:
            Output chunks
        """
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        if replay and self._backlog:
            queue.put_nowait(self.backlog)
        if self._finished.is_set():
            queue.put_nowait(None)
        else:
            self._subscribers.add(queue)
        try:
            while (chunk := await queue.get()) is not None:
                yield chunk
        finally:
            self._subscribers.discard(queue)

    async def aclose(self) -> None:
        """Stop reading; subscribers receive EOF."""
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
