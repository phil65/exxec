"""Tests for StreamCapture (local provider output forwarding)."""

import asyncio
import io

from exxec.local_provider.utils import StreamCapture


def test_stream_capture_forwards_str_and_bytes() -> None:
    """Both str and bytes writes reach the original stream and the queue."""
    original = io.StringIO()
    queue: asyncio.Queue[str] = asyncio.Queue()
    capture = StreamCapture(original, queue)

    capture.write("hello\n")
    capture.write(b"world\n")

    assert original.getvalue() == "hello\nworld\n"
    assert [queue.get_nowait(), queue.get_nowait()] == ["hello", "world"]
    assert queue.empty()


async def test_stream_capture_hands_thread_writes_to_the_loop() -> None:
    """Writes from a worker thread reach a waiting getter via the event loop."""
    original = io.StringIO()
    queue: asyncio.Queue[str] = asyncio.Queue()
    capture = StreamCapture(original, queue)
    getter = asyncio.ensure_future(queue.get())
    await asyncio.sleep(0)
    calls: list[object] = []
    loop = asyncio.get_running_loop()
    real = loop.call_soon_threadsafe

    def recording(*args: object) -> object:
        calls.append(args)
        return real(*args)  # type: ignore[arg-type]

    loop.call_soon_threadsafe = recording  # type: ignore[method-assign]
    try:
        await asyncio.to_thread(capture.write, "from thread\n")
        assert await asyncio.wait_for(getter, timeout=1) == "from thread"
    finally:
        del loop.call_soon_threadsafe
    assert any(getattr(c[0], "__name__", "") == "_put" for c in calls)  # type: ignore[index]
