"""Tests for StreamCapture (local provider output forwarding)."""

from __future__ import annotations

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
