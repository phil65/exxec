"""Line splitting of spawn output streams."""

import time
from typing import TYPE_CHECKING

import pytest

from exxec.spawn_stream import _iter_lines


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.parametrize(
    ("chunks", "lines"),
    [
        ((b"a\nb", b"c\n", b"d"), ["a", "bc", "d"]),
        ((b"x\r\n", b"\n"), ["x", ""]),
        ((b"\xc3", b"\xa4\n"), ["ä"]),
        ((b"bad\xff\n",), ["bad�"]),
        ((b"", b"a", b"", b"b"), ["ab"]),
    ],
)
async def test_iter_lines(chunks: tuple[bytes, ...], lines: list[str]):
    assert [line async for line in _iter_lines(_chunks(*chunks))] == lines


async def test_iter_lines_long_line_is_linear():
    chunk = b"x" * 65536
    count = 400  # one 26 MB line
    start = time.perf_counter()
    lines = [line async for line in _iter_lines(_chunks(*[chunk] * count, b"\n"))]
    assert len(lines) == 1
    assert len(lines[0]) == len(chunk) * count
    assert time.perf_counter() - start < 1.0
