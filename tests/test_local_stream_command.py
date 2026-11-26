"""Tests for `LocalExecutionEnvironment.stream_command()` built on `spawn()`."""

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import pytest

from exxec import LocalExecutionEnvironment
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent


if TYPE_CHECKING:
    from pathlib import Path


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX shell syntax")
PY = sys.executable
WAIT_TIMEOUT = 0x102
SYNCHRONIZE_AND_QUERY = 0x00100000 | 0x1000


async def test_lines_and_completion(tmp_path: Path) -> None:
    script = tmp_path / "emit.py"
    script.write_text("import sys\nsys.stdout.write('a\\r\\nb\\n\\nlast')\n")
    async with LocalExecutionEnvironment() as env:
        events = [event async for event in env.stream_command(f'"{PY}" "{script}"')]
    lines = [event.data for event in events if isinstance(event, OutputEvent)]
    assert lines == ["a", "b", "", "last"]
    assert isinstance(events[-1], ProcessCompletedEvent)


async def test_nonzero_exit_code() -> None:
    async with LocalExecutionEnvironment() as env:
        events = [event async for event in env.stream_command("exit 3")]
    assert isinstance(events[-1], ProcessErrorEvent)
    assert events[-1].exit_code == 3  # noqa: PLR2004
    assert events[-1].error_type == "CommandError"


@posix_only
async def test_honours_env_cwd(tmp_path: Path) -> None:
    async with LocalExecutionEnvironment(cwd=str(tmp_path)) as env:
        lines = [e.data async for e in env.stream_command("pwd") if isinstance(e, OutputEvent)]
    assert os.path.realpath(lines[0]) == os.path.realpath(tmp_path)


@posix_only
async def test_timeout_is_total_and_kills_children() -> None:
    async with LocalExecutionEnvironment() as env:
        # Produces output continuously, so an inactivity timeout would never fire.
        command = "sleep 30 & echo $!; while true; do echo tick; sleep 0.05; done"
        events = [event async for event in env.stream_command(command, timeout=0.5)]
    assert isinstance(events[-1], ProcessErrorEvent)
    assert events[-1].error_type == "TimeoutError"
    child_pid = int(next(e.data for e in events if isinstance(e, OutputEvent)))
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@posix_only
async def test_closing_stream_kills_process() -> None:
    async with LocalExecutionEnvironment(default_command_timeout=None) as env:
        stream = env.stream_command("echo $$; sleep 30")
        pid = None
        async for event in stream:
            if isinstance(event, OutputEvent):
                pid = int(event.data)
                break
        await stream.aclose()  # type: ignore[attr-defined]
    assert pid is not None
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_slow_consumer_does_not_hit_timeout_in_consumer_code() -> None:
    async with LocalExecutionEnvironment() as env:
        events = []
        async for event in env.stream_command("echo one", timeout=5):
            await asyncio.sleep(0.05)  # consumer work must never be cancelled by the timeout
            events.append(event)
    assert isinstance(events[-1], ProcessCompletedEvent)


def _alive(pid: int) -> bool:
    """Whether a process runs (without signalling it: os.kill(pid, 0) kills on Windows)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(SYNCHRONIZE_AND_QUERY, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_closing_stream_kills_descendants_on_every_platform(tmp_path: Path) -> None:
    """A cancelled command leaves nothing behind: not the shell, not what it started."""
    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    async with LocalExecutionEnvironment(default_command_timeout=None) as env:
        stream = env.stream_command(f'"{PY}" "{script}"')
        grandchild = None
        async for event in stream:
            if isinstance(event, OutputEvent):
                grandchild = int(event.data)
                break
        assert grandchild is not None
        assert _alive(grandchild)
        await stream.aclose()  # type: ignore[attr-defined]
    for _ in range(50):
        if not _alive(grandchild):
            break
        await asyncio.sleep(0.1)
    assert not _alive(grandchild)
