"""Tests for the spawn-based process and PTY managers and the output broadcaster."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import pytest

from exxec import LocalExecutionEnvironment
from exxec.output_broadcast import OutputBroadcaster
from exxec.process_manager import SpawnProcessManager
from exxec.pty_manager import PtySize
from exxec.spawn_pty_manager import SpawnPtyManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX PTYs")
PY = sys.executable


async def collect_until(stream: AsyncIterator[bytes], marker: bytes, timeout: float = 5) -> bytes:
    buffer = b""

    async def consume() -> None:
        nonlocal buffer
        async for chunk in stream:
            buffer += chunk
            if marker in buffer:
                return

    await asyncio.wait_for(consume(), timeout=timeout)
    return buffer


async def test_process_manager_has_no_implicit_timeout() -> None:
    async with LocalExecutionEnvironment(default_command_timeout=0.3) as env:
        pm = env.process_manager
        assert isinstance(pm, SpawnProcessManager)
        script = "import time; time.sleep(1); print('survived')"
        process_id = await pm.start_process(PY, args=["-c", script])
        assert await pm.wait_for_exit(process_id) == 0
        assert "survived" in (await pm.get_output(process_id)).combined


async def test_process_manager_rejects_unknown_ids() -> None:
    async with LocalExecutionEnvironment() as env:
        with pytest.raises(ValueError, match="not found"):
            await env.process_manager.get_output("proc_missing")


@posix_only
async def test_broadcaster_fans_out_and_replays() -> None:
    async with LocalExecutionEnvironment() as env:
        script = "import sys, time; print('one', flush=True); time.sleep(0.3); print('two')"
        handle = await env.spawn(PY, ["-c", script])
        async with handle:
            broadcaster = OutputBroadcaster(handle)
            first = broadcaster.subscribe()
            assert b"one" in await collect_until(first, b"one")
            # Joins late: replays the backlog, then continues live.
            late = broadcaster.subscribe(replay=True)
            rest_first = b"".join([chunk async for chunk in first])
            rest_late = b"".join([chunk async for chunk in late])
            await broadcaster.wait_finished()
            assert broadcaster.finished
    assert b"two" in rest_first
    assert rest_late.replace(b"\r", b"") == b"one\ntwo\n"


async def test_broadcaster_backlog_limit() -> None:
    async with LocalExecutionEnvironment() as env:
        script = "import sys; [print('x' * 99, flush=True) for _ in range(50)]"
        async with await env.spawn(PY, ["-c", script]) as handle:
            broadcaster = OutputBroadcaster(handle, backlog_limit=1000)
            await broadcaster.wait_finished()
            backlog = broadcaster.backlog
    # Whole chunks are evicted, so the backlog may exceed the limit by < 1 chunk.
    assert 0 < len(backlog) < 1000 + 5000


@posix_only
async def test_pty_kill_hangs_up_interactive_shell_quickly() -> None:
    async with LocalExecutionEnvironment() as env:
        manager = env.get_pty_manager()
        info = await manager.create(command="bash", args=["--norc", "-i"])
        await asyncio.sleep(0.2)
        loop = asyncio.get_running_loop()
        started = loop.time()
        assert await manager.kill(info.id) is True
        # Interactive bash ignores SIGTERM; SIGHUP ends it without the kill grace period.
        assert loop.time() - started < 2  # noqa: PLR2004


@posix_only
async def test_pty_manager_default_shell_roundtrip() -> None:
    async with LocalExecutionEnvironment() as env:
        manager = env.get_pty_manager()
        assert isinstance(manager, SpawnPtyManager)
        info = await manager.create(size=PtySize(rows=20, cols=60))
        assert info.status == "running"
        assert info.pid > 0
        await manager.write(info.id, b"echo hello-$((40 + 2))\r")
        output = await collect_until(manager.stream(info.id), b"hello-42")
        assert b"hello-42" in output
        assert await manager.kill(info.id) is True
        assert await manager.get_info(info.id) is None
        assert await manager.kill(info.id) is False


@posix_only
async def test_pty_manager_replays_early_output_and_records_exit() -> None:
    async with LocalExecutionEnvironment() as env:
        manager = env.get_pty_manager()
        info = await manager.create(command=PY, args=["-c", "print('early'); raise SystemExit(3)"])
        assert isinstance(manager, SpawnPtyManager)
        assert await manager.wait(info.id) == 3  # noqa: PLR2004
        # Subscribing after the process already finished still yields its output.
        output = b"".join([chunk async for chunk in manager.stream(info.id)])
        assert b"early" in output
        session = await manager.get_info(info.id)
        assert session is not None
        assert session.status == "exited"
        assert session.exit_code == 3  # noqa: PLR2004


@posix_only
async def test_pty_manager_concurrent_streams_and_read() -> None:
    async with LocalExecutionEnvironment() as env:
        manager = env.get_pty_manager()
        script = (
            "import time; [print(f'line{i}', flush=True) or time.sleep(0.02) for i in range(20)]"
        )
        info = await manager.create(command=PY, args=["-c", script])

        async def drain() -> bytes:
            return b"".join([chunk async for chunk in manager.stream(info.id)])

        first, second = await asyncio.gather(drain(), drain())
        read_data = b""
        while chunk := await manager.read(info.id, 7):
            assert len(chunk) <= 7  # noqa: PLR2004
            read_data += chunk
    assert first == second
    assert b"line19" in first
    assert read_data == first


@posix_only
async def test_pty_manager_resize_and_list() -> None:
    async with LocalExecutionEnvironment() as env:
        manager = env.get_pty_manager()
        script = (
            "import os, signal, sys\n"
            "def on_winch(*_):\n"
            "    size = os.get_terminal_size()\n"
            "    print('size', size.lines, size.columns, flush=True)\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGWINCH, on_winch)\n"
            "print('ready', flush=True)\n"
            "signal.pause()\n"
        )
        info = await manager.create(command=PY, args=["-c", script])
        assert [session.id for session in await manager.list_sessions()] == [info.id]
        stream = manager.stream(info.id)
        await collect_until(stream, b"ready")
        await manager.resize(info.id, PtySize(rows=12, cols=34))
        rest = b"".join([chunk async for chunk in stream])
        assert b"size 12 34" in rest
        session = await manager.get_info(info.id)
        assert session is not None
        assert session.size == PtySize(rows=12, cols=34)


@posix_only
async def test_env_exit_kills_pty_sessions() -> None:
    async with LocalExecutionEnvironment() as env:
        manager = env.get_pty_manager()
        info = await manager.create(command=PY, args=["-c", "import time; time.sleep(30)"])
    assert await manager.get_info(info.id) is None


@posix_only
async def test_unknown_pty_session_raises() -> None:
    async with LocalExecutionEnvironment() as env:
        with pytest.raises(KeyError):
            await env.get_pty_manager().write("pty_missing", b"x")


async def test_pty_manager_unavailable_without_pty_support() -> None:
    from exxec.base import ExecutionEnvironment

    class NoPty(LocalExecutionEnvironment):
        supports_pty = False

    assert ExecutionEnvironment.supports_pty is False
    async with NoPty() as env:
        with pytest.raises(NotImplementedError):
            env.get_pty_manager()
