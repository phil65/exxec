"""Tests for `DockerExecutionEnvironment.spawn()` against a real container."""

from __future__ import annotations

import asyncio
import platform
import shutil
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from exxec.pty_manager import PtySize


pytest.importorskip("testcontainers")

from exxec import DockerExecutionEnvironment


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.process_handle import ProcessHandle


pytestmark = [
    pytest.mark.skipif(platform.system() != "Linux", reason="Docker tests require Linux"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available"),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def docker_env() -> AsyncIterator[DockerExecutionEnvironment]:
    async with DockerExecutionEnvironment(env_vars={"BASE": "b"}) as env:
        yield env


async def read_all(handle: ProcessHandle) -> bytes:
    return b"".join([chunk async for chunk in handle.output()])


async def read_until(iterator: AsyncIterator[bytes], marker: bytes, timeout: float = 10) -> bytes:
    buffer = b""

    async def consume() -> None:
        nonlocal buffer
        async for chunk in iterator:
            buffer += chunk
            if marker in buffer:
                return

    await asyncio.wait_for(consume(), timeout=timeout)
    return buffer


async def test_capabilities() -> None:
    assert DockerExecutionEnvironment.supports_spawn is True
    assert DockerExecutionEnvironment.supports_pty is True


async def test_pipe_output_cwd_env_and_exit_code(docker_env: DockerExecutionEnvironment) -> None:
    script = (
        "import os, sys\n"
        "print(os.getcwd(), flush=True)\n"
        "print(os.environ['BASE'], os.environ['CALL'], sys.argv[1], flush=True)\n"
        "print('to stderr', file=sys.stderr, flush=True)\n"
        "sys.exit(5)\n"
    )
    handle = await docker_env.spawn("python", ["-c", script, "a 'b' c"], env={"CALL": "x y"})
    async with handle:
        assert handle.is_pty is False
        lines = (await read_all(handle)).decode().splitlines()
        assert await handle.wait() == 5  # noqa: PLR2004
    # Docker delivers stdout and stderr on separate streams: no relative ordering.
    assert sorted(lines) == sorted(["/workspace", "b x y a 'b' c", "to stderr"])
    assert lines.index("/workspace") < lines.index("b x y a 'b' c")


async def test_pipe_stdin_roundtrip(docker_env: DockerExecutionEnvironment) -> None:
    script = "import sys; print(sys.stdin.read().upper(), end='')"
    async with await docker_env.spawn("python", ["-c", script]) as handle:
        await handle.write(b"in docker")
        await handle.close_stdin()
        assert await read_all(handle) == b"IN DOCKER"
        assert await handle.wait() == 0


async def test_shell_command_line_and_cwd(docker_env: DockerExecutionEnvironment) -> None:
    async with await docker_env.spawn("pwd && echo a | tr a b", cwd="/tmp") as handle:
        assert (await read_all(handle)).decode().split() == ["/tmp", "b"]
        assert await handle.wait() == 0


async def test_large_output_is_demultiplexed(docker_env: DockerExecutionEnvironment) -> None:
    script = "import sys; sys.stdout.write('x' * 300000); sys.stdout.flush()"
    async with await docker_env.spawn("python", ["-c", script]) as handle:
        output = await read_all(handle)
        assert await handle.wait() == 0
    assert output == b"x" * 300000


async def test_kill_with_sigterm(docker_env: DockerExecutionEnvironment) -> None:
    script = "import time; print('ready', flush=True); time.sleep(60)"
    async with await docker_env.spawn("python", ["-c", script]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=5) == 143  # noqa: PLR2004
        assert handle.pid is not None


async def test_kill_escalates_to_sigkill(docker_env: DockerExecutionEnvironment) -> None:
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    async with await docker_env.spawn("python", ["-c", script]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=0.5) == 137  # noqa: PLR2004


async def test_pty_size_term_and_resize(docker_env: DockerExecutionEnvironment) -> None:
    script = (
        "import os, signal, sys\n"
        "size = os.get_terminal_size()\n"
        "print('term', os.environ['TERM'], sys.stdout.isatty(), flush=True)\n"
        "print('size', size.lines, size.columns, flush=True)\n"
        "def on_winch(*_):\n"
        "    size = os.get_terminal_size()\n"
        "    print('resized', size.lines, size.columns, flush=True)\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGWINCH, on_winch)\n"
        "print('ready', flush=True)\n"
        "signal.pause()\n"
    )
    size = PtySize(rows=33, cols=77)
    async with await docker_env.spawn("python", ["-c", script], pty=size) as handle:
        assert handle.is_pty is True
        iterator = handle.output()
        before = await read_until(iterator, b"ready")
        await handle.resize(PtySize(rows=44, cols=144))
        after = b"".join([chunk async for chunk in iterator])
        assert await handle.wait() == 0
    assert "term xterm-256color True" in before.decode()
    assert "size 33 77" in before.decode()
    assert "resized 44 144" in after.decode()


async def test_pty_input_and_eof(docker_env: DockerExecutionEnvironment) -> None:
    script = "import sys; print('got:', input()); data = sys.stdin.read(); print('rest', len(data))"
    async with await docker_env.spawn("python", ["-c", script], pty=PtySize()) as handle:
        await handle.write(b"typed\r")
        await handle.write(b"ab\r")
        await handle.close_stdin()
        text = (await read_all(handle)).decode()
        assert await handle.wait() == 0
    assert "got: typed" in text
    assert "rest 3" in text


async def test_context_exit_kills_process(docker_env: DockerExecutionEnvironment) -> None:
    handle = await docker_env.spawn("python", ["-c", "import time; time.sleep(60)"])
    async with handle:
        pass
    assert handle.returncode == 143  # noqa: PLR2004


async def test_process_manager_runs_in_container(docker_env: DockerExecutionEnvironment) -> None:
    from exxec.process_manager import SpawnProcessManager

    pm = docker_env.process_manager
    assert isinstance(pm, SpawnProcessManager)
    process_id = await pm.start_process("test -f /.dockerenv && echo inside")
    assert await pm.wait_for_exit(process_id) == 0
    assert (await pm.get_output(process_id)).combined.strip() == "inside"
    long_running = await pm.start_process("python", args=["-c", "import time; time.sleep(60)"])
    await pm.kill_process(long_running)
    assert (await pm.get_process_info(long_running))["is_running"] is False
    await pm.release_process(process_id)
    await pm.release_process(long_running)


async def test_pty_manager_in_container(docker_env: DockerExecutionEnvironment) -> None:
    from exxec.spawn_pty_manager import SpawnPtyManager

    manager = docker_env.get_pty_manager()
    assert isinstance(manager, SpawnPtyManager)
    info = await manager.create()
    await manager.write(info.id, b"echo container-$((6 * 7))\r")
    assert b"container-42" in await read_until(manager.stream(info.id), b"container-42")
    assert await manager.kill(info.id) is True
