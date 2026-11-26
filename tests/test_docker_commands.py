"""Docker command/code paths (quoting, exit codes, timeouts, event loop) on a real container."""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio


pytest.importorskip("testcontainers")

from exxec import DockerExecutionEnvironment
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _docker_daemon_available() -> bool:
    if platform.system() != "Linux" or shutil.which("docker") is None:
        return False
    try:
        out = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
    except subprocess.TimeoutExpired:
        return False
    return out.returncode == 0


pytestmark = [
    pytest.mark.skipif(not _docker_daemon_available(), reason="Docker daemon not available"),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def docker_env() -> AsyncIterator[DockerExecutionEnvironment]:
    async with DockerExecutionEnvironment(default_command_timeout=20) as env:
        yield env


async def test_execute_command_keeps_quoting(docker_env: DockerExecutionEnvironment):
    result = await docker_env.execute_command("echo 'a   b'")
    assert result.stdout == "a   b\n"
    result = await docker_env.execute_command('echo "it\'s"')
    assert result.success, result.error
    assert result.stdout == "it's\n"


async def test_execute_command_runs_in_workspace(docker_env: DockerExecutionEnvironment):
    result = await docker_env.execute_command("pwd")
    assert result.stdout == "/workspace\n"


async def test_execute_command_timeout(docker_env: DockerExecutionEnvironment):
    result = await docker_env.execute_command("sleep 30", timeout=0.5)
    assert result.error_type == "TimeoutError"
    list_processes = r"for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline; echo; done"
    check = await docker_env.execute_command(f"{list_processes} | grep -c '^sleep 30' || true")
    assert check.stdout is not None
    assert check.stdout.strip() == "0"


async def test_execute_command_non_utf8(docker_env: DockerExecutionEnvironment):
    result = await docker_env.execute_command(r"printf 'ok\377\n'; echo more")
    assert result.success, result.error
    assert result.stdout == "ok�\nmore\n"


async def test_stream_command_exit_code(docker_env: DockerExecutionEnvironment):
    events = [e async for e in docker_env.stream_command("echo out; exit 3")]
    assert [e.data for e in events if isinstance(e, OutputEvent)] == ["out"]
    assert isinstance(events[-1], ProcessErrorEvent)
    assert events[-1].exit_code == 3  # noqa: PLR2004


async def test_stream_command_builtins(docker_env: DockerExecutionEnvironment):
    events = [e async for e in docker_env.stream_command("cd /tmp && pwd")]
    assert [e.data for e in events if isinstance(e, OutputEvent)] == ["/tmp"]
    assert isinstance(events[-1], ProcessCompletedEvent)


async def test_execute_does_not_block_the_event_loop(docker_env: DockerExecutionEnvironment):
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        result = await docker_env.execute("import time\ntime.sleep(1)\n_result = 1")
    finally:
        task.cancel()
    assert result.success, result.error
    assert result.result == 1
    assert ticks > 20  # noqa: PLR2004


async def test_concurrent_executes_use_their_own_scripts(docker_env: DockerExecutionEnvironment):
    codes = [f"import time\ntime.sleep(0.5)\n_result = {i}" for i in range(3)]
    results = await asyncio.gather(*(docker_env.execute(c) for c in codes))
    assert [r.result for r in results] == [0, 1, 2]


async def test_execute_timeout(docker_env: DockerExecutionEnvironment):
    env = docker_env
    old = env.default_command_timeout
    env.default_command_timeout = 0.5
    try:
        result = await env.execute("import time\ntime.sleep(30)")
    finally:
        env.default_command_timeout = old
    assert not result.success
    assert result.error_type == "TimeoutError"


async def test_stream_code_exit_code(docker_env: DockerExecutionEnvironment):
    events = [e async for e in docker_env.stream_code("print('hi')\nraise SystemExit(4)")]
    assert any(isinstance(e, OutputEvent) and e.data == "hi" for e in events)
    assert isinstance(events[-1], ProcessErrorEvent)
    assert events[-1].exit_code == 4  # noqa: PLR2004
