"""Local / SRT environments: sandbox wrapping of code, timeouts that kill, decoding."""

import asyncio
import os
import shutil
import subprocess
import sys
import uuid

import pytest

from exxec import LocalExecutionEnvironment
from exxec.srt_provider import SRTExecutionEnvironment
from exxec_config import SandboxConfig


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


class MarkingEnvironment(LocalExecutionEnvironment):
    """Stands in for a sandbox: wrapped processes see SANDBOXED=1."""

    def wrap_argv(self, argv: list[str]) -> list[str]:
        return ["env", "SANDBOXED=1", *argv]

    def wrap_command(self, command: str) -> str:
        return f"SANDBOXED=1 {command}"


SANDBOX_CHECK = "import os\n_result = os.environ.get('SANDBOXED')"


@posix_only
async def test_execute_goes_through_wrap_argv():
    async with MarkingEnvironment(isolated=True) as env:
        result = await env.execute(SANDBOX_CHECK)
    assert result.success, result.error
    assert result.result == "1"


@posix_only
async def test_stream_code_goes_through_wrap_argv():
    code = "import os\nprint('marker', os.environ.get('SANDBOXED'))"
    async with MarkingEnvironment(isolated=True) as env:
        events = [e async for e in env.stream_code(code)]
    assert any(getattr(e, "data", "") == "marker 1" for e in events), events


def _srt_works() -> bool:
    if not shutil.which("srt"):
        return False
    env = SRTExecutionEnvironment(SandboxConfig())
    try:
        out = subprocess.run(env.wrap_argv(["true"]), capture_output=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        return False
    return out.returncode == 0


@pytest.mark.skipif(not _srt_works(), reason="srt not installed or not usable here")
async def test_srt_execute_honours_deny_read(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    config = SandboxConfig(deny_read=[str(tmp_path)])
    code = f"_result = open({str(secret)!r}).read()"
    async with SRTExecutionEnvironment(config) as env:
        result = await env.execute(code)
    assert result.result != "top secret"
    assert not result.success


def _alive(marker: str) -> bool:
    out = subprocess.run(["pgrep", "-f", marker], capture_output=True, check=False)
    return out.returncode == 0


@posix_only
async def test_execute_command_timeout_kills_the_process_group():
    marker = f"41.{uuid.uuid4().int % 10**6}"
    env = LocalExecutionEnvironment()
    result = await env.execute_command(f"sleep {marker}; echo hi", timeout=0.5)
    assert result.error_type == "TimeoutError"
    await asyncio.sleep(0.2)
    assert not _alive(f"sleep {marker}")


@posix_only
async def test_isolated_execute_timeout_kills_the_process_group():
    marker = f"42.{uuid.uuid4().int % 10**6}"
    code = f"import subprocess\nsubprocess.run(['sleep', '{marker}'])"
    env = LocalExecutionEnvironment(isolated=True, default_command_timeout=0.5)
    result = await env.execute(code)
    assert not result.success
    await asyncio.sleep(0.2)
    assert not _alive(f"sleep {marker}")


async def test_execute_command_with_non_utf8_output():
    env = LocalExecutionEnvironment()
    cmd = f"{sys.executable} -c \"import sys; sys.stdout.buffer.write(b'ok\\xff\\n')\""
    result = await env.execute_command(cmd)
    assert result.success, result.error
    assert result.exit_code == 0
    assert result.stdout == "ok\ufffd\n"


async def test_isolated_execute_with_non_utf8_output():
    code = "import sys\nsys.stderr.buffer.write(b'\\xff')\n_result = 1"
    async with LocalExecutionEnvironment(isolated=True) as env:
        result = await env.execute(code)
    assert result.success, result.error
    assert result.result == 1


@posix_only
async def test_concurrent_execution_timeout_kills_its_own_process():
    marker = f"43.{uuid.uuid4().int % 10**6}"
    env = LocalExecutionEnvironment(isolated=True, default_command_timeout=1.0)

    async def quick() -> object:
        await asyncio.sleep(0.1)  # starts after the slow one
        return await env.execute("_result = 'done'")

    slow_code = f"import subprocess\nsubprocess.run(['sleep', '{marker}'])"
    slow, fast = await asyncio.gather(env.execute(slow_code), quick())
    assert not slow.success
    assert fast.success  # type: ignore[union-attr]
    await asyncio.sleep(0.2)
    assert not _alive(f"sleep {marker}")


def test_os_environ_untouched():
    assert "SANDBOXED" not in os.environ


@posix_only
async def test_aexit_ends_background_children_of_finished_commands():
    marker = f"44.{uuid.uuid4().int % 10**6}"
    async with LocalExecutionEnvironment() as env:
        result = await env.execute_command(f"sleep {marker} >/dev/null 2>&1 & echo started")
        assert result.success, result.error
        await asyncio.sleep(0.2)
        assert _alive(f"sleep {marker}")  # the command's leader is gone, its child not
    await asyncio.sleep(0.2)
    assert not _alive(f"sleep {marker}")


@posix_only
async def test_aexit_ends_background_children_of_finished_spawned_processes():
    marker = f"45.{uuid.uuid4().int % 10**6}"
    async with LocalExecutionEnvironment() as env:
        handle = await env.spawn(f"sleep {marker} >/dev/null 2>&1 &")
        assert await handle.wait() == 0
        await asyncio.sleep(0.2)
        assert _alive(f"sleep {marker}")
    await asyncio.sleep(0.2)
    assert not _alive(f"sleep {marker}")


@posix_only
async def test_aexit_ends_background_children_of_isolated_code():
    marker = f"46.{uuid.uuid4().int % 10**6}"
    code = (
        "import subprocess\n"
        f"subprocess.Popen(['sleep', '{marker}'], stdout=subprocess.DEVNULL,"
        " stderr=subprocess.DEVNULL)\n_result = 1"
    )
    async with LocalExecutionEnvironment(isolated=True) as env:
        result = await env.execute(code)
        assert result.success, result.error
        await asyncio.sleep(0.2)
        assert _alive(f"sleep {marker}")
    await asyncio.sleep(0.2)
    assert not _alive(f"sleep {marker}")


@posix_only
async def test_finished_groups_are_forgotten():
    async with LocalExecutionEnvironment() as env:
        for _ in range(3):
            assert (await env.execute_command("true")).success
        assert len(env._process_groups) <= 1
