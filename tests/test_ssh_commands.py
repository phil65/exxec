"""SSH command/code paths against the in-process asyncssh test server."""

from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING

import pytest


pytest.importorskip("asyncssh")

from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent
from tests.test_spawn_ssh import ssh_server  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from exxec.ssh_provider import SshExecutionEnvironment


pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="test server requires POSIX"),
    pytest.mark.skipif(shutil.which("uv") is None, reason="ssh env verifies uv on the remote"),
]


@pytest.fixture
async def env(ssh_server: int, tmp_path: Path) -> AsyncIterator[SshExecutionEnvironment]:  # noqa: F811
    from exxec.ssh_provider import SshExecutionEnvironment

    work_dir = tmp_path / "work dir"
    work_dir.mkdir()
    ssh_env = SshExecutionEnvironment(
        host="127.0.0.1",
        port=ssh_server,
        username="tester",
        password="secret",
        cwd=str(work_dir),
        env_vars={"VAL": "it's $(echo expanded)"},
        known_hosts=None,
        default_command_timeout=None,
    )
    async with ssh_env:
        yield ssh_env


async def test_write_file_content_with_eof_line(env: SshExecutionEnvironment, tmp_path: Path):
    target = tmp_path / "out.txt"
    marker = tmp_path / "injected"
    content = f"before\nEOF\ntouch {marker}\nafter"
    await env.write_file(str(target), content)
    assert target.read_text() == content
    assert not marker.exists()


async def test_env_values_are_not_expanded(env: SshExecutionEnvironment):
    result = await env.execute_command('printf %s "$VAL"')
    assert result.stdout == "it's $(echo expanded)"


async def test_cwd_with_spaces(env: SshExecutionEnvironment, tmp_path: Path):
    result = await env.execute_command("pwd")
    assert result.success, result.error
    assert result.stdout == f"{tmp_path / 'work dir'}\n"


@pytest.mark.skipif(shutil.which("timeout") is None, reason="coreutils timeout")
async def test_execute_command_builtins_with_timeout(env: SshExecutionEnvironment):
    result = await env.execute_command("cd / && pwd", timeout=5)
    assert result.success, result.error
    assert result.stdout == "/\n"


async def test_stream_command_exit_code_and_order(env: SshExecutionEnvironment):
    cmd = "echo one; echo two >&2; echo three; exit 3"
    events = [e async for e in env.stream_command(cmd)]
    assert [e.data for e in events if isinstance(e, OutputEvent)] == ["one", "two", "three"]
    assert isinstance(events[-1], ProcessErrorEvent)
    assert events[-1].exit_code == 3  # noqa: PLR2004


async def test_stream_command_builtins(env: SshExecutionEnvironment):
    events = [e async for e in env.stream_command("cd / && pwd")]
    assert [e.data for e in events if isinstance(e, OutputEvent)] == ["/"]
    assert isinstance(events[-1], ProcessCompletedEvent)


async def test_execute_python_with_eof_line(env: SshExecutionEnvironment):
    code = 'doc = """\nEOF\necho INJECTED\n"""\nprint(doc.split()[0])'
    result = await env.execute(code)
    assert result.success, result.error
    assert result.stdout == "EOF\n"
