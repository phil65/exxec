"""Command lines built by remote providers, run by a local shell instead of the sandbox."""

import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from exxec.command_line import with_timeout


posix_only = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("timeout") is None,
    reason="POSIX shell with coreutils timeout (as in the remote sandboxes)",
)


def _run_argv(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, capture_output=True, timeout=10, check=False)


@posix_only
@pytest.mark.parametrize(
    ("timeout", "command", "stdout"),
    [(5, "cd / && pwd", b"/\n"), (None, "cd / && pwd", b"/\n"), (5, "echo 'a  b'", b"a  b\n")],
)
def test_with_timeout(timeout: float | None, command: str, stdout: bytes):
    out = _run_argv(["sh", "-c", with_timeout(command, timeout)])
    assert out.stdout == stdout


@posix_only
def test_with_timeout_limits_the_whole_line():
    out = _run_argv(["sh", "-c", with_timeout("true && sleep 5", 0.2)])
    assert out.returncode == 124  # noqa: PLR2004


class FakeVercelSandbox:
    async def run_command(self, cmd: str, args: list[str] | None, **_: Any) -> Any:
        out = _run_argv([cmd, *(args or [])])
        return SimpleNamespace(stdout=out.stdout, stderr=out.stderr, returncode=out.returncode)


@posix_only
@pytest.mark.parametrize("timeout", [5, None])
async def test_vercel_execute_command_is_a_shell_command(timeout: float | None):
    from exxec.vercel_provider import VercelExecutionEnvironment

    env = VercelExecutionEnvironment(default_command_timeout=timeout)
    env.sandbox = FakeVercelSandbox()  # type: ignore[assignment]
    result = await env.execute_command("cd / && pwd")
    assert result.success, result.error
    assert result.stdout == "/\n"


class FakeMicrosandbox:
    async def shell(self, command: str) -> Any:
        out = _run_argv(["sh", "-c", command])
        return SimpleNamespace(
            stdout_text=out.stdout.decode(),
            stderr_text=out.stderr.decode(),
            exit_code=out.returncode,
            success=out.returncode == 0,
        )


@posix_only
async def test_microsandbox_execute_command_env_and_builtins():
    from exxec.microsandbox_provider import MicrosandboxExecutionEnvironment

    env = MicrosandboxExecutionEnvironment(
        default_command_timeout=5, env_vars={"VAL": "it's $(echo x)"}
    )
    env.sandbox = FakeMicrosandbox()  # type: ignore[assignment]
    result = await env.execute_command('cd / && printf "%s %s" "$PWD" "$VAL"')
    assert result.success, result.error
    assert result.stdout == "/ it's $(echo x)"
