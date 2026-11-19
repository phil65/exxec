"""Tests for `SshExecutionEnvironment.spawn()` against an in-process asyncssh server.

The test server executes each session's command locally via
`LocalExecutionEnvironment.spawn()` and bridges I/O, PTY window changes and
signal requests, so the full SSH wire protocol path of the client is exercised.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import sys
from typing import TYPE_CHECKING

import pytest


pytest.importorskip("asyncssh")

import asyncssh

from exxec import LocalExecutionEnvironment
from exxec.process_handle import parse_signal_name
from exxec.pty_manager import PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from exxec.process_handle import ProcessHandle
    from exxec.ssh_provider import SshExecutionEnvironment


pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="test server requires POSIX"),
    pytest.mark.skipif(shutil.which("uv") is None, reason="ssh env verifies uv on the remote"),
]

PY = sys.executable
SIGNALS_BY_EXIT_CODE = {129: "HUP", 130: "INT", 137: "KILL", 143: "TERM"}


class _PasswordServer(asyncssh.SSHServer):
    def begin_auth(self, username: str) -> bool:
        return True

    def password_auth_supported(self) -> bool:
        return True

    def validate_password(self, username: str, password: str) -> bool:
        return True


async def _serve_session(process: asyncssh.SSHServerProcess[bytes]) -> None:
    """Run the session command locally, bridging I/O, resizes and signals."""
    term_type = process.term_type
    command = process.command or ""
    width, height, _, _ = process.term_size
    size = PtySize(rows=height, cols=width) if term_type else None
    async with (
        LocalExecutionEnvironment(default_command_timeout=None) as env,
        await env.spawn("/bin/sh", ["-c", command], pty=size) as handle,
    ):

        async def pump_out() -> None:
            async for chunk in handle.output():
                process.stdout.write(chunk)

        async def pump_in() -> None:
            while True:
                try:
                    data = await process.stdin.read(65536)
                except asyncssh.TerminalSizeChanged as e:
                    await handle.resize(PtySize(rows=e.height, cols=e.width))
                    continue
                except asyncssh.SignalReceived as e:
                    if (sig := parse_signal_name(e.signal)) is not None:
                        await handle.send_signal(sig)
                    continue
                except asyncssh.BreakReceived:
                    continue
                if not data:
                    if not handle.is_pty:
                        await handle.close_stdin()
                    return
                await handle.write(data)

        out_task = asyncio.create_task(pump_out())
        in_task = asyncio.create_task(pump_in())
        code = await handle.wait()
        await out_task
        in_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await in_task
    if code in SIGNALS_BY_EXIT_CODE:
        process.exit_with_signal(SIGNALS_BY_EXIT_CODE[code])
    else:
        process.exit(code)


@pytest.fixture
async def ssh_server() -> AsyncIterator[int]:
    host_key = asyncssh.generate_private_key("ssh-ed25519")
    server = await asyncssh.listen(
        "127.0.0.1",
        0,
        server_host_keys=[host_key],
        server_factory=_PasswordServer,
        process_factory=_serve_session,
        sftp_factory=True,
        encoding=None,
        line_editor=False,
    )
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def ssh_env(ssh_server: int, tmp_path: Path) -> AsyncIterator[SshExecutionEnvironment]:
    from exxec.ssh_provider import SshExecutionEnvironment

    env = SshExecutionEnvironment(
        host="127.0.0.1",
        port=ssh_server,
        username="tester",
        password="secret",
        cwd=str(tmp_path),
        env_vars={"BASE": "b"},
        known_hosts=None,
    )
    async with env:
        yield env


async def read_all(handle: ProcessHandle) -> bytes:
    return b"".join([chunk async for chunk in handle.output()])


async def read_until(iterator: AsyncIterator[bytes], marker: bytes, timeout: float = 5) -> bytes:
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
    from exxec.ssh_provider import SshExecutionEnvironment

    assert SshExecutionEnvironment.supports_spawn is True
    assert SshExecutionEnvironment.supports_pty is True


async def test_pipe_output_cwd_env_and_exit_code(
    ssh_env: SshExecutionEnvironment, tmp_path: Path
) -> None:
    script = (
        "import os, sys\n"
        "print(os.getcwd(), flush=True)\n"
        "print(os.environ['BASE'], os.environ['CALL'], sys.argv[1], flush=True)\n"
        "print('to stderr', file=sys.stderr, flush=True)\n"
        "sys.exit(4)\n"
    )
    handle = await ssh_env.spawn(PY, ["-c", script, "arg with 'quotes'"], env={"CALL": "c d"})
    async with handle:
        assert handle.is_pty is False
        lines = (await read_all(handle)).decode().splitlines()
        assert await handle.wait() == 4  # noqa: PLR2004
    assert lines == [str(tmp_path), "b c d arg with 'quotes'", "to stderr"]


async def test_pipe_stdin_roundtrip(ssh_env: SshExecutionEnvironment) -> None:
    script = "import sys; print(sys.stdin.read().upper(), end='')"
    async with await ssh_env.spawn(PY, ["-c", script]) as handle:
        await handle.write(b"over ssh")
        await handle.close_stdin()
        assert await read_all(handle) == b"OVER SSH"
        assert await handle.wait() == 0


async def test_shell_command_line(ssh_env: SshExecutionEnvironment) -> None:
    async with await ssh_env.spawn("echo a | tr a b") as handle:
        assert (await read_all(handle)).strip() == b"b"
        assert await handle.wait() == 0


async def test_signal_reports_exit_signal(ssh_env: SshExecutionEnvironment) -> None:
    script = "import time; print('ready', flush=True); time.sleep(30)"
    async with await ssh_env.spawn(PY, ["-c", script]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=5) == 143  # noqa: PLR2004


async def test_kill_escalates_by_closing_channel(ssh_env: SshExecutionEnvironment) -> None:
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    async with await ssh_env.spawn(PY, ["-c", script]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=0.5) == 137  # noqa: PLR2004


async def test_pty_size_term_and_resize(ssh_env: SshExecutionEnvironment) -> None:
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
    size = PtySize(rows=25, cols=90)
    async with await ssh_env.spawn(PY, ["-c", script], pty=size) as handle:
        assert handle.is_pty is True
        iterator = handle.output()
        before = await read_until(iterator, b"ready")
        await handle.resize(PtySize(rows=50, cols=132))
        after = b"".join([chunk async for chunk in iterator])
        assert await handle.wait() == 0
    assert "term xterm-256color True" in before.decode()
    assert "size 25 90" in before.decode()
    assert "resized 50 132" in after.decode()


async def test_pty_input(ssh_env: SshExecutionEnvironment) -> None:
    script = "print('got:', input())"
    async with await ssh_env.spawn(PY, ["-c", script], pty=PtySize()) as handle:
        await handle.write(b"typed\r")
        assert "got: typed" in (await read_all(handle)).decode()
        assert await handle.wait() == 0


async def test_context_exit_kills_remote_process(ssh_env: SshExecutionEnvironment) -> None:
    handle = await ssh_env.spawn(PY, ["-c", "import time; time.sleep(30)"])
    async with handle:
        pass
    assert handle.returncode == 143  # noqa: PLR2004


async def test_process_manager_over_ssh(ssh_env: SshExecutionEnvironment) -> None:
    from exxec.process_manager import SpawnProcessManager

    pm = ssh_env.process_manager
    assert isinstance(pm, SpawnProcessManager)
    process_id = await pm.start_process("echo $BASE-$CALL", env={"CALL": "c"})
    assert await pm.wait_for_exit(process_id) == 0
    assert (await pm.get_output(process_id)).combined.strip() == "b-c"


async def test_pty_manager_over_ssh(ssh_env: SshExecutionEnvironment) -> None:
    from exxec.spawn_pty_manager import SpawnPtyManager

    manager = ssh_env.get_pty_manager()
    assert isinstance(manager, SpawnPtyManager)
    info = await manager.create(command="sh")
    await manager.write(info.id, b"echo remote-$((6 * 7))\r")
    assert b"remote-42" in await read_until(manager.stream(info.id), b"remote-42")
    assert await manager.kill(info.id) is True


async def test_terminal_session_over_ssh(ssh_env: SshExecutionEnvironment) -> None:
    pytest.importorskip("pyghostty")
    from exxec.terminal import TerminalSession

    async with await TerminalSession.start(ssh_env, "sh") as term:
        await term.send_line("echo over-$((40 + 2))")
        await term.wait_for("over-42")
        await term.resize(PtySize(rows=30, cols=100))
        await term.send_line("stty size")
        await term.wait_for("30 100")
