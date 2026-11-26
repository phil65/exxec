"""Tests for `LocalExecutionEnvironment.spawn()` and the local process handles."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import pytest

from exxec import LocalExecutionEnvironment
from exxec.process_handle import (
    ProcessHandleError,
    normalize_returncode,
    parse_signal_name,
    signal_exit_code,
)
from exxec.pty_manager import PtySize


if TYPE_CHECKING:
    from pathlib import Path

    from exxec.process_handle import ProcessHandle


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX")
PY = sys.executable


async def read_all(handle: ProcessHandle) -> bytes:
    return b"".join([chunk async for chunk in handle.output()])


async def read_until(handle: ProcessHandle, marker: bytes, timeout: float = 5.0) -> bytes:
    """Read output until `marker` appears (output iterator is claimed)."""
    buffer = b""

    async def consume() -> None:
        nonlocal buffer
        async for chunk in handle.output():
            buffer += chunk
            if marker in buffer:
                return

    await asyncio.wait_for(consume(), timeout=timeout)
    return buffer


def test_normalize_returncode() -> None:
    assert normalize_returncode(0) == 0
    assert normalize_returncode(3) == 3  # noqa: PLR2004
    assert normalize_returncode(-15) == 143  # noqa: PLR2004


def test_parse_signal_name() -> None:
    assert parse_signal_name("SIGTERM") == "TERM"
    assert parse_signal_name("INT") == "INT"
    assert parse_signal_name("USR1") is None


def test_signal_exit_code() -> None:
    assert signal_exit_code("TERM") == 143  # noqa: PLR2004
    assert signal_exit_code("SIGKILL") == 137  # noqa: PLR2004
    assert signal_exit_code("NOT_A_SIGNAL") == 143  # noqa: PLR2004


async def test_capabilities() -> None:
    assert LocalExecutionEnvironment.supports_spawn is True
    assert LocalExecutionEnvironment.supports_pty is (sys.platform != "win32")


async def test_pipe_argv_output_and_exit_code() -> None:
    async with LocalExecutionEnvironment() as env:
        script = "import sys; print('out', flush=True); print('err', file=sys.stderr); sys.exit(3)"
        async with await env.spawn(PY, ["-c", script]) as handle:
            assert handle.is_pty is False
            assert handle.pid is not None
            output = await read_all(handle)
            assert await handle.wait() == 3  # noqa: PLR2004
            assert handle.returncode == 3  # noqa: PLR2004
    assert output.decode().splitlines() == ["out", "err"]


async def test_pipe_shell_command_line() -> None:
    async with (
        LocalExecutionEnvironment() as env,
        await env.spawn("echo one && echo two") as handle,
    ):
        output = await read_all(handle)
        assert await handle.wait() == 0
    assert output.decode().split() == ["one", "two"]


async def test_pipe_stdin_roundtrip() -> None:
    async with LocalExecutionEnvironment() as env:
        script = "import sys; data = sys.stdin.read(); print(data.upper(), end='')"
        async with await env.spawn(PY, ["-c", script]) as handle:
            await handle.write(b"hello")
            await handle.close_stdin()
            output = await read_all(handle)
            assert await handle.wait() == 0
    assert output == b"HELLO"


async def test_pipe_cwd_and_env(tmp_path: Path) -> None:
    async with LocalExecutionEnvironment(env_vars={"BASE": "b"}, inherit_env=True) as env:
        script = "import os; print(os.getcwd()); print(os.environ['BASE'], os.environ['CALL'])"
        async with await env.spawn(
            PY, ["-c", script], cwd=str(tmp_path), env={"CALL": "c"}
        ) as handle:
            lines = (await read_all(handle)).decode().splitlines()
    assert os.path.realpath(lines[0]) == os.path.realpath(tmp_path)
    assert lines[1] == "b c"


async def test_output_single_consumer() -> None:
    async with LocalExecutionEnvironment() as env, await env.spawn(PY, ["-c", "pass"]) as handle:
        handle.output()
        with pytest.raises(ProcessHandleError):
            handle.output()


async def test_pipe_resize_rejected() -> None:
    async with LocalExecutionEnvironment() as env, await env.spawn(PY, ["-c", "pass"]) as handle:
        with pytest.raises(ProcessHandleError):
            await handle.resize(PtySize(rows=10, cols=10))


@posix_only
async def test_pipe_kill_uses_shell_exit_convention() -> None:
    async with LocalExecutionEnvironment() as env:
        handle = await env.spawn(PY, ["-c", "import time; time.sleep(30)"])
        async with handle:
            assert await handle.kill(grace=2) == 143  # noqa: PLR2004


@posix_only
async def test_pipe_kill_escalates_to_sigkill() -> None:
    async with LocalExecutionEnvironment() as env:
        script = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\n"
            "time.sleep(30)\n"
        )
        async with await env.spawn(PY, ["-c", script]) as handle:
            await read_until(handle, b"ready")
            assert await handle.kill(grace=0.3) == 137  # noqa: PLR2004


@posix_only
async def test_pipe_kill_terminates_process_group() -> None:
    async with (
        LocalExecutionEnvironment() as env,
        await env.spawn("sleep 30 & echo $!; wait") as handle,
    ):
        output = await read_until(handle, b"\n")
        child_pid = int(output.strip())
        await handle.kill(grace=2)
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@posix_only
@pytest.mark.parametrize("pty", [None, PtySize(rows=24, cols=80)])
async def test_kill_signals_group_after_leader_exited(pty: PtySize | None) -> None:
    async with LocalExecutionEnvironment() as env:
        # nohup: a terminal hangup (the PTY leader exiting) must not be what ends it
        handle = await env.spawn("nohup sleep 30 >/dev/null 2>&1 & echo $!; sleep 0.2", pty=pty)
        async with handle:
            output = await read_until(handle, b"\n")
            child_pid = int(output.strip())
            assert await handle.wait() == 0  # the leader is gone, its child is not
            os.kill(child_pid, 0)
            assert await handle.kill(grace=2) == 0
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@posix_only
async def test_context_exit_kills_running_process() -> None:
    async with LocalExecutionEnvironment() as env:
        handle = await env.spawn(PY, ["-c", "import time; time.sleep(30)"])
        async with handle:
            pass
        assert handle.returncode == 143  # noqa: PLR2004


@posix_only
async def test_wait_is_cancel_safe() -> None:
    async with LocalExecutionEnvironment() as env:
        script = "import time; time.sleep(0.5)"
        async with await env.spawn(PY, ["-c", script]) as handle:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(handle.wait(), timeout=0.05)
            assert await handle.wait() == 0


@posix_only
async def test_pty_is_a_terminal_with_size() -> None:
    async with LocalExecutionEnvironment() as env:
        script = (
            "import os, sys\n"
            "size = os.get_terminal_size()\n"
            "print(sys.stdin.isatty(), sys.stdout.isatty(), size.lines, size.columns)\n"
        )
        size = PtySize(rows=30, cols=100)
        async with await env.spawn(PY, ["-c", script], pty=size) as handle:
            assert handle.is_pty is True
            assert handle.pty_size == size
            output = await read_all(handle)
            assert await handle.wait() == 0
    assert output.decode().split() == ["True", "True", "30", "100"]


@posix_only
async def test_pty_sets_term_and_echoes_input() -> None:
    async with LocalExecutionEnvironment() as env:
        script = "import os; print(os.environ['TERM']); print('got:', input())"
        async with await env.spawn(PY, ["-c", script], pty=PtySize()) as handle:
            await handle.write(b"abc\r")
            output = await read_all(handle)
            assert await handle.wait() == 0
    text = output.decode()
    assert "xterm-256color" in text
    assert "got: abc" in text


@posix_only
async def test_pty_resize_delivers_sigwinch() -> None:
    async with LocalExecutionEnvironment() as env:
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
        async with await env.spawn(PY, ["-c", script], pty=PtySize()) as handle:
            chunks: list[bytes] = []
            iterator = handle.output()
            async for chunk in iterator:
                chunks.append(chunk)
                if b"ready" in b"".join(chunks):
                    break
            await handle.resize(PtySize(rows=40, cols=120))
            assert handle.pty_size == PtySize(rows=40, cols=120)
            chunks.extend([chunk async for chunk in iterator])
            assert await handle.wait() == 0
    assert "size 40 120" in b"".join(chunks).decode()


@posix_only
async def test_pty_close_stdin_sends_eof() -> None:
    async with LocalExecutionEnvironment() as env:
        script = "import sys; data = sys.stdin.read(); print('len', len(data))"
        async with await env.spawn(PY, ["-c", script], pty=PtySize()) as handle:
            await handle.write(b"xy\r")
            await handle.close_stdin()
            output = await read_all(handle)
            assert await handle.wait() == 0
    assert "len 3" in output.decode()


@posix_only
async def test_pty_shell_command_line_and_kill() -> None:
    async with LocalExecutionEnvironment() as env:
        handle = await env.spawn("echo started; sleep 30", pty=PtySize())
        async with handle:
            await read_until(handle, b"started")
            assert await handle.kill(grace=2) == 143  # noqa: PLR2004


@posix_only
async def test_pty_nonzero_exit_code() -> None:
    async with (
        LocalExecutionEnvironment() as env,
        await env.spawn("exit 7", pty=PtySize()) as handle,
    ):
        await read_all(handle)
        assert await handle.wait() == 7  # noqa: PLR2004


@posix_only
async def test_pty_reader_unblocks_on_close() -> None:
    async with LocalExecutionEnvironment() as env:
        handle = await env.spawn(PY, ["-c", "import time; time.sleep(30)"], pty=PtySize())
        reader = asyncio.create_task(read_all(handle))
        await asyncio.sleep(0.1)
        await handle.aclose()
        await asyncio.wait_for(reader, timeout=5)


@posix_only
async def test_signal_immediately_after_spawn_is_delivered() -> None:
    async with LocalExecutionEnvironment() as env:
        for size in (None, PtySize()):
            handle = await env.spawn(PY, ["-c", "import time; time.sleep(30)"], pty=size)
            async with handle:
                await handle.send_signal("HUP")
                assert await asyncio.wait_for(handle.wait(), timeout=5) == 129  # noqa: PLR2004
