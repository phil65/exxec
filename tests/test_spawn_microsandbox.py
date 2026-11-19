"""Tests for `MicrosandboxExecutionEnvironment.spawn()` against real microVMs.

Requires the microsandbox runtime and access to /dev/kvm (Linux) or
Hypervisor.framework (macOS); skipped otherwise.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from exxec.microsandbox_provider import MicrosandboxExecutionEnvironment
from exxec.pty_manager import PtySize


pytest.importorskip("microsandbox")

from microsandbox import is_runtime_installed


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.process_handle import ProcessHandle


def _vm_available() -> bool:
    if sys.platform == "linux" and not os.access("/dev/kvm", os.R_OK | os.W_OK):
        return False
    return sys.platform in ("linux", "darwin") and is_runtime_installed()


pytestmark = [
    pytest.mark.skipif(not _vm_available(), reason="microsandbox runtime / KVM not available"),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def msb_env() -> AsyncIterator[MicrosandboxExecutionEnvironment]:
    from microsandbox import MicrosandboxError

    env = MicrosandboxExecutionEnvironment(env_vars={"BASE": "b"}, cwd="/tmp")
    try:
        await env.__aenter__()
    except MicrosandboxError as e:
        # e.g. CI runners with the runtime installed but no usable hypervisor
        pytest.skip(f"microVMs cannot start here: {e}")
    try:
        yield env
    finally:
        await env.__aexit__(None, None, None)


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
    assert MicrosandboxExecutionEnvironment.supports_spawn is True
    assert MicrosandboxExecutionEnvironment.supports_pty is True


async def test_pipe_output_cwd_env_stdin_exit(msb_env: MicrosandboxExecutionEnvironment) -> None:
    script = (
        "import os, sys\n"
        "print(os.getcwd(), os.environ['BASE'], os.environ['CALL'], sys.argv[1], flush=True)\n"
        "print(sys.stdin.read().upper(), flush=True)\n"
        "sys.exit(6)\n"
    )
    handle = await msb_env.spawn("python", ["-c", script, "a 'b'"], env={"CALL": "c d"})
    async with handle:
        assert handle.is_pty is False
        await handle.write(b"piped in")
        await handle.close_stdin()
        lines = (await read_all(handle)).decode().splitlines()
        assert await handle.wait() == 6  # noqa: PLR2004
        assert handle.pid is not None
    assert lines == ["/tmp b c d a 'b'", "PIPED IN"]


async def test_shell_command_line(msb_env: MicrosandboxExecutionEnvironment) -> None:
    async with await msb_env.spawn("echo a | tr a b && exit 3") as handle:
        assert (await read_all(handle)).strip() == b"b"
        assert await handle.wait() == 3  # noqa: PLR2004


async def test_exit_code_known_without_reading_output(
    msb_env: MicrosandboxExecutionEnvironment,
) -> None:
    async with await msb_env.spawn("sh", ["-c", "echo unread; exit 9"]) as handle:
        assert await asyncio.wait_for(handle.wait(), timeout=10) == 9  # noqa: PLR2004


async def test_kill_reports_signal_exit_codes(msb_env: MicrosandboxExecutionEnvironment) -> None:
    script = "import time; print('ready', flush=True); time.sleep(60)"
    async with await msb_env.spawn("python", ["-c", script]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=5) == 143  # noqa: PLR2004
    ignoring = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    async with await msb_env.spawn("python", ["-c", ignoring]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=0.5) == 137  # noqa: PLR2004


async def test_pty_size_input_and_resize(msb_env: MicrosandboxExecutionEnvironment) -> None:
    script = (
        "import os, signal, sys\n"
        "s = os.get_terminal_size()\n"
        "print('tty', sys.stdout.isatty(), s.lines, s.columns, flush=True)\n"
        "print('got', input(), flush=True)\n"
        "def on_winch(*_):\n"
        "    s = os.get_terminal_size()\n"
        "    print('resized', s.lines, s.columns, flush=True)\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGWINCH, on_winch)\n"
        "print('ready', flush=True)\n"
        "signal.pause()\n"
    )
    size = PtySize(rows=31, cols=91)
    async with await msb_env.spawn("python", ["-c", script], pty=size) as handle:
        iterator = handle.output()
        first = await read_until(iterator, b"tty")
        await handle.write(b"typed\r")
        before = first + await read_until(iterator, b"ready")
        await handle.resize(PtySize(rows=41, cols=121))
        after = b"".join([chunk async for chunk in iterator])
        assert await handle.wait() == 0
    text = (before + after).decode()
    assert "tty True 31 91" in text
    assert "got typed" in text
    assert "resized 41 121" in after.decode()


async def test_stream_command_is_real_time(msb_env: MicrosandboxExecutionEnvironment) -> None:
    from exxec.events import OutputEvent, ProcessCompletedEvent

    loop = asyncio.get_running_loop()
    stamps: list[float] = []
    events = []
    async for event in msb_env.stream_command("echo one; sleep 1; echo two"):
        events.append(event)
        if isinstance(event, OutputEvent):
            stamps.append(loop.time())
    lines = [e.data for e in events if isinstance(e, OutputEvent)]
    assert lines == ["one", "two"]
    assert isinstance(events[-1], ProcessCompletedEvent)
    # "one" arrives before the sleep finishes, i.e. streamed rather than buffered.
    assert stamps[1] - stamps[0] > 0.5  # noqa: PLR2004


async def test_managers_use_spawn(msb_env: MicrosandboxExecutionEnvironment) -> None:
    from exxec.process_manager import SpawnProcessManager
    from exxec.spawn_pty_manager import SpawnPtyManager

    pm = msb_env.process_manager
    assert isinstance(pm, SpawnProcessManager)
    process_id = await pm.start_process("echo $BASE")
    assert await pm.wait_for_exit(process_id) == 0
    assert (await pm.get_output(process_id)).combined.strip() == "b"
    manager = msb_env.get_pty_manager()
    assert isinstance(manager, SpawnPtyManager)
    info = await manager.create(command="sh")
    await manager.write(info.id, b"echo vm-$((6 * 7))\r")
    assert b"vm-42" in await read_until(manager.stream(info.id), b"vm-42")
    assert await manager.kill(info.id) is True


async def test_terminal_session_in_vm(msb_env: MicrosandboxExecutionEnvironment) -> None:
    pytest.importorskip("pyghostty")
    from exxec.terminal import TerminalSession

    async with await TerminalSession.start(msb_env, "python", ["-q"]) as term:
        await term.wait_for(">>>")
        await term.send_line("import platform; platform.machine()")
        await term.wait_for(re.compile(r"'(x86_64|aarch64|arm64)'"))
        await term.send_keys("C-d")
        assert await term.wait() == 0
