"""Integration tests for `E2bExecutionEnvironment.spawn()` against real e2b sandboxes.

Run with `pytest -m integration`; requires E2B_API_KEY.
"""

import asyncio
import os
import re
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from exxec.e2b_provider import E2bExecutionEnvironment
from exxec.pty_manager import PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.process_handle import ProcessHandle


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("E2B_API_KEY"), reason="E2B_API_KEY not set"),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def e2b_env() -> AsyncIterator[E2bExecutionEnvironment]:
    pytest.importorskip("e2b")
    async with E2bExecutionEnvironment(env_vars={"BASE": "b"}, timeout=600) as env:
        yield env


async def read_all(handle: ProcessHandle) -> bytes:
    return b"".join([chunk async for chunk in handle.output()])


async def read_until(iterator: AsyncIterator[bytes], marker: bytes, timeout: float = 20) -> bytes:
    buffer = b""

    async def consume() -> None:
        nonlocal buffer
        async for chunk in iterator:
            buffer += chunk
            if marker in buffer:
                return

    await asyncio.wait_for(consume(), timeout=timeout)
    return buffer


async def test_pipe_output_cwd_env_stdin_exit(e2b_env: E2bExecutionEnvironment) -> None:
    script = (
        "import os, sys\n"
        "print(os.getcwd(), os.environ['BASE'], os.environ['CALL'], sys.argv[1], flush=True)\n"
        "print(sys.stdin.read().upper(), flush=True)\n"
        "sys.exit(6)\n"
    )
    handle = await e2b_env.spawn(
        "python3", ["-c", script, "a 'b'"], cwd="/tmp", env={"CALL": "c d"}
    )
    async with handle:
        assert handle.is_pty is False
        assert handle.pid is not None
        await handle.write(b"piped in")
        await handle.close_stdin()
        lines = (await read_all(handle)).decode().splitlines()
        assert await handle.wait() == 6  # noqa: PLR2004
    assert lines == ["/tmp b c d a 'b'", "PIPED IN"]


async def test_shell_command_line(e2b_env: E2bExecutionEnvironment) -> None:
    async with await e2b_env.spawn("echo a | tr a b; exit 3") as handle:
        assert (await read_all(handle)).strip() == b"b"
        assert await handle.wait() == 3  # noqa: PLR2004


async def test_kill_reports_signal_exit_codes(e2b_env: E2bExecutionEnvironment) -> None:
    script = "import time; print('ready', flush=True); time.sleep(120)"
    async with await e2b_env.spawn("python3", ["-c", script]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=10) == 143  # noqa: PLR2004
    ignoring = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(120)\n"
    )
    async with await e2b_env.spawn("python3", ["-c", ignoring]) as handle:
        await read_until(handle.output(), b"ready")
        assert await handle.kill(grace=1) == 137  # noqa: PLR2004


async def test_pty_command_size_input_resize(e2b_env: E2bExecutionEnvironment) -> None:
    script = (
        "import os, signal, sys\n"
        "s = os.get_terminal_size()\n"
        "print('tty', sys.stdout.isatty(), s.lines, s.columns, flush=True)\n"
        "print('got', input(), flush=True)\n"
        "def on_winch(*_):\n"
        "    s = os.get_terminal_size()\n"
        "    print('resized', s.lines, s.columns, flush=True)\n"
        "    sys.exit(4)\n"
        "signal.signal(signal.SIGWINCH, on_winch)\n"
        "print('ready', flush=True)\n"
        "signal.pause()\n"
    )
    size = PtySize(rows=33, cols=99)
    async with await e2b_env.spawn("python3", ["-c", script], pty=size) as handle:
        iterator = handle.output()
        first = await read_until(iterator, b"tty True")
        await handle.write(b"typed\r")
        before = first + await read_until(iterator, b"ready")
        await handle.resize(PtySize(rows=44, cols=111))
        after = b"".join([chunk async for chunk in iterator])
        assert await handle.wait() == 4  # noqa: PLR2004
    text = (before + after).decode(errors="replace")
    assert "tty True 33 99" in text
    assert "got typed" in text
    assert "resized 44 111" in text


async def test_managers_use_spawn(e2b_env: E2bExecutionEnvironment) -> None:
    from exxec.process_manager import SpawnProcessManager
    from exxec.spawn_pty_manager import SpawnPtyManager

    pm = e2b_env.process_manager
    assert isinstance(pm, SpawnProcessManager)
    process_id = await pm.start_process("echo $BASE-$CALL", env={"CALL": "c"})
    assert await pm.wait_for_exit(process_id) == 0
    assert (await pm.get_output(process_id)).combined.strip() == "b-c"
    manager = e2b_env.get_pty_manager()
    assert isinstance(manager, SpawnPtyManager)
    info = await manager.create()  # e2b's own login shell
    await manager.write(info.id, b"echo cloud-$((6 * 7))\r")
    assert b"cloud-42" in await read_until(manager.stream(info.id), b"cloud-42")
    assert await manager.kill(info.id) is True


async def test_terminal_session(e2b_env: E2bExecutionEnvironment) -> None:
    pytest.importorskip("pyghostty")
    from exxec.terminal import TerminalSession

    async with await TerminalSession.start(e2b_env, "python3", ["-q"]) as term:
        await term.wait_for(">>>", timeout=20)
        # The shell's startup line was cleared before exec'ing the program.
        assert "exec " not in term.text()
        await term.send_line("6 * 7")
        await term.wait_for(re.compile(r"^42$", re.MULTILINE), timeout=20)
        await term.send_keys("C-d")
        assert await term.wait() == 0
