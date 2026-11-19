"""Integration tests for `BeamExecutionEnvironment.spawn()` against real Beam sandboxes.

Run with `pytest -m integration`; requires BEAM_TOKEN.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from exxec.process_handle import ProcessHandleError
from exxec.pty_manager import PtySize


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.beam_provider import BeamExecutionEnvironment
    from exxec.process_handle import ProcessHandle


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("BEAM_TOKEN"), reason="BEAM_TOKEN not set"),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def beam_env() -> AsyncIterator[BeamExecutionEnvironment]:
    pytest.importorskip("beam")
    from exxec.beam_provider import BeamExecutionEnvironment

    async with BeamExecutionEnvironment(env_vars={"BASE": "b"}) as env:
        yield env


async def read_all(handle: ProcessHandle) -> bytes:
    return b"".join([chunk async for chunk in handle.output()])


async def test_capabilities() -> None:
    from exxec.beam_provider import BeamExecutionEnvironment

    assert BeamExecutionEnvironment.supports_spawn is True
    assert BeamExecutionEnvironment.supports_pty is False


async def test_pipe_output_cwd_env_exit(beam_env: BeamExecutionEnvironment) -> None:
    script = (
        "import os, sys\n"
        "print(os.getcwd(), os.environ['BASE'], os.environ['CALL'], sys.argv[1], flush=True)\n"
        "sys.exit(6)\n"
    )
    handle = await beam_env.spawn(
        "python3", ["-c", script, "a 'b'"], cwd="/tmp", env={"CALL": "c d"}
    )
    async with handle:
        assert handle.pid is not None
        assert (await read_all(handle)).decode().splitlines() == ["/tmp b c d a 'b'"]
        assert await handle.wait() == 6  # noqa: PLR2004


async def test_no_stdin_and_no_pty(beam_env: BeamExecutionEnvironment) -> None:
    async with await beam_env.spawn("true") as handle:
        with pytest.raises(ProcessHandleError, match="stdin"):
            await handle.write(b"x")
    with pytest.raises(NotImplementedError):
        await beam_env.spawn("true", pty=PtySize())


async def test_kill_reports_signal_exit_codes(beam_env: BeamExecutionEnvironment) -> None:
    async with await beam_env.spawn("sleep", ["120"]) as handle:
        await asyncio.sleep(1)
        assert await handle.kill(grace=10) == 143  # noqa: PLR2004
    ignoring = "trap '' TERM; sleep 120"
    async with await beam_env.spawn(ignoring) as handle:
        await asyncio.sleep(1)
        assert await handle.kill(grace=2) == 137  # noqa: PLR2004


async def test_streaming_does_not_block_event_loop(beam_env: BeamExecutionEnvironment) -> None:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        events = [event async for event in beam_env.stream_command("sleep 2; echo done")]
    finally:
        task.cancel()
    assert any(getattr(event, "data", None) == "done" for event in events)
    # The loop kept running while Beam was polled (previously it blocked for ~2s).
    assert ticks >= 20  # noqa: PLR2004


async def test_process_manager(beam_env: BeamExecutionEnvironment) -> None:
    from exxec.process_manager import SpawnProcessManager

    pm = beam_env.process_manager
    assert isinstance(pm, SpawnProcessManager)
    process_id = await pm.start_process("echo $BASE-$CALL", env={"CALL": "c"})
    assert await pm.wait_for_exit(process_id) == 0
    assert (await pm.get_output(process_id)).combined.strip() == "b-c"
