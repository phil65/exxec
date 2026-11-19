"""Tests that background processes run in the right environment.

Covers `LocalProcessManager` (command wrapping, cwd, env vars, cleanup) and
`StreamingProcessManager` (routing through the environment's `stream_command`).
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import pytest

from exxec import LocalExecutionEnvironment, MockExecutionEnvironment
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.process_manager import (
    KILLED_EXIT_CODE,
    LocalProcessManager,
    StreamingProcessManager,
    build_command_line,
    build_remote_command_line,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from exxec.events import ExecutionEvent


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX shell syntax")


class WrappingLocalEnvironment(LocalExecutionEnvironment):
    """Local environment whose wrapper marks every command it runs."""

    def wrap_command(self, command: str) -> str:
        return f"echo WRAPPED && {command}"


class RecordingEnvironment(MockExecutionEnvironment):
    """Mock environment recording commands passed to `stream_command`."""

    def __init__(self, events: list[ExecutionEvent] | None = None, block: bool = False) -> None:
        super().__init__()
        self.commands: list[str] = []
        self.events = events
        self.block = block
        self.stream_closed = asyncio.Event()

    async def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        _ = timeout
        self.commands.append(command)
        try:
            yield ProcessStartedEvent(process_id="p", command=command)
            for event in self.events or []:
                yield event
            if self.block:
                await asyncio.Event().wait()
        finally:
            self.stream_closed.set()


async def test_local_env_uses_local_process_manager() -> None:
    async with LocalExecutionEnvironment() as env:
        assert isinstance(env.process_manager, LocalProcessManager)


async def test_remote_env_defaults_to_streaming_process_manager() -> None:
    env = RecordingEnvironment()
    # MockExecutionEnvironment overrides process_manager, so check the base default.
    manager = super(MockExecutionEnvironment, env).process_manager
    assert isinstance(manager, StreamingProcessManager)


@posix_only
async def test_local_process_applies_wrap_command() -> None:
    async with WrappingLocalEnvironment(default_command_timeout=None) as env:
        pm = env.process_manager
        process_id = await pm.start_process("echo", args=["inner"])
        assert await pm.wait_for_exit(process_id) == 0
        output = await pm.get_output(process_id)
        assert output.combined.splitlines() == ["WRAPPED", "inner"]
        info = await pm.get_process_info(process_id)
        assert info["command"] == "echo"
        assert info["args"] == ["inner"]


@posix_only
async def test_local_process_uses_env_cwd(tmp_path: Path) -> None:
    async with LocalExecutionEnvironment(cwd=str(tmp_path)) as env:
        pm = env.process_manager
        process_id = await pm.start_process("pwd")
        await pm.wait_for_exit(process_id)
        output = await pm.get_output(process_id)
        assert output.combined.strip() == str(tmp_path.resolve())


@posix_only
async def test_local_process_cwd_argument_overrides_env_cwd(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    async with LocalExecutionEnvironment(cwd=str(tmp_path)) as env:
        pm = env.process_manager
        process_id = await pm.start_process("pwd", cwd=other)
        await pm.wait_for_exit(process_id)
        output = await pm.get_output(process_id)
        assert output.combined.strip() == str(other.resolve())


@posix_only
async def test_local_process_env_vars() -> None:
    async with LocalExecutionEnvironment(env_vars={"ENV_LEVEL": "env"}, inherit_env=True) as env:
        pm = env.process_manager
        process_id = await pm.start_process(
            'echo "$ENV_LEVEL $CALL_LEVEL"', env={"CALL_LEVEL": "call"}
        )
        await pm.wait_for_exit(process_id)
        output = await pm.get_output(process_id)
        assert output.combined.strip() == "env call"


async def test_local_process_output_without_newline() -> None:
    async with LocalExecutionEnvironment() as env:
        pm = env.process_manager
        script = "import sys, time; sys.stdout.write('prompt> '); sys.stdout.flush(); time.sleep(5)"
        process_id = await pm.start_process("python", args=["-c", script])
        for _ in range(50):
            if (await pm.get_output(process_id)).combined:
                break
            await asyncio.sleep(0.1)
        assert (await pm.get_output(process_id)).combined == "prompt> "
        await pm.kill_process(process_id)


@posix_only
async def test_local_kill_terminates_child_processes() -> None:
    async with LocalExecutionEnvironment() as env:
        pm = env.process_manager
        # The shell spawns a child; killing must take down the whole group.
        process_id = await pm.start_process("sleep 30 & echo $!; wait")
        for _ in range(50):
            if (await pm.get_output(process_id)).combined.strip():
                break
            await asyncio.sleep(0.1)
        child_pid = int((await pm.get_output(process_id)).combined.strip())
        await pm.kill_process(process_id)
        assert (await pm.get_process_info(process_id))["exit_code"] == KILLED_EXIT_CODE
        await asyncio.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)


async def test_env_exit_cleans_up_running_processes() -> None:
    async with LocalExecutionEnvironment() as env:
        pm = env.process_manager
        process_id = await pm.start_process("python", args=["-c", "import time; time.sleep(30)"])
        assert (await pm.get_process_info(process_id))["is_running"] is True
    assert await pm.list_processes() == []


async def test_streaming_routes_through_stream_command() -> None:
    env = RecordingEnvironment(
        events=[
            OutputEvent(process_id="p", data="line one", stream="combined"),
            OutputEvent(process_id="p", data="line two", stream="combined"),
            ProcessCompletedEvent(process_id="p", exit_code=0),
        ]
    )
    pm = StreamingProcessManager(env)
    process_id = await pm.start_process(
        "server", args=["--port", "8000"], cwd="/work dir", env={"MODE": "dev mode"}
    )
    assert await pm.wait_for_exit(process_id) == 0
    assert env.commands == ["cd '/work dir' && export MODE='dev mode' && server --port 8000"]
    output = await pm.get_output(process_id)
    assert output.combined == "line one\nline two\n"
    assert output.exit_code == 0


async def test_streaming_nonzero_exit_code() -> None:
    env = RecordingEnvironment(
        events=[
            ProcessErrorEvent(
                process_id="p",
                error="Command exited with code 3",
                error_type="CommandError",
                exit_code=3,
            )
        ]
    )
    pm = StreamingProcessManager(env)
    process_id = await pm.start_process("false")
    assert await pm.wait_for_exit(process_id) == 3  # noqa: PLR2004
    assert (await pm.get_output(process_id)).combined == ""


async def test_streaming_error_event_is_reported() -> None:
    env = RecordingEnvironment(
        events=[
            ProcessErrorEvent(
                process_id="p", error="Command timed out", error_type="TimeoutError", exit_code=1
            )
        ]
    )
    pm = StreamingProcessManager(env)
    process_id = await pm.start_process("slow")
    assert await pm.wait_for_exit(process_id) == 1
    assert "Command timed out" in (await pm.get_output(process_id)).combined


async def test_streaming_kill_closes_stream() -> None:
    env = RecordingEnvironment(block=True)
    pm = StreamingProcessManager(env)
    process_id = await pm.start_process("daemon")
    await asyncio.sleep(0.05)
    assert (await pm.get_process_info(process_id))["is_running"] is True
    await pm.kill_process(process_id)
    await asyncio.wait_for(env.stream_closed.wait(), timeout=1)
    info = await pm.get_process_info(process_id)
    assert info["is_running"] is False
    assert info["exit_code"] == KILLED_EXIT_CODE


def test_build_remote_command_line_rejects_invalid_env_names() -> None:
    with pytest.raises(ValueError, match="Invalid environment variable names"):
        build_remote_command_line("true", None, {"BAD NAME": "x"})


def test_build_command_line_quoting() -> None:
    assert build_command_line("echo", ["a b"]) == "echo 'a b'"
    assert build_command_line("echo", ["a b"], windows=True) == 'echo "a b"'
    assert build_command_line("echo a | cat", None) == "echo a | cat"


def test_build_remote_command_line_without_prefixes() -> None:
    assert build_remote_command_line("echo hi", None, None) == "echo hi"
