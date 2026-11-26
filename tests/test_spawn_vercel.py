"""Tests for the Vercel process handles.

Unit tests use fakes that mirror the vercel SDK's runtime API (including the rule
that an interactive session must be exited in the task that entered it). The
integration test runs against a real sandbox when credentials work.
"""

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING, Any, cast

import anyio
import pytest

from exxec.process_handle import ProcessHandleError
from exxec.pty_manager import PtySize


pytest.importorskip("vercel")

from exxec.vercel_provider.process_handle import (
    VercelInteractiveHandle,
    VercelProcessHandle,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from vercel.sandbox import Sandbox

    from exxec.process_handle import ProcessHandle


class FakeReader:
    def __init__(self) -> None:
        self.chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.closed = False

    async def receive(self, max_bytes: int = 65536) -> bytes:
        chunk = await self.chunks.get()
        if chunk is None:
            raise anyio.EndOfStream
        return chunk[:max_bytes]

    async def aclose(self) -> None:
        self.closed = True


class FakeStdin:
    def __init__(self, command: FakeCommand) -> None:
        self.command = command
        self.data = b""
        self.closed = False

    async def send(self, item: bytes) -> None:
        self.data += item

    async def aclose(self) -> None:
        self.closed = True
        # Echo stdin upper-cased, then exit, like `tr a-z A-Z`.
        self.command.stdout.chunks.put_nowait(self.data.upper())
        self.command.finish(0)


class FakeCommand:
    def __init__(self, argv: list[str], kwargs: dict[str, Any]) -> None:
        self.argv = argv
        self.kwargs = kwargs
        self.stdout = FakeReader()
        self.stdin = FakeStdin(self)
        self.returncode: int | None = None
        self.signals: list[str] = []
        self._done = asyncio.Event()

    def finish(self, code: int) -> None:
        self.returncode = code
        self.stdout.chunks.put_nowait(None)
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    async def send_signal(self, sig: str) -> None:
        self.signals.append(sig)
        self.finish(-15)

    async def kill(self) -> None:
        self.signals.append("KILL")
        self.finish(-9)


class FakeStream:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.sent: list[bytes] = []

    async def receive(self, max_bytes: int = 65536) -> bytes:
        chunk = await self.incoming.get()
        if chunk is None:
            raise anyio.EndOfStream
        return chunk

    async def send(self, item: bytes) -> None:
        self.sent.append(item)


class FakeSession:
    def __init__(self) -> None:
        self.stream = FakeStream()
        self.returncode: int | None = None
        self.sizes: list[tuple[int, int]] = []
        self._done = asyncio.Event()

    async def resize(self, cols: int, rows: int) -> None:
        self.sizes.append((cols, rows))

    async def wait(self) -> int | None:
        await self._done.wait()
        return self.returncode

    def exit(self, code: int | None) -> None:
        self.returncode = code
        self.stream.incoming.put_nowait(None)
        self._done.set()


class FakeInteractiveOperation:
    """Mirrors InteractiveSessionOperation: must exit in the task that entered."""

    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.entered_in: asyncio.Task[Any] | None = None
        self.exited = False

    async def __aenter__(self) -> FakeSession:
        self.entered_in = asyncio.current_task()
        return self.session

    async def __aexit__(self, *exc: object) -> None:
        assert asyncio.current_task() is self.entered_in, "exited in a different task"
        self.exited = True
        if self.session.returncode is None:
            self.session.exit(None)  # hang-up: no exit status


class FakeSandbox:
    def __init__(self) -> None:
        self.commands: list[FakeCommand] = []
        self.operations: list[FakeInteractiveOperation] = []
        self.interactive_calls: list[dict[str, Any]] = []

    async def create_command(self, command: str, args: list[str], **kwargs: Any) -> FakeCommand:
        fake = FakeCommand([command, *args], kwargs)
        self.commands.append(fake)
        return fake

    def open_interactive(
        self, command: str, args: list[str], **kwargs: Any
    ) -> FakeInteractiveOperation:
        self.interactive_calls.append({"argv": [command, *args], **kwargs})
        operation = FakeInteractiveOperation(FakeSession())
        self.operations.append(operation)
        return operation


def as_sandbox(fake: FakeSandbox) -> Sandbox:
    """The fakes implement the subset of the SDK the handles use."""
    return cast("Sandbox", fake)


async def read_all(handle: ProcessHandle) -> bytes:
    return b"".join([chunk async for chunk in handle.output()])


async def test_pipe_handle_stdin_output_and_exit() -> None:
    import subprocess

    sandbox = FakeSandbox()
    handle = await VercelProcessHandle.start(
        as_sandbox(sandbox),
        ["tr", "a-z", "A-Z"],
        cwd="/work",
        env={"A": "b"},
    )
    command = sandbox.commands[0]
    assert command.argv == ["tr", "a-z", "A-Z"]
    assert command.kwargs["stdin"] == subprocess.PIPE
    assert command.kwargs["stderr"] == subprocess.STDOUT
    assert command.kwargs["cwd"] == "/work"
    async with handle:
        await handle.write(b"hello")
        await handle.close_stdin()
        assert await read_all(handle) == b"HELLO"
        assert await handle.wait() == 0
        with pytest.raises(ProcessHandleError):
            await handle.write(b"late")


async def test_pipe_handle_signals_map_to_shell_exit_codes() -> None:
    sandbox = FakeSandbox()
    handle = await VercelProcessHandle.start(
        as_sandbox(sandbox), ["sleep", "60"], cwd=None, env=None
    )
    async with handle:
        assert await handle.kill(grace=1) == 143  # noqa: PLR2004
    assert sandbox.commands[0].signals == ["TERM"]
    handle = await VercelProcessHandle.start(
        as_sandbox(sandbox), ["sleep", "60"], cwd=None, env=None
    )
    await handle.send_signal("KILL")
    assert await handle.wait() == 137  # noqa: PLR2004
    await handle.aclose()


async def test_interactive_handle_io_resize_and_exit() -> None:
    sandbox = FakeSandbox()
    handle = await VercelInteractiveHandle.start(
        as_sandbox(sandbox),
        ["python3", "-q"],
        cwd="/w",
        env=None,
        size=PtySize(rows=30, cols=100),
    )
    call = sandbox.interactive_calls[0]
    assert call["argv"] == ["python3", "-q"]
    assert (call["cols"], call["rows"]) == (100, 30)
    session = sandbox.operations[0].session
    async with handle:
        assert handle.is_pty
        await handle.write(b"print(1)\r")
        assert session.stream.sent == [b"print(1)\r"]
        await handle.resize(PtySize(rows=40, cols=120))
        assert session.sizes == [(120, 40)]
        await handle.send_signal("INT")
        assert session.stream.sent[-1] == b"\x03"
        session.stream.incoming.put_nowait(b">>> ")
        session.exit(4)
        assert await read_all(handle) == b">>> "
        assert await handle.wait() == 4  # noqa: PLR2004
    # Exited in the owner task (the fake asserts this) when the handle closed.
    assert sandbox.operations[0].exited


async def test_interactive_kill_hangs_up_session_in_owner_task() -> None:
    sandbox = FakeSandbox()
    handle = await VercelInteractiveHandle.start(
        as_sandbox(sandbox),
        ["bash"],
        cwd=None,
        env=None,
        size=PtySize(),
    )
    reader = asyncio.create_task(read_all(handle))
    # Hang-up comes from a different task than the one that opened the session.
    assert await asyncio.create_task(handle.kill(grace=1, signal="HUP")) == 129  # noqa: PLR2004
    await asyncio.wait_for(reader, timeout=1)
    await handle.aclose()
    assert sandbox.operations[0].exited


async def test_interactive_start_propagates_open_errors() -> None:
    class FailingSandbox(FakeSandbox):
        def open_interactive(self, command: str, args: list[str], **kwargs: Any) -> Any:
            @contextlib.asynccontextmanager
            async def failing() -> AsyncIterator[FakeSession]:
                raise ConnectionError("refused")
                yield FakeSession()

            return failing()

    with pytest.raises(ConnectionError, match="refused"):
        await VercelInteractiveHandle.start(
            as_sandbox(FailingSandbox()),
            ["bash"],
            cwd=None,
            env=None,
            size=PtySize(),
        )


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("VERCEL_TOKEN"), reason="VERCEL_TOKEN not set")
async def test_vercel_spawn_integration() -> None:
    from vercel.sandbox import SandboxApiError

    from exxec.vercel_provider import VercelExecutionEnvironment

    env = VercelExecutionEnvironment(
        token=os.environ["VERCEL_TOKEN"],
        project_id=os.environ.get("VERCEL_PROJECT_ID"),
        team_id=os.environ.get("VERCEL_TEAM_ID"),
    )
    try:
        await env.__aenter__()
    except SandboxApiError as e:
        pytest.skip(f"Vercel credentials rejected: {e}")
    try:
        async with await env.spawn("sh", ["-c", "read x; echo got-$x; exit 3"]) as handle:
            await handle.write(b"value\n")
            await handle.close_stdin()
            assert b"got-value" in await read_all(handle)
            assert await handle.wait() == 3  # noqa: PLR2004
        size = PtySize(rows=30, cols=90)
        async with await env.spawn("sh", ["-c", "stty size; exit 5"], pty=size) as handle:
            assert b"30 90" in await read_all(handle)
            assert await handle.wait() == 5  # noqa: PLR2004
    finally:
        await env.__aexit__(None, None, None)
