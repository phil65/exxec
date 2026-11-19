"""Tests for the Modal process handle.

`LocalModalSandbox` mirrors the subset of Modal's `Sandbox.exec()` API the handle
uses, but runs every exec as a real local process (via the local `spawn()`), so
the handle's shell wrappers (PID file, `stty` sizing, TTY recording, `stty -F`
resize, `kill`) are exercised for real. The integration test uses Modal itself.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, cast

import pytest

from exxec import LocalExecutionEnvironment
from exxec.pty_manager import PtySize


if sys.platform == "win32":
    # Importing modal on Windows switches the global event loop policy to the
    # selector loop (no subprocess support), which would break other tests.
    pytest.skip("requires POSIX shell and PTYs", allow_module_level=True)

pytest.importorskip("modal")

from exxec.modal_provider.process_handle import ModalProcessHandle


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from modal import Sandbox

    from exxec.process_handle import ProcessHandle


PY = sys.executable


class _Method:
    """Callable with an `.aio` coroutine variant, like Modal's synchronicity wrappers."""

    def __init__(self, coro: Callable[..., Awaitable[Any]]) -> None:
        self.aio = coro


class _Stdout:
    def __init__(self, handle: ProcessHandle) -> None:
        self._handle = handle
        self._iterator = handle.output()

        async def read() -> bytes:
            return b"".join([chunk async for chunk in self._iterator])

        self.read = _Method(read)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterator


class _Stdin:
    def __init__(self, handle: ProcessHandle) -> None:
        self._handle = handle
        self._buffer = b""
        self._eof = False

        async def drain() -> None:
            if self._buffer:
                data, self._buffer = self._buffer, b""
                await self._handle.write(data)
            if self._eof:
                await self._handle.close_stdin()

        self.drain = _Method(drain)

    def write(self, data: bytes) -> None:
        self._buffer += data

    def write_eof(self) -> None:
        self._eof = True


class _ContainerProcess:
    def __init__(self, handle: ProcessHandle) -> None:
        self.stdout = _Stdout(handle)
        self.stdin = _Stdin(handle)

        async def poll() -> int | None:
            return handle.returncode

        self.wait = _Method(handle.wait)
        self.poll = _Method(poll)


class LocalModalSandbox:
    def __init__(self, env: LocalExecutionEnvironment) -> None:
        self._env = env
        self.calls: list[dict[str, Any]] = []
        self.handles: list[ProcessHandle] = []

        async def exec_(*args: str, **kwargs: Any) -> _ContainerProcess:
            self.calls.append({"args": list(args), **kwargs})
            size = PtySize() if kwargs.get("pty") else None
            handle = await env.spawn(
                args[0], list(args[1:]), cwd=kwargs.get("workdir"), env=kwargs.get("env"), pty=size
            )
            self.handles.append(handle)
            return _ContainerProcess(handle)

        self.exec = _Method(exec_)


def as_sandbox(fake: LocalModalSandbox) -> Sandbox:
    return cast("Sandbox", fake)


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


async def test_pipe_stdin_cwd_env_and_exit(tmp_path: Path) -> None:
    async with LocalExecutionEnvironment() as env:
        sandbox = LocalModalSandbox(env)
        script = (
            "import os, sys\n"
            "print(os.getcwd(), os.environ['CALL'], flush=True)\n"
            "print(sys.stdin.read().upper(), flush=True)\n"
            "sys.exit(6)\n"
        )
        handle = await ModalProcessHandle.start(
            as_sandbox(sandbox), [PY, "-c", script], cwd=str(tmp_path), env={"CALL": "c"}, size=None
        )
        async with handle:
            await handle.write(b"piped")
            await handle.close_stdin()
            lines = (await read_all(handle)).decode().splitlines()
            assert await handle.wait() == 6  # noqa: PLR2004
        call = sandbox.calls[0]
        assert call["text"] is False
        assert call["pty"] is False
    assert os.path.realpath(lines[0].split()[0]) == os.path.realpath(tmp_path)
    assert lines == [lines[0], "PIPED"]
    assert lines[0].endswith(" c")


async def test_signals_via_pid_file() -> None:
    async with LocalExecutionEnvironment() as env:
        sandbox = LocalModalSandbox(env)
        script = "import time; print('ready', flush=True); time.sleep(60)"
        handle = await ModalProcessHandle.start(
            as_sandbox(sandbox), [PY, "-c", script], cwd=None, env=None, size=None
        )
        async with handle:
            await read_until(handle.output(), b"ready")
            assert await handle.kill(grace=5) == 143  # noqa: PLR2004
            # The PID read from the file is the target's (the wrapper shell exec'd it).
            assert handle.pid == sandbox.handles[0].pid


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="the local stand-in runs host tools; resize needs GNU `stty -F` (Modal runs Linux)",
)
async def test_pty_initial_size_and_resize_via_stty() -> None:
    async with LocalExecutionEnvironment() as env:
        sandbox = LocalModalSandbox(env)
        script = (
            "import os, signal, sys\n"
            "s = os.get_terminal_size()\n"
            "print('tty', sys.stdout.isatty(), s.lines, s.columns, flush=True)\n"
            "def on_winch(*_):\n"
            "    s = os.get_terminal_size()\n"
            "    print('resized', s.lines, s.columns, flush=True)\n"
            "    sys.exit(4)\n"
            "signal.signal(signal.SIGWINCH, on_winch)\n"
            "print('ready', flush=True)\n"
            "signal.pause()\n"
        )
        handle = await ModalProcessHandle.start(
            as_sandbox(sandbox),
            [PY, "-c", script],
            cwd=None,
            env=None,
            size=PtySize(rows=33, cols=99),
        )
        async with handle:
            iterator = handle.output()
            before = await read_until(iterator, b"ready")
            await handle.resize(PtySize(rows=44, cols=111))
            after = await read_until(iterator, b"resized", timeout=10)
            assert await asyncio.wait_for(handle.wait(), timeout=10) == 4  # noqa: PLR2004
        assert sandbox.calls[0]["pty"] is True
    assert "tty True 33 99" in before.decode()
    assert "resized 44 111" in after.decode()


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("MODAL_TOKEN_ID") or (Path.home() / ".modal.toml").exists()),
    reason="Modal credentials not configured",
)
async def test_modal_spawn_integration() -> None:
    from exxec.modal_provider import ModalExecutionEnvironment

    async with ModalExecutionEnvironment() as env:
        async with await env.spawn("sh", ["-c", "read x; echo got-$x; exit 3"]) as handle:
            await handle.write(b"value\n")
            await handle.close_stdin()
            assert b"got-value" in await read_all(handle)
            assert await handle.wait() == 3  # noqa: PLR2004
        size = PtySize(rows=30, cols=90)
        async with await env.spawn("sh", ["-c", "stty size; exit 5"], pty=size) as handle:
            assert b"30 90" in await read_all(handle)
            assert await handle.wait() == 5  # noqa: PLR2004
