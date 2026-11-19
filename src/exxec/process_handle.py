"""Provider-neutral handle for a single spawned process.

`ExecutionEnvironment.spawn()` returns a `ProcessHandle`. It is the one primitive
providers implement; process managers, PTY sessions and terminal emulation are
built on top of it.

Semantics shared by all implementations:

- **Output** is a single-owner byte stream: `output()` may be called once.
  For pipe processes stdout and stderr are merged; for PTY processes the
  terminal output is returned raw (escape sequences included).
- **Exit codes** follow the POSIX shell convention: a process terminated by
  signal N reports `128 + N`.
- **Termination** targets the whole process (group) where the provider allows it.
- **Cleanup**: use the handle as an async context manager, or call `aclose()`.
  Leaving the context kills a still-running process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import signal
from typing import TYPE_CHECKING, Literal, Self


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from exxec.pty_manager import PtySize


SignalName = Literal["INT", "TERM", "KILL", "HUP"]
"""Signals that can be sent to a spawned process."""

SIGNAL_NUMBERS: dict[SignalName, int] = {"INT": 2, "TERM": 15, "KILL": 9, "HUP": 1}
"""POSIX signal numbers, independent of the host platform."""

_SIGNAL_NUMBERS_BY_NAME: dict[str, int] = dict(SIGNAL_NUMBERS.items())

DEFAULT_KILL_GRACE = 5.0
"""Seconds to wait after SIGTERM before escalating to SIGKILL."""


class ProcessHandleError(RuntimeError):
    """Raised for invalid operations on a process handle."""


def normalize_returncode(code: int) -> int:
    """Convert a Python-style return code into the POSIX shell convention.

    Python's subprocess APIs report death by signal N as `-N`; shells report
    `128 + N`. All handles use the shell convention.

    Args:
        code: Return code as reported by the platform API

    Returns:
        Exit code following the shell convention
    """
    return 128 - code if code < 0 else code


def parse_signal_name(name: str) -> SignalName | None:
    """Map a signal name (with or without `SIG` prefix) to a supported `SignalName`.

    Args:
        name: Signal name, e.g. "TERM" or "SIGINT"

    Returns:
        The matching `SignalName`, or None if the signal is not supported
    """
    match name.removeprefix("SIG"):
        case "INT":
            return "INT"
        case "TERM":
            return "TERM"
        case "KILL":
            return "KILL"
        case "HUP":
            return "HUP"
        case _:
            return None


def signal_exit_code(name: str) -> int:
    """Exit code for a process terminated by the named signal.

    Args:
        name: Signal name with or without `SIG` prefix (e.g. "TERM", "SIGKILL")

    Returns:
        `128 + signal number`, or 128 + SIGTERM for unknown signal names
    """
    bare = name.removeprefix("SIG")
    if (number := _SIGNAL_NUMBERS_BY_NAME.get(bare)) is not None:
        return 128 + number
    try:
        return 128 + signal.Signals[f"SIG{bare}"].value
    except KeyError:
        return 128 + SIGNAL_NUMBERS["TERM"]


class ProcessHandle(ABC):
    """A process running inside an execution environment."""

    def __init__(self, *, pid: int | None, pty: PtySize | None) -> None:
        """Initialize common handle state.

        Args:
            pid: Process ID inside the environment, if known
            pty: Terminal size when attached to a PTY, None for pipe processes
        """
        self.pid = pid
        self._pty_size = pty
        self._output_claimed = False
        self._returncode: int | None = None
        self._wait_task: asyncio.Task[int] | None = None
        self._closed = False

    @property
    def is_pty(self) -> bool:
        """Whether the process is attached to a pseudo-terminal."""
        return self._pty_size is not None

    @property
    def pty_size(self) -> PtySize | None:
        """Current terminal size, None for pipe processes."""
        return self._pty_size

    @property
    def returncode(self) -> int | None:
        """Exit code once the process has exited (shell convention), else None."""
        return self._returncode

    def output(self) -> AsyncIterator[bytes]:
        """Stream process output until EOF.

        Returns:
            Iterator over raw output chunks

        Raises:
            ProcessHandleError: If output was already claimed by another consumer
        """
        if self._output_claimed:
            msg = "output() can only be consumed once per process"
            raise ProcessHandleError(msg)
        self._output_claimed = True
        return self._iter_output()

    async def wait(self) -> int:
        """Wait for the process to exit.

        Safe to call concurrently and to cancel; the underlying wait keeps running.

        Returns:
            Exit code (shell convention)
        """
        if self._wait_task is None:
            self._wait_task = asyncio.ensure_future(self._wait_and_record())
        return await asyncio.shield(self._wait_task)

    async def _wait_and_record(self) -> int:
        code = await self._wait()
        self._returncode = code
        return code

    async def resize(self, size: PtySize) -> None:
        """Resize the pseudo-terminal.

        Raises:
            ProcessHandleError: If the process is not attached to a PTY
        """
        if self._pty_size is None:
            msg = "Cannot resize a process that is not attached to a PTY"
            raise ProcessHandleError(msg)
        await self._resize(size)
        self._pty_size = size

    async def kill(self, *, grace: float = DEFAULT_KILL_GRACE) -> int:
        """Terminate the process: SIGTERM, then SIGKILL after `grace` seconds.

        Args:
            grace: Seconds to wait for a graceful exit before force-killing

        Returns:
            Exit code (shell convention)
        """
        if self._returncode is not None:
            return self._returncode
        await self.send_signal("TERM")
        try:
            return await asyncio.wait_for(self.wait(), timeout=grace)
        except TimeoutError:
            await self.send_signal("KILL")
            return await self.wait()

    async def aclose(self) -> None:
        """Kill the process if still running and release all resources."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._returncode is None and not await self._has_exited():
                await self.kill()
        finally:
            await self._release()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _has_exited(self) -> bool:
        """Non-blocking exit check; override when the provider can tell cheaply."""
        return self._wait_task is not None and self._wait_task.done()

    @abstractmethod
    def _iter_output(self) -> AsyncIterator[bytes]:
        """Yield raw output chunks until EOF."""

    @abstractmethod
    async def write(self, data: bytes) -> None:
        """Write to the process' stdin (or the terminal input for PTYs)."""

    @abstractmethod
    async def close_stdin(self) -> None:
        """Signal end of input (for PTYs: send EOT, i.e. Ctrl-D)."""

    @abstractmethod
    async def send_signal(self, sig: SignalName) -> None:
        """Send a signal to the process (group). No-op if it already exited."""

    @abstractmethod
    async def _wait(self) -> int:
        """Wait for exit and return the exit code (shell convention)."""

    @abstractmethod
    async def _resize(self, size: PtySize) -> None:
        """Apply a new terminal size (only called for PTY processes)."""

    @abstractmethod
    async def _release(self) -> None:
        """Release sockets, file descriptors and channels."""
