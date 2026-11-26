"""Interactive terminal sessions with a live, readable screen.

A `TerminalSession` runs a program on a PTY inside any environment with
`supports_pty`, feeds its output into a `ScreenEmulator` and exposes the screen as
plain text, so programs written for humans (shells, REPLs, TUIs, prompts) can be
driven programmatically, e.g. by agents:

    async with await TerminalSession.start(env, "python3") as term:
        await term.wait_for(">>> ")
        await term.send_line("1 + 1")
        screen = await term.wait_for_quiet()

Raw output stays available to other consumers (e.g. a websocket to a GUI terminal)
via `subscribe()`; all consumers see identical output.
"""

import asyncio
import contextlib
import re
from typing import TYPE_CHECKING, Self

from exxec.output_broadcast import OutputBroadcaster
from exxec.pty_manager import PtySize
from exxec.spawn_pty_manager import DEFAULT_SHELL_COMMAND
from exxec.terminal.keys import encode_key
from exxec.terminal.screen import (
    MODE_ALTERNATE_SCREEN,
    MODE_APPLICATION_CURSOR_KEYS,
    MODE_BRACKETED_PASTE,
    GhosttyScreen,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from types import TracebackType

    from exxec.base import ExecutionEnvironment
    from exxec.process_handle import ProcessHandle
    from exxec.terminal.screen import CursorPosition, ScreenEmulator


DEFAULT_QUIET_PERIOD = 0.3
"""Seconds without output after which the screen is considered settled."""

DEFAULT_WAIT_TIMEOUT = 10.0
"""Default upper bound for `wait_for` / `wait_for_quiet`."""

_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"


class TerminalSession:
    """A program running on a PTY, with an emulated screen."""

    def __init__(
        self,
        handle: ProcessHandle,
        screen: ScreenEmulator,
        *,
        answer_queries: bool = True,
    ) -> None:
        """Attach to a running PTY process. Prefer `TerminalSession.start()`.

        Args:
            handle: PTY process handle (its output is claimed by the session)
            screen: Screen emulator matching the PTY size
            answer_queries: Write the emulator's replies to terminal queries back to
                the process. Disable when a real terminal (e.g. a GUI client) is
                attached and answers queries itself.
        """
        if not handle.is_pty:
            msg = "TerminalSession requires a process attached to a PTY"
            raise ValueError(msg)
        self.handle = handle
        self.screen = screen
        self._broadcaster = OutputBroadcaster(handle)
        self._changed = asyncio.Condition()
        self._last_output = asyncio.get_running_loop().time()
        self._replies: asyncio.Queue[bytes] = asyncio.Queue()
        self._closed = False
        if answer_queries:
            screen.on_reply(self._replies.put_nowait)
        self._feeder = asyncio.create_task(self._feed())
        self._reply_writer = asyncio.create_task(self._write_replies())

    @classmethod
    async def start(
        cls,
        env: ExecutionEnvironment,
        command: str | None = None,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        size: PtySize | None = None,
        screen_factory: Callable[[PtySize], ScreenEmulator] = GhosttyScreen,
        answer_queries: bool = True,
    ) -> TerminalSession:
        """Spawn a program on a PTY in `env` and attach a screen to it.

        Args:
            env: Environment with `supports_pty`
            command: Program (with `args`) or shell command line (without);
                defaults to the user's login shell
            args: Program arguments
            cwd: Working directory
            env_vars: Additional environment variables
            size: Terminal size (defaults to 24x80)
            screen_factory: Creates the screen emulator for a given size
            answer_queries: See `__init__`

        Returns:
            The running session
        """
        size = size or PtySize()
        screen = screen_factory(size)
        try:
            handle = await env.spawn(
                command or DEFAULT_SHELL_COMMAND, args or None, cwd=cwd, env=env_vars, pty=size
            )
        except BaseException:
            screen.close()
            raise
        return cls(handle, screen, answer_queries=answer_queries)

    # --- output ---------------------------------------------------------------

    async def _feed(self) -> None:
        async for chunk in self._broadcaster.subscribe(replay=True):
            self.screen.feed(chunk)
            self._last_output = asyncio.get_running_loop().time()
            async with self._changed:
                self._changed.notify_all()
        async with self._changed:
            self._changed.notify_all()

    async def _write_replies(self) -> None:
        while True:
            reply = await self._replies.get()
            with contextlib.suppress(Exception):
                await self.handle.write(reply)

    def subscribe(self, *, replay: bool = True) -> AsyncIterator[bytes]:
        """Raw terminal output for additional consumers (e.g. a GUI terminal).

        Args:
            replay: Start with the retained recent output
        """
        return self._broadcaster.subscribe(replay=replay)

    @property
    def output_finished(self) -> bool:
        """Whether the program closed the terminal (usually: it exited)."""
        return self._broadcaster.finished and self._feeder.done()

    # --- screen ---------------------------------------------------------------

    def text(self) -> str:
        """The visible screen as plain text."""
        return self.screen.text()

    def scrollback(self) -> str:
        """Scrollback plus visible screen as plain text."""
        return self.screen.contents()

    @property
    def cursor(self) -> CursorPosition:
        """Current cursor position."""
        return self.screen.cursor

    @property
    def size(self) -> PtySize:
        """Current terminal size."""
        assert self.handle.pty_size is not None
        return self.handle.pty_size

    @property
    def alternate_screen(self) -> bool:
        """Whether a full-screen program (vim, less, htop, ...) is active."""
        return self.screen.mode(MODE_ALTERNATE_SCREEN)

    async def wait_for(
        self, pattern: str | re.Pattern[str], *, timeout: float = DEFAULT_WAIT_TIMEOUT
    ) -> re.Match[str]:
        """Wait until the visible screen matches `pattern`.

        The screen text has trailing blanks trimmed from each line, so match a
        prompt like `">>> "` as `">>>"`.

        Args:
            pattern: Compiled regular expression, or a plain string matched literally
            timeout: Seconds to wait

        Returns:
            The match on the screen text

        Raises:
            TimeoutError: If the pattern did not appear in time, or the program
                ended without producing it
        """
        regex = pattern if isinstance(pattern, re.Pattern) else re.compile(re.escape(pattern))
        async with asyncio.timeout(timeout):
            async with self._changed:
                while (match := regex.search(self.screen.text())) is None:
                    if self.output_finished:
                        msg = f"Terminal output ended without matching {regex.pattern!r}"
                        raise TimeoutError(msg)
                    await self._changed.wait()
        return match

    async def wait_for_quiet(
        self,
        quiet: float = DEFAULT_QUIET_PERIOD,
        *,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
    ) -> str:
        """Wait until no output arrived for `quiet` seconds and return the screen.

        Args:
            quiet: Seconds without output that count as settled
            timeout: Upper bound; the current screen is returned when reached

        Returns:
            The visible screen as plain text
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self.output_finished:
            now = loop.time()
            remaining_quiet = self._last_output + quiet - now
            if remaining_quiet <= 0 or now >= deadline:
                break
            await asyncio.sleep(min(remaining_quiet, deadline - now))
        return self.screen.text()

    # --- input ----------------------------------------------------------------

    async def send(self, data: str | bytes) -> None:
        """Send raw input (text is UTF-8 encoded; no key translation)."""
        await self.handle.write(data.encode() if isinstance(data, str) else data)

    async def send_line(self, text: str) -> None:
        """Type `text` followed by Enter."""
        await self.send(text.encode() + b"\r")

    async def send_keys(self, *keys: str) -> None:
        """Press keys by name, e.g. `send_keys("C-c")` or `send_keys("Up", "Enter")`.

        See `exxec.terminal.keys` for the supported names.
        """
        application_cursor = self.screen.mode(MODE_APPLICATION_CURSOR_KEYS)
        data = b"".join(encode_key(key, application_cursor=application_cursor) for key in keys)
        await self.handle.write(data)

    async def paste(self, text: str) -> None:
        """Paste text, using bracketed paste when the program enabled it."""
        data = text.encode()
        if self.screen.mode(MODE_BRACKETED_PASTE):
            data = _PASTE_START + data + _PASTE_END
        await self.handle.write(data)

    async def resize(self, size: PtySize) -> None:
        """Resize the terminal (program and screen)."""
        await self.handle.resize(size)
        self.screen.resize(size)

    # --- lifecycle ------------------------------------------------------------

    async def wait(self) -> int:
        """Wait for the program to exit and return its exit code."""
        return await self.handle.wait()

    @property
    def returncode(self) -> int | None:
        """Exit code once the program exited, else None."""
        return self.handle.returncode

    async def aclose(self) -> None:
        """Hang up the program (if running) and release all resources."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.handle.aclose()
        finally:
            await self._broadcaster.aclose()
            for task in (self._feeder, self._reply_writer):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self.screen.on_reply(None)
            self.screen.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
