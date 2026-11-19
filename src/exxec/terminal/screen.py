"""Terminal screen emulation: turning raw PTY output into screen state.

`ScreenEmulator` is the interface `TerminalSession` relies on. `GhosttyScreen`
implements it with libghostty-vt via pyghostty (install `exxec[vt]`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Callable

    from exxec.pty_manager import PtySize


DEFAULT_SCROLLBACK_LINES = 10000

# DEC private modes commonly needed by callers.
MODE_APPLICATION_CURSOR_KEYS = 1
MODE_ALTERNATE_SCREEN = 1049
MODE_BRACKETED_PASTE = 2004


@dataclass(frozen=True)
class CursorPosition:
    """Zero-based cursor position on the visible screen."""

    x: int
    y: int


class ScreenEmulator(Protocol):
    """A virtual terminal that interprets output and exposes the resulting screen."""

    def feed(self, data: bytes) -> None:
        """Process output written by the program."""
        ...

    def text(self) -> str:
        """Visible screen as plain text (rows as displayed, trailing blanks trimmed)."""
        ...

    def contents(self) -> str:
        """Scrollback plus visible screen as plain text, soft-wraps joined."""
        ...

    @property
    def cursor(self) -> CursorPosition:
        """Current cursor position."""
        ...

    def resize(self, size: PtySize) -> None:
        """Change the screen size (reflowing the primary screen)."""
        ...

    def mode(self, mode: int) -> bool:
        """Whether a DEC private mode (e.g. 1049, 2004) is currently enabled."""
        ...

    def on_reply(self, callback: Callable[[bytes], None] | None) -> None:
        """Register a callback for replies the terminal sends back to the program.

        Programs query the terminal (cursor position, device attributes, ...) and
        expect an answer on their input; without a real terminal attached, these
        replies must be written back to the process.
        """
        ...

    def close(self) -> None:
        """Release native resources."""
        ...


class GhosttyScreen:
    """`ScreenEmulator` backed by libghostty-vt (pyghostty)."""

    def __init__(self, size: PtySize, *, scrollback: int = DEFAULT_SCROLLBACK_LINES) -> None:
        """Create a virtual terminal.

        Args:
            size: Initial screen size
            scrollback: Lines of scrollback to retain

        Raises:
            ImportError: If pyghostty is not installed (install `exxec[vt]`)
        """
        try:
            import pyghostty  # type: ignore[import-untyped]
        except ImportError as e:
            msg = "Terminal emulation requires pyghostty: install `exxec[vt]`"
            raise ImportError(msg) from e
        self._terminal = pyghostty.Terminal(cols=size.cols, rows=size.rows, scrollback=scrollback)
        self._closed = False

    def feed(self, data: bytes) -> None:
        self._terminal.feed(data)

    def text(self) -> str:
        return str(self._terminal.text())

    def contents(self) -> str:
        return str(self._terminal.contents())

    @property
    def cursor(self) -> CursorPosition:
        x, y = self._terminal.cursor
        return CursorPosition(x=int(x), y=int(y))

    def resize(self, size: PtySize) -> None:
        self._terminal.resize(size.cols, size.rows)

    def mode(self, mode: int) -> bool:
        return bool(self._terminal.mode(mode))

    def on_reply(self, callback: Callable[[bytes], None] | None) -> None:
        self._terminal.on_reply(callback)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._terminal.close()
