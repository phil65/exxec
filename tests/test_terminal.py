"""Tests for key encoding, the Ghostty screen emulator and `TerminalSession`."""

import re
import sys

import pytest

from exxec import LocalExecutionEnvironment
from exxec.pty_manager import PtySize
from exxec.terminal import TerminalSession, encode_key


PY = sys.executable

needs_vt = [
    pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX PTYs and libghostty-vt")
]


def test_encode_named_and_control_keys() -> None:
    assert encode_key("Enter") == b"\r"
    assert encode_key("tab") == b"\t"
    assert encode_key("C-c") == b"\x03"
    assert encode_key("C-Z") == b"\x1a"
    assert encode_key("C-[") == b"\x1b"
    assert encode_key("M-x") == b"\x1bx"
    assert encode_key("M-C-c") == b"\x1b\x03"
    assert encode_key("F5") == b"\x1b[15~"
    assert encode_key("ä") == "ä".encode()


def test_encode_cursor_keys_respect_application_mode() -> None:
    assert encode_key("Up") == b"\x1b[A"
    assert encode_key("Up", application_cursor=True) == b"\x1bOA"
    assert encode_key("End", application_cursor=True) == b"\x1bOF"


def test_encode_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        encode_key("Hyper")
    with pytest.raises(ValueError, match="Unknown control key"):
        encode_key("C-1")


@needs_vt[0]
def test_screen_collapses_redraws_and_strips_styles() -> None:
    pytest.importorskip("pyghostty")
    from exxec.terminal import GhosttyScreen

    screen = GhosttyScreen(PtySize(rows=5, cols=30))
    screen.feed(b"start\r\nprogress  10%\rprogress 100%\r\n\x1b[1;32mdone\x1b[0m")
    assert screen.text() == "start\nprogress 100%\ndone"
    assert screen.cursor.y == 2  # noqa: PLR2004
    screen.close()


@needs_vt[0]
@pytest.mark.skipif(
    sys.platform == "linux" and sys.version_info >= (3, 14),
    reason="intermittently hangs on Linux/3.14 when run after the full suite",
)
async def test_python_repl_roundtrip() -> None:
    pytest.importorskip("pyghostty")
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(env, PY, ["-q"]) as term,
    ):
        await term.wait_for(">>>")
        await term.send_line("6 * 7")
        match = await term.wait_for(re.compile(r"^42$", re.MULTILINE), timeout=5)
        assert match.group(0) == "42"
        await term.send_keys("C-d")
        assert await term.wait() == 0


@needs_vt[0]
async def test_ctrl_c_interrupts_running_command() -> None:
    pytest.importorskip("pyghostty")
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(env, PY, ["-q"]) as term,
    ):
        await term.wait_for(">>>")
        await term.send_line("import time; time.sleep(30)")
        await term.wait_for_quiet(0.3)
        await term.send_keys("C-c")
        await term.wait_for("KeyboardInterrupt")


@needs_vt[0]
async def test_progress_output_shows_final_state() -> None:
    pytest.importorskip("pyghostty")
    script = (
        "import sys, time\n"
        "for i in range(0, 101, 20):\n"
        "    sys.stdout.write(f'\\rprogress {i:3d}%'); sys.stdout.flush(); time.sleep(0.02)\n"
        "print('\\nfinished')\n"
    )
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(env, PY, ["-c", script]) as term,
    ):
        assert await term.wait() == 0
        await term.wait_for("finished")
        assert term.text().splitlines() == ["progress 100%", "finished"]


@needs_vt[0]
async def test_alternate_screen_detection_and_restore() -> None:
    pytest.importorskip("pyghostty")
    script = (
        "import sys\n"
        "print('before', flush=True)\n"
        "sys.stdout.write('\\x1b[?1049h\\x1b[2J\\x1b[HFULLSCREEN APP'); sys.stdout.flush()\n"
        "input()\n"
        "sys.stdout.write('\\x1b[?1049l'); sys.stdout.flush()\n"
        "print('after')\n"
    )
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(env, PY, ["-c", script]) as term,
    ):
        await term.wait_for("FULLSCREEN APP")
        assert term.alternate_screen is True
        assert "before" not in term.text()
        await term.send_keys("Enter")
        await term.wait_for("after")
        assert term.alternate_screen is False
        assert "before" in term.text()


@needs_vt[0]
async def test_answers_cursor_position_queries() -> None:
    pytest.importorskip("pyghostty")
    # Asks the terminal for the cursor position and blocks until the reply arrives,
    # like many TUIs and line editors do on startup.
    script = (
        "import sys, termios, tty\n"
        "old = termios.tcgetattr(0)\n"
        "tty.setraw(0)\n"
        "sys.stdout.write('abc\\x1b[6n'); sys.stdout.flush()\n"
        "reply = b''\n"
        "while not reply.endswith(b'R'):\n"
        "    reply += sys.stdin.buffer.read(1)\n"
        "termios.tcsetattr(0, termios.TCSADRAIN, old)\n"
        "print('\\r\\nreply', reply[1:].decode())\n"
    )
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(env, PY, ["-c", script]) as term,
    ):
        await term.wait_for("reply", timeout=5)
        assert "reply [1;4R" in term.text()


@needs_vt[0]
async def test_wait_for_raises_when_program_exits_without_match() -> None:
    pytest.importorskip("pyghostty")
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(env, PY, ["-c", "print('bye')"]) as term,
    ):
        with pytest.raises(TimeoutError, match="ended without matching"):
            await term.wait_for("never-printed", timeout=5)


@needs_vt[0]
async def test_raw_subscribers_see_same_output() -> None:
    pytest.importorskip("pyghostty")
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(env, PY, ["-c", "print('shared output')"]) as term,
    ):
        raw = b"".join([chunk async for chunk in term.subscribe()])
        await term.wait_for("shared output")
    assert b"shared output" in raw


@needs_vt[0]
async def test_bracketed_paste_and_resize() -> None:
    pytest.importorskip("pyghostty")
    script = (
        "import os, sys\n"
        "sys.stdout.write('\\x1b[?2004h'); sys.stdout.flush()\n"
        "print('ready', flush=True)\n"
        "print('got', repr(input()), flush=True)\n"
        "input()\n"
        "size = os.get_terminal_size()\n"
        "print('size', size.lines, size.columns)\n"
    )
    async with (
        LocalExecutionEnvironment() as env,
        await TerminalSession.start(
            env, PY, ["-c", script], size=PtySize(rows=10, cols=60)
        ) as term,
    ):
        await term.wait_for("ready")
        await term.paste("pasted")
        await term.send_keys("Enter")
        await term.wait_for("got '\\x1b[200~pasted\\x1b[201~'")
        await term.resize(PtySize(rows=15, cols=70))
        assert term.size == PtySize(rows=15, cols=70)
        await term.send_keys("Enter")
        await term.wait_for("size 15 70")
