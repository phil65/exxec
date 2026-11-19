"""Process handles for local execution.

- `LocalPipeProcessHandle`: asyncio subprocess with piped stdin/stdout
  (stderr merged into stdout), started in its own process group on POSIX.
- `LocalPtyProcessHandle`: process attached to a pseudo-terminal (POSIX only).
  No Python-level fork is involved (forking a multi-threaded process is unsafe):
  asyncio's subprocess machinery starts a minimal isolated Python helper with the
  terminal's slave side as stdio; the helper makes it the controlling terminal via
  `os.login_tty()` and execs the target program. I/O uses the non-blocking master
  file descriptor with event-loop readiness callbacks, and exit is observed via
  asyncio's child watcher, so no threads are involved at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import struct
import sys
from typing import TYPE_CHECKING

from exxec.process_handle import (
    SIGNAL_NUMBERS,
    ProcessHandle,
    ProcessHandleError,
    normalize_returncode,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


READ_CHUNK_SIZE = 65536
EOT = b"\x04"

_LOGIN_TTY_HELPER = "import os, sys; os.login_tty(0); os.execvp(sys.argv[1], sys.argv[1:])"
"""Run in the child: become session leader with stdin's tty as controlling terminal."""


class LocalPipeProcessHandle(ProcessHandle):
    """A local subprocess with piped I/O."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        """Wrap a started asyncio subprocess.

        Args:
            process: Process started with stdin=PIPE, stdout=PIPE, stderr=STDOUT
        """
        super().__init__(pid=process.pid, pty=None)
        self._process = process

    @classmethod
    async def start(
        cls,
        argv: list[str] | None,
        command_line: str | None,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> LocalPipeProcessHandle:
        """Start a process either from argv (direct exec) or a shell command line.

        Args:
            argv: Program and arguments for direct execution
            command_line: Shell command line, used when argv is None
            cwd: Working directory
            env: Full environment for the child (None inherits)

        Returns:
            Handle for the started process
        """
        new_session = sys.platform != "win32"
        pipe = asyncio.subprocess.PIPE
        if argv is not None:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=pipe,
                stdout=pipe,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=env,
                start_new_session=new_session,
            )
        else:
            assert command_line is not None
            process = await asyncio.create_subprocess_shell(
                command_line,
                stdin=pipe,
                stdout=pipe,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                env=env,
                start_new_session=new_session,
            )
        return cls(process)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        stdout = self._process.stdout
        if stdout is None:
            return
        while chunk := await stdout.read(READ_CHUNK_SIZE):
            yield chunk

    async def write(self, data: bytes) -> None:
        stdin = self._process.stdin
        if stdin is None or stdin.is_closing():
            msg = "stdin is closed"
            raise ProcessHandleError(msg)
        stdin.write(data)
        await stdin.drain()

    async def close_stdin(self) -> None:
        stdin = self._process.stdin
        if stdin is not None and not stdin.is_closing():
            stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await stdin.wait_closed()

    async def send_signal(self, sig: SignalName) -> None:
        if self._process.returncode is not None:
            return
        try:
            if sys.platform == "win32":
                # Windows has no process groups/POSIX signals for subprocesses.
                if sig == "KILL":
                    self._process.kill()
                else:
                    self._process.terminate()
            else:
                os.killpg(self._process.pid, SIGNAL_NUMBERS[sig])
        except ProcessLookupError:
            pass

    async def _wait(self) -> int:
        return normalize_returncode(await self._process.wait())

    async def _has_exited(self) -> bool:
        return self._process.returncode is not None

    async def _resize(self, size: PtySize) -> None:
        msg = "Pipe processes have no terminal"
        raise ProcessHandleError(msg)

    async def _release(self) -> None:
        await self.close_stdin()


class LocalPtyProcessHandle(ProcessHandle):
    """A local process attached to a pseudo-terminal (POSIX only)."""

    def __init__(self, process: asyncio.subprocess.Process, master_fd: int, size: PtySize) -> None:
        """Wrap a started process and the master side of its terminal.

        Args:
            process: Process whose stdio is the terminal's slave side
            master_fd: Master file descriptor (owned by the handle from now on)
            size: Initial terminal size
        """
        super().__init__(pid=process.pid, pty=size)
        self._process = process
        self._fd = master_fd
        os.set_blocking(self._fd, False)
        self._fd_closed = False
        self._pending: set[asyncio.Future[None]] = set()

    @classmethod
    async def start(
        cls,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str],
        size: PtySize,
    ) -> LocalPtyProcessHandle:
        """Spawn argv attached to a new pseudo-terminal.

        Args:
            argv: Program and arguments
            cwd: Working directory
            env: Full environment for the child
            size: Initial terminal size

        Returns:
            Handle for the spawned process

        Raises:
            ProcessHandleError: On platforms without PTY support
        """
        if sys.platform == "win32":
            msg = "PTY processes are not supported on Windows"
            raise ProcessHandleError(msg)
        master_fd, slave_fd = os.openpty()
        try:
            _set_window_size(master_fd, size)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                "-c",
                _LOGIN_TTY_HELPER,
                *argv,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=cwd,
                env=env,
            )
        except BaseException:
            os.close(master_fd)
            raise
        finally:
            # The child holds its own copies; the parent only needs the master.
            os.close(slave_fd)
        return cls(process, master_fd, size)

    async def _wait_fd(self, *, writable: bool) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def ready() -> None:
            if not future.done():
                future.set_result(None)

        if writable:
            loop.add_writer(self._fd, ready)
        else:
            loop.add_reader(self._fd, ready)
        self._pending.add(future)
        try:
            await future
        finally:
            self._pending.discard(future)
            if not self._fd_closed:
                if writable:
                    loop.remove_writer(self._fd)
                else:
                    loop.remove_reader(self._fd)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        while not self._fd_closed:
            try:
                chunk = os.read(self._fd, READ_CHUNK_SIZE)
            except BlockingIOError:
                await self._wait_fd(writable=False)
                continue
            except OSError as e:
                # Linux reports EIO on the master once all slave fds are closed.
                if e.errno == errno.EIO:
                    return
                raise
            if not chunk:
                return
            yield chunk

    async def write(self, data: bytes) -> None:
        if self._fd_closed:
            msg = "Terminal is closed"
            raise ProcessHandleError(msg)
        view = memoryview(data)
        while view:
            try:
                written = os.write(self._fd, view)
            except BlockingIOError:
                await self._wait_fd(writable=True)
                if self._fd_closed:
                    msg = "Terminal is closed"
                    raise ProcessHandleError(msg) from None
                continue
            view = view[written:]

    async def close_stdin(self) -> None:
        # A terminal has no separate input stream; EOT makes the line
        # discipline deliver EOF to the reading program.
        await self.write(EOT)

    async def send_signal(self, sig: SignalName) -> None:
        if self._process.returncode is not None:
            return
        if sys.platform == "win32":
            msg = "PTY processes are not supported on Windows"
            raise ProcessHandleError(msg)
        # login_tty() makes the child a session (and process group) leader.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self._process.pid, SIGNAL_NUMBERS[sig])

    async def _wait(self) -> int:
        return normalize_returncode(await self._process.wait())

    async def _has_exited(self) -> bool:
        return self._process.returncode is not None

    async def _resize(self, size: PtySize) -> None:
        # The kernel delivers SIGWINCH to the terminal's foreground process group.
        _set_window_size(self._fd, size)

    async def _release(self) -> None:
        if self._fd_closed:
            return
        loop = asyncio.get_running_loop()
        loop.remove_reader(self._fd)
        loop.remove_writer(self._fd)
        self._fd_closed = True
        # Wake readers/writers waiting on the fd so they observe the closed state.
        for future in self._pending:
            if not future.done():
                future.set_result(None)
        os.close(self._fd)


def _set_window_size(fd: int, size: PtySize) -> None:
    if sys.platform == "win32":
        msg = "PTY processes are not supported on Windows"
        raise ProcessHandleError(msg)
    import fcntl
    import termios

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", size.rows, size.cols, 0, 0))
