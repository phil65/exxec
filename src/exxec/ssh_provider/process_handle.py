"""Process handle for SSH execution via asyncssh.

Each spawned process is its own SSH session channel:

- PTY processes request a terminal (`term_type` / `term_size`); resizes are sent as
  SSH window-change requests.
- Pipe processes redirect stderr into stdout on the channel.
- Signals are sent as SSH signal requests (supported by OpenSSH >= 7.9). Because
  servers may ignore them, a KILL additionally closes the channel, which makes
  sshd hang up the session.
"""

import contextlib
from typing import TYPE_CHECKING, Any

from exxec.process_handle import ProcessHandle, ProcessHandleError, signal_exit_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from asyncssh import SSHClientConnection, SSHClientProcess

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


READ_CHUNK_SIZE = 65536
EOT = b"\x04"

CONNECTION_LOST_EXIT_CODE = 255
"""Exit code when the channel closed without an exit status (ssh(1) convention)."""


class SshProcessHandle(ProcessHandle):
    """A process running in an SSH session channel."""

    def __init__(self, process: SSHClientProcess[Any], size: PtySize | None) -> None:
        """Wrap a started asyncssh client process.

        Args:
            process: Process created with `encoding=None` (bytes I/O)
            size: Terminal size when a PTY was requested
        """
        super().__init__(pid=None, pty=size)
        self._process = process
        self._force_closed = False
        self._stdin_closed = False

    @classmethod
    async def start(
        cls,
        connection: SSHClientConnection,
        command_line: str,
        *,
        size: PtySize | None,
    ) -> SshProcessHandle:
        """Open a session channel running `command_line`.

        Args:
            connection: Established SSH connection
            command_line: Full POSIX shell command line (cwd/env already applied)
            size: Request a PTY of this size, None for pipes

        Returns:
            Handle for the started process
        """
        import asyncssh

        if size is not None:
            process = await connection.create_process(
                command_line,
                term_type="xterm-256color",
                term_size=(size.cols, size.rows),
                encoding=None,
            )
        else:
            process = await connection.create_process(
                command_line,
                stderr=asyncssh.STDOUT,
                encoding=None,
            )
        return cls(process, size)

    async def _iter_output(self) -> AsyncIterator[bytes]:
        while chunk := await self._process.stdout.read(READ_CHUNK_SIZE):
            yield chunk

    async def write(self, data: bytes) -> None:
        if self._stdin_closed or self._force_closed:
            msg = "stdin is closed"
            raise ProcessHandleError(msg)
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def close_stdin(self) -> None:
        if self.is_pty:
            await self.write(EOT)
            return
        if not self._force_closed and not self._stdin_closed:
            self._stdin_closed = True
            with contextlib.suppress(BrokenPipeError, OSError):
                self._process.stdin.write_eof()

    async def send_signal(self, sig: SignalName) -> None:
        if self._returncode is not None or self._force_closed:
            return
        with contextlib.suppress(OSError):
            self._process.send_signal(sig)
        if sig == "KILL":
            # Not every server honours signal requests; closing the channel
            # makes sshd hang up the session as a last resort.
            self._force_closed = True
            self._process.close()

    async def _wait(self) -> int:
        await self._process.wait_closed()
        # asyncssh reports exit_status -1 for signal deaths, so check the signal first.
        if self._process.exit_signal is not None:
            name, _core_dumped, _msg, _lang = self._process.exit_signal
            return signal_exit_code(name)
        status = self._process.exit_status
        if status is not None and status >= 0:
            return int(status)
        if self._force_closed:
            return signal_exit_code("KILL")
        return CONNECTION_LOST_EXIT_CODE

    async def _resize(self, size: PtySize) -> None:
        self._process.change_terminal_size(size.cols, size.rows)

    async def _release(self) -> None:
        self._process.close()
        await self._process.wait_closed()
