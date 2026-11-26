"""Process handle for Docker containers via the Docker Engine exec API.

- I/O goes over the hijacked exec socket, used non-blocking with the event loop.
  Without a TTY Docker multiplexes stdout/stderr into framed chunks
  (8-byte header: stream type, 3 padding bytes, big-endian payload size), which
  are demultiplexed and merged; with a TTY the stream is raw terminal output.
  Docker reads stdout and stderr from separate pipes, so their relative order in
  the merged output is not guaranteed (within each stream it is).
- Docker has no API to signal an exec'd process. The command is started through
  `sh`, which records its PID in a pidfile and then `exec`s the target, so the
  recorded PID is the target's. Signals are delivered with `kill` via a second
  exec, to the process group first and the PID as fallback.
- Exit codes come from `exec_inspect` once the process stopped running.
"""

import asyncio
import contextlib
import socket
import struct
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from exxec.process_handle import ProcessHandle, ProcessHandleError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.process_handle import SignalName
    from exxec.pty_manager import PtySize


READ_CHUNK_SIZE = 65536
EOT = b"\x04"
_FRAME_HEADER = struct.Struct(">BxxxL")
_STDOUT, _STDERR = 1, 2
_PIDFILE_WAIT_SECONDS = 2.0
_INSPECT_INTERVAL_MAX = 0.2


class DockerExecAPI(Protocol):
    """The subset of `docker.APIClient` used for exec-based processes."""

    def exec_create(
        self,
        container: str,
        cmd: list[str],
        *,
        stdin: bool,
        tty: bool,
        environment: dict[str, str] | None,
        workdir: str | None,
    ) -> dict[str, Any]: ...

    def exec_start(self, exec_id: str, *, tty: bool, socket: bool, demux: bool) -> Any: ...

    def exec_resize(self, exec_id: str, *, height: int, width: int) -> None: ...

    def exec_inspect(self, exec_id: str) -> dict[str, Any]: ...


def _raw_socket(stream: Any) -> socket.socket:
    """Extract the underlying socket from docker-py's exec socket wrapper."""
    if isinstance(stream, socket.socket):
        return stream
    # docker-py wraps unix/tcp sockets in socket.SocketIO; `_sock` is its public-in-practice
    # attribute and the only way to obtain the raw socket for non-blocking use.
    raw = stream._sock
    if not isinstance(raw, socket.socket):
        msg = f"Unsupported Docker exec socket type: {type(raw).__name__}"
        raise ProcessHandleError(msg)
    return raw


class DockerProcessHandle(ProcessHandle):
    """A process running in a Docker container via `docker exec`."""

    def __init__(
        self,
        api: DockerExecAPI,
        container_id: str,
        exec_id: str,
        sock: socket.socket,
        pidfile: str,
        size: PtySize | None,
    ) -> None:
        """Wrap a started exec instance.

        Args:
            api: Docker API client
            container_id: Container the process runs in
            exec_id: Exec instance ID
            sock: Raw hijacked exec socket (set to non-blocking)
            pidfile: Path inside the container holding the process PID
            size: Terminal size when started with a TTY
        """
        super().__init__(pid=None, pty=size)
        self._api = api
        self._container_id = container_id
        self._exec_id = exec_id
        self._sock = sock
        self._pidfile = pidfile
        self._stdin_closed = False
        self._sock_closed = False

    @classmethod
    async def start(
        cls,
        api: DockerExecAPI,
        container_id: str,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        size: PtySize | None,
    ) -> DockerProcessHandle:
        """Create and start an exec instance running argv.

        Args:
            api: Docker API client
            container_id: Target container
            argv: Program and arguments (use `["sh", "-c", line]` for shell lines)
            cwd: Working directory inside the container
            env: Environment variables for the process
            size: Allocate a TTY of this size, None for pipes

        Returns:
            Handle for the started process
        """
        pidfile = f"/tmp/.exxec-{uuid.uuid4().hex}.pid"
        # `$0` is the pidfile; "$@" is the target argv, which replaces the shell.
        wrapped = ["sh", "-c", 'echo $$ > "$0" && exec "$@"', pidfile, *argv]
        tty = size is not None
        if tty:
            env = {"TERM": "xterm-256color", **(env or {})}
        created = await asyncio.to_thread(
            api.exec_create,
            container_id,
            wrapped,
            stdin=True,
            tty=tty,
            environment=env or None,
            workdir=cwd,
        )
        exec_id = str(created["Id"])
        stream = await asyncio.to_thread(api.exec_start, exec_id, tty=tty, socket=True, demux=False)
        sock = _raw_socket(stream)
        sock.setblocking(False)
        handle = cls(api, container_id, exec_id, sock, pidfile, size)
        if size is not None:
            await handle._resize(size)
        return handle

    async def _recv(self) -> bytes:
        if self._sock_closed:
            return b""
        loop = asyncio.get_running_loop()
        try:
            return await loop.sock_recv(self._sock, READ_CHUNK_SIZE)
        except ConnectionResetError, OSError:
            if self._sock_closed:
                return b""
            raise

    async def _iter_output(self) -> AsyncIterator[bytes]:
        if self.is_pty:
            while chunk := await self._recv():
                yield chunk
            return
        buffer = bytearray()
        while data := await self._recv():
            buffer += data
            while len(buffer) >= _FRAME_HEADER.size:
                stream_type, length = _FRAME_HEADER.unpack_from(buffer)
                end = _FRAME_HEADER.size + length
                if len(buffer) < end:
                    break
                payload = bytes(buffer[_FRAME_HEADER.size : end])
                del buffer[:end]
                if stream_type in (_STDOUT, _STDERR) and payload:
                    yield payload

    async def write(self, data: bytes) -> None:
        if self._stdin_closed or self._sock_closed:
            msg = "stdin is closed"
            raise ProcessHandleError(msg)
        await asyncio.get_running_loop().sock_sendall(self._sock, data)

    async def close_stdin(self) -> None:
        if self.is_pty:
            await self.write(EOT)
            return
        if not self._stdin_closed and not self._sock_closed:
            self._stdin_closed = True
            with contextlib.suppress(OSError):
                self._sock.shutdown(socket.SHUT_WR)

    async def _read_pid(self) -> int | None:
        if self.pid is not None:
            return self.pid
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _PIDFILE_WAIT_SECONDS
        while True:
            output = await self._run_helper(["cat", self._pidfile])
            if output.strip().isdigit():
                self.pid = int(output.strip())
                return self.pid
            if loop.time() >= deadline or await self._has_exited():
                return None
            await asyncio.sleep(0.05)

    async def _run_helper(self, argv: list[str]) -> str:
        """Run a short helper command in the container and return its stdout."""
        created = await asyncio.to_thread(
            self._api.exec_create,
            self._container_id,
            argv,
            stdin=False,
            tty=False,
            environment=None,
            workdir=None,
        )
        exec_id = str(created["Id"])
        stream = await asyncio.to_thread(
            self._api.exec_start, exec_id, tty=False, socket=False, demux=True
        )
        stdout = stream[0] if isinstance(stream, tuple) else stream
        return stdout.decode(errors="replace") if isinstance(stdout, bytes) else ""

    async def send_signal(self, sig: SignalName) -> None:
        if self._returncode is not None or await self._has_exited():
            return
        pid = await self._read_pid()
        if pid is None:
            return
        script = f"kill -s {sig} -- -{pid} 2>/dev/null || kill -s {sig} {pid} 2>/dev/null; true"
        await self._run_helper(["sh", "-c", script])

    async def _inspect(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._api.exec_inspect, self._exec_id)

    async def _has_exited(self) -> bool:
        if self._returncode is not None:
            return True
        return not bool((await self._inspect()).get("Running"))

    async def _wait(self) -> int:
        interval = 0.02
        while True:
            info = await self._inspect()
            if not info.get("Running") and info.get("ExitCode") is not None:
                return int(info["ExitCode"])
            await asyncio.sleep(interval)
            interval = min(interval * 2, _INSPECT_INTERVAL_MAX)

    async def _resize(self, size: PtySize) -> None:
        await asyncio.to_thread(
            self._api.exec_resize, self._exec_id, height=size.rows, width=size.cols
        )

    async def _release(self) -> None:
        if not self._sock_closed:
            self._sock_closed = True
            self._sock.close()
        with contextlib.suppress(Exception):
            await self._run_helper(["rm", "-f", self._pidfile])
