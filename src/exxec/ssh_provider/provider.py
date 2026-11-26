"""SSH execution environment that runs code on remote machines."""

import contextlib
from contextlib import AsyncExitStack
import shlex
import time
from typing import TYPE_CHECKING, Any, Protocol, Self

from exxec.base import ExecutionEnvironment
from exxec.command_line import build_remote_command_line, with_timeout
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.exceptions import NotInitializedError
from exxec.models import ExecutionResult
from exxec.parse_output import wrap_command


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from asyncssh import SSHClientConnection, SSHCompletedProcess
    from asyncssh.misc import _ACMWrapper
    from upathtools.filesystems.base.wrapper import WrapperFileSystem

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo
    from exxec.process_handle import ProcessHandle
    from exxec.pty_manager import PtySize


class _AsyncClosable(Protocol):
    async def close(self) -> None: ...


class SshExecutionEnvironment(ExecutionEnvironment):
    """Executes code on remote machines via SSH using asyncssh."""

    supports_spawn = True
    supports_pty = True

    def __init__(
        self,
        host: str,
        username: str,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        password: str | None = None,
        private_key_path: str | None = None,
        port: int = 22,
        default_command_timeout: float | None = 60.0,
        language: Language = "python",
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
        **ssh_kwargs: Any,
    ) -> None:
        """Initialize SSH environment.

        Args:
            host: Remote host to connect to
            username: SSH username
            password: SSH password (if not using key auth)
            dependencies: List of dependencies to install
            lifespan_handler: lifespan handler during execution
            private_key_path: Path to SSH private key file
            port: SSH port
            default_command_timeout: Default timeout for command execution in seconds.
                If None, commands run without timeout unless explicitly specified.
            language: Programming language to use
            cwd: Remote working directory (auto-generated if None)
            env_vars: Environment variables to set for all executions
            inherit_env: If True, inherit environment variables from os.environ
            **ssh_kwargs: Additional arguments passed to asyncssh.connect()
        """
        super().__init__(
            lifespan_handler=lifespan_handler,
            dependencies=dependencies,
            cwd=cwd,
            env_vars=env_vars,
            inherit_env=inherit_env,
            default_command_timeout=default_command_timeout,
        )
        self.host = host
        self.username = username
        self.password = password
        self.private_key_path = private_key_path
        self.port = port
        self.language = language
        self.ssh_kwargs = ssh_kwargs

        self._connection_cm: _ACMWrapper[SSHClientConnection] | None = None
        self.connection: SSHClientConnection | None = None
        self._remote_work_dir: str | None = None
        self._fs: WrapperFileSystem | None = None
        # The SFTP filesystem uses its own connection; tracked to close it on exit.
        self._sftp_pool: _AsyncClosable | None = None
        self._sftp_stack: AsyncExitStack | None = None

    def _ensure_connected(self) -> SSHClientConnection:
        """Validate that SSH connection is established.

        Returns:
            The SSH connection.

        Raises:
            NotInitializedError: If not connected via async context manager.
        """
        if self.connection is None:
            raise NotInitializedError("SSH")
        return self.connection

    def _prepend_env_vars(self, command: str) -> str:
        """Prepend (shell-quoted) environment variable exports to a command line."""
        return build_remote_command_line(command, None, self.get_env() or None)

    async def run(self, command: str, *, input: str | None = None) -> SSHCompletedProcess:  # noqa: A002
        """Run a command line on the remote machine with login shell.

        Args:
            command: Shell command line
            input: Data for the command's stdin
        """
        connection = self._ensure_connected()
        command = self._prepend_env_vars(command)
        return await connection.run(wrap_command(command), input=input)

    async def __aenter__(self) -> Self:
        """Establish SSH connection and set up remote environment."""
        import asyncssh

        # Build connection arguments
        self.connect_kwargs = {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            **self.ssh_kwargs,
        }
        # Add authentication
        if self.private_key_path:
            self.connect_kwargs["client_keys"] = [self.private_key_path]
        elif self.password:
            self.connect_kwargs["password"] = self.password
        # Create and enter the asyncssh connection context manager
        self._connection_cm = asyncssh.connect(**self.connect_kwargs)
        self.connection = await self._connection_cm.__aenter__()
        try:
            await self._set_up()
        except BaseException as e:
            # Don't leak the connection (or the temp dir) when setup fails.
            with contextlib.suppress(Exception):
                await self.__aexit__(type(e), e, e.__traceback__)
            raise
        return self

    async def _set_up(self) -> None:
        # Base setup (tool server, OS detection) runs commands, so it needs the connection.
        await super().__aenter__()
        # Set up remote working directory
        if self.cwd:
            self._remote_work_dir = self.cwd
        else:
            # Create temporary directory
            result = await self.run("mktemp -d")
            if result.returncode != 0:
                stderr = (
                    result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
                )
                msg = f"Failed to create remote temp directory: {stderr}"
                raise RuntimeError(msg)
            assert result.stdout
            stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
            self._remote_work_dir = stdout.strip()

        await self.run(f"mkdir -p {self._work_dir_arg}")
        await self._verify_tools()
        if self.dependencies:  # Install dependencies if specified
            await self._install_dependencies()
        # Create filesystem with proper event loop
        self._fs = await self._create_fs()

    @property
    def _work_dir_arg(self) -> str:
        """The remote working directory, quoted for a shell command line."""
        return shlex.quote(self._remote_work_dir or ".")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up remote environment and close SSH connection."""
        try:
            if self.connection and self._connection_cm:
                try:
                    # Clean up temporary working directory if we created it
                    if not self.cwd and self._remote_work_dir:
                        await self.run(f"rm -rf {self._work_dir_arg}")
                finally:
                    cm, self._connection_cm = self._connection_cm, None
                    self.connection = None
                    await cm.__aexit__(exc_type, exc_val, exc_tb)
            await self._close_fs()
        finally:
            await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _close_fs(self) -> None:
        """Close the SFTP channel pool and the filesystem's own SSH connection."""
        pool, stack = self._sftp_pool, self._sftp_stack
        self._sftp_pool = self._sftp_stack = None
        self._fs = None
        if pool is not None:
            await pool.close()
        if stack is not None:
            # A partially initialized SFTP client may fail on close; mirror sshfs.
            with contextlib.suppress(BrokenPipeError):
                await stack.aclose()

    async def _create_fs(self) -> WrapperFileSystem:
        """Create SSHFileSystem with proper event loop integration."""
        import asyncio

        from fsspec.dircache import DirCache
        from sshfs import SSHFileSystem  # type: ignore[import-untyped]
        from sshfs.pools import SFTPSoftChannelPool  # type: ignore[import-untyped]
        from upathtools.filesystems.base.wrapper import WrapperFileSystem

        # SSH filesystem doesnt work with existing connections, so we hack around it.
        loop = asyncio.get_running_loop()
        fs = SSHFileSystem.__new__(SSHFileSystem)
        fs._loop = loop
        fs._intrans = False
        fs._transaction = None
        fs.dircache = DirCache()
        fs.active_executors = 0
        stack = AsyncExitStack()
        fs._stack = stack
        fs._client, fs._pool = await fs._connect(
            self.connect_kwargs["host"],
            SFTPSoftChannelPool,
            max_sftp_channels=8,
            connect_args={k: v for k, v in self.connect_kwargs.items() if k != "host"},
            sftp_client_kwargs={},
        )
        self._sftp_pool = fs._pool
        self._sftp_stack = stack
        return WrapperFileSystem(fs)

    def get_fs(self) -> WrapperFileSystem:
        """Return a SSHFileSystem instance for the remote machine."""
        if self._fs is None:
            msg = "Filesystem not available. Use 'async with' context manager first."
            raise RuntimeError(msg)
        return self._fs

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pty: PtySize | None = None,
    ) -> ProcessHandle:
        """Start a process on the remote machine in its own SSH session channel.

        Environment variables are exported in the command line (servers usually
        restrict which variables clients may set via SSH env requests).

        Args:
            command: Program (with `args`) or shell command line (without)
            args: Program arguments; the program replaces the shell via `exec`
            cwd: Remote working directory (defaults to the environment's directory)
            env: Additional environment variables, merged over `env_vars`
            pty: Request a pseudo-terminal of this size

        Returns:
            Handle to the running process
        """
        from exxec.command_line import build_spawn_command_line
        from exxec.ssh_provider.process_handle import SshProcessHandle

        connection = self._ensure_connected()
        merged_env = {**self.env_vars, **(env or {})}
        command_line = build_spawn_command_line(
            command, args, cwd or self._remote_work_dir, merged_env or None
        )
        return await SshProcessHandle.start(connection, command_line, size=pty)

    async def _verify_tools(self) -> None:
        """Verify that required tools are available on the remote machine."""
        if self.language == "python":
            # Require uv to be available - use login shell to load profile
            uv_result = await self.run("which uv")
            if uv_result.returncode != 0:
                msg = "uv not found on remote machine. Please install uv first."
                raise RuntimeError(msg)
        elif self.language in ("javascript", "typescript"):
            node_result = await self.run("which node")
            if node_result.returncode != 0:
                msg = "Node.js not found on remote machine"
                raise RuntimeError(msg)

    async def _install_dependencies(self) -> None:
        """Install dependencies on the remote machine."""
        # For Python, dependencies are handled via uv run --with
        # For JS/TS, we still need to install them in the working directory
        if self.language in ("javascript", "typescript") and self.dependencies:
            cmd = f"npm init -y && {shlex.join(['npm', 'install', *self.dependencies])}"
            result = await self.run_in_working_dir(cmd)
            if result.returncode != 0:
                stderr = (
                    result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
                )
                msg = f"Failed to install Node.js dependencies: {stderr}"
                raise RuntimeError(msg)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code on the remote machine."""
        self._ensure_connected()
        start_time = time.time()
        try:
            if self.language == "python":
                result = await self._execute_python(code)
            elif self.language == "javascript":
                result = await self._execute_javascript(code)
            elif self.language == "typescript":
                result = await self._execute_typescript(code)
            else:
                msg = f"Unsupported language: {self.language}"
                raise ValueError(msg)  # noqa: TRY301

            success = result.returncode == 0
            # Add tool server URL to code if available
            if self.server_info and self.language == "python":
                code = self._inject_tool_server(code)

            return ExecutionResult(
                result=result.stdout if success else None,
                duration=time.time() - start_time,
                success=success,
                error=result.stderr.decode()
                if isinstance(result.stderr, bytes)
                else result.stderr
                if not success
                else None,
                error_type="RemoteExecutionError" if not success else None,
                exit_code=result.returncode,
                stdout=result.stdout.decode()
                if isinstance(result.stdout, bytes)
                else result.stdout,
                stderr=result.stderr.decode()
                if isinstance(result.stderr, bytes)
                else result.stderr,
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def run_in_working_dir(
        self,
        cmd: str,
        timeout: float | None = None,
    ) -> SSHCompletedProcess:
        """Run a command in the working directory.

        Args:
            cmd: Command to run
            timeout: Optional timeout in seconds (None means no timeout)
        """
        return await self.run(f"cd {self._work_dir_arg} && {with_timeout(cmd, timeout)}")

    async def _execute_python(self, code: str) -> SSHCompletedProcess:
        """Execute Python code using uv run --with for dependencies."""
        self._ensure_connected()
        script_path = f"{self._remote_work_dir}/script.py"
        await self.write_file(script_path, code)
        # Build uv run command with dependencies
        with_args = [arg for dep in self.dependencies for arg in ("--with", dep)]
        cmd = shlex.join(["uv", "run", *with_args, "python", script_path])
        return await self.run_in_working_dir(cmd, timeout=self.default_command_timeout)

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file on the remote server (exactly, via stdin)."""
        result = await self.run(f"cat > {shlex.quote(path)}", input=content)
        if result.returncode != 0:
            msg = f"Failed to write {path}: {result.stderr}"
            raise RuntimeError(msg)

    async def _execute_javascript(self, code: str) -> Any:
        """Execute JavaScript code using node."""
        self._ensure_connected()
        script_path = f"{self._remote_work_dir}/script.js"
        await self.write_file(script_path, code)
        return await self.run_in_working_dir(
            shlex.join(["node", script_path]), timeout=self.default_command_timeout
        )

    async def _execute_typescript(self, code: str) -> Any:
        """Execute TypeScript code using ts-node or similar."""
        self._ensure_connected()
        script_path = f"{self._remote_work_dir}/script.ts"
        await self.write_file(script_path, code)
        # Try ts-node first, fall back to tsc + node
        ts_node_result = await self.run("which ts-node")
        if ts_node_result.returncode == 0:
            cmd = shlex.join(["ts-node", script_path])
        else:
            # Compile and run
            cmd = f"{shlex.join(['npx', 'tsc', script_path])} && node script.js"

        return await self.run_in_working_dir(cmd, timeout=self.default_command_timeout)

    def _inject_tool_server(self, code: str) -> str:
        """Inject tool server URL into Python code if available."""
        if not self.server_info:
            return code

        injection = f"""
# Tool server configuration injected by anyenv
import os
os.environ['TOOL_SERVER_URL'] = '{self.server_info.url}'
os.environ['TOOL_SERVER_PORT'] = '{self.server_info.port}'

"""
        return injection + code

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a shell command on the remote machine."""
        self._ensure_connected()
        start_time = time.time()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout

        try:
            result = await self.run_in_working_dir(command, timeout=effective_timeout)
            success = result.returncode == 0
            stderr = err.decode() if isinstance(err := result.stderr, bytes) else err
            stdout = out.decode() if isinstance(out := result.stdout, bytes) else out
            timed_out = effective_timeout is not None and result.returncode == 124  # noqa: PLR2004
            error: str | None = None
            if timed_out:
                error = f"Command timed out after {effective_timeout} seconds"
            elif not success:
                error = stderr
            error_type = "TimeoutError" if timed_out else "RemoteCommandError"
            return ExecutionResult(
                result=result.stdout if success else None,
                duration=time.time() - start_time,
                success=success,
                error=error,
                error_type=error_type if not success else None,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events over SSH."""
        connection = self._ensure_connected()
        process_id = f"ssh_{id(connection)}"
        yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")

        try:
            if self.language == "python":
                result = await self._execute_python(code)
            elif self.language == "javascript":
                result = await self._execute_javascript(code)
            elif self.language == "typescript":
                result = await self._execute_typescript(code)
            else:
                msg = f"Unsupported language: {self.language}"
                yield ProcessErrorEvent(process_id=process_id, error=msg, error_type="ValueError")
                return

            stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
            stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr

            if stdout:
                yield OutputEvent(process_id=process_id, data=stdout, stream="stdout")
            if stderr:
                yield OutputEvent(process_id=process_id, data=stderr, stream="stderr")

            exit_code = result.returncode or 0
            if exit_code == 0:
                yield ProcessCompletedEvent(process_id=process_id, exit_code=exit_code)
            else:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=f"Process exited with code {exit_code}",
                    error_type="ProcessError",
                    exit_code=exit_code,
                )

        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute a command (login shell, working dir) and stream its output line by line.

        Args:
            command: Shell command line
            timeout: Total time limit in seconds (defaults to `default_command_timeout`);
                on expiry the process is killed.

        Yields:
            Started, output (one per line, stdout and stderr in order) and
            completed/error events
        """
        from exxec.spawn_stream import stream_command_via_spawn

        connection = self._ensure_connected()
        async for event in stream_command_via_spawn(
            self,
            wrap_command(command),
            timeout=timeout,
            process_id=f"ssh_cmd_{id(connection)}",
            display_command=command,
        ):
            yield event


if __name__ == "__main__":

    async def _main() -> None:
        async with SshExecutionEnvironment("91.99.102.138", "root") as sandbox:
            fs = sandbox.get_fs()
            print(await fs._ls("/"))

    import asyncio

    asyncio.run(_main())
