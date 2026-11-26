"""Vercel sandbox execution environment that runs code in cloud sandboxes."""

import contextlib
import subprocess
import time
from typing import TYPE_CHECKING, Self
import uuid

from exxec.base import ExecutionEnvironment
from exxec.command_line import with_timeout
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.exceptions import NotInitializedError
from exxec.models import ExecutionResult
from exxec.parse_output import get_script_path, parse_output, wrap_code


# Vercel API minimum timeout requirement
MIN_TIMEOUT_SECONDS = 1
DEFAULT_TIMEOUT_SECONDS = 60


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from upathtools.filesystems import VercelFS
    from vercel.sandbox import (
        ByteReader,
        CreateSandboxOperation,
        Sandbox,
        SandboxClient,
        SandboxCredentials,
        SandboxResources,
    )

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo
    from exxec.process_handle import ProcessHandle
    from exxec.pty_manager import PtySize


class VercelExecutionEnvironment(ExecutionEnvironment):
    """Executes code in a Vercel cloud sandbox."""

    supports_spawn = True
    supports_pty = True

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        image: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        resources: SandboxResources | None = None,
        ports: list[int] | None = None,
        language: Language = "python",
        token: str | None = None,
        project_id: str | None = None,
        team_id: str | None = None,
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
        default_command_timeout: float | None = None,
    ) -> None:
        """Initialize Vercel sandbox environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of packages to install via pip / npm
            image: Vercel Container Registry image reference (uses Vercel default if None)
            timeout: Sandbox execution time limit in seconds (minimum 1)
            resources: Resource configuration for the sandbox
            ports: List of ports to expose
            language: Programming language to use
            token: Vercel API token (uses environment if None)
            project_id: Vercel project ID (uses environment if None)
            team_id: Vercel team ID (uses environment if None)
            cwd: Working directory for the sandbox
            env_vars: Environment variables to set for all executions
            inherit_env: If True, inherit environment variables from os.environ
            default_command_timeout: Default timeout for command execution in seconds
        """
        super().__init__(
            lifespan_handler=lifespan_handler,
            dependencies=dependencies,
            cwd=cwd,
            env_vars=env_vars,
            inherit_env=inherit_env,
            default_command_timeout=default_command_timeout,
        )
        self.image = image
        if timeout < MIN_TIMEOUT_SECONDS:
            error_msg = f"Vercel requires timeout >= 1 second, got {timeout} seconds"
            raise ValueError(error_msg)
        self.timeout = timeout

        self.resources = resources
        self.ports = ports or [3000]
        self.language: Language = language
        # Vercel sandboxes run Linux
        self._os_type = "Linux"
        self.token = token
        self.project_id = project_id
        self.team_id = team_id
        self.sandbox: Sandbox | None = None
        self._client: SandboxClient | None = None
        self._create_operation: CreateSandboxOperation | None = None

    def _ensure_initialized(self) -> Sandbox:
        """Validate that the environment is properly initialized.

        Returns:
            The sandbox instance.

        Raises:
            NotInitializedError: If environment not entered via async context manager.
        """
        if self.sandbox is None:
            raise NotInitializedError("Vercel")
        return self.sandbox

    async def __aenter__(self) -> Self:
        """Setup Vercel sandbox."""
        from vercel.sandbox import SandboxClient, SandboxServiceOptions

        # Start tool server via base class
        await super().__aenter__()
        if self.token and self.project_id and self.team_id:
            options = SandboxServiceOptions(credentials_factory=self._get_credentials)
            self._client = SandboxClient.create(options=options)
        else:
            self._client = SandboxClient.create()
        self._create_operation = self._client.create_sandbox(
            image=self.image,
            ports=self.ports,
            execution_time_limit=self.timeout,
            resources=self.resources,
        )
        self.sandbox = await self._create_operation.__aenter__()

        # Install Python dependencies if specified
        if self.dependencies and self.language == "python":
            try:
                args = ["install", *self.dependencies]
                install_result = await self.sandbox.run_command("pip", args)
                if install_result.returncode != 0:
                    # Log warning but don't fail - code might still work
                    pass
            except Exception:  # noqa: BLE001
                # Log warning but don't fail - code might still work
                pass

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup sandbox."""
        if self._create_operation:
            with contextlib.suppress(Exception):
                await self._create_operation.__aexit__(exc_type, exc_val, exc_tb)
            self._create_operation = None
            self.sandbox = None
        if self._client:
            await self._client.aclose()
            self._client = None

        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _get_credentials(self) -> SandboxCredentials:
        from vercel.sandbox import SandboxCredentials

        assert self.token
        assert self.project_id
        assert self.team_id
        return SandboxCredentials(
            token=self.token, project_id=self.project_id, team_id=self.team_id
        )

    async def get_domain(self, port: int) -> str:
        """Get domain for the Vercel sandbox."""
        sandbox = self._ensure_initialized()
        for route in sandbox.routes:
            if route.port == port:
                return route.url
        msg = f"No route exposed for port {port}"
        raise ValueError(msg)

    def get_fs(self) -> VercelFS:
        """Return a VercelFS instance for the sandbox."""
        from upathtools.filesystems import VercelFS

        sandbox = self._ensure_initialized()
        return VercelFS(
            sandbox_name=sandbox.name,
            token=self.token,
            project_id=self.project_id,
            team_id=self.team_id,
        )

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in the Vercel sandbox."""
        sandbox = self._ensure_initialized()
        start_time = time.time()
        try:
            script_path, wrapped_code = self._prepare_code_execution(code)
            await sandbox.fs.write_text(script_path, wrapped_code)
            cmd, args = self._get_execution_command(script_path)
            result = await sandbox.run_command(cmd, args, env=self.get_env(), capture_output=True)
            stdout = (result.stdout or b"").decode()
            stderr = (result.stderr or b"").decode()
            execution_result, error_info = parse_output(stdout)
            if result.returncode == 0 and error_info is None:
                return ExecutionResult(
                    result=execution_result,
                    duration=time.time() - start_time,
                    success=True,
                    stdout=stdout,
                    exit_code=result.returncode,
                    stderr=stderr,
                )

            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                error=(error_info or {}).get("error", "Command execution failed"),
                exit_code=result.returncode,
                error_type=(error_info or {}).get("type", "ExecutionError"),
                stdout=stdout,
                stderr=stderr,
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a terminal command in the Vercel sandbox."""
        sandbox = self._ensure_initialized()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        # A shell command line, time-limited as a whole if a timeout is set
        cmd, args = "sh", ["-c", with_timeout(command, effective_timeout)]
        start_time = time.time()
        try:
            result = await sandbox.run_command(
                cmd, args or None, env=self.get_env(), capture_output=True
            )
            stdout = (result.stdout or b"").decode()
            stderr = (result.stderr or b"").decode()
            # Exit code 124 is timeout's exit code when command times out
            if result.returncode == 124:  # noqa: PLR2004
                return ExecutionResult(
                    result=None,
                    duration=time.time() - start_time,
                    success=False,
                    error=f"Command timed out after {effective_timeout} seconds",
                    error_type="TimeoutError",
                    exit_code=124,
                    stdout=stdout,
                    stderr=stderr,
                )
            success = result.returncode == 0
            return ExecutionResult(
                result=stdout if success else None,
                duration=time.time() - start_time,
                success=success,
                error=stderr if not success else None,
                error_type="CommandError" if not success else None,
                stdout=stdout,
                stderr=stderr,
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events in the Vercel sandbox."""
        sandbox = self._ensure_initialized()
        process_id: str | None = None
        try:
            script_path, wrapped_code = self._prepare_code_execution(code)
            await sandbox.fs.write_text(script_path, wrapped_code)
            cmd, args = self._get_execution_command(script_path)
            command = await sandbox.create_command(
                cmd, args, env=self.get_env(), stderr=subprocess.STDOUT
            )
            process_id = command.id
            yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")
            if command.stdout:
                async for line in _iter_lines(command.stdout):
                    yield OutputEvent(process_id=process_id, data=line, stream="combined")

            exit_code = await command.wait()
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
            error_id = process_id or str(uuid.uuid4())[:8]
            yield ProcessErrorEvent.failed(e, process_id=error_id)

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pty: PtySize | None = None,
    ) -> ProcessHandle:
        """Start a process in the Vercel sandbox.

        Pipe processes use `create_command()`, PTY processes `open_interactive()`.

        Args:
            command: Program (with `args`) or shell command line (without)
            args: Program arguments, passed without shell interpretation
            cwd: Working directory (defaults to the environment's)
            env: Additional environment variables, merged over `env_vars`
            pty: Attach to a pseudo-terminal of this size

        Returns:
            Handle to the running process
        """
        from exxec.vercel_provider.process_handle import (
            VercelInteractiveHandle,
            VercelProcessHandle,
        )

        sandbox = self._ensure_initialized()
        argv = [command, *args] if args else ["sh", "-c", command]
        merged_env = {**self.env_vars, **(env or {})} or None
        work_dir = cwd or self.cwd
        if pty is not None:
            return await VercelInteractiveHandle.start(
                sandbox, argv, cwd=work_dir, env=merged_env, size=pty
            )
        return await VercelProcessHandle.start(sandbox, argv, cwd=work_dir, env=merged_env)

    async def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute a terminal command and stream events in the Vercel sandbox."""
        sandbox = self._ensure_initialized()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        # A shell command line, time-limited as a whole if a timeout is set
        cmd, args = "sh", ["-c", with_timeout(command, effective_timeout)]
        process_id = f"vercel_cmd_{id(sandbox)}"
        yield ProcessStartedEvent(process_id=process_id, command=command)
        try:
            async_cmd = await sandbox.create_command(
                cmd, args or None, env=self.get_env(), stderr=subprocess.STDOUT
            )
            if async_cmd.stdout:
                async for line in _iter_lines(async_cmd.stdout):
                    yield OutputEvent(process_id=process_id, data=line, stream="combined")

            exit_code = await async_cmd.wait()
            # Exit code 124 is timeout's exit code when command times out
            if exit_code == 124:  # noqa: PLR2004
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=f"Command timed out after {effective_timeout} seconds",
                    error_type="TimeoutError",
                    exit_code=exit_code,
                )
            elif exit_code == 0:
                yield ProcessCompletedEvent(process_id=process_id, exit_code=exit_code)
            else:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=f"Command exited with code {exit_code}",
                    error_type="CommandError",
                    exit_code=exit_code,
                )

        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    def _prepare_code_execution(self, code: str) -> tuple[str, str]:
        """Prepare code for execution, returning script path and wrapped code."""
        script_path = get_script_path(self.language)
        wrapped_code = wrap_code(code, self.language)
        return script_path, wrapped_code

    def _get_execution_command(self, script_path: str) -> tuple[str, list[str]]:
        """Get execution command based on language.

        Returns:
            Tuple of (cmd, args) where cmd is the executable and args is the arg list.
        """
        match self.language:
            case "javascript":
                return ("node", [script_path])
            case "typescript":
                return ("npx", ["ts-node", script_path])
            case _:
                return ("python3", [script_path])


async def _iter_lines(reader: ByteReader) -> AsyncIterator[str]:
    """Yield non-empty decoded lines from a Vercel command output reader."""
    while line := await reader.readline():
        if (text := line.decode().rstrip("\r\n")).strip():
            yield text


if __name__ == "__main__":

    async def _main() -> None:
        async with VercelExecutionEnvironment() as sandbox:
            await sandbox.execute_command("mkdir test")
            result = await sandbox.execute_command("ls")
            print(result)

    import asyncio

    asyncio.run(_main())
