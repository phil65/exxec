"""Daytona execution environment that runs code in remote sandboxes."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import TYPE_CHECKING, Any, Self

from exxec.base import ExecutionEnvironment
from exxec.daytona_provider.helpers import convert_language
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.models import ExecutionResult
from exxec.parse_output import parse_output, wrap_python_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from asyncssh import SSHClientConnection
    from daytona._async.sandbox import AsyncSandbox
    from upathtools.filesystems import DaytonaFS

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo
    from exxec.process_handle import ProcessHandle
    from exxec.pty_manager import PtySize


DEFAULT_DAYTONA_API_URL = "https://app.daytona.io/api"
DAYTONA_SSH_HOST = "ssh.app.daytona.io"
"""Daytona's SSH gateway (token as username)."""


class DaytonaExecutionEnvironment(ExecutionEnvironment):
    """Executes code in a Daytona sandbox with isolated environment."""

    supports_spawn = True
    supports_pty = True

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        target: str | None = None,
        image: str = "python:3.13-slim",
        timeout: float = 300.0,
        keep_alive: bool = False,
        language: Language = "python",
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
        default_command_timeout: float | None = None,
        ssh_known_hosts: str | None = None,
    ) -> None:
        """Initialize Daytona environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of packages to install via pip / npm
            api_url: Daytona API server URL (uses DAYTONA_API_URL env var if None)
            api_key: API key for authentication (uses DAYTONA_API_KEY env var if None)
            target: Target location (uses DAYTONA_TARGET env var if None)
            image: Docker image to use for the sandbox
            timeout: Sandbox lifetime timeout in seconds
            keep_alive: Keep sandbox running after execution
            language: Programming language to use for execution
            cwd: Working directory for the sandbox
            env_vars: Environment variables to set for all executions
            inherit_env: If True, inherit environment variables from os.environ
            default_command_timeout: Default timeout for command execution in seconds
            ssh_known_hosts: known_hosts file used to verify Daytona's SSH gateway for
                `spawn()`. None disables host key verification: the gateway
                authenticates with a short-lived, sandbox-scoped token that is revoked
                when the environment exits.
        """
        from daytona import AsyncDaytona, DaytonaConfig

        super().__init__(
            lifespan_handler=lifespan_handler,
            dependencies=dependencies,
            cwd=cwd,
            env_vars=env_vars,
            inherit_env=inherit_env,
            default_command_timeout=default_command_timeout,
        )
        self.image = image
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.language: Language = language
        # Without explicit settings Daytona reads DAYTONA_* env vars itself; passing a
        # config without api_url would make it fall back to the deprecated server_url.
        if api_url or api_key or target:
            config = DaytonaConfig(
                api_url=api_url or os.environ.get("DAYTONA_API_URL") or DEFAULT_DAYTONA_API_URL,
                api_key=api_key,
                target=target,
            )
            self.daytona = AsyncDaytona(config)
        else:
            self.daytona = AsyncDaytona()
        self._sandbox: AsyncSandbox | None = None
        # Daytona sandboxes run Linux containers
        self._os_type = "Linux"
        self.ssh_known_hosts = ssh_known_hosts
        self._ssh: SSHClientConnection | None = None
        self._ssh_token: str | None = None
        self._ssh_lock = asyncio.Lock()

    @property
    def sandbox(self) -> AsyncSandbox:
        """Return the initialized sandbox."""
        assert self._sandbox is not None, "Sandbox not initialized"
        return self._sandbox

    async def __aenter__(self) -> Self:
        """Setup Daytona client and create sandbox."""
        await super().__aenter__()
        from daytona.common.daytona import (
            CreateSandboxFromImageParams,
        )

        language = convert_language(self.language)
        params = CreateSandboxFromImageParams(image=self.image, language=language)
        self._sandbox = await self.daytona.create(params)
        await self.sandbox.start(timeout=self.timeout)
        if self.dependencies and self.language == "python":
            deps_str = " ".join(self.dependencies)
            result = await self.sandbox.process.exec(f"pip install {deps_str}")
            if result.exit_code != 0:
                msg = f"Failed to install dependencies: {deps_str}"
                raise RuntimeError(msg)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup sandbox, SSH access and the API client."""
        # Base cleanup first: it terminates spawned processes, which need the connection.
        await super().__aexit__(exc_type, exc_val, exc_tb)
        await self._close_ssh()
        if not self.keep_alive and self._sandbox is not None:
            with contextlib.suppress(Exception):
                await self.sandbox.stop()
                await self.sandbox.delete()
        # Closes the HTTP sessions (unannotated upstream).
        await self.daytona.close()  # type: ignore[no-untyped-call]

    async def _ssh_connection(self) -> SSHClientConnection:
        """Return the SSH connection to the sandbox, creating access on first use."""
        import asyncssh

        async with self._ssh_lock:
            if self._ssh is not None and not self._ssh.is_closed():
                return self._ssh
            if self._ssh_token is None:
                access = await self.sandbox.create_ssh_access()
                self._ssh_token = access.token
            self._ssh = await asyncssh.connect(
                DAYTONA_SSH_HOST, username=self._ssh_token, known_hosts=self.ssh_known_hosts
            )
            return self._ssh

    async def _close_ssh(self) -> None:
        if self._ssh is not None:
            self._ssh.close()
            with contextlib.suppress(Exception):
                await self._ssh.wait_closed()
            self._ssh = None
        if self._ssh_token is not None and self._sandbox is not None:
            with contextlib.suppress(Exception):
                await self.sandbox.revoke_ssh_access(self._ssh_token)
            self._ssh_token = None

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pty: PtySize | None = None,
    ) -> ProcessHandle:
        """Start a process in the sandbox over Daytona's SSH gateway.

        The SSH connection (and its access token) is created on first use and shared
        by all spawned processes; each process gets its own session channel.

        Args:
            command: Program (with `args`) or shell command line (without)
            args: Program arguments; the program replaces the shell via `exec`
            cwd: Working directory (defaults to the environment's, else the user's home)
            env: Additional environment variables, merged over `env_vars`
            pty: Request a pseudo-terminal of this size

        Returns:
            Handle to the running process
        """
        from exxec.command_line import build_spawn_command_line
        from exxec.ssh_provider.process_handle import SshProcessHandle

        connection = await self._ssh_connection()
        merged_env = {**self.env_vars, **(env or {})} or None
        command_line = build_spawn_command_line(command, args, cwd or self.cwd, merged_env)
        return await SshProcessHandle.start(connection, command_line, size=pty)

    async def get_domain(self, port: int) -> str:
        """Return the domain name for the sandbox."""
        info = await self.sandbox.get_preview_link(port)
        assert isinstance(info.url, str)
        return info.url

    def get_fs(self) -> DaytonaFS:
        """Return a DaytonaFS instance for the sandbox."""
        from upathtools.filesystems import DaytonaFS

        return DaytonaFS(sandbox_id=self.sandbox.id)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in the Daytona sandbox."""
        start_time = time.time()
        wrapped_code = wrap_python_code(code)
        try:
            response = await self.sandbox.process.exec(
                f"python -c '{wrapped_code}'",
                timeout=int(self.timeout),
                env=self.get_env(),
            )
            # Parse execution results
            if response.exit_code == 0:
                result, error_info = parse_output(response.result)
                if error_info is None:
                    return ExecutionResult(
                        result=result,
                        duration=time.time() - start_time,
                        success=True,
                        stdout=response.result,
                        stderr="",
                        exit_code=int(response.exit_code),
                    )
                return ExecutionResult(
                    result=None,
                    duration=time.time() - start_time,
                    success=False,
                    error=error_info.get("error", "Unknown error"),
                    error_type=error_info.get("type", "ExecutionError"),
                    stdout=response.result,
                    stderr="",
                    exit_code=int(response.exit_code),
                )

            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                error=response.result if response.result else "Command execution failed",
                exit_code=int(response.exit_code),
                error_type="CommandError",
                stdout=response.result,
                stderr="",
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a terminal command in the Daytona sandbox."""
        start_time = time.time()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        try:
            exec_kwargs: dict[str, Any] = {"env": self.get_env()}
            if effective_timeout is not None:
                exec_kwargs["timeout"] = int(effective_timeout)
            response = await self.sandbox.process.exec(command, **exec_kwargs)
            success = response.exit_code == 0
            return ExecutionResult(
                result=response.result if success else None,
                duration=time.time() - start_time,
                success=success,
                error=response.result if not success else None,
                error_type="CommandError" if not success else None,
                exit_code=int(response.exit_code),
                stdout=response.result,
                stderr="",  # Daytona combines stdout/stderr in result
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events in the Daytona sandbox."""
        process_id = f"daytona_{id(self.sandbox)}"
        yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")

        try:
            result = await self.execute(code)

            if result.stdout:
                yield OutputEvent(process_id=process_id, data=result.stdout, stream="combined")

            if result.success:
                yield ProcessCompletedEvent(process_id=process_id, exit_code=result.exit_code or 0)
            else:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=result.error or "Unknown error",
                    error_type=result.error_type or "ExecutionError",
                    exit_code=result.exit_code,
                )

        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute terminal command and stream events in the Daytona sandbox."""
        process_id = f"daytona_cmd_{id(self.sandbox)}"
        yield ProcessStartedEvent(process_id=process_id, command=command)
        try:
            result = await self.execute_command(command, timeout=timeout)
            if result.stdout:
                yield OutputEvent(process_id=process_id, data=result.stdout, stream="combined")
            if result.success:
                yield ProcessCompletedEvent(process_id=process_id, exit_code=result.exit_code or 0)
            else:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=result.error or "Unknown error",
                    error_type=result.error_type or "CommandError",
                    exit_code=result.exit_code,
                )
        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)


if __name__ == "__main__":

    async def _main() -> None:
        async with DaytonaExecutionEnvironment() as sandbox:
            async for p in sandbox.execute_command_stream("echo 'Hello, Daytona!'"):
                print(p)

    import asyncio

    asyncio.run(_main())
