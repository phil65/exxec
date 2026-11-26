"""Microsandbox execution environment that runs code in lightweight sandboxes."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Self
import uuid

from exxec.base import ExecutionEnvironment
from exxec.command_line import build_remote_command_line, with_timeout
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.exceptions import NotInitializedError
from exxec.models import ExecutionResult


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from microsandbox import Sandbox
    from upathtools.filesystems import MicrosandboxFS

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo
    from exxec.process_handle import ProcessHandle
    from exxec.pty_manager import PtySize


def _inject_env_vars_to_code(env_vars: dict[str, str], code: str) -> str:
    """Inject environment variables into Python code."""
    if not env_vars:
        return code
    # Prepend os.environ updates
    env_setup = "import os\n"
    for key, value in env_vars.items():
        env_setup += f"os.environ[{key!r}] = {value!r}\n"
    return env_setup + code


DEFAULT_IMAGES: dict[str, str] = {
    "python": "python:3.13-slim",
    "javascript": "node:22-slim",
    "typescript": "node:22-slim",
}
"""OCI images used when no image is configured, by language."""


class MicrosandboxExecutionEnvironment(ExecutionEnvironment):
    """Executes code in a Microsandbox microVM."""

    supports_spawn = True
    supports_pty = True

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        namespace: str = "exxec",
        api_key: str | None = None,
        memory: int = 512,
        cpus: int = 1,
        timeout: float = 180.0,
        language: Language = "python",
        image: str | None = None,
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
        default_command_timeout: float | None = None,
    ) -> None:
        """Initialize Microsandbox environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of packages to install via pip / npm
            namespace: Prefix for the sandbox name (a unique suffix is appended)
            api_key: API key for authentication (uses MSB_API_KEY env var if None)
            memory: Memory limit in MiB
            cpus: Number of virtual CPUs
            timeout: Sandbox start timeout in seconds
            language: Programming language to use
            image: OCI image (defaults to a slim image for the language)
            cwd: Working directory in the sandbox
            env_vars: Environment variables for all processes in the sandbox
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
        self.namespace = namespace
        self.api_key = api_key
        self.memory = memory
        self.cpus = cpus
        self.timeout = timeout
        self.language = language
        self.image = image
        self.sandbox: Sandbox | None = None
        # Microsandbox runs Linux containers
        self._os_type = "Linux"

    def _ensure_initialized(self) -> Sandbox:
        """Validate that the environment is properly initialized.

        Returns:
            The sandbox instance.

        Raises:
            NotInitializedError: If environment not entered via async context manager.
        """
        if self.sandbox is None:
            raise NotInitializedError("Microsandbox")
        return self.sandbox

    async def __aenter__(self) -> Self:
        """Setup Microsandbox environment."""
        # Start tool server via base class
        from microsandbox import Sandbox

        await super().__aenter__()
        self.sandbox = await asyncio.wait_for(
            Sandbox.create(
                f"{self.namespace}-{uuid.uuid4().hex[:12]}",
                image=self.image or DEFAULT_IMAGES.get(self.language, DEFAULT_IMAGES["python"]),
                memory=self.memory,
                cpus=self.cpus,
                workdir=self.cwd,
                # Only the configured variables: the host environment is meaningless in the VM.
                env=self.env_vars or None,
                ephemeral=True,
            ),
            timeout=self.timeout,
        )
        if self.dependencies and self.language == "python":
            deps_str = " ".join(self.dependencies)
            install_result = await self.sandbox.shell(f"pip install {deps_str}")
            if install_result.exit_code != 0:
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
        if self.sandbox:
            with contextlib.suppress(Exception):
                await self.sandbox.stop()

        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pty: PtySize | None = None,
    ) -> ProcessHandle:
        """Start a process in the microVM via `exec_stream()`.

        Args:
            command: Program (with `args`) or shell command line (without)
            args: Program arguments, passed without shell interpretation
            cwd: Working directory (defaults to the sandbox workdir)
            env: Additional environment variables (sandbox-wide `env_vars` apply too)
            pty: Allocate a TTY of this size

        Returns:
            Handle to the running process
        """
        from exxec.microsandbox_provider.process_handle import MicrosandboxProcessHandle

        sandbox = self._ensure_initialized()
        argv = [command, *args] if args else ["sh", "-c", command]
        return await MicrosandboxProcessHandle.start(
            sandbox, argv, cwd=cwd or self.cwd, env=env or None, size=pty
        )

    def get_fs(self) -> MicrosandboxFS:
        """Return a MicrosandboxFS instance for the sandbox."""
        from upathtools.filesystems import MicrosandboxFS

        sandbox = self._ensure_initialized()
        return MicrosandboxFS(sandbox=sandbox)

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in the Microsandbox environment."""
        sandbox = self._ensure_initialized()
        start_time = time.time()
        try:
            # Inject environment variables into code for Python
            if self.language == "python":
                code = _inject_env_vars_to_code(self.get_env() or {}, code)
            execution = await sandbox.exec("python", ["-c", code])
            stdout = execution.stdout_text
            stderr = execution.stderr_text
            if execution.success:
                return ExecutionResult(
                    result=stdout if stdout else None,
                    duration=time.time() - start_time,
                    success=True,
                    stdout=stdout,
                    stderr=stderr,
                )

            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                error=stderr or "Code execution failed",
                error_type="ExecutionError",
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
        """Execute a terminal command in the Microsandbox environment."""
        sandbox = self._ensure_initialized()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        # Export environment variables, limit the whole line if a timeout is set
        full_command = build_remote_command_line(
            with_timeout(command, effective_timeout), None, self.get_env() or None
        )
        start_time = time.time()
        try:
            execution = await sandbox.shell(full_command)
            stdout = execution.stdout_text
            stderr = execution.stderr_text
            # Exit code 124 indicates timeout
            if execution.exit_code == 124:  # noqa: PLR2004
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
            success = execution.success
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
        """Execute code and emit combined events (no real-time streaming)."""
        process_id = f"microsandbox_{id(self.sandbox)}"
        yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")

        try:
            result = await self.execute(code)  # Emit output as single combined event
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
        """Execute a shell command and stream its output line by line, in real time."""
        from exxec.spawn_stream import stream_command_via_spawn

        async for event in stream_command_via_spawn(
            self,
            command,
            timeout=timeout,
            process_id=f"microsandbox_cmd_{id(self.sandbox)}",
        ):
            yield event
