"""Hopx execution environment that runs code in cloud VM sandboxes."""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any, Self

from exxec.base import ExecutionEnvironment
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.exceptions import NotInitializedError
from exxec.models import ExecutionResult
from exxec.parse_output import get_script_path, parse_output, wrap_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from hopx_ai import AsyncSandbox  # type: ignore[import-untyped]
    from upathtools.filesystems import HopXFS

    from exxec.events import ExecutionEvent
    from exxec.hopx_provider.pty_manager import HopxPtyManager
    from exxec.models import Language, ServerInfo


def _get_execution_command(language: Language, script_path: str) -> str:
    """Get execution command based on language."""
    match language:
        case "python":
            return f"python {script_path}"
        case "javascript":
            return f"node {script_path}"
        case "typescript":
            return f"npx ts-node {script_path}"
        case _:
            return f"python {script_path}"


class HopxExecutionEnvironment(ExecutionEnvironment):
    """Executes code in a Hopx cloud VM sandbox."""

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        template: str | None = None,
        template_id: str | None = None,
        timeout: float = 300.0,
        default_command_timeout: float | None = None,
        keep_alive: bool = False,
        language: Language = "python",
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
        api_key: str | None = None,
        base_url: str = "https://api.hopx.dev",
        region: str | None = None,
        internet_access: bool | None = None,
    ) -> None:
        """Initialize Hopx environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of packages to install via pip / npm
            template: Hopx template name (e.g., "code-interpreter", "base")
            template_id: Hopx template ID (alternative to template name)
            timeout: Sandbox lifetime in seconds (auto-kill timeout)
            default_command_timeout: Default timeout for command execution in seconds.
                If None, commands run without timeout unless explicitly specified.
            keep_alive: Keep sandbox running after execution
            language: Programming language to use
            cwd: Working directory for the sandbox
            env_vars: Environment variables to set for all executions
            inherit_env: If True, inherit environment variables from os.environ
            api_key: Hopx API key (or use HOPX_API_KEY env var)
            base_url: Hopx API base URL
            region: Preferred region for sandbox creation
            internet_access: Enable internet access in the sandbox
        """
        super().__init__(
            lifespan_handler=lifespan_handler,
            dependencies=dependencies,
            cwd=cwd,
            env_vars=env_vars,
            inherit_env=inherit_env,
            default_command_timeout=default_command_timeout,
        )
        self.template = template
        self.template_id = template_id
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.language: Language = language
        self.api_key = api_key
        self.base_url = base_url
        self.region = region
        self.internet_access = internet_access
        self.sandbox: AsyncSandbox | None = None
        # Hopx sandboxes run Linux
        self._os_type = "Linux"
        self._pty_manager: HopxPtyManager | None = None

    def _ensure_initialized(self) -> AsyncSandbox:
        """Ensure the sandbox has been created."""
        if self.sandbox is None:
            raise NotInitializedError("Hopx")
        return self.sandbox

    async def __aenter__(self) -> Self:
        """Setup Hopx sandbox."""
        from hopx_ai import AsyncSandbox

        await super().__aenter__()

        create_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        if self.template:
            create_kwargs["template"] = self.template
        if self.template_id:
            create_kwargs["template_id"] = self.template_id
        if self.region:
            create_kwargs["region"] = self.region
        if self.internet_access is not None:
            create_kwargs["internet_access"] = self.internet_access
        if self.timeout:
            create_kwargs["timeout_seconds"] = int(self.timeout)
        if self.env_vars:
            create_kwargs["env_vars"] = self.env_vars

        self.sandbox = await AsyncSandbox.create(**create_kwargs)

        if self.dependencies:
            deps_str = " ".join(self.dependencies)
            match self.language:
                case "python":
                    await self.sandbox.commands.run(f"pip install {deps_str}")
                case "javascript" | "typescript":
                    await self.sandbox.commands.run(f"npm install {deps_str}")

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup sandbox."""
        if self.sandbox and not self.keep_alive:
            with contextlib.suppress(Exception):
                await self.sandbox.kill()
        await super().__aexit__(exc_type, exc_val, exc_tb)

    def get_fs(self) -> HopXFS:
        """Return a HopXFS instance for the sandbox."""
        from upathtools.filesystems import HopXFS

        sandbox = self._ensure_initialized()
        return HopXFS(sandbox_id=sandbox.sandbox_id, api_key=self.api_key)

    def get_pty_manager(self) -> HopxPtyManager:
        """Return a HopxPtyManager for interactive terminal sessions."""
        if self._pty_manager is None:
            from exxec.hopx_provider.pty_manager import HopxPtyManager

            sandbox = self._ensure_initialized()
            self._pty_manager = HopxPtyManager(sandbox)
        return self._pty_manager

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in the Hopx sandbox."""
        sandbox = self._ensure_initialized()
        start_time = time.time()
        try:
            wrapped_code = wrap_code(code, language=self.language)
            script_path = get_script_path(self.language)
            await sandbox.files.write(script_path, wrapped_code)
            command = _get_execution_command(self.language, script_path)
            result = await sandbox.commands.run(
                command,
                env=self.get_env(),
                working_dir=self.cwd or "/workspace",
            )
            execution_result, error_info = parse_output(result.stdout)
            if result.exit_code == 0 and error_info is None:
                return ExecutionResult(
                    result=execution_result,
                    duration=time.time() - start_time,
                    success=True,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                exit_code=result.exit_code,
                error=(error_info or {}).get("error", "Command execution failed"),
                error_type=(error_info or {}).get("type", "ExecutionError"),
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a terminal command in the Hopx sandbox."""
        sandbox = self._ensure_initialized()
        start_time = time.time()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        try:
            run_kwargs: dict[str, Any] = {
                "working_dir": self.cwd or "/workspace",
            }
            if self.get_env():
                run_kwargs["env"] = self.get_env()
            if effective_timeout is not None:
                run_kwargs["timeout_seconds"] = int(effective_timeout)
            result = await sandbox.commands.run(command, **run_kwargs)
            success = result.exit_code == 0
            return ExecutionResult(
                result=result.stdout if success else None,
                duration=time.time() - start_time,
                success=success,
                error=result.stderr if not success else None,
                error_type="CommandError" if not success else None,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events in the Hopx sandbox."""
        sandbox = self._ensure_initialized()
        process_id = f"hopx_{id(sandbox)}"
        wrapped_code = wrap_code(code, language=self.language)
        script_path = get_script_path(self.language)
        await sandbox.files.write(script_path, wrapped_code)
        command = _get_execution_command(self.language, script_path)
        yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")
        try:
            result = await sandbox.commands.run(
                command,
                env=self.get_env(),
                working_dir=self.cwd or "/workspace",
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stdout")
            if result.stderr:
                for line in result.stderr.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stderr")

            if result.exit_code != 0:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=result.stderr or f"Command exited with code {result.exit_code}",
                    error_type="ExecutionError",
                    exit_code=result.exit_code,
                )
            else:
                yield ProcessCompletedEvent(process_id=process_id, exit_code=0)

        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute a terminal command and stream events in the Hopx sandbox."""
        sandbox = self._ensure_initialized()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        process_id = f"hopx_cmd_{id(sandbox)}"
        yield ProcessStartedEvent(process_id=process_id, command=command)
        try:
            run_kwargs: dict[str, Any] = {
                "working_dir": self.cwd or "/workspace",
            }
            if self.get_env():
                run_kwargs["env"] = self.get_env()
            if effective_timeout is not None:
                run_kwargs["timeout_seconds"] = int(effective_timeout)
            result = await sandbox.commands.run(command, **run_kwargs)

            if result.stdout:
                for line in result.stdout.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stdout")
            if result.stderr:
                for line in result.stderr.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stderr")

            if result.exit_code == 0:
                yield ProcessCompletedEvent(process_id=process_id, exit_code=result.exit_code)
            else:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=f"Command exited with code {result.exit_code}",
                    error_type="CommandError",
                    exit_code=result.exit_code,
                )

        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def set_file_content(self, path: str, content: str | bytes) -> None:
        """Set file content in the Hopx sandbox filesystem."""
        sandbox = self._ensure_initialized()
        if isinstance(content, bytes):
            await sandbox.files.write_bytes(path, content)
        else:
            await sandbox.files.write(path, content)

    async def get_file_content(self, path: str) -> bytes:
        """Get file content from the Hopx sandbox filesystem."""
        sandbox = self._ensure_initialized()
        return await sandbox.files.read_bytes(path)  # type: ignore[no-any-return]


if __name__ == "__main__":

    async def _main() -> None:
        async with HopxExecutionEnvironment(template="code-interpreter") as env:
            result = await env.execute_command("echo hello")
            print(result)

    import asyncio

    asyncio.run(_main())
