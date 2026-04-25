"""Sprites execution environment that runs code in Fly.io cloud VMs."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any, Self

from exxec.base import ExecutionEnvironment
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.exceptions import NotInitializedError
from exxec.models import ExecutionResult
from exxec.parse_output import get_script_path, parse_output, wrap_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from sprites import Sprite, SpritesClient
    from upathtools.filesystems import SpritesFS

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo
    from exxec.sprites_provider.pty_manager import SpritesPtyManager


def _get_execution_command(language: Language, script_path: str) -> list[str]:
    """Get execution command parts based on language."""
    match language:
        case "python":
            return ["python", script_path]
        case "javascript":
            return ["node", script_path]
        case "typescript":
            return ["npx", "ts-node", script_path]
        case _:
            return ["python", script_path]


async def _run_in_thread[T](func: Callable[..., T], *args: Any) -> T:
    """Run a sync function in a thread to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


class SpritesExecutionEnvironment(ExecutionEnvironment):
    """Executes code in a Sprites (Fly.io) cloud VM."""

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        name: str | None = None,
        timeout: float = 300.0,
        default_command_timeout: float | None = None,
        keep_alive: bool = False,
        language: Language = "python",
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
        token: str | None = None,
        base_url: str = "https://api.sprites.dev",
        ram_mb: int | None = None,
        cpus: int | None = None,
        region: str | None = None,
        storage_gb: int | None = None,
    ) -> None:
        """Initialize Sprites environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of packages to install via pip / npm
            name: Sprite name (auto-generated if None)
            timeout: Default command timeout in seconds
            default_command_timeout: Default timeout for command execution in seconds.
                If None, commands run without timeout unless explicitly specified.
            keep_alive: Keep sprite running after exiting context manager
            language: Programming language to use
            cwd: Working directory for the sprite
            env_vars: Environment variables to set for all executions
            inherit_env: If True, inherit environment variables from os.environ
            token: Sprites API token (or use SPRITES_TOKEN env var)
            base_url: Sprites API base URL
            ram_mb: RAM in MB for the sprite
            cpus: Number of CPUs for the sprite
            region: Preferred region for sprite creation
            storage_gb: Storage in GB for the sprite
        """
        super().__init__(
            lifespan_handler=lifespan_handler,
            dependencies=dependencies,
            cwd=cwd,
            env_vars=env_vars,
            inherit_env=inherit_env,
            default_command_timeout=default_command_timeout,
        )
        self.name = name
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.language: Language = language
        self.token = token
        self.base_url = base_url
        self.ram_mb = ram_mb
        self.cpus = cpus
        self.region = region
        self.storage_gb = storage_gb
        self.sprites_client: SpritesClient | None = None
        self.sprite: Sprite | None = None
        self._auto_created = False
        # Sprites VMs run Linux
        self._os_type = "Linux"
        self._pty_manager: SpritesPtyManager | None = None

    def _ensure_initialized(self) -> Sprite:
        """Ensure the sprite has been created/connected."""
        if self.sprite is None:
            raise NotInitializedError("Sprite")
        return self.sprite

    async def __aenter__(self) -> Self:
        """Setup Sprites environment."""
        import os

        from sprites import SpritesClient
        from sprites.types import SpriteConfig

        await super().__aenter__()

        token = self.token or os.environ.get("SPRITES_TOKEN", "")
        self.sprites_client = SpritesClient(
            token=token,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        config = SpriteConfig(
            ram_mb=self.ram_mb,
            cpus=self.cpus,
            region=self.region,
            storage_gb=self.storage_gb,
        )

        if self.name:
            # Try to connect to existing sprite, create if not found
            try:
                self.sprite = await _run_in_thread(self.sprites_client.get_sprite, self.name)
            except Exception:  # noqa: BLE001
                self.sprite = await _run_in_thread(
                    self.sprites_client.create_sprite, self.name, config
                )
                self._auto_created = True
        else:
            # Auto-generate name and create
            import uuid

            self.name = f"exxec-{uuid.uuid4().hex[:8]}"
            self.sprite = await _run_in_thread(self.sprites_client.create_sprite, self.name, config)
            self._auto_created = True

        # Install dependencies if specified
        if self.dependencies:
            match self.language:
                case "python":
                    await self._run_command_async("pip", "install", *self.dependencies)
                case "javascript" | "typescript":
                    await self._run_command_async("npm", "install", *self.dependencies)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup sprite."""
        if self.sprite and not self.keep_alive and self._auto_created:
            with contextlib.suppress(Exception):
                await _run_in_thread(self.sprite.delete)
        if self.sprites_client:
            with contextlib.suppress(Exception):
                await _run_in_thread(self.sprites_client.close)
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _run_command_async(self, *args: str, **kwargs: Any) -> Any:
        """Run a sprite command in a thread."""
        from sprites.exec import run

        sprite = self._ensure_initialized()
        env = kwargs.pop("env", None) or self.get_env()
        cwd_val = kwargs.pop("cwd", None) or self.cwd
        timeout_val = kwargs.pop("timeout", None)

        def _do_run() -> Any:
            return run(
                sprite,
                *args,
                capture_output=True,
                timeout=timeout_val,
                env=env or {},
                cwd=cwd_val,
            )

        return await _run_in_thread(_do_run)

    def get_fs(self) -> SpritesFS:
        """Return a SpritesFS instance for the sprite."""
        from upathtools.filesystems import SpritesFS

        self._ensure_initialized()
        assert self.name is not None
        token = self.token or ""
        return SpritesFS(
            sprite_name=self.name,
            token=token,
            base_url=self.base_url,
        )

    def get_pty_manager(self) -> SpritesPtyManager:
        """Return a SpritesPtyManager for interactive terminal sessions."""
        if self._pty_manager is None:
            from exxec.sprites_provider.pty_manager import SpritesPtyManager

            sprite = self._ensure_initialized()
            self._pty_manager = SpritesPtyManager(sprite)
        return self._pty_manager

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in the Sprites VM."""
        sprite = self._ensure_initialized()
        start_time = time.time()
        try:
            wrapped_code = wrap_code(code, language=self.language)
            script_path = get_script_path(self.language)

            # Write code to file via filesystem API
            fs = sprite.filesystem("/")
            await _run_in_thread(lambda: fs.path(script_path).write_text(wrapped_code))

            # Execute the script
            cmd_parts = _get_execution_command(self.language, script_path)
            result = await self._run_command_async(*cmd_parts)

            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

            execution_result, error_info = parse_output(stdout)
            if result.returncode == 0 and error_info is None:
                return ExecutionResult(
                    result=execution_result,
                    duration=time.time() - start_time,
                    success=True,
                    exit_code=result.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                exit_code=result.returncode,
                error=(error_info or {}).get("error", "Command execution failed"),
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
        """Execute a terminal command in the Sprites VM."""
        self._ensure_initialized()
        start_time = time.time()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        try:
            result = await self._run_command_async(
                "bash",
                "-c",
                command,
                timeout=effective_timeout,
            )
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            success = result.returncode == 0
            return ExecutionResult(
                result=stdout if success else None,
                duration=time.time() - start_time,
                success=success,
                error=stderr if not success else None,
                error_type="CommandError" if not success else None,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
            )
        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events in the Sprites VM."""
        sprite = self._ensure_initialized()
        process_id = f"sprites_{id(sprite)}"
        wrapped_code = wrap_code(code, language=self.language)
        script_path = get_script_path(self.language)

        # Write code to file
        fs = sprite.filesystem("/")
        await _run_in_thread(lambda: fs.path(script_path).write_text(wrapped_code))

        cmd_parts = _get_execution_command(self.language, script_path)
        yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")
        try:
            result = await self._run_command_async(*cmd_parts)
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

            if stdout:
                for line in stdout.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stdout")
            if stderr:
                for line in stderr.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stderr")

            if result.returncode != 0:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=stderr or f"Command exited with code {result.returncode}",
                    error_type="ExecutionError",
                    exit_code=result.returncode,
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
        """Execute a terminal command and stream events in the Sprites VM."""
        sprite = self._ensure_initialized()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        process_id = f"sprites_cmd_{id(sprite)}"
        yield ProcessStartedEvent(process_id=process_id, command=command)
        try:
            result = await self._run_command_async(
                "bash",
                "-c",
                command,
                timeout=effective_timeout,
            )
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

            if stdout:
                for line in stdout.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stdout")
            if stderr:
                for line in stderr.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stderr")

            if result.returncode == 0:
                yield ProcessCompletedEvent(
                    process_id=process_id,
                    exit_code=result.returncode,
                )
            else:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=f"Command exited with code {result.returncode}",
                    error_type="CommandError",
                    exit_code=result.returncode,
                )

        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def set_file_content(self, path: str, content: str | bytes) -> None:
        """Set file content in the Sprites VM filesystem."""
        sprite = self._ensure_initialized()
        fs = sprite.filesystem("/")
        if isinstance(content, str):
            await _run_in_thread(lambda: fs.path(path).write_text(content))
        else:
            await _run_in_thread(lambda: fs.path(path).write_bytes(content))

    async def get_file_content(self, path: str) -> bytes:
        """Get file content from the Sprites VM filesystem."""
        sprite = self._ensure_initialized()
        fs = sprite.filesystem("/")
        return await _run_in_thread(lambda: fs.path(path).read_bytes())


if __name__ == "__main__":

    async def _main() -> None:
        async with SpritesExecutionEnvironment() as env:
            result = await env.execute_command("echo hello")
            print(result)

    import asyncio

    asyncio.run(_main())
