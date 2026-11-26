"""Docker execution environment that runs code in isolated containers."""

import asyncio
import contextlib
from pathlib import Path
import shlex
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Literal, Self
import uuid

from exxec.base import ExecutionEnvironment
from exxec.events import ProcessErrorEvent, ProcessStartedEvent
from exxec.log import get_logger
from exxec.models import ExecutionResult
from exxec.parse_output import parse_output, wrap_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from fsspec.implementations.dirfs import DirFileSystem  # type: ignore[import-untyped]
    from testcontainers.core.container import DockerContainer

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo
    from exxec.process_handle import ProcessHandle
    from exxec.pty_manager import PtySize


logger = get_logger(__name__)
_WORKSPACE_ENTRIES = "/workspace/..?* /workspace/.[!.]* /workspace/*"


class DockerExecutionEnvironment(ExecutionEnvironment):
    """Executes code in a Docker container with HTTP tool callbacks."""

    supports_spawn = True
    supports_pty = True

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        image: str = "python:3.13-slim",
        default_command_timeout: float | None = 60.0,
        language: Language = "python",
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """Initialize Docker environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of packages to install (pip for Python, npm for JS/TS)
            image: Docker image to use
            default_command_timeout: Default timeout for command execution in seconds.
                If None, commands run without timeout unless explicitly specified.
            language: Programming language to use
            cwd: Working directory for the sandbox
            env_vars: Environment variables to set for all executions
            inherit_env: If True, inherit environment variables from os.environ
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
        self.language: Language = language
        self.container: DockerContainer | None = None
        self.host_workdir: str | None = None
        # Docker containers are always Linux
        self._os_type = "Linux"

    async def __aenter__(self) -> Self:
        from testcontainers.core.container import DockerContainer

        await super().__aenter__()
        self.host_workdir = tempfile.mkdtemp()  # Create temp dir on host for shared fs
        self.container = DockerContainer(self.image).with_volume_mapping(
            self.host_workdir, "/workspace", "rw"
        )
        # Add environment variables if specified
        env = self.get_env() or {}
        for key, value in env.items():
            self.container = self.container.with_env(key, value)
        if self.server_info:
            self.container = self.container.with_kwargs(network_mode="host")
        self.container = self.container.with_command(["sh", "-c", "sleep infinity"])
        await asyncio.to_thread(self.container.start)

        # Install once, before the first command can run (not in the background).
        install_commands: list[str] = []
        if self.server_info:
            install_commands.append("pip install httpx2")
        if self.dependencies:
            match self.language:
                case "python":
                    install_commands.append(shlex.join(["pip", "install", *self.dependencies]))
                case "javascript" | "typescript":
                    install_commands.append(shlex.join(["npm", "install", *self.dependencies]))
        if install_commands:
            argv = ["sh", "-c", " && ".join(install_commands)]
            result = await asyncio.to_thread(self.container.exec, argv)
            if result.exit_code != 0:
                output = _decode(result.output)
                logger.warning("Installing dependencies failed: %s", output)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.container:  # Cleanup container
            container = self.container

            def stop() -> None:
                # Files the container created in the bind mount belong to its (root)
                # user: remove them from inside, so the host directory can go too.
                with contextlib.suppress(Exception):
                    container.exec([
                        "sh",
                        "-c",
                        "rm -rf /workspace/..?* /workspace/.[!.]* /workspace/*",
                    ])
                with contextlib.suppress(Exception):
                    container.stop()

            await asyncio.to_thread(stop)
        if self.host_workdir:  # Cleanup host working directory
            with contextlib.suppress(Exception):
                shutil.rmtree(self.host_workdir)
        await super().__aexit__(exc_type, exc_val, exc_tb)

    def get_fs(self) -> DirFileSystem:
        """Return a DirFileSystem instance for the shared host directory."""
        from fsspec.implementations.dirfs import DirFileSystem
        from upathtools.filesystems import AsyncLocalFileSystem

        if not self.host_workdir:
            msg = "Docker environment not started"
            raise RuntimeError(msg)
        return DirFileSystem(path=self.host_workdir, fs=AsyncLocalFileSystem())

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pty: PtySize | None = None,
    ) -> ProcessHandle:
        """Start a process in the container via `docker exec`.

        Args:
            command: Program (with `args`) or shell command line (without)
            args: Program arguments, passed without shell interpretation
            cwd: Working directory in the container (defaults to /workspace)
            env: Additional environment variables, merged over `env_vars`
            pty: Allocate a TTY of this size

        Returns:
            Handle to the running process

        Raises:
            RuntimeError: If the container has not been started
        """
        from exxec.docker_provider.process_handle import DockerProcessHandle

        if not self.container:
            msg = "Docker environment not started"
            raise RuntimeError(msg)
        wrapped = self.container.get_wrapped_container()
        if wrapped.client is None:
            msg = "Docker container has no API client"
            raise RuntimeError(msg)
        argv = [command, *args] if args else ["sh", "-c", command]
        merged_env = {**self.env_vars, **(env or {})}
        return await DockerProcessHandle.start(
            wrapped.client.api,
            str(wrapped.id),
            argv,
            cwd=cwd or "/workspace",
            env=merged_env or None,
            size=pty,
        )

    async def _run(
        self, command: str, args: list[str] | None, timeout: float | None
    ) -> tuple[str, int]:
        """Run a process via `spawn()` and collect its (combined) output.

        Raises:
            TimeoutError: After killing the process, if it ran longer than `timeout`
        """
        handle = await self.spawn(command, args)
        async with handle:
            chunks: list[bytes] = []
            try:
                async with asyncio.timeout(timeout):
                    chunks.extend([chunk async for chunk in handle.output()])
                    exit_code = await handle.wait()
            except TimeoutError:
                await handle.kill(grace=1.0)
                raise
        return _decode(b"".join(chunks)), exit_code

    def _write_script(self, code: str) -> str:
        """Write wrapped code to a script of its own in the shared workdir; return its name."""
        if not self.host_workdir:
            error_msg = "Host working directory not initialized"
            raise RuntimeError(error_msg)
        lang: Literal["python", "javascript", "typescript"] = (
            "javascript" if self.language == "typescript" else self.language
        )
        ext = {"python": "py", "javascript": "js", "typescript": "ts"}.get(self.language, "py")
        name = f"script-{uuid.uuid4().hex}.{ext}"
        Path(self.host_workdir, name).write_text(wrap_code(code, lang))
        return name

    def _remove_script(self, name: str) -> None:
        if self.host_workdir:
            with contextlib.suppress(OSError):
                Path(self.host_workdir, name).unlink()

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in Docker container."""
        start_time = time.time()
        if not self.container:
            error_msg = "Docker environment not properly initialized"
            raise RuntimeError(error_msg)
        if not self.host_workdir:
            error_msg = "Host working directory not initialized"
            raise RuntimeError(error_msg)
        script = self._write_script(code)
        try:
            program, *args = get_execution_argv(self.language, script)
            output, exit_code = await self._run(program, args, self.default_command_timeout)
        except TimeoutError:
            msg = f"Execution timed out after {self.default_command_timeout} seconds"
            return ExecutionResult.failed(TimeoutError(msg), start_time)
        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)
        finally:
            self._remove_script(script)
        duration = time.time() - start_time
        execution_result, error_info = parse_output(output)
        if exit_code == 0 and error_info is None:
            return ExecutionResult(
                result=execution_result,
                duration=duration,
                success=True,
                exit_code=exit_code,
                stdout=output,
                stderr="",
            )
        return ExecutionResult(
            result=None,
            duration=duration,
            success=False,
            error=error_info.get("error", "Container execution failed")
            if error_info
            else "Container execution failed",
            error_type=error_info.get("type", "ContainerError") if error_info else "ContainerError",
            exit_code=exit_code,
            stdout=output,
            stderr="",
        )

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a shell command in the container's /workspace and return the result."""
        start_time = time.time()
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        try:
            if not self.container:
                error_msg = "Docker environment not properly initialized"
                raise RuntimeError(error_msg)  # noqa: TRY301
            stdout, exit_code = await self._run(command, None, effective_timeout)
        except TimeoutError:
            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                error=f"Command timed out after {effective_timeout} seconds",
                error_type="TimeoutError",
                exit_code=1,
                stdout="",
                stderr="",
            )
        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)
        success = exit_code == 0
        return ExecutionResult(
            result=stdout if success else None,
            duration=time.time() - start_time,
            success=success,
            error=stdout if not success else None,  # stderr is merged into stdout
            error_type="CommandError" if not success else None,
            exit_code=exit_code,
            stdout=stdout,
            stderr="",
        )

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events in the Docker container."""
        from exxec.spawn_stream import stream_command_via_spawn

        process_id = f"docker_{id(self.container)}"
        if not self.container or not self.host_workdir:
            yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")
            msg = "Docker container not initialized"
            yield ProcessErrorEvent(process_id=process_id, error=msg, error_type="RuntimeError")
            return
        script = self._write_script(code)
        try:
            async for event in stream_command_via_spawn(
                self,
                shlex.join(get_execution_argv(self.language, script)),
                timeout=None,
                process_id=process_id,
                display_command=f"execute({len(code)} chars)",
            ):
                yield event
        finally:
            self._remove_script(script)

    async def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute a shell command in /workspace and stream its output line by line.

        Args:
            command: Shell command line
            timeout: Total time limit in seconds (defaults to `default_command_timeout`);
                on expiry the process group is killed.

        Yields:
            Started, output (one per line) and completed/error events
        """
        from exxec.spawn_stream import stream_command_via_spawn

        process_id = f"docker_cmd_{id(self.container)}"
        if not self.container:
            yield ProcessStartedEvent(process_id=process_id, command=command)
            yield ProcessErrorEvent(
                process_id=process_id,
                error="Docker container not initialized",
                error_type="RuntimeError",
            )
            return
        async for event in stream_command_via_spawn(
            self, command, timeout=timeout, process_id=process_id
        ):
            yield event


def _decode(data: bytes | None) -> str:
    return data.decode(errors="replace") if data else ""


def get_execution_argv(language: Language, script: str) -> list[str]:
    """Get the program and arguments that run a script (in /workspace) of the language."""
    match language:
        case "javascript":
            return ["node", script]
        case "typescript":
            return ["npx", "ts-node", script]
        case _:
            return ["python", script]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Example."""
        async with DockerExecutionEnvironment() as provider:
            await provider.execute_command("mkdir test")
            async for line in provider.execute_command_stream("ls"):
                print(line)

    asyncio.run(main())
