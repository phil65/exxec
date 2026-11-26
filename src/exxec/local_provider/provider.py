"""Local execution environment that runs code locally."""

import asyncio
import contextlib
import inspect
import os
import platform
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Literal, Self

from anyenv.processes import create_process, create_shell_process

from exxec.base import ExecutionEnvironment, OSType
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.local_provider.utils import StreamCapture, find_executable
from exxec.log import get_logger
from exxec.models import ExecutionResult
from exxec.parse_output import parse_output, wrap_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from fsspec.asyn import AsyncFileSystem  # type: ignore[import-untyped]

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo
    from exxec.process_handle import ProcessHandle
    from exxec.pty_manager import PtySize


logger = get_logger(__name__)
NEW_SESSION = sys.platform != "win32"
"""Start subprocesses in their own session, so a timeout can kill the whole group."""

PYTHON_EXECUTABLES = [
    "python3",
    "python",
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.14",
]


def _decode(data: bytes | None) -> str:
    return data.decode(errors="replace") if data else ""


async def _kill_process_group(process: asyncio.subprocess.Process, grace: float = 1.0) -> None:
    """Terminate a process started with `NEW_SESSION` and everything in its group.

    Sends TERM to the group, then KILL after `grace` seconds, and reaps the process.
    """
    from exxec.local_provider.process_handle import _kill_tree_windows

    if process.returncode is not None:
        return
    if sys.platform == "win32":
        await _kill_tree_windows(process)
    else:
        import signal

        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=grace)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    await process.wait()


async def _kill_process_groups(pgids: set[int], grace: float) -> None:
    """Terminate whole process groups whose leaders may have exited: TERM, then KILL.

    The members aren't necessarily our children, so their exit is polled.
    """
    import signal

    from exxec.local_provider.process_handle import _group_alive, _signal_group

    alive = {pgid for pgid in pgids if _group_alive(pgid)}
    for pgid in alive:
        _signal_group(pgid, signal.SIGTERM)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace
    while alive := {pgid for pgid in alive if _group_alive(pgid)}:
        if loop.time() >= deadline:
            for pgid in alive:
                _signal_group(pgid, signal.SIGKILL)
            return
        await asyncio.sleep(0.05)


def _platform_os_type() -> OSType:
    """Map `platform.system()` to the supported OS types (other Unixes count as Linux)."""
    match platform.system():
        case "Windows":
            return "Windows"
        case "Darwin":
            return "Darwin"
        case _:
            return "Linux"


class LocalExecutionEnvironment(ExecutionEnvironment):
    """Executes code in the same process or isolated subprocess.

    On POSIX every process it starts leads its own session; leaving the environment ends
    all those process groups, also those of commands that already finished (their
    background children). Nothing started here outlives the environment.
    """

    supports_spawn = True
    supports_pty = sys.platform != "win32"

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        default_command_timeout: float | None = 30.0,
        isolated: bool = False,
        executable: str | None = None,
        language: Language = "python",
        root_path: str | None = None,
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """Initialize local environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of Python packages to install via pip / npm
            default_command_timeout: Default timeout for command execution in seconds.
                If None, commands run without timeout unless explicitly specified.
            isolated: If True, run code in subprocess; if False, run in same process
            executable: Executable to use for isolated mode (if None, auto-detect)
            language: Programming language to use (for isolated mode)
            root_path: Path to become to root of the filesystem
            cwd: Working directory for the environment
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
        self.isolated = isolated
        self.language: Language = language
        self.executable = executable or (find_executable(language) if isolated else None)
        self.process: asyncio.subprocess.Process | None = None
        """The last started isolated process (kept for compatibility)."""
        self._processes: set[asyncio.subprocess.Process] = set()
        self._process_groups: set[int] = set()
        """Process groups started here (each process starts its own session): all of
        them, with whatever they left running, end when the environment closes."""
        self.root_path = root_path
        # Local provider knows OS statically
        self._os_type = _platform_os_type()

    async def __aenter__(self) -> Self:
        # Start tool server via base class
        await super().__aenter__()

        # Install dependencies if specified and in isolated mode
        if self.isolated and self.dependencies and self.language == "python" and self.executable:
            # Into the interpreter that runs the code. The dependency list is the
            # environment's own configuration, so this runs outside a sandbox.
            argv = [self.executable, "-m", "pip", "install", *self.dependencies]
            try:
                process = await create_process(
                    *argv, stdout="pipe", stderr="pipe", start_new_session=NEW_SESSION
                )
                self._track_group(process.pid)
                try:
                    _, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=self.default_command_timeout
                    )
                finally:
                    await _kill_process_group(process)
                if process.returncode:
                    logger.warning("Installing %s failed: %s", self.dependencies, _decode(stderr))
            except Exception as e:  # noqa: BLE001
                # Don't fail: the code might still work
                logger.warning("Installing %s failed: %s", self.dependencies, e)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            for process in list(self._processes):
                await _kill_process_group(process, grace=5.0)
            # Including the groups of commands that already finished: their background
            # children would otherwise outlive the environment.
            await _kill_process_groups(self._process_groups, grace=5.0)
            self._process_groups.clear()
        finally:
            # Cleanup server via base class
            await super().__aexit__(exc_type, exc_val, exc_tb)

    def _track_group(self, pid: int) -> None:
        """Remember the process group a new session leader `pid` heads.

        Groups left empty are dropped first: their id is free again and could be
        reused by an unrelated group.
        """
        if not NEW_SESSION:
            return
        from exxec.local_provider.process_handle import _group_alive

        self._process_groups = {g for g in self._process_groups if _group_alive(g)}
        self._process_groups.add(pid)

    def get_fs(self) -> AsyncFileSystem:
        """Return an AsyncLocalFileSystem for the current working directory."""
        from fsspec.implementations.dirfs import DirFileSystem  # type: ignore[import-untyped]
        from upathtools.filesystems import AsyncLocalFileSystem

        fs = AsyncLocalFileSystem()
        if self.root_path:
            return DirFileSystem(self.root_path, fs)
        return fs

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in same process or isolated subprocess."""
        if self.isolated:
            return await self._execute_subprocess(code)
        return await self._execute_local(code)

    async def _execute_local(self, code: str) -> ExecutionResult:
        """Execute code directly in current process."""
        start_time = time.time()

        try:
            namespace = {"__builtins__": __builtins__}
            exec(code, namespace)

            # Try to get result from main() function
            if "main" in namespace and callable(namespace["main"]):
                main_func = namespace["main"]
                if inspect.iscoroutinefunction(main_func):
                    # Run async function in executor to handle blocking calls properly
                    def run_in_thread() -> Any:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(main_func())
                        finally:
                            loop.close()

                    result = await asyncio.wait_for(
                        asyncio.to_thread(run_in_thread), timeout=self.default_command_timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(main_func), timeout=self.default_command_timeout
                    )
            else:
                result = namespace.get("_result")
            return ExecutionResult(result=result, duration=time.time() - start_time, success=True)

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def _start_code_process(
        self, *, stderr: Literal["pipe", "stdout"]
    ) -> asyncio.subprocess.Process:
        """Start the interpreter for isolated code, through `wrap_argv()` (sandboxes)."""
        process = await create_process(
            *self.wrap_argv(self._get_subprocess_args()),
            stdin="pipe",
            stdout="pipe",
            stderr=stderr,
            env=self.get_env(),
            start_new_session=NEW_SESSION,
        )
        self._track_group(process.pid)
        self.process = process
        self._processes.add(process)
        return process

    async def _finish_process(self, process: asyncio.subprocess.Process) -> None:
        """Kill what's left of a process (timeout, cancellation) and forget it."""
        try:
            await _kill_process_group(process)
        finally:
            self._processes.discard(process)

    async def _execute_subprocess(self, code: str) -> ExecutionResult:
        """Execute code in subprocess with communication via stdin/stdout."""
        start_time = time.time()
        process: asyncio.subprocess.Process | None = None
        try:
            wrapped_code = wrap_code(code, self.language)
            process = await self._start_code_process(stderr="pipe")
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(wrapped_code.encode()),
                    timeout=self.default_command_timeout,
                )
            except TimeoutError:
                msg = f"Execution timed out after {self.default_command_timeout} seconds"
                return ExecutionResult.failed(TimeoutError(msg), start_time)
            stdout = _decode(stdout_data)
            stderr = _decode(stderr_data)
            if process.returncode == 0:
                execution_result, error_info = parse_output(stdout)
                if error_info is None:
                    return ExecutionResult(
                        result=execution_result,
                        duration=time.time() - start_time,
                        success=True,
                        exit_code=process.returncode,
                        stdout=stdout,
                        stderr=stderr,
                    )
                return ExecutionResult(
                    result=None,
                    duration=time.time() - start_time,
                    success=False,
                    error=error_info.get("error", "Subprocess execution failed"),
                    error_type=error_info.get("type", "SubprocessError"),
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                error=stderr or "Subprocess execution failed",
                error_type="SubprocessError",
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)
        finally:
            if process is not None:
                await self._finish_process(process)

    def _get_subprocess_args(self) -> list[str]:
        """Get subprocess arguments based on language."""
        if not self.executable:
            msg = "No executable found for subprocess execution"
            raise RuntimeError(msg)

        match self.language:
            case "python":
                return [self.executable]
            case "javascript":
                return [self.executable]
            case "typescript":
                if shutil.which("ts-node"):
                    return ["ts-node"]
                if shutil.which("tsx"):
                    return ["tsx"]
                return ["npx", "ts-node"]
            case _:
                return [self.executable]

    def wrap_command(self, command: str) -> str:
        """Wrap a shell command before execution.

        Subclasses can override to add sandboxing, containers, etc.
        Default implementation returns command unchanged.

        Args:
            command: Shell command to wrap

        Returns:
            Wrapped command string
        """
        return command

    def wrap_argv(self, argv: list[str]) -> list[str]:
        """Wrap a program invocation (argv form) before direct execution.

        Counterpart of `wrap_command` for processes executed without a shell.
        Subclasses adding sandboxing must override both.
        Default implementation returns argv unchanged.

        Args:
            argv: Program and arguments

        Returns:
            Wrapped argv
        """
        return argv

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pty: PtySize | None = None,
    ) -> ProcessHandle:
        """Start a local process and return a handle to it.

        With `args`, the program is executed directly after `wrap_argv()`;
        otherwise `command` runs as a shell command line after `wrap_command()`.
        Both hooks keep sandboxing subclasses in effect.

        Args:
            command: Program (with `args`) or shell command line (without)
            args: Program arguments
            cwd: Working directory (defaults to the environment's cwd)
            env: Additional environment variables
            pty: Attach to a pseudo-terminal of this size (POSIX only)

        Returns:
            Handle to the running process
        """
        from exxec.local_provider.process_handle import (
            LocalPipeProcessHandle,
            LocalPtyProcessHandle,
        )

        self._check_spawn_pty(pty)
        work_dir = cwd or self.cwd
        base_env = self.get_env()
        if pty is not None:
            pty_env = {**(base_env if base_env is not None else os.environ), **(env or {})}
            pty_env.setdefault("TERM", "xterm-256color")
            if args:
                argv = self.wrap_argv([command, *args])
            else:
                argv = ["/bin/sh", "-c", self.wrap_command(command)]
            handle: ProcessHandle = await LocalPtyProcessHandle.start(
                argv, cwd=work_dir, env=pty_env, size=pty
            )
        else:
            pipe_env = (
                {**(base_env if base_env is not None else os.environ), **env} if env else base_env
            )
            if args:
                handle = await LocalPipeProcessHandle.start(
                    self.wrap_argv([command, *args]), None, cwd=work_dir, env=pipe_env
                )
            else:
                handle = await LocalPipeProcessHandle.start(
                    None, self.wrap_command(command), cwd=work_dir, env=pipe_env
                )
        if handle.pid is not None:
            self._track_group(handle.pid)
        return handle

    async def execute_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a shell command and return result with metadata."""
        start_time = time.time()
        command = self.wrap_command(command)
        effective_timeout = timeout if timeout is not None else self.default_command_timeout

        try:
            process = await create_shell_process(
                command,
                stdout="pipe",
                stderr="pipe",
                cwd=self.cwd,
                env=self.get_env(),
                start_new_session=NEW_SESSION,
            )
        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)
        self._track_group(process.pid)
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(), timeout=effective_timeout
            )
        except TimeoutError:
            msg = f"Command timed out after {effective_timeout} seconds"
            return ExecutionResult.failed(TimeoutError(msg), start_time)
        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)
        finally:
            # On timeout or cancellation: the shell and everything it started.
            await _kill_process_group(process)

        stdout = _decode(stdout_data)
        stderr = _decode(stderr_data)
        success = process.returncode == 0
        return ExecutionResult(
            result=stdout if success else None,
            duration=time.time() - start_time,
            success=success,
            error=stderr if not success else None,
            error_type="CommandError" if not success else None,
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
        )

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events."""
        from exxec.events import ProcessErrorEvent

        process_id = f"local_{id(self)}"
        yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")

        try:
            if self.isolated:
                async for event in self._stream_code_subprocess(code, process_id):
                    yield event
            else:
                async for event in self._stream_code_local(code, process_id):
                    yield event
        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def _stream_code_subprocess(
        self, code: str, process_id: str
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute code in subprocess and stream events."""
        try:
            process = await self._start_code_process(stderr="stdout")
        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)
            return
        try:
            if process.stdin:
                wrapped_code = wrap_code(code, self.language)
                process.stdin.write(wrapped_code.encode())
                process.stdin.close()

            if process.stdout:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            process.stdout.readline(), timeout=self.default_command_timeout
                        )
                    except TimeoutError:
                        await _kill_process_group(process)
                        yield ProcessErrorEvent(
                            process_id=process_id,
                            error=f"Process timed out after {self.default_command_timeout} seconds",
                            error_type="TimeoutError",
                            exit_code=1,
                        )
                        return
                    if not line:
                        break
                    data = _decode(line).rstrip("\n\r")
                    yield OutputEvent(process_id=process_id, data=data, stream="combined")

            exit_code = await process.wait()
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
        finally:
            await self._finish_process(process)

    async def _stream_code_local(self, code: str, process_id: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code locally and stream events."""
        try:
            output_queue: asyncio.Queue[str] = asyncio.Queue()
            stdout_capture = StreamCapture(sys.stdout, output_queue)
            stderr_capture = StreamCapture(sys.stderr, output_queue)
            execution_done = False

            async def execute_code() -> None:
                nonlocal execution_done
                try:
                    namespace = {"__builtins__": __builtins__}

                    with (
                        contextlib.redirect_stdout(stdout_capture),
                        contextlib.redirect_stderr(stderr_capture),
                    ):
                        lang: Literal["python", "javascript", "typescript"] = (
                            "javascript" if self.language == "typescript" else "python"
                        )
                        wrapped_code = wrap_code(code, lang)

                        if self.language == "python":
                            exec(wrapped_code, namespace)

                            if "main" in namespace and callable(namespace["main"]):
                                main_func = namespace["main"]
                                if inspect.iscoroutinefunction(main_func):
                                    result = await asyncio.wait_for(
                                        main_func(), timeout=self.default_command_timeout
                                    )
                                else:
                                    result = await asyncio.wait_for(
                                        asyncio.to_thread(main_func),
                                        timeout=self.default_command_timeout,
                                    )

                                if result is not None:
                                    print(f"Result: {result}")
                            else:
                                result = namespace.get("_result")
                                if result is not None:
                                    print(f"Result: {result}")

                except Exception as e:  # noqa: BLE001
                    print(f"ERROR: {e}", file=sys.stderr)
                finally:
                    execution_done = True
                    with contextlib.suppress(asyncio.QueueFull):
                        output_queue.put_nowait("__EXECUTION_COMPLETE__")

            execute_task = asyncio.create_task(execute_code())

            try:
                while True:
                    try:
                        line = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                        if line == "__EXECUTION_COMPLETE__":
                            break
                        if line and line.strip():
                            yield OutputEvent(
                                process_id=process_id,
                                data=line.rstrip("\n\r"),
                                stream="combined",
                            )
                    except TimeoutError:
                        if execution_done and output_queue.empty():
                            break
                        continue

                await execute_task
                yield ProcessCompletedEvent(process_id=process_id, exit_code=0)

            except Exception as e:  # noqa: BLE001
                if not execute_task.done():
                    execute_task.cancel()
                yield ProcessErrorEvent(
                    process_id=process_id, error=str(e), error_type=type(e).__name__
                )

        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute a shell command and stream its output line by line.

        The command runs via `spawn()`, i.e. wrapped by `wrap_command()`, in the
        environment's cwd and in its own process group. Closing or cancelling the
        stream terminates the process.

        Args:
            command: Shell command line
            timeout: Total time limit in seconds (defaults to `default_command_timeout`;
                no limit if both are None). On expiry the process group is killed.

        Yields:
            Started, output (one per line) and completed/error events
        """
        from exxec.spawn_stream import stream_command_via_spawn

        async for event in stream_command_via_spawn(
            self,
            command,
            timeout=timeout,
            process_id=f"local_cmd_{id(self)}",
            display_command=self.wrap_command(command),
        ):
            yield event


if __name__ == "__main__":
    import asyncio

    provider = LocalExecutionEnvironment()

    async def main() -> None:
        """Example."""
        async for line in provider.execute_command_stream("ls -l"):
            print(line)

    asyncio.run(main())
