"""Process management for background command execution.

Two implementations are provided:

- `LocalProcessManager` spawns subprocesses on the local machine. It is used by
  `LocalExecutionEnvironment` and its subclasses and honours their command
  wrapping (e.g. the `srt` sandbox), working directory and environment variables.
- `StreamingProcessManager` runs commands through the environment's own
  `stream_command()`, so the process runs *inside* the environment (container,
  remote host, sandbox) instead of on the host. It is the default for all
  environments that do not provide a dedicated process manager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import codecs
import contextlib
from dataclasses import dataclass, field
import os
import re
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING, Any
import uuid

from anyenv.process_manager.models import ProcessOutput
from anyenv.process_manager.process_manager import BaseTerminal
from anyenv.process_manager.protocol import ProcessManagerProtocol

from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent
from exxec.log import get_logger


if TYPE_CHECKING:
    from pathlib import Path

    from exxec.base import ExecutionEnvironment
    from exxec.local_provider import LocalExecutionEnvironment


logger = get_logger(__name__)

DEFAULT_OUTPUT_LIMIT = 1048576
"""Default maximum number of output bytes retained per process."""

KILLED_EXIT_CODE = 130
"""Exit code reported for processes terminated via `kill_process`."""

_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class TerminalTask(BaseTerminal):
    """Tracks a background process and the task collecting its output."""

    task: asyncio.Task[None] | None = field(default=None)
    process: asyncio.subprocess.Process | None = field(default=None)

    def is_running(self) -> bool:
        """Check if the output collection task is still running."""
        return self.task is not None and not self.task.done()


def build_command_line(command: str, args: list[str] | None, *, windows: bool = False) -> str:
    """Build a shell command line from a command and optional arguments.

    Args:
        command: Program name, or a full shell command string when no args are given
        args: Arguments, quoted for the target shell when joined
        windows: Quote for `cmd.exe` / the Windows C runtime instead of a POSIX shell

    Returns:
        A command line suitable for execution by the target shell
    """
    if not args:
        return command
    if windows:
        return subprocess.list2cmdline([command, *args])
    return shlex.join([command, *args])


def build_remote_command_line(
    command_line: str,
    cwd: str | None,
    env: dict[str, str] | None,
) -> str:
    """Prefix a command line with directory change and environment exports.

    Used for environments where cwd and env can only be passed as part of the
    shell command itself.

    Args:
        command_line: Shell command line to run
        cwd: Working directory to change into first
        env: Environment variables to export first

    Returns:
        Combined POSIX shell command line

    Raises:
        ValueError: If an environment variable name is not a valid identifier
    """
    parts: list[str] = []
    if cwd:
        parts.append(f"cd {shlex.quote(cwd)}")
    if env:
        invalid = [name for name in env if not _ENV_VAR_NAME.match(name)]
        if invalid:
            msg = f"Invalid environment variable names: {', '.join(invalid)}"
            raise ValueError(msg)
        assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
        parts.append(f"export {assignments}")
    parts.append(command_line)
    return " && ".join(parts)


class BaseProcessManager(ProcessManagerProtocol, ABC):
    """Shared bookkeeping for process managers backed by `TerminalTask` records."""

    def __init__(self) -> None:
        """Initialize the process registry."""
        self._terminals: dict[str, TerminalTask] = {}

    def _get(self, process_id: str) -> TerminalTask:
        if process_id not in self._terminals:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)
        return self._terminals[process_id]

    @abstractmethod
    async def _terminate(self, terminal: TerminalTask) -> None:
        """Terminate the underlying process of a running terminal."""

    async def get_output(self, process_id: str) -> ProcessOutput:
        """Get current output from a process."""
        terminal = self._get(process_id)
        output = terminal.get_output()
        return ProcessOutput(
            stdout=output,
            stderr="",
            combined=output,
            truncated=terminal._truncated,
            exit_code=terminal.get_exit_code(),
        )

    async def wait_for_exit(self, process_id: str) -> int:
        """Wait for process to complete and return its exit code."""
        terminal = self._get(process_id)
        if terminal.task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(terminal.task)
        exit_code = terminal.get_exit_code()
        return exit_code if exit_code is not None else 1

    async def kill_process(self, process_id: str) -> None:
        """Kill a running process."""
        terminal = self._get(process_id)
        if terminal.is_running():
            await self._terminate(terminal)
            if terminal.task is not None and not terminal.task.done():
                terminal.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await terminal.task
            terminal.set_exit_code(KILLED_EXIT_CODE)
        logger.info("Killed process %s", process_id)

    async def release_process(self, process_id: str) -> None:
        """Kill the process if needed and drop all tracking state."""
        terminal = self._get(process_id)
        if terminal.is_running():
            await self.kill_process(process_id)
        del self._terminals[process_id]
        logger.info("Released process %s", process_id)

    async def list_processes(self) -> list[str]:
        """List all tracked process IDs."""
        return list(self._terminals.keys())

    async def get_process_info(self, process_id: str) -> dict[str, Any]:
        """Get information about a specific process."""
        terminal = self._get(process_id)
        return {
            "terminal_id": process_id,
            "command": terminal.command,
            "args": terminal.args,
            "cwd": terminal.cwd,
            "created_at": terminal.created_at.isoformat(),
            "is_running": terminal.is_running(),
            "exit_code": terminal.get_exit_code(),
            "output_limit": terminal.output_limit,
        }

    async def cleanup(self) -> None:
        """Kill all running processes and clear tracking state."""
        for process_id in list(self._terminals):
            await self.release_process(process_id)


class LocalProcessManager(BaseProcessManager):
    """Runs background processes as local subprocesses.

    Commands are passed through the environment's `wrap_command()`, so sandboxing
    subclasses (like `SRTExecutionEnvironment`) also apply to background processes.
    """

    def __init__(self, env: LocalExecutionEnvironment) -> None:
        """Initialize with a local execution environment."""
        super().__init__()
        self.env = env

    def _build_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        base = self.env.get_env()
        if not env:
            return base
        return {**(base if base is not None else os.environ), **env}

    async def start_process(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> str:
        """Start a background process on the local machine.

        Args:
            command: Program name, or a shell command string when no args are given
            args: Arguments for the program
            cwd: Working directory (defaults to the environment's cwd)
            env: Additional environment variables
            output_limit: Maximum bytes of output to retain

        Returns:
            Process ID for tracking

        Raises:
            OSError: If the process could not be spawned
        """
        from anyenv.processes import create_shell_process

        process_id = f"proc_{uuid.uuid4().hex[:8]}"
        args = args or []
        work_dir = str(cwd) if cwd else self.env.cwd
        command_line = self.env.wrap_command(
            build_command_line(command, args, windows=sys.platform == "win32")
        )
        process = await create_shell_process(
            command_line,
            stdout="pipe",
            stderr="stdout",
            cwd=work_dir,
            env=self._build_env(env),
            start_new_session=sys.platform != "win32",
        )
        terminal = TerminalTask(
            terminal_id=process_id,
            command=command,
            args=args,
            cwd=work_dir,
            env=env or {},
            output_limit=output_limit or DEFAULT_OUTPUT_LIMIT,
            process=process,
        )
        terminal.task = asyncio.create_task(self._collect(terminal, process))
        self._terminals[process_id] = terminal
        logger.info("Started local process %s: %s", process_id, command_line)
        return process_id

    async def _collect(self, terminal: TerminalTask, process: asyncio.subprocess.Process) -> None:
        """Collect output until EOF, then record the exit code.

        Reads in chunks rather than lines so output without trailing newlines
        (prompts, progress indicators) is captured immediately.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            if process.stdout is not None:
                while chunk := await process.stdout.read(8192):
                    terminal.add_output(decoder.decode(chunk))
                terminal.add_output(decoder.decode(b"", final=True))
            terminal.set_exit_code(await process.wait())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            terminal.add_output(f"Process error: {e}\n")
            terminal.set_exit_code(1)
            logger.exception("Error in process %s", terminal.terminal_id)

    async def _terminate(self, terminal: TerminalTask) -> None:
        from anyenv.processes import hard_kill

        if terminal.process is not None:
            await hard_kill(terminal.process)


class StreamingProcessManager(BaseProcessManager):
    """Runs background processes through the environment's `stream_command()`.

    This executes commands inside the environment (container, remote host,
    sandbox). Limitations, inherent to `stream_command()`:

    - The environment's `default_command_timeout` applies to the process.
    - `kill_process` cancels the event stream; whether the remote process is
      terminated as well depends on the provider's cancellation handling.
    - Output is line-buffered and stdout/stderr are combined.
    """

    def __init__(self, env: ExecutionEnvironment) -> None:
        """Initialize with an execution environment."""
        super().__init__()
        self.env = env

    async def start_process(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> str:
        """Start a background process inside the execution environment.

        Args:
            command: Program name, or a shell command string when no args are given
            args: Arguments for the program
            cwd: Working directory inside the environment
            env: Additional environment variables
            output_limit: Maximum bytes of output to retain

        Returns:
            Process ID for tracking

        Raises:
            ValueError: If an environment variable name is invalid
        """
        process_id = f"proc_{uuid.uuid4().hex[:8]}"
        args = args or []
        work_dir = str(cwd) if cwd else None
        command_line = build_remote_command_line(build_command_line(command, args), work_dir, env)
        terminal = TerminalTask(
            terminal_id=process_id,
            command=command,
            args=args,
            cwd=work_dir,
            env=env or {},
            output_limit=output_limit or DEFAULT_OUTPUT_LIMIT,
        )
        terminal.task = asyncio.create_task(self._collect(terminal, command_line))
        self._terminals[process_id] = terminal
        logger.info("Started process %s in %s: %s", process_id, type(self.env).__name__, command)
        return process_id

    async def _collect(self, terminal: TerminalTask, command_line: str) -> None:
        """Consume execution events and record output and exit code."""
        try:
            async for event in self.env.stream_command(command_line):
                match event:
                    case OutputEvent(data=data):
                        terminal.add_output(f"{data}\n")
                    case ProcessCompletedEvent(exit_code=exit_code):
                        terminal.set_exit_code(exit_code)
                    case ProcessErrorEvent(exit_code=int() as exit_code, error_type="CommandError"):
                        terminal.set_exit_code(exit_code)
                    case ProcessErrorEvent(error=error, exit_code=exit_code):
                        terminal.add_output(f"Process error: {error}\n")
                        terminal.set_exit_code(exit_code if exit_code is not None else 1)
                    case _:
                        pass
            if terminal.get_exit_code() is None:
                terminal.set_exit_code(0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            terminal.add_output(f"Process error: {e}\n")
            terminal.set_exit_code(1)
            logger.exception("Error in process %s", terminal.terminal_id)

    async def _terminate(self, terminal: TerminalTask) -> None:
        # Cancelling the collection task closes the stream_command generator,
        # which is the only cancellation mechanism stream_command offers.
        if terminal.task is not None and not terminal.task.done():
            terminal.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await terminal.task


EnvironmentTerminalManager = StreamingProcessManager
"""Deprecated alias for `StreamingProcessManager`."""
