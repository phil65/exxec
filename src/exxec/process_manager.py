"""Process management for background command execution.

Two implementations are provided:

- `SpawnProcessManager` runs processes via `ExecutionEnvironment.spawn()`. It is
  the default for environments with `supports_spawn` (local and its sandboxing
  subclasses, ssh, docker): no timeout, unbuffered output, real termination.
- `StreamingProcessManager` runs commands through the environment's own
  `stream_command()`, so the process still runs *inside* the environment. It is
  the fallback for environments without `spawn()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import codecs
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import uuid

from anyenv.process_manager.models import ProcessOutput
from anyenv.process_manager.process_manager import BaseTerminal
from anyenv.process_manager.protocol import ProcessManagerProtocol

from exxec.command_line import build_command_line, build_remote_command_line
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent
from exxec.log import get_logger


if TYPE_CHECKING:
    from pathlib import Path

    from exxec.base import ExecutionEnvironment
    from exxec.process_handle import ProcessHandle


logger = get_logger(__name__)

DEFAULT_OUTPUT_LIMIT = 1048576
"""Default maximum number of output bytes retained per process."""

KILLED_EXIT_CODE = 130
"""Exit code reported for processes terminated via `kill_process`."""


@dataclass
class TerminalTask(BaseTerminal):
    """Tracks a background process and the task collecting its output."""

    task: asyncio.Task[None] | None = field(default=None)
    handle: ProcessHandle | None = field(default=None)

    def is_running(self) -> bool:
        """Check if the output collection task is still running."""
        return self.task is not None and not self.task.done()


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


class SpawnProcessManager(BaseProcessManager):
    """Runs background processes via `ExecutionEnvironment.spawn()`.

    The default for environments with `supports_spawn`. Processes run inside the
    environment without any timeout, output is captured as it arrives (no line
    buffering), and `kill_process` terminates the process for real.
    """

    def __init__(self, env: ExecutionEnvironment) -> None:
        """Initialize with an execution environment that supports `spawn()`."""
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
            cwd: Working directory (defaults to the environment's)
            env: Additional environment variables
            output_limit: Maximum bytes of output to retain

        Returns:
            Process ID for tracking
        """
        process_id = f"proc_{uuid.uuid4().hex[:8]}"
        args = args or []
        work_dir = str(cwd) if cwd else None
        handle = await self.env.spawn(command, args or None, cwd=work_dir, env=env)
        terminal = TerminalTask(
            terminal_id=process_id,
            command=command,
            args=args,
            cwd=work_dir,
            env=env or {},
            output_limit=output_limit or DEFAULT_OUTPUT_LIMIT,
            handle=handle,
        )
        terminal.task = asyncio.create_task(self._collect(terminal, handle))
        self._terminals[process_id] = terminal
        logger.info("Started process %s in %s: %s", process_id, type(self.env).__name__, command)
        return process_id

    async def _collect(self, terminal: TerminalTask, handle: ProcessHandle) -> None:
        """Collect output until EOF, then record the exit code and release the handle."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            async for chunk in handle.output():
                terminal.add_output(decoder.decode(chunk))
            terminal.add_output(decoder.decode(b"", final=True))
            terminal.set_exit_code(await handle.wait())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            terminal.add_output(f"Process error: {e}\n")
            terminal.set_exit_code(1)
            logger.exception("Error in process %s", terminal.terminal_id)
        finally:
            await handle.aclose()

    async def _terminate(self, terminal: TerminalTask) -> None:
        if terminal.handle is not None:
            await terminal.handle.kill()


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
