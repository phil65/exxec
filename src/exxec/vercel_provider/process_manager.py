"""Vercel-specific terminal manager using detached command execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import uuid

from anyenv.process_manager import ProcessManagerProtocol, ProcessOutput
from anyenv.process_manager.process_manager import BaseTerminal

from exxec.log import get_logger


if TYPE_CHECKING:
    from pathlib import Path

    from vercel.sandbox import Command, Sandbox


logger = get_logger(__name__)


@dataclass
class VercelTerminal(BaseTerminal):
    """Terminal implementation for Vercel provider."""

    command_id: str | None = None
    _command: Command | None = None
    _task: asyncio.Task[None] | None = None

    def is_running(self) -> bool:
        """Check if the terminal is still running."""
        if self._command is None:
            return False
        try:
            # Vercel commands don't seem to have a direct status check
            # We'll consider it running if we haven't gotten an exit code
            return self.get_exit_code() is None
        except Exception:  # noqa: BLE001
            return False


class VercelTerminalManager(ProcessManagerProtocol):
    """Terminal manager that uses Vercel's detached command execution."""

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize with a Vercel sandbox instance."""
        self.sandbox = sandbox
        self._terminals: dict[str, VercelTerminal] = {}

    async def start_process(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> str:
        """Create a new terminal session using Vercel's detached commands."""
        terminal_id = f"vercel_term_{uuid.uuid4().hex[:8]}"
        args = args or []

        # Create terminal object
        terminal = VercelTerminal(
            terminal_id=terminal_id,
            command=command,
            args=args,
            cwd=str(cwd) if cwd else None,
            env=env or {},
            output_limit=output_limit or 1048576,
        )

        cmd = " ".join([command, *args])
        try:
            cwd = str(cwd) if cwd else None
            detached_cmd = await self.sandbox.create_command(command, args, cwd=cwd, env=env)
            terminal._command = detached_cmd
            terminal.command_id = detached_cmd.id
            self._terminals[terminal_id] = terminal
            task = asyncio.create_task(self._collect_output(terminal))
            terminal._task = task
            msg = "Started Vercel terminal %s: %s"
            logger.info(msg, terminal_id, cmd)
        except Exception as e:
            logger.exception("Failed to start Vercel command: %s", cmd)
            terminal.add_output(f"Failed to start command: {e}\n")
            terminal.set_exit_code(1)
            self._terminals[terminal_id] = terminal

        return terminal_id

    async def _collect_output(self, terminal: VercelTerminal) -> None:
        """Collect output from Vercel command."""
        try:
            if command := terminal._command:
                # Collect stdout
                if command.stdout and (stdout := await command.stdout.read()):
                    terminal.add_output(stdout.decode())
                # Collect stderr
                if command.stderr and (stderr := await command.stderr.read()):
                    terminal.add_output(f"STDERR: {stderr.decode()}")

                terminal.set_exit_code(await command.wait())

        except Exception as e:
            logger.exception("Error collecting output for %s", terminal.terminal_id)
            terminal.add_output(f"Terminal error: {e}\n")
            terminal.set_exit_code(1)

    async def get_output(self, process_id: str) -> ProcessOutput:
        """Get current output from a process."""
        if process_id not in self._terminals:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        terminal = self._terminals[process_id]
        # Try to update status if command exists
        if terminal._command and terminal.is_running():
            try:
                await terminal._command.refresh()
                if terminal._command.returncode is not None:
                    terminal.set_exit_code(terminal._command.returncode)
            except Exception:  # noqa: BLE001
                pass  # Best effort

        output = terminal.get_output()
        exit_code = terminal.get_exit_code()
        return ProcessOutput(stdout=output, stderr="", combined=output, exit_code=exit_code)

    async def wait_for_exit(self, process_id: str) -> int:
        """Wait for process to complete."""
        if process_id not in self._terminals:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        terminal = self._terminals[process_id]
        cmd = terminal._command
        try:
            if cmd and terminal.is_running():
                terminal.set_exit_code(await cmd.wait())

        except Exception:
            logger.exception("Error waiting for process %s", process_id)
            terminal.set_exit_code(1)

        return terminal.get_exit_code() or 0

    async def kill_process(self, process_id: str) -> None:
        """Kill a running process."""
        if process_id not in self._terminals:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        terminal = self._terminals[process_id]
        if terminal._command and terminal.is_running():
            await terminal._command.kill()
            terminal.set_exit_code(137)  # SIGKILL exit code
            logger.info("Killed Vercel process %s", process_id)

    async def release_process(self, process_id: str) -> None:
        """Release process resources."""
        if process_id not in self._terminals:
            msg = f"Process {process_id} not found"
            raise ValueError(msg)

        terminal = self._terminals[process_id]
        if terminal.is_running():  # Kill if still running
            await self.kill_process(process_id)
        del self._terminals[process_id]
        logger.info("Released process %s", process_id)

    async def list_processes(self) -> list[str]:
        """List all tracked terminals."""
        return list(self._terminals.keys())

    async def get_process_info(self, process_id: str) -> dict[str, Any]:
        """Get information about a specific process."""
        terminal = self._terminals[process_id]
        return {
            "command": terminal.command,
            "args": terminal.args,
            "cwd": str(terminal.cwd) if terminal.cwd else None,
            "created_at": terminal.created_at.isoformat(),
            "is_running": terminal.is_running(),
            "exit_code": terminal.get_exit_code(),
            "command_id": terminal.command_id,
        }

    # async def list_processes(self) -> dict[str, dict[str, Any]]:
    #     """List all tracked terminals and their status."""
    #     return {
    #         terminal_id: await self.get_process_info(terminal_id)
    #         for terminal_id in self._terminals
    #     }
