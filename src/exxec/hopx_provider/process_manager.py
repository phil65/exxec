"""Hopx-specific terminal manager using sandbox process management."""

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

    from hopx_ai import AsyncSandbox  # type: ignore[import-untyped]


logger = get_logger(__name__)


@dataclass(kw_only=True)
class HopxTerminal(BaseTerminal):
    """Represents a terminal session using Hopx's process management."""

    process_id: str | None = None
    _background_result: dict[str, Any] | None = None

    def is_running(self) -> bool:
        """Check if terminal is still running."""
        return self._exit_code is None


class HopxTerminalManager(ProcessManagerProtocol):
    """Terminal manager that uses Hopx's process management."""

    def __init__(self, sandbox: AsyncSandbox) -> None:
        """Initialize with a Hopx sandbox instance."""
        self.sandbox = sandbox
        self._terminals: dict[str, HopxTerminal] = {}

    async def start_process(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> str:
        """Create a new terminal session using Hopx's background commands."""
        terminal_id = f"hopx_term_{uuid.uuid4().hex[:8]}"
        args = args or []
        full_cmd = f"{command} {' '.join(args)}" if args else command
        terminal = HopxTerminal(
            terminal_id=terminal_id,
            command=command,
            args=args,
            cwd=str(cwd) if cwd else None,
            env=env or {},
            output_limit=output_limit or 1048576,
        )
        self._terminals[terminal_id] = terminal

        try:
            result = await self.sandbox.commands.run(
                full_cmd,
                background=True,
                env=env,
                working_dir=str(cwd) if cwd else "/workspace",
            )
            terminal._background_result = {
                "stdout": result.stdout,
                "process_id": getattr(result, "process_id", None),
            }
            if result.stdout:
                terminal.add_output(result.stdout)
            logger.info("Created Hopx terminal %s: %s", terminal_id, full_cmd)
        except Exception as e:
            self._terminals.pop(terminal_id, None)
            msg = f"Failed to create Hopx terminal: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        else:
            return terminal_id

    async def get_output(self, process_id: str) -> ProcessOutput:
        """Get current output from a process."""
        terminal = self._get_terminal(process_id)
        output = terminal.get_output()
        exit_code = terminal.get_exit_code()
        return ProcessOutput(stdout=output, stderr="", combined=output, exit_code=exit_code)

    async def wait_for_exit(self, process_id: str) -> int:
        """Wait for process to complete."""
        terminal = self._get_terminal(process_id)
        # Background processes in Hopx return immediately;
        # poll via list_processes if needed
        if terminal._exit_code is None:
            terminal.set_exit_code(0)
        return terminal.get_exit_code() or 0

    async def kill_process(self, process_id: str) -> None:
        """Kill a running process."""
        terminal = self._get_terminal(process_id)
        try:
            hopx_pid = terminal.process_id
            if hopx_pid and terminal.is_running():
                await self.sandbox.kill_process(hopx_pid)
                terminal.set_exit_code(130)
                logger.info("Killed Hopx process %s", process_id)
        except Exception:
            logger.exception("Error killing process %s", process_id)
            terminal.set_exit_code(1)

    async def release_process(self, process_id: str) -> None:
        """Release process resources."""
        terminal = self._get_terminal(process_id)
        if terminal.is_running():
            await self.kill_process(process_id)
        del self._terminals[process_id]
        logger.info("Released process %s", process_id)

    async def list_processes(self) -> list[str]:
        """List all tracked terminals."""
        return list(self._terminals.keys())

    async def get_process_info(self, process_id: str) -> dict[str, Any]:
        """Get information about a specific process."""
        terminal = self._get_terminal(process_id)
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

    def _get_terminal(self, terminal_id: str) -> HopxTerminal:
        """Get terminal by ID."""
        if terminal_id not in self._terminals:
            msg = f"Process {terminal_id} not found"
            raise ValueError(msg)
        return self._terminals[terminal_id]

    async def cleanup(self) -> None:
        """Clean up all terminals."""
        logger.info("Cleaning up %s Hopx terminals", len(self._terminals))
        if cleanup_tasks := [self.release_process(id_) for id_ in list(self._terminals)]:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        logger.info("Hopx terminal cleanup completed")
