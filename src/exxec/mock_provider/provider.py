"""Mock execution environment for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from fsspec.implementations.asyn_wrapper import (  # type: ignore[import-untyped]
    AsyncFileSystemWrapper,
)
from fsspec.implementations.memory import MemoryFileSystem  # type: ignore[import-untyped]

from exxec.base import ExecutionEnvironment
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.mock_provider.process_manager import MockProcessManager
from exxec.models import ExecutionResult


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anyenv.process_manager.models import ProcessOutput
    from fsspec.asyn import AsyncFileSystem  # type: ignore[import-untyped]

    from exxec.events import ExecutionEvent


class MockExecutionEnvironment(ExecutionEnvironment):
    """Mock execution environment for testing with memory FS and fake processes."""

    def __init__(
        self,
        code_results: dict[str, ExecutionResult] | None = None,
        command_results: dict[str, ExecutionResult] | None = None,
        default_result: ExecutionResult | None = None,
        process_outputs: dict[str, ProcessOutput] | None = None,
        default_process_output: ProcessOutput | None = None,
        code_exceptions: dict[str, Exception] | None = None,
        command_exceptions: dict[str, Exception] | None = None,
        cwd: str | None = None,
        deterministic_ids: bool = False,
    ) -> None:
        """Initialize mock execution environment.

        Args:
            code_results: Map of code string -> result for execute()
            command_results: Map of command string -> result for execute_command()
            default_result: Default result when no match found
            process_outputs: Map of command -> output for process manager
            default_process_output: Default output for process manager
            code_exceptions: Map of code string -> exception to raise during stream_code
            command_exceptions: Map of command -> exception to raise during stream_command
            cwd: Working directory for the sandbox
            deterministic_ids: If True, use sequential IDs instead of UUIDs for processes
        """
        super().__init__(cwd=cwd)
        self._code_results = code_results or {}
        self._command_results = command_results or {}
        self._code_exceptions = code_exceptions or {}
        self._command_exceptions = command_exceptions or {}
        self._default_result = default_result or ExecutionResult(
            result=None,
            duration=0.001,
            success=True,
            stdout="",
            stderr="",
            exit_code=0,
        )
        self._sync_fs = MemoryFileSystem()
        self._fs = AsyncFileSystemWrapper(self._sync_fs)
        self._mock_process_manager = MockProcessManager(
            default_output=default_process_output,
            command_outputs=process_outputs,
        )
        # Mock environment defaults to Linux
        self._os_type = "Linux"
        self._deterministic_ids = deterministic_ids
        self._process_counter = 0

    @property
    def process_manager(self) -> MockProcessManager:
        """Get the mock process manager."""
        return self._mock_process_manager

    def _generate_process_id(self, prefix: str) -> str:
        """Generate a process ID, either deterministic or random.

        Args:
            prefix: Prefix for the process ID (e.g., 'stream', 'cmd')

        Returns:
            Process ID string
        """
        if self._deterministic_ids:
            self._process_counter += 1
            return f"{prefix}_{self._process_counter:04d}"
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def get_fs(self) -> AsyncFileSystem:
        """Return the async-wrapped memory filesystem."""
        return self._fs

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code and return mock result."""
        return self._code_results.get(code, self._default_result)

    async def execute_command(self, command: str) -> ExecutionResult:
        """Execute command and return mock result."""
        return self._command_results.get(command, self._default_result)

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Stream mock code execution events."""
        # Check for exception simulation first
        if code in self._code_exceptions:
            raise self._code_exceptions[code]

        result = self._code_results.get(code, self._default_result)
        process_id = self._generate_process_id("code")

        yield ProcessStartedEvent(process_id=process_id, command="python", pid=12345)

        if result.stdout:
            yield OutputEvent(process_id=process_id, data=result.stdout, stream="stdout")
        if result.stderr:
            yield OutputEvent(process_id=process_id, data=result.stderr, stream="stderr")

        # If there's an error, yield ProcessErrorEvent instead of ProcessCompletedEvent
        if not result.success and result.error:
            yield ProcessErrorEvent(
                process_id=process_id,
                error=result.error,
                error_type=result.error_type or "error",
                exit_code=result.exit_code or 1,
            )
        else:
            yield ProcessCompletedEvent(
                process_id=process_id,
                exit_code=result.exit_code or 0,
                duration=result.duration,
            )

    async def stream_command(self, command: str) -> AsyncIterator[ExecutionEvent]:
        """Stream mock command execution events."""
        # Check for exception simulation first
        if command in self._command_exceptions:
            raise self._command_exceptions[command]

        result = self._command_results.get(command, self._default_result)
        process_id = self._generate_process_id("cmd")

        yield ProcessStartedEvent(process_id=process_id, command=command, pid=12345)

        if result.stdout:
            yield OutputEvent(process_id=process_id, data=result.stdout, stream="stdout")
        if result.stderr:
            yield OutputEvent(process_id=process_id, data=result.stderr, stream="stderr")

        # If there's an error, yield ProcessErrorEvent instead of ProcessCompletedEvent
        if not result.success and result.error:
            yield ProcessErrorEvent(
                process_id=process_id,
                error=result.error,
                error_type=result.error_type,
                exit_code=result.exit_code or 1,
            )
        else:
            yield ProcessCompletedEvent(
                process_id=process_id,
                exit_code=result.exit_code or 0,
                duration=result.duration,
            )
