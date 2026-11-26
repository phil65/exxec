"""Base execution environment interface."""

from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING, ClassVar, Literal, Self


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType
    from typing import Any

    from anyenv.process_manager import ProcessManagerProtocol
    from fsspec.asyn import AsyncFileSystem  # type: ignore[import-untyped]

    from exxec.events import ExecutionEvent
    from exxec.models import ExecutionResult, ServerInfo
    from exxec.process_handle import ProcessHandle
    from exxec.pty_manager import PtyManagerProtocol, PtySize
    from exxec.spawn_pty_manager import SpawnPtyManager


OSType = Literal["Windows", "Darwin", "Linux"]


class ExecutionEnvironment(ABC):
    """Abstract base class for code execution environments."""

    supports_spawn: ClassVar[bool] = False
    """Whether `spawn()` is implemented for this environment."""

    supports_pty: ClassVar[bool] = False
    """Whether `spawn(pty=...)` can attach processes to a pseudo-terminal."""

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
        default_command_timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize execution environment with optional lifespan handler.

        Args:
            lifespan_handler: Optional async context manager for tool server
            dependencies: Optional list of dependencies to install
            cwd: Working directory for the environment (None means use default/auto)
            env_vars: Environment variables to set for all executions
            inherit_env: If True, inherit environment variables from os.environ
            default_command_timeout: Default timeout for command execution in seconds.
                If None, commands run without timeout unless explicitly specified.
                This is separate from sandbox/instance lifetime timeouts.
            **kwargs: Additional keyword arguments for specific providers
        """
        self.lifespan_handler = lifespan_handler
        self.server_info: ServerInfo | None = None
        self.dependencies = dependencies or []
        self.cwd = cwd
        self.env_vars = env_vars or {}
        self.inherit_env = inherit_env
        self.default_command_timeout = default_command_timeout
        self._process_manager: ProcessManagerProtocol | None = None
        self._spawn_pty_manager: SpawnPtyManager | None = None
        self._os_type: OSType | None = None

    def get_env(self) -> dict[str, str] | None:
        """Get environment variables, optionally merged with os.environ.

        Returns:
            Merged environment dict if inherit_env=True or env_vars set,
            None otherwise.
        """
        if not self.env_vars and not self.inherit_env:
            return None
        if self.inherit_env:
            return {**os.environ, **(self.env_vars or {})}
        return self.env_vars or None

    async def __aenter__(self) -> Self:
        """Setup environment (start server, spawn process, etc.)."""
        # Start tool server if provided
        if self.lifespan_handler:
            self.server_info = await self.lifespan_handler.__aenter__()
        # Detect OS type if not already set by subclass
        if self._os_type is None:
            try:
                self._os_type = await self._detect_os_type()
            except BaseException as e:
                if self.lifespan_handler:
                    await self.lifespan_handler.__aexit__(type(e), e, e.__traceback__)
                raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup (stop server, kill process, etc.)."""
        from exxec.process_manager import BaseProcessManager

        try:
            if isinstance(self._process_manager, BaseProcessManager):
                await self._process_manager.cleanup()
            if self._spawn_pty_manager is not None:
                await self._spawn_pty_manager.cleanup()
        finally:
            # Cleanup server if provided, even if the cleanup above failed
            if self.lifespan_handler:
                await self.lifespan_handler.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def process_manager(self) -> ProcessManagerProtocol:
        """Get the process manager for this execution environment.

        Environments with `supports_spawn` get a `SpawnProcessManager`; others fall
        back to `StreamingProcessManager`, which runs processes through
        `stream_command()`. Both run processes inside the environment.
        """
        if self._process_manager is None:
            from exxec.process_manager import SpawnProcessManager, StreamingProcessManager

            self._process_manager = (
                SpawnProcessManager(self) if self.supports_spawn else StreamingProcessManager(self)
            )
        return self._process_manager

    @property
    def os_type(self) -> OSType:
        """Get the OS type of the execution environment.

        Returns:
            "Windows", "Darwin", or "Linux"

        Raises:
            RuntimeError: If accessed before entering the async context manager.
        """
        if self._os_type is None:
            msg = "OS type not detected. Use 'async with' context manager first."
            raise RuntimeError(msg)
        return self._os_type

    async def _detect_os_type(self) -> OSType:
        """Detect OS type by running commands in the environment.

        Providers can override this if they know the OS statically.

        Returns:
            "Windows", "Darwin", or "Linux"
        """
        # Try uname first (works on Linux/macOS)
        result = await self.execute_command("uname -s")
        if result.exit_code == 0 and result.stdout:
            uname = result.stdout.strip()
            if uname == "Darwin":
                return "Darwin"
            if uname == "Linux":
                return "Linux"

        # Check for Windows
        result = await self.execute_command("ver")
        if result.exit_code == 0 and result.stdout and "Windows" in result.stdout:
            return "Windows"

        # Default to Linux for Unix-like systems
        return "Linux"

    @abstractmethod
    async def execute(self, code: str) -> ExecutionResult:
        """Execute code and return result with metadata."""
        ...

    def get_fs(self) -> AsyncFileSystem:
        """Return an async fsspec filesystem for the environment.

        Raises:
            NotImplementedError: If the provider has no filesystem access.
        """
        msg = "VFS is not supported"
        raise NotImplementedError(msg)

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        pty: PtySize | None = None,
    ) -> ProcessHandle:
        """Start a process inside the environment and return a handle to it.

        This is the low-level primitive process managers and terminals build on.
        Unlike `stream_command()`, no timeout is applied; the caller owns the
        process lifetime (use the handle as an async context manager).

        Args:
            command: Program to execute when `args` is given, otherwise a shell
                command line (pipes, redirects, etc. are supported)
            args: Program arguments; passed without shell interpretation
            cwd: Working directory (defaults to the environment's working directory)
            env: Additional environment variables, merged over the environment's own
            pty: Attach the process to a pseudo-terminal of this size. Requires
                `supports_pty`. Output then contains raw terminal escape sequences.

        Returns:
            Handle to the running process

        Raises:
            NotImplementedError: If the environment does not support spawning
                (see `supports_spawn`), or a PTY was requested without `supports_pty`
        """
        msg = f"{type(self).__name__} does not support spawn()"
        raise NotImplementedError(msg)

    def _check_spawn_pty(self, pty: PtySize | None) -> None:
        """Raise if a PTY was requested but the environment cannot provide one."""
        if pty is not None and not self.supports_pty:
            msg = f"{type(self).__name__} does not support PTY processes"
            raise NotImplementedError(msg)

    def get_pty_manager(self) -> PtyManagerProtocol:
        """Return a PTY manager for interactive terminal sessions.

        Environments with `supports_pty` get a `SpawnPtyManager`. Providers with a
        native PTY API that do not implement `spawn()` yet override this.

        Returns:
            A PtyManagerProtocol implementation for this environment.

        Raises:
            NotImplementedError: If PTY is not supported by this provider.
        """
        if not self.supports_pty:
            msg = f"{type(self).__name__} does not support PTY sessions"
            raise NotImplementedError(msg)
        if self._spawn_pty_manager is None:
            from exxec.spawn_pty_manager import SpawnPtyManager

            self._spawn_pty_manager = SpawnPtyManager(self)
        return self._spawn_pty_manager

    async def set_file_content(self, path: str, content: str | bytes) -> None:
        """Set file content in the filesystem.

        Args:
            path: Path in the filesystem
            content: File content (str or bytes)
        """
        if isinstance(content, str):
            content = content.encode()
        await self.get_fs()._pipe_file(path, content)

    async def get_file_content(self, path: str) -> bytes:
        """Get file content from the filesystem.

        Args:
            path: Path in the filesystem

        Returns:
            File content as bytes
        """
        return await self.get_fs()._cat_file(path)  # type: ignore[no-any-return]

    @abstractmethod
    def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events (required).

        Args:
            code: Code to execute

        Yields:
            ExecutionEvent objects as they occur
        """
        ...

    @abstractmethod
    def stream_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Execute a terminal command and stream events (required).

        Args:
            command: Terminal command to execute
            timeout: Command timeout in seconds. If None, uses default_command_timeout.
                If both are None, command runs without timeout.

        Yields:
            ExecutionEvent objects as they occur
        """
        ...

    async def execute_stream(self, code: str) -> AsyncIterator[str]:
        """Execute code and stream output line by line.

        Default implementation delegates to stream_code() and filters OutputEvents.

        Args:
            code: Code to execute

        Yields:
            Lines of output as they are produced
        """
        from exxec.events import OutputEvent

        async for event in self.stream_code(code):
            if isinstance(event, OutputEvent):
                yield event.data

    @abstractmethod
    async def execute_command(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute a terminal command and return result with metadata.

        Args:
            command: Terminal command to execute
            timeout: Command timeout in seconds. If None, uses default_command_timeout.
                If both are None, command runs without timeout.

        Returns:
            ExecutionResult with command output and metadata
        """
        ...

    async def execute_command_stream(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """Execute a terminal command and stream output line by line.

        Default implementation delegates to stream_command() and filters OutputEvents.

        Args:
            command: Terminal command to execute
            timeout: Command timeout in seconds. If None, uses default_command_timeout.
                If both are None, command runs without timeout.

        Yields:
            Lines of output as they are produced
        """
        from exxec.events import OutputEvent

        async for event in self.stream_command(command, timeout=timeout):
            if isinstance(event, OutputEvent):
                yield event.data

    @classmethod
    async def execute_script(cls, script_content: str, **kwargs: Any) -> ExecutionResult:
        """Execute a PEP 723 script with automatic dependency management.

        Creates a new execution environment configured for the script's dependencies.

        Args:
            script_content: Python source code with PEP 723 metadata
            **kwargs: Additional keyword arguments for the execution environment

        Returns:
            ExecutionResult with script output and metadata

        Raises:
            ScriptError: If the script metadata is invalid or malformed
        """
        from exxec.pep723 import parse_script_metadata

        metadata = parse_script_metadata(script_content)
        async with cls(dependencies=metadata.dependencies, **kwargs) as env:
            return await env.execute(script_content)

    @classmethod
    async def execute_script_stream(cls, script_content: str, **kwargs: Any) -> AsyncIterator[str]:
        """Execute a PEP 723 script and stream output with dependency management.

        Creates a new execution environment configured for the script's dependencies.

        Args:
            script_content: Python source code with PEP 723 metadata
            **kwargs: Additional keyword arguments for the execution environment

        Yields:
            Lines of output as they are produced

        Raises:
            ScriptError: If the script metadata is invalid or malformed
        """
        from exxec.pep723 import parse_script_metadata

        metadata = parse_script_metadata(script_content)
        async with cls(dependencies=metadata.dependencies, **kwargs) as env:
            async for line in env.execute_stream(script_content):
                yield line
