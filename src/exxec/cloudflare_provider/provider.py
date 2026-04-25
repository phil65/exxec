"""Cloudflare sandbox execution environment using Cloudflare Workers HTTP API."""

from __future__ import annotations

import contextlib
import re
import shlex
import time
from typing import TYPE_CHECKING, Any, Self
import uuid

import anyenv
import httpx

from exxec.base import ExecutionEnvironment
from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent
from exxec.exceptions import NotInitializedError
from exxec.models import ExecutionResult
from exxec.parse_output import get_script_path, parse_output, wrap_code


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from upathtools.filesystems import CloudflareFS

    from exxec.events import ExecutionEvent
    from exxec.models import Language, ServerInfo


_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HTTP_ERROR_THRESHOLD = 400
_HTTP_NOT_FOUND = 404


def _apply_env_vars_to_command(
    command: str,
    env_vars: dict[str, str] | None,
) -> str:
    """Prepend environment variable exports to a command string."""
    if not env_vars:
        return command
    exports = " && ".join(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in env_vars.items()
        if _ENV_VAR_NAME_RE.match(key)
    )
    return f"{exports} && {command}" if exports else command


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


class CloudflareExecutionEnvironment(ExecutionEnvironment):
    """Executes code in a Cloudflare Worker sandbox via HTTP API.

    Requires a deployed Cloudflare Sandbox Worker that exposes the
    standard sandbox HTTP API (session/create, execute, file/write, etc.).
    """

    def __init__(
        self,
        lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None,
        dependencies: list[str] | None = None,
        base_url: str = "",
        api_token: str | None = None,
        account_id: str | None = None,
        session_id: str | None = None,
        timeout: float = 30.0,
        default_command_timeout: float | None = None,
        language: Language = "python",
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """Initialize Cloudflare sandbox environment.

        Args:
            lifespan_handler: Async context manager for tool server (optional)
            dependencies: List of packages to install via pip / npm
            base_url: Base URL of the Cloudflare Sandbox Worker deployment
            api_token: API token for authentication (optional)
            account_id: Cloudflare account ID (optional)
            session_id: Explicit session ID (auto-generated if None)
            timeout: HTTP request timeout in seconds
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
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.account_id = account_id
        self.session_id = session_id or f"exxec-{uuid.uuid4().hex[:12]}"
        self.timeout = timeout
        self.language: Language = language
        self._client: httpx.AsyncClient | None = None
        # Cloudflare sandboxes run Linux
        self._os_type = "Linux"

    def _get_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if self.account_id:
            headers["CF-Account-ID"] = self.account_id
        return headers

    def get_fs(self) -> CloudflareFS:
        """Return a CloudflareFS instance for the sandbox."""
        from upathtools.filesystems import CloudflareFS

        return CloudflareFS(
            base_url=self.base_url,
            session_id=self.session_id,
            api_token=self.api_token,
            account_id=self.account_id,
            working_dir=self.cwd or "/workspace",
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            raise NotInitializedError("CloudFlare")
        return self._client

    async def _post(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a POST request to the sandbox API."""
        client = self._ensure_client()
        url = f"{self.base_url}{path}"
        response = await client.post(url, json=payload, headers=self._get_headers())
        if response.status_code >= _HTTP_ERROR_THRESHOLD:
            msg = f"Cloudflare API error ({response.status_code}): {response.text}"
            raise RuntimeError(msg)
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()  # type: ignore[no-any-return]
        return {}

    async def _get(self, path: str) -> dict[str, Any]:
        """Make a GET request to the sandbox API."""
        client = self._ensure_client()
        url = f"{self.base_url}{path}"
        response = await client.get(url, headers=self._get_headers())
        if response.status_code >= _HTTP_ERROR_THRESHOLD:
            msg = f"Cloudflare API error ({response.status_code}): {response.text}"
            raise RuntimeError(msg)
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()  # type: ignore[no-any-return]
        return {}

    async def __aenter__(self) -> Self:
        """Setup Cloudflare sandbox session."""
        await super().__aenter__()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))

        # Create the session
        payload: dict[str, Any] = {
            "id": self.session_id,
            "env": self.env_vars or {},
            "cwd": self.cwd or "/workspace",
            "isolation": True,
        }
        await self._post("/api/session/create", payload=payload)

        # Install dependencies
        if self.dependencies:
            deps_str = " ".join(self.dependencies)
            match self.language:
                case "python":
                    await self._execute_raw(f"pip install {deps_str}")
                case "javascript" | "typescript":
                    await self._execute_raw(f"npm install {deps_str}")

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup sandbox session."""
        if self._client:
            with contextlib.suppress(Exception):
                # Kill all processes in the session
                url = f"{self.base_url}/api/process/kill-all"
                await self._client.delete(
                    url,
                    params={"session": self.session_id},
                    headers=self._get_headers(),
                )
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _execute_raw(self, command: str) -> dict[str, Any]:
        """Execute a raw command in the sandbox, returning the API response."""
        env = self.get_env()
        command_to_run = _apply_env_vars_to_command(command, env)
        return await self._post(
            "/api/execute",
            payload={"id": self.session_id, "command": command_to_run},
        )

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code in the Cloudflare sandbox."""
        start_time = time.time()
        try:
            wrapped_code = wrap_code(code, language=self.language)
            script_path = get_script_path(self.language)

            # Write the script file
            await self._post(
                "/api/file/write",
                payload={
                    "id": self.session_id,
                    "path": script_path,
                    "content": wrapped_code,
                },
            )

            # Execute the script
            command = _get_execution_command(self.language, script_path)
            data = await self._execute_raw(command)

            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            exit_code = data.get("exitCode", data.get("exit_code", 0))

            execution_result, error_info = parse_output(stdout)
            if exit_code == 0 and error_info is None:
                return ExecutionResult(
                    result=execution_result,
                    duration=time.time() - start_time,
                    success=True,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                )

            return ExecutionResult(
                result=None,
                duration=time.time() - start_time,
                success=False,
                exit_code=exit_code,
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
        """Execute a terminal command in the Cloudflare sandbox."""
        start_time = time.time()
        try:
            data = await self._execute_raw(command)
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            exit_code = data.get("exitCode", data.get("exit_code", 0))
            success = exit_code == 0
            return ExecutionResult(
                result=stdout if success else None,
                duration=time.time() - start_time,
                success=success,
                error=stderr if not success else None,
                error_type="CommandError" if not success else None,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        except Exception as e:  # noqa: BLE001
            return ExecutionResult.failed(e, start_time)

    async def stream_code(self, code: str) -> AsyncIterator[ExecutionEvent]:
        """Execute code and stream events in the Cloudflare sandbox."""
        process_id = f"cf_{self.session_id}"
        wrapped_code = wrap_code(code, language=self.language)
        script_path = get_script_path(self.language)

        # Write the script file
        await self._post(
            "/api/file/write",
            payload={
                "id": self.session_id,
                "path": script_path,
                "content": wrapped_code,
            },
        )

        command = _get_execution_command(self.language, script_path)
        yield ProcessStartedEvent(process_id=process_id, command=f"execute({len(code)} chars)")

        try:
            # Try SSE streaming first
            client = self._ensure_client()
            url = f"{self.base_url}/api/execute/stream"
            env = self.get_env()
            command_to_run = _apply_env_vars_to_command(command, env)
            payload = {"id": self.session_id, "command": command_to_run}
            headers = {**self._get_headers(), "Accept": "text/event-stream"}

            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code == _HTTP_NOT_FOUND:
                    # SSE not supported, fall back to regular execution
                    async for event in self._stream_fallback(command, process_id):
                        yield event
                    return

                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        event_data = anyenv.load_json(data, return_type=dict)
                        if "stdout" in event_data:
                            yield OutputEvent(
                                process_id=process_id,
                                data=event_data["stdout"],
                                stream="stdout",
                            )
                        if "stderr" in event_data:
                            yield OutputEvent(
                                process_id=process_id,
                                data=event_data["stderr"],
                                stream="stderr",
                            )
                    except anyenv.JsonLoadError:
                        yield OutputEvent(process_id=process_id, data=data, stream="stdout")

                yield ProcessCompletedEvent(process_id=process_id, exit_code=0)

        except httpx.HTTPError:
            # Fall back to regular execution on HTTP errors
            async for event in self._stream_fallback(command, process_id):
                yield event
        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def _stream_fallback(
        self,
        command: str,
        process_id: str,
    ) -> AsyncIterator[ExecutionEvent]:
        """Fallback streaming by running command and splitting output."""
        try:
            data = await self._execute_raw(command)
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            exit_code = data.get("exitCode", data.get("exit_code", 0))

            if stdout:
                for line in stdout.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stdout")
            if stderr:
                for line in stderr.splitlines():
                    if line:
                        yield OutputEvent(process_id=process_id, data=line, stream="stderr")

            if exit_code != 0:
                yield ProcessErrorEvent(
                    process_id=process_id,
                    error=stderr or f"Command exited with code {exit_code}",
                    error_type="ExecutionError",
                    exit_code=exit_code,
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
        """Execute a terminal command and stream events in the Cloudflare sandbox."""
        process_id = f"cf_cmd_{self.session_id}"
        yield ProcessStartedEvent(process_id=process_id, command=command)

        try:
            # Try SSE streaming first
            client = self._ensure_client()
            url = f"{self.base_url}/api/execute/stream"
            env = self.get_env()
            command_to_run = _apply_env_vars_to_command(command, env)
            payload = {"id": self.session_id, "command": command_to_run}
            headers = {**self._get_headers(), "Accept": "text/event-stream"}

            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code == _HTTP_NOT_FOUND:
                    async for event in self._stream_fallback(command, process_id):
                        yield event
                    return

                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        event_data = anyenv.load_json(data, return_type=dict)
                        if "stdout" in event_data:
                            yield OutputEvent(
                                process_id=process_id,
                                data=event_data["stdout"],
                                stream="stdout",
                            )
                        if "stderr" in event_data:
                            yield OutputEvent(
                                process_id=process_id,
                                data=event_data["stderr"],
                                stream="stderr",
                            )
                    except anyenv.JsonLoadError:
                        yield OutputEvent(
                            process_id=process_id,
                            data=data,
                            stream="stdout",
                        )

                yield ProcessCompletedEvent(process_id=process_id, exit_code=0)

        except httpx.HTTPError:
            async for event in self._stream_fallback(command, process_id):
                yield event
        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)

    async def set_file_content(self, path: str, content: str | bytes) -> None:
        """Set file content in the Cloudflare sandbox."""
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        await self._post(
            "/api/file/write",
            payload={"id": self.session_id, "path": path, "content": text},
        )

    async def get_file_content(self, path: str) -> bytes:
        """Get file content from the Cloudflare sandbox."""
        data = await self._post(
            "/api/file/read",
            payload={"id": self.session_id, "path": path},
        )
        content = data.get("content", "")
        return content.encode("utf-8") if isinstance(content, str) else content


if __name__ == "__main__":

    async def _main() -> None:
        async with CloudflareExecutionEnvironment(
            base_url="https://my-sandbox.workers.dev",
        ) as env:
            result = await env.execute_command("echo hello")
            print(result)

    import asyncio

    asyncio.run(_main())
