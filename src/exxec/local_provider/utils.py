"""Local execution environment that runs code locally."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import shutil
import sys
import threading
from typing import TYPE_CHECKING, TextIO


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from exxec.models import Language


PYTHON_EXECUTABLES = [
    "python3",
    "python",
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.14",
]


class StreamCapture(io.StringIO):
    """Capture and forward output to a queue.

    Writes from other threads (sync code run via `asyncio.to_thread`) are handed to
    the event loop the capture was created on: asyncio queues are not thread-safe.
    The redirect this is used with (`contextlib.redirect_stdout`) is process-global,
    so output of other tasks and threads during a run is captured too.
    """

    def __init__(self, original_stream: TextIO, queue: asyncio.Queue[str]) -> None:
        super().__init__()
        self.original_stream = original_stream
        self.queue = queue
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._loop_thread = threading.get_ident()

    def _put(self, line: str) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(line)

    def write[S: (str, bytes)](self, s: S) -> int:
        """Capture and forward output to a queue."""
        text = s if isinstance(s, str) else s.decode()
        result = self.original_stream.write(text)
        if text:
            lines = text.splitlines(keepends=True)
            for line in lines:
                if not line.strip():
                    continue
                if self._loop is None or threading.get_ident() == self._loop_thread:
                    self._put(line.rstrip("\n\r"))
                else:
                    with contextlib.suppress(RuntimeError):  # loop already closed
                        self._loop.call_soon_threadsafe(self._put, line.rstrip("\n\r"))
        return result

    def flush(self) -> None:
        """Flush the stream."""
        return self.original_stream.flush()


async def execute_stream_local(code: str, timeout: float) -> AsyncIterator[str]:
    """Execute code in same process and stream output line by line."""
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
                    exec(code, namespace)

                    if "main" in namespace and callable(namespace["main"]):
                        main_func = namespace["main"]
                        if inspect.iscoroutinefunction(main_func):
                            result = await asyncio.wait_for(main_func(), timeout=timeout)
                        else:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(main_func), timeout=timeout
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

        while True:
            try:
                line = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                if line == "__EXECUTION_COMPLETE__":
                    break
                yield line
            except TimeoutError:
                if execution_done and output_queue.empty():
                    break
                continue
            except Exception as e:  # noqa: BLE001
                yield f"ERROR: {e}"
                break

        try:
            await execute_task
        except Exception as e:  # noqa: BLE001
            yield f"ERROR: {e}"

    except Exception as e:  # noqa: BLE001
        yield f"ERROR: {e}"


def find_executable(language: Language) -> str:
    """Find the best available executable for the given language."""
    match language:
        case "python":
            for candidate in PYTHON_EXECUTABLES:
                if shutil.which(candidate):
                    return candidate
            error_msg = "No Python executable found"
            raise RuntimeError(error_msg)

        case "javascript":
            candidates = ["node", "nodejs"]
            for candidate in candidates:
                if shutil.which(candidate):
                    return candidate
            error_msg = "No Node.js executable found"
            raise RuntimeError(error_msg)

        case "typescript":
            node_candidates = ["node", "nodejs"]
            node_exe = None
            for candidate in node_candidates:
                if shutil.which(candidate):
                    node_exe = candidate
                    break

            if not node_exe:
                error_msg = "No Node.js executable found (required for TypeScript)"
                raise RuntimeError(error_msg)

            # Check for TypeScript runners
            ts_runners = ["ts-node", "tsx"]
            for runner in ts_runners:
                if shutil.which(runner):
                    return node_exe

            return node_exe

        case _:
            candidates = ["python3", "python"]
            for candidate in candidates:
                if shutil.which(candidate):
                    return candidate
            error_msg = f"No suitable executable found for language: {language}"
            raise RuntimeError(error_msg)
