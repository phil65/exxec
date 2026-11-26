"""`stream_command()` implemented on top of `spawn()`.

Environments with `supports_spawn` can delegate `stream_command()` here to get real
streaming with uniform semantics: one output event per line, a total time limit
that kills the process (group), and termination when the stream is closed.
"""

from __future__ import annotations

import asyncio
import codecs
from typing import TYPE_CHECKING

from exxec.events import OutputEvent, ProcessCompletedEvent, ProcessErrorEvent, ProcessStartedEvent


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable

    from exxec.base import ExecutionEnvironment
    from exxec.events import ExecutionEvent


async def stream_command_via_spawn(
    env: ExecutionEnvironment,
    command: str,
    *,
    timeout: float | None,
    process_id: str,
    display_command: str | None = None,
) -> AsyncIterator[ExecutionEvent]:
    """Run a shell command via `env.spawn()` and stream its output line by line.

    Args:
        env: Environment with `supports_spawn`
        command: Shell command line
        timeout: Total time limit in seconds (defaults to the environment's
            `default_command_timeout`; no limit if both are None)
        process_id: ID used in the emitted events
        display_command: Command shown in the started event (defaults to `command`)

    Yields:
        Started, output (one per line) and completed/error events
    """
    effective_timeout = timeout if timeout is not None else env.default_command_timeout
    yield ProcessStartedEvent(process_id=process_id, command=display_command or command)
    try:
        handle = await env.spawn(command)
    except Exception as e:  # noqa: BLE001
        yield ProcessErrorEvent.failed(e, process_id=process_id)
        return
    loop = asyncio.get_running_loop()
    deadline = None if effective_timeout is None else loop.time() + effective_timeout

    async def before_deadline[T](awaitable: Awaitable[T]) -> T:
        # Bounded per await (never across a yield), so a timeout cannot cancel the
        # consumer's code while this generator is suspended.
        if deadline is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=max(deadline - loop.time(), 0))

    async with handle:
        lines = _iter_lines(handle.output())
        try:
            while (line := await before_deadline(anext(lines, None))) is not None:
                yield OutputEvent(process_id=process_id, data=line, stream="combined")
            exit_code = await before_deadline(handle.wait())
        except TimeoutError:
            await handle.kill(grace=1.0)
            yield ProcessErrorEvent(
                process_id=process_id,
                error=f"Command timed out after {effective_timeout} seconds",
                error_type="TimeoutError",
                exit_code=1,
            )
            return
        except Exception as e:  # noqa: BLE001
            yield ProcessErrorEvent.failed(e, process_id=process_id)
            return
        finally:
            await lines.aclose()
    if exit_code == 0:
        yield ProcessCompletedEvent(process_id=process_id, exit_code=exit_code)
    else:
        yield ProcessErrorEvent(
            process_id=process_id,
            error=f"Command exited with code {exit_code}",
            error_type="CommandError",
            exit_code=exit_code,
        )


async def _iter_lines(chunks: AsyncIterator[bytes]) -> AsyncGenerator[str]:
    """Split a byte stream into decoded lines (without line terminators)."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    # Parts of the current line; only new text is searched, so a long line is linear.
    pending: list[str] = []
    async for chunk in chunks:
        text = decoder.decode(chunk)
        if "\n" not in text:
            if text:
                pending.append(text)
            continue
        first, *rest = text.split("\n")
        pending.append(first)
        yield "".join(pending).rstrip("\r")
        *complete, last = rest
        for line in complete:
            yield line.rstrip("\r")
        pending = [last] if last else []
    tail = "".join(pending) + decoder.decode(b"", final=True)
    if tail:
        yield tail.rstrip("\r")
