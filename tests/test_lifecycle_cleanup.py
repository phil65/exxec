"""Setup/teardown failures still release what was acquired."""

from __future__ import annotations

import contextlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import pytest

from exxec import LocalExecutionEnvironment
from exxec.srt_provider import SRTExecutionEnvironment


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _lifespan(events: list[str]) -> Any:
    @contextlib.asynccontextmanager
    async def lifespan() -> AsyncIterator[Any]:
        events.append("enter")
        try:
            yield None
        finally:
            events.append("exit")

    return lifespan()


class FailingDetection(LocalExecutionEnvironment):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._os_type = None

    async def _detect_os_type(self) -> Any:
        raise RuntimeError("detection failed")


async def test_failed_os_detection_exits_the_lifespan():
    events: list[str] = []
    with pytest.raises(RuntimeError, match="detection failed"):
        async with FailingDetection(lifespan_handler=_lifespan(events)):
            pass
    assert events == ["enter", "exit"]


async def test_failed_cleanup_still_exits_the_lifespan():
    class Broken:
        async def cleanup(self) -> None:
            raise RuntimeError("cleanup failed")

    events: list[str] = []
    env = LocalExecutionEnvironment(lifespan_handler=_lifespan(events))
    with pytest.raises(RuntimeError, match="cleanup failed"):
        async with env:
            env._spawn_pty_manager = Broken()  # type: ignore[assignment]
    assert events == ["enter", "exit"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/fd")
def test_srt_settings_file_does_not_leak_an_fd():
    before = len(list(Path("/proc/self/fd").iterdir()))
    envs = [SRTExecutionEnvironment() for _ in range(5)]
    after = len(list(Path("/proc/self/fd").iterdir()))
    for env in envs:
        env._cleanup_settings_file()
    assert after == before
