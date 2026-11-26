"""Docker environment setup and teardown with a fake container (no daemon needed)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest


pytest.importorskip("testcontainers")

from exxec import DockerExecutionEnvironment
from exxec.events import ProcessErrorEvent


class FakeContainer:
    """Stands in for testcontainers' DockerContainer; `exec` answers from `results`."""

    instances: ClassVar[list[FakeContainer]] = []
    results: ClassVar[list[Any]] = []

    def __init__(self, image: str) -> None:
        self.image = image
        self.started = False
        self.stopped = False
        self.execs: list[list[str]] = []
        FakeContainer.instances.append(self)

    def with_volume_mapping(self, *_args: Any) -> Self:
        return self

    def with_env(self, *_args: Any) -> Self:
        return self

    def with_kwargs(self, **_kwargs: Any) -> Self:
        return self

    def with_command(self, *_args: Any) -> Self:
        return self

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def exec(self, argv: list[str]) -> Any:
        self.execs.append(argv)
        result = FakeContainer.results.pop(0) if FakeContainer.results else 0
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(exit_code=result, output=b"")


@pytest.fixture
def fake_container(monkeypatch: pytest.MonkeyPatch) -> type[FakeContainer]:
    import testcontainers.core.container

    FakeContainer.instances = []
    FakeContainer.results = []
    monkeypatch.setattr(testcontainers.core.container, "DockerContainer", FakeContainer)
    return FakeContainer


async def test_failed_setup_stops_the_container_and_removes_the_workdir(fake_container):
    fake_container.results = [RuntimeError("install broke")]
    env = DockerExecutionEnvironment(dependencies=["requests"])
    with pytest.raises(RuntimeError, match="install broke"):
        await env.__aenter__()
    assert fake_container.instances[0].stopped
    assert env.host_workdir is None or not Path(env.host_workdir).exists()


async def test_each_install_runs_even_if_an_earlier_one_fails(fake_container):
    fake_container.results = [1, 0]
    env = DockerExecutionEnvironment(dependencies=["requests"])
    env.server_info = SimpleNamespace(url="http://localhost:1")  # type: ignore[assignment]
    async with env:
        execs = fake_container.instances[0].execs
    installs = [argv for argv in execs if "install" in " ".join(argv)]
    assert len(installs) == 2  # noqa: PLR2004
    assert "requests" in " ".join(installs[1])


async def test_unwritable_script_is_a_failed_result_not_an_exception(fake_container, tmp_path):
    async with DockerExecutionEnvironment() as env:
        env.host_workdir = str(tmp_path / "missing")
        result = await env.execute("print(1)")
        assert not result.success
        events = [event async for event in env.stream_code("print(1)")]
        env.host_workdir = None  # nothing for __aexit__ to remove
    assert isinstance(events[-1], ProcessErrorEvent)
