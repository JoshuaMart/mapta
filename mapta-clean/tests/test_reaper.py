"""The Docker sandbox reaper: a SIGKILLed scan must not leak its container."""

import os

import pytest

from mapta.config import SandboxSettings
from mapta.infrastructure.sandbox import docker as docker_mod


@pytest.fixture
def fake_docker(monkeypatch):
    """Record docker invocations and replay a canned `docker ps`."""
    calls: list[list[str]] = []
    listing = {"value": ""}

    async def fake_run(argv, *, timeout, stdin=None):
        calls.append(argv)
        if argv[:3] == ["docker", "ps", "-a"]:
            # Real `docker ps` ignores --format when -q is present, so the
            # fake refuses to answer a query that would come back id-only.
            assert "-q" not in argv and "-aq" not in argv, "-q drops --format"
            return 0, listing["value"], ""
        return 0, "", ""

    monkeypatch.setattr(docker_mod, "_run", fake_run)
    return calls, listing


async def test_a_container_of_a_dead_process_is_removed(fake_docker):
    calls, listing = fake_docker
    dead_pid = 999999  # no such process
    listing["value"] = f"abc123def456 {dead_pid}\n"

    assert await docker_mod.reap_orphaned_sandboxes() == 1
    assert ["docker", "rm", "--force", "--volumes", "abc123def456"] in calls


async def test_a_container_of_a_live_process_is_left_alone(fake_docker):
    calls, listing = fake_docker
    listing["value"] = f"abc123def456 {os.getpid()}\n"

    assert await docker_mod.reap_orphaned_sandboxes() == 0
    assert not any(c[:2] == ["docker", "rm"] for c in calls)


async def test_an_unlabelled_container_is_left_alone(fake_docker):
    calls, listing = fake_docker
    listing["value"] = "abc123def456 \n"

    assert await docker_mod.reap_orphaned_sandboxes() == 0
    assert not any(c[:2] == ["docker", "rm"] for c in calls)


async def test_a_missing_docker_cli_is_not_an_error(monkeypatch):
    async def boom(argv, *, timeout, stdin=None):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(docker_mod, "_run", boom)
    assert await docker_mod.reap_orphaned_sandboxes() == 0


async def test_containers_are_labelled_with_the_creating_process(monkeypatch):
    seen: list[list[str]] = []

    async def fake_run(argv, *, timeout, stdin=None):
        seen.append(argv)
        return 0, "container-id", ""

    monkeypatch.setattr(docker_mod, "_run", fake_run)
    sandbox = docker_mod.DockerSandbox(SandboxSettings(bootstrap=False))
    await sandbox.start()

    run_argv = next(a for a in seen if a[:2] == ["docker", "run"])
    assert "mapta.sandbox=1" in run_argv
    assert f"mapta.pid={os.getpid()}" in run_argv


async def test_the_factory_sweeps_once_per_process(monkeypatch):
    sweeps = []

    async def fake_reap() -> int:
        sweeps.append(1)
        return 0

    async def fake_start(self) -> None:
        return None

    monkeypatch.setattr(docker_mod, "reap_orphaned_sandboxes", fake_reap)
    monkeypatch.setattr(docker_mod.DockerSandbox, "start", fake_start)

    factory = docker_mod.DockerSandboxFactory(SandboxSettings())
    await factory.create()
    await factory.create()
    assert len(sweeps) == 1
