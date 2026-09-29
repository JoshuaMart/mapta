"""Sandbox backed by a local Docker container.

Every call goes through an :mod:`asyncio` subprocess: a blocking
``subprocess.run`` here would stall the event loop and serialise the scans that
are supposed to run in parallel.

A container is an isolation convenience, not a security boundary.
"""

import asyncio
import logging
import os
import shlex
import uuid

from ...config import SandboxSettings
from ...domain import CommandResult, SandboxError

__all__ = ["DockerSandbox", "DockerSandboxFactory"]

logger = logging.getLogger(__name__)

#: Marks our containers so an orphan can be told from anything else running.
_LABEL = "mapta.sandbox"

#: Slack given to the in-container `timeout` before the outer deadline fires.
_TIMEOUT_GRACE = 10.0

#: Tools the agent's commands routinely reach for.
BASE_APT_PACKAGES = "curl wget git netcat-openbsd dnsutils iputils-ping jq unzip ca-certificates"
BASE_PIP_PACKAGES = "requests httpx beautifulsoup4 pyjwt"


class DockerSandbox:
    """An ephemeral Docker container used as one scan's execution sandbox."""

    def __init__(self, settings: SandboxSettings) -> None:
        self._settings = settings
        self._container_id: str | None = None
        self._has_timeout = False

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Create the container and install the tooling the tools assume."""
        name = f"mapta-sandbox-{uuid.uuid4().hex[:12]}"
        settings = self._settings
        argv = [
            "docker", "run", "--detach",
            "--name", name,
            "--workdir", settings.workdir,
            "--network", settings.network,
            "--memory", settings.memory,
            "--cpus", settings.cpus,
            # A runaway agent should not be able to fork-bomb the host.
            "--pids-limit", settings.pids_limit,
            "--label", f"{_LABEL}=1",
            "--label", f"mapta.pid={os.getpid()}",
            settings.image,
            "sleep", "infinity",
        ]
        try:
            code, stdout, stderr = await _run(argv, timeout=300)
        except FileNotFoundError as exc:
            raise SandboxError(
                "The `docker` CLI was not found. Install Docker, or set "
                "SANDBOX_PROVIDER=none to run without a sandbox."
            ) from exc
        if code != 0:
            raise SandboxError(f"Failed to start the sandbox container: {stderr.strip()}")

        self._container_id = stdout.strip()
        logger.info("Sandbox container started: %s (%s)", name, settings.image)

        # `docker exec` only kills its local client on timeout, leaving the real
        # process to eat the container's pid and memory budget. Wrapping in
        # `timeout` fixes that, where the image provides it.
        self._has_timeout = (await self.run("command -v timeout", timeout=15)).exit_code == 0
        if not self._has_timeout:
            logger.warning(
                "No `timeout` in %s: commands that time out keep running in the container.",
                settings.image,
            )

        if settings.bootstrap:
            await self._bootstrap()

    async def aclose(self) -> None:
        """Remove the container. Safe to call more than once."""
        container_id, self._container_id = self._container_id, None
        if not container_id:
            return
        try:
            code, _, stderr = await _run(
                ["docker", "rm", "--force", "--volumes", container_id], timeout=120
            )
        except TimeoutError:
            code, stderr = 124, "docker rm timed out"
        if code == 0:
            logger.info("Sandbox container removed: %s", container_id[:12])
        else:
            logger.warning(
                "Sandbox container %s was NOT removed (exit %s): %s. Remove it with "
                "`docker rm -f %s`.",
                container_id[:12], code, stderr.strip()[:200], container_id[:12],
            )

    async def __aenter__(self) -> DockerSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # --- the Sandbox port -------------------------------------------------

    async def run(
        self, command: str, *, timeout: float | None = None, user: str = "root"
    ) -> CommandResult:
        """Execute ``command`` with bash inside the container."""
        limit = self._timeout(timeout)
        argv = self._exec_argv(user=user, command=command, kill_after=limit)
        try:
            # The outer deadline is a backstop for the inner one.
            code, stdout, stderr = await _run(argv, timeout=limit + _TIMEOUT_GRACE)
        except TimeoutError:
            return CommandResult(
                stdout="",
                stderr=f"[sandbox] command timed out after {limit}s",
                exit_code=124,
            )
        return CommandResult(stdout=stdout, stderr=stderr, exit_code=code)

    def _timeout(self, requested: float | None) -> float:
        """Clamp a caller-supplied timeout into something workable.

        An agent can ask for zero, a negative number or an hour; none of those
        should be handed to ``asyncio.timeout`` as-is.
        """
        settings = self._settings
        if not isinstance(requested, int | float) or isinstance(requested, bool):
            return settings.default_timeout
        if requested <= 0:
            return settings.default_timeout
        return min(float(requested), settings.max_timeout)

    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path`` inside the container."""
        directory = "/".join(path.rsplit("/", 1)[:-1]) or "."
        # Stream the payload over stdin so no quoting or length limit applies.
        argv = self._exec_argv(
            user="root",
            command=f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}",
            interactive=True,
        )
        code, _, stderr = await _run(
            argv, timeout=self._settings.default_timeout, stdin=content.encode()
        )
        if code != 0:
            raise SandboxError(f"Failed to write {path}: {stderr.strip()}")

    async def read_file(self, path: str) -> str:
        """Read ``path`` from the container."""
        result = await self.run(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            raise SandboxError(f"Failed to read {path}: {result.stderr.strip()}")
        return result.stdout

    # --- helpers ----------------------------------------------------------

    async def _bootstrap(self) -> None:
        settings = self._settings
        apt = f"{BASE_APT_PACKAGES} {settings.apt_packages}".strip()
        pip = f"{BASE_PIP_PACKAGES} {settings.pip_packages}".strip()
        workdir = shlex.quote(settings.workdir)

        steps = (
            (
                "system packages",
                "command -v apt-get >/dev/null && "
                "(DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                f"--no-install-recommends {apt}) "
                "|| echo '[sandbox] apt-get unavailable, skipping system packages'",
                900,
            ),
            # `sandbox_run_python` sources `.venv/bin/activate` from the working
            # directory, so the venv has to live there.
            (
                "python venv",
                f"cd {workdir} && (python3 -m venv .venv || python -m venv .venv) && "
                ".venv/bin/pip install -q --upgrade pip",
                600,
            ),
            ("python packages", f"cd {workdir} && .venv/bin/pip install -q {pip}", 900),
        )

        for label, command, step_timeout in steps:
            result = await self.run(command, timeout=step_timeout)
            if result.exit_code != 0:
                logger.warning(
                    "Sandbox bootstrap step %r failed (exit %s): %s",
                    label,
                    result.exit_code,
                    result.stderr.strip()[:500],
                )
            else:
                logger.info("Sandbox bootstrap step %r done", label)

    def _exec_argv(
        self,
        *,
        user: str,
        command: str,
        interactive: bool = False,
        kill_after: float | None = None,
    ) -> list[str]:
        if not self._container_id:
            raise SandboxError("Sandbox container is not running (it was already closed).")
        argv = ["docker", "exec"]
        if interactive:
            argv.append("--interactive")
        argv += ["--user", user, "--workdir", self._settings.workdir, self._container_id]
        if kill_after is not None and self._has_timeout:
            argv += ["timeout", "-k", "5", str(int(kill_after))]
        argv += ["bash", "-lc", command]
        return argv


async def _run(
    argv: list[str], *, timeout: float, stdin: bytes | None = None
) -> tuple[int, str, str]:
    """Run a process and capture its output, killing it on timeout."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout):
            out, err = await process.communicate(stdin)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    return (
        process.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


async def reap_orphaned_sandboxes() -> int:
    """Remove sandboxes whose creating process is gone, and report how many.

    ``aclose`` runs in a ``finally`` block, which SIGKILL bypasses -- a killed
    or OOM-reaped scan therefore leaves its container holding memory, CPU and
    pid reservations until somebody notices. Containers of live processes, our
    own included, are left alone, so concurrent scans are safe.
    """
    try:
        code, out, _ = await _run(
            # No -q here: it forces id-only output and silently drops --format.
            ["docker", "ps", "-a", "--filter", f"label={_LABEL}=1",
             "--format", '{{.ID}} {{.Label "mapta.pid"}}'],
            timeout=30,
        )
    except (TimeoutError, FileNotFoundError, OSError):
        return 0
    if code != 0:
        return 0

    reaped = 0
    for line in out.splitlines():
        container_id, _, raw_pid = line.partition(" ")
        if not container_id:
            continue
        if _process_alive(raw_pid.strip()):
            continue
        await _run(["docker", "rm", "--force", "--volumes", container_id], timeout=60)
        logger.warning(
            "Removed an orphaned sandbox container (%s) left by a killed run.",
            container_id[:12],
        )
        reaped += 1
    return reaped


def _process_alive(raw_pid: str) -> bool:
    """True when the labelled pid still exists; unknown pids count as alive."""
    if not raw_pid.isdigit():
        return True  # no label to judge by, so leave it be
    try:
        os.kill(int(raw_pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class DockerSandboxFactory:
    """Creates one started :class:`DockerSandbox` per scan."""

    def __init__(self, settings: SandboxSettings) -> None:
        self._settings = settings
        self._swept = False

    async def create(self) -> DockerSandbox:
        if not self._swept:
            # Once per run, before the first container of this process.
            self._swept = True
            await reap_orphaned_sandboxes()
        sandbox = DockerSandbox(self._settings)
        try:
            await sandbox.start()
        except BaseException:
            # Bootstrap can run for minutes; if it fails or the scan is
            # cancelled, nobody else holds the container to remove it.
            await sandbox.aclose()
            raise
        return sandbox
