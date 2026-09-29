"""Reference sandbox provider for MAPTA, backed by a local Docker container.

`main.py` does not ship a sandbox: it loads one through the ``SANDBOX_FACTORY``
environment variable, in the form ``module:function``. This module provides a
working implementation so the project runs out of the box::

    export SANDBOX_FACTORY="sandbox_docker:create_sandbox"

A sandbox object must expose:

* ``files.write(path, content)``
* ``commands.run(command, timeout=..., user=...)`` -> object with ``stdout``,
  ``stderr`` and ``exit_code``
* ``set_timeout(ms)``  (optional)
* ``kill()``           (optional)

Configuration (all optional):

======================  ==============================================
SANDBOX_IMAGE           Docker image. Default: ``python:3.12-slim``
SANDBOX_WORKDIR         Working directory. Default: ``/home/user``
SANDBOX_NETWORK         Docker network mode. Default: ``bridge``
SANDBOX_MEMORY          Memory limit, e.g. ``2g``. Default: ``2g``
SANDBOX_CPUS            CPU limit, e.g. ``2``. Default: ``2``
SANDBOX_PACKAGES        Extra apt packages, space separated
SANDBOX_PIP_PACKAGES    Extra pip packages installed in the venv
SANDBOX_SKIP_BOOTSTRAP  Set to ``1`` to skip apt/venv setup (for images
                        that already provide them)
======================  ==============================================

The container is started detached and is removed by ``kill()``, which
``main.run_continuously`` calls when a scan finishes.

Note: a Docker container is an isolation convenience, not a security boundary.
Only point MAPTA at targets you are authorised to test.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = os.getenv("SANDBOX_IMAGE", "python:3.12-slim")
DEFAULT_WORKDIR = os.getenv("SANDBOX_WORKDIR", "/home/user")
# Tools the agent's commands routinely reach for.
BASE_PACKAGES = "curl wget git netcat-openbsd dnsutils iputils-ping jq unzip ca-certificates"
BASE_PIP_PACKAGES = "requests httpx beautifulsoup4 pyjwt"


@dataclass
class CommandResult:
    """Result of a command executed in the sandbox."""

    stdout: str
    stderr: str
    exit_code: int


class _Files:
    """The ``sandbox.files`` namespace."""

    def __init__(self, sandbox: "DockerSandbox") -> None:
        self._sandbox = sandbox

    def write(self, path: str, content: str) -> None:
        """Write ``content`` to ``path`` inside the container."""
        self._sandbox._ensure_running()
        directory = os.path.dirname(path) or "/"
        # Stream the payload over stdin so no quoting or length limit applies.
        argv = self._sandbox._exec_argv(
            user="root",
            command=f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}",
            interactive=True,
        )
        completed = subprocess.run(
            argv,
            input=content.encode("utf-8"),
            capture_output=True,
            timeout=self._sandbox.default_timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to write {path}: {completed.stderr.decode('utf-8', 'replace').strip()}"
            )

    def read(self, path: str) -> str:
        """Read ``path`` from the container."""
        result = self._sandbox.commands.run(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to read {path}: {result.stderr.strip()}")
        return result.stdout


class _Commands:
    """The ``sandbox.commands`` namespace."""

    def __init__(self, sandbox: "DockerSandbox") -> None:
        self._sandbox = sandbox

    def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        user: Optional[str] = None,
    ) -> CommandResult:
        """Run ``command`` with bash inside the container."""
        self._sandbox._ensure_running()
        argv = self._sandbox._exec_argv(user=user or "root", command=command)
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout or self._sandbox.default_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", "replace")
            stderr = (exc.stderr or b"").decode("utf-8", "replace")
            return CommandResult(
                stdout=stdout,
                stderr=f"{stderr}\n[sandbox] command timed out after {timeout or self._sandbox.default_timeout}s",
                exit_code=124,
            )
        return CommandResult(
            stdout=completed.stdout.decode("utf-8", "replace"),
            stderr=completed.stderr.decode("utf-8", "replace"),
            exit_code=completed.returncode,
        )


class DockerSandbox:
    """An ephemeral Docker container used as the agent's execution sandbox."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        workdir: str = DEFAULT_WORKDIR,
        network: Optional[str] = None,
        memory: Optional[str] = None,
        cpus: Optional[str] = None,
        bootstrap: Optional[bool] = None,
        default_timeout: int = 120,
    ) -> None:
        self.image = image
        self.workdir = workdir
        self.network = network or os.getenv("SANDBOX_NETWORK", "bridge")
        self.memory = memory or os.getenv("SANDBOX_MEMORY", "2g")
        self.cpus = cpus or os.getenv("SANDBOX_CPUS", "2")
        self.default_timeout = default_timeout
        self.lifetime_seconds: Optional[int] = None
        self.container_id: Optional[str] = None

        if bootstrap is None:
            bootstrap = os.getenv("SANDBOX_SKIP_BOOTSTRAP", "").lower() not in ("1", "true", "yes")
        self.bootstrap = bootstrap

        self.files = _Files(self)
        self.commands = _Commands(self)

        self._start()
        if self.bootstrap:
            self._bootstrap()

    # --- lifecycle -------------------------------------------------------

    def _start(self) -> None:
        name = f"mapta-sandbox-{uuid.uuid4().hex[:12]}"
        argv = [
            "docker", "run", "--detach",
            "--name", name,
            "--workdir", self.workdir,
            "--network", self.network,
            "--memory", self.memory,
            "--cpus", self.cpus,
            # A runaway agent should not be able to fork-bomb the host.
            "--pids-limit", "512",
            self.image,
            "sleep", "infinity",
        ]
        try:
            completed = subprocess.run(argv, capture_output=True, timeout=300)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The `docker` CLI was not found. Install Docker, or point SANDBOX_FACTORY "
                "at a different provider."
            ) from exc

        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to start the sandbox container: "
                f"{completed.stderr.decode('utf-8', 'replace').strip()}"
            )

        self.container_id = completed.stdout.decode().strip()
        logger.info("Sandbox container started: %s (%s)", name, self.image)

    def _bootstrap(self) -> None:
        """Install the tooling the agent's tools assume is present."""
        packages = f"{BASE_PACKAGES} {os.getenv('SANDBOX_PACKAGES', '')}".strip()
        pip_packages = f"{BASE_PIP_PACKAGES} {os.getenv('SANDBOX_PIP_PACKAGES', '')}".strip()

        steps: List[tuple] = [
            (
                "system packages",
                "command -v apt-get >/dev/null && "
                "(DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends {packages}) "
                "|| echo '[sandbox] apt-get unavailable, skipping system packages'",
                900,
            ),
            # `sandbox_run_python` runs `source .venv/bin/activate` from the
            # working directory, so the venv has to live there.
            (
                "python venv",
                f"cd {shlex.quote(self.workdir)} && "
                "(python3 -m venv .venv || python -m venv .venv) && "
                ".venv/bin/pip install -q --upgrade pip",
                600,
            ),
            (
                "python packages",
                f"cd {shlex.quote(self.workdir)} && .venv/bin/pip install -q {pip_packages}",
                900,
            ),
        ]

        for label, command, timeout in steps:
            result = self.commands.run(command, timeout=timeout, user="root")
            if result.exit_code != 0:
                logger.warning(
                    "Sandbox bootstrap step '%s' failed (exit %s): %s",
                    label, result.exit_code, result.stderr.strip()[:500],
                )
            else:
                logger.info("Sandbox bootstrap step '%s' done", label)

    def set_timeout(self, timeout: int) -> None:
        """Record the requested sandbox lifetime.

        `main.py` calls this to extend how long the sandbox may live, mirroring
        hosted providers that reap idle sandboxes. A local container lives until
        `kill()`, so this only records the value -- in particular it must not
        shorten the per-command timeout. Values above 10000 are read as
        milliseconds, matching the hosted-provider convention.
        """
        self.lifetime_seconds = timeout // 1000 if timeout > 10000 else timeout
        logger.debug("Sandbox lifetime hint: %ss (no-op for a local container)", self.lifetime_seconds)

    def kill(self) -> None:
        """Remove the container. Safe to call more than once."""
        if not self.container_id:
            return
        subprocess.run(
            ["docker", "rm", "--force", "--volumes", self.container_id],
            capture_output=True,
            timeout=120,
        )
        logger.info("Sandbox container removed: %s", self.container_id[:12])
        self.container_id = None

    # --- helpers ---------------------------------------------------------

    def _ensure_running(self) -> None:
        if not self.container_id:
            raise RuntimeError("Sandbox container is not running (it was already killed).")

    def _exec_argv(self, user: str, command: str, interactive: bool = False) -> List[str]:
        argv = ["docker", "exec"]
        if interactive:
            argv.append("--interactive")
        argv += ["--user", user, "--workdir", self.workdir, self.container_id, "bash", "-lc", command]
        return argv

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, *exc_info) -> None:
        self.kill()


def create_sandbox() -> DockerSandbox:
    """SANDBOX_FACTORY entry point: ``sandbox_docker:create_sandbox``."""
    return DockerSandbox()


if __name__ == "__main__":
    # Smoke test: python sandbox_docker.py
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with create_sandbox() as sandbox:
        print(sandbox.commands.run("whoami && python3 --version && curl --version | head -1").stdout)
        sandbox.files.write(f"{DEFAULT_WORKDIR}/hello.py", "print('hello from the sandbox')\n")
        print(sandbox.commands.run("source .venv/bin/activate && python3 hello.py").stdout)
