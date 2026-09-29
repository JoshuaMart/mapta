"""Adapter for third-party sandbox providers loaded via ``module:function``.

Hosted providers expose a synchronous object with ``files.write(...)`` and
``commands.run(...)``. This wraps one so it satisfies the async sandbox port,
running the blocking calls off the event loop.
"""

import asyncio
import importlib
import logging
from typing import Any

from ...domain import CommandResult, SandboxError

__all__ = ["LegacySandboxAdapter", "LegacySandboxFactory"]

logger = logging.getLogger(__name__)


class LegacySandboxAdapter:
    """Wraps a synchronous provider sandbox behind the async port."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    async def run(
        self, command: str, *, timeout: float | None = None, user: str = "root"
    ) -> CommandResult:
        result = await asyncio.to_thread(
            self._sandbox.commands.run, command, timeout=timeout, user=user
        )
        return CommandResult(
            stdout=getattr(result, "stdout", "") or "",
            stderr=getattr(result, "stderr", "") or "",
            exit_code=getattr(result, "exit_code", 0) or 0,
        )

    async def write_file(self, path: str, content: str) -> None:
        await asyncio.to_thread(self._sandbox.files.write, path, content)

    async def aclose(self) -> None:
        if kill := getattr(self._sandbox, "kill", None):
            await asyncio.to_thread(kill)


class LegacySandboxFactory:
    """Loads ``module:function`` and adapts whatever it returns.

    Args:
        spec: Import path of the factory, e.g. ``my_provider:create_sandbox``.
        lifetime_ms: Optional lifetime hint passed to ``set_timeout`` when the
            provider supports it, mirroring hosted providers that reap idle
            sandboxes.
    """

    def __init__(self, spec: str, *, lifetime_ms: int = 12_000) -> None:
        if ":" not in spec:
            raise SandboxError(
                f"Invalid sandbox factory {spec!r}: expected the form 'module:function'."
            )
        self._spec = spec
        self._lifetime_ms = lifetime_ms

    async def create(self) -> LegacySandboxAdapter:
        return LegacySandboxAdapter(await asyncio.to_thread(self._build))

    def _build(self) -> Any:
        module_name, func_name = self._spec.rsplit(":", 1)
        try:
            factory = getattr(importlib.import_module(module_name), func_name)
        except (ImportError, AttributeError) as exc:
            raise SandboxError(f"Could not load sandbox factory {self._spec!r}: {exc}") from exc

        sandbox = factory()
        if set_timeout := getattr(sandbox, "set_timeout", None):
            try:
                set_timeout(self._lifetime_ms)
            except Exception as exc:  # a hint, never a reason to fail the scan
                logger.debug("set_timeout not honoured by %s: %s", self._spec, exc)
        return sandbox
