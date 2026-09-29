"""A sandbox that refuses to run anything, used when none is configured.

It keeps the agent loop working (and honest) instead of crashing: the model is
told plainly that execution is unavailable.
"""

from ...domain import CommandResult

__all__ = ["NullSandbox", "NullSandboxFactory"]

_MESSAGE = (
    "No sandbox is configured for this run, so nothing can be executed. "
    "Set SANDBOX_PROVIDER=docker (or SANDBOX_FACTORY=module:function) and re-run."
)


class NullSandbox:
    """Implements the sandbox port with a clear error for every operation."""

    async def run(
        self, command: str, *, timeout: float | None = None, user: str = "root"
    ) -> CommandResult:
        return CommandResult(stdout="", stderr=_MESSAGE, exit_code=127)

    async def write_file(self, path: str, content: str) -> None:
        raise RuntimeError(_MESSAGE)

    async def aclose(self) -> None:
        return None


class NullSandboxFactory:
    async def create(self) -> NullSandbox:
        return NullSandbox()
