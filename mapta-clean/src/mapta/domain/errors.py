"""Domain-level errors.

Every failure MAPTA raises on purpose derives from :class:`MaptaError`, so the
composition root can turn them into exit codes without catching ``Exception``.
"""


class MaptaError(Exception):
    """Base class for every error MAPTA raises deliberately."""


class ConfigurationError(MaptaError):
    """The environment or CLI flags do not describe a runnable configuration."""


class SandboxError(MaptaError):
    """The sandbox could not be created, or refused to execute a command."""


class ToolArgumentError(MaptaError):
    """A model produced tool arguments that could not be decoded."""


class ToolUnavailable(MaptaError):
    """A tool needs a collaborator this run was not configured with."""


class LLMError(MaptaError):
    """The model provider rejected a request or returned an unusable answer."""
