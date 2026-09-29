"""Sandbox adapters and the factory that picks one from configuration."""

import logging

from ...config import SandboxSettings
from ...domain import SandboxFactory
from .docker import DockerSandbox, DockerSandboxFactory
from .legacy import LegacySandboxAdapter, LegacySandboxFactory
from .null import NullSandbox, NullSandboxFactory

__all__ = [
    "DockerSandbox",
    "DockerSandboxFactory",
    "LegacySandboxAdapter",
    "LegacySandboxFactory",
    "NullSandbox",
    "NullSandboxFactory",
    "build_sandbox_factory",
]

logger = logging.getLogger(__name__)


def build_sandbox_factory(settings: SandboxSettings) -> SandboxFactory:
    """Resolve ``SANDBOX_PROVIDER`` into a concrete factory.

    Accepted values: ``docker``, ``none``, or ``module:function`` for a
    third-party provider.
    """
    match settings.provider.strip().lower():
        case "docker" | "":
            return DockerSandboxFactory(settings)
        case "none" | "null" | "off":
            logger.warning("Running without a sandbox: execution tools will refuse to run.")
            return NullSandboxFactory()
        case _:
            return LegacySandboxFactory(settings.provider)
