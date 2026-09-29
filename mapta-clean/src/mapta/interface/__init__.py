"""Interface layer: CLI and the composition root."""

from .cli import main
from .container import Container, build_scan_service

__all__ = ["Container", "build_scan_service", "main"]
