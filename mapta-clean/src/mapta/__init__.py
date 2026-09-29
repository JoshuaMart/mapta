"""MAPTA - autonomous multi-agent penetration testing.

Layers, outermost last::

    domain          data structures, errors, ports (no I/O, no dependencies)
    application     use cases: the agent loop, the scan service, the tools
    infrastructure  adapters: OpenRouter, Docker, Telegram, mail.tm, files
    interface       CLI and the composition root that wires it all together

Dependencies only ever point inwards.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
