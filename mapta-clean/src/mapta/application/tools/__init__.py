"""Tool definitions and the registry that exposes them to the agents."""

from .agents import run_sandbox_agent, run_validator_agent
from .base import SubAgentRunner, Tool, ToolContext, ToolRegistry, tool
from .mail import (
    create_email_account,
    get_message_by_id,
    get_registered_emails,
    list_account_messages,
    set_email_jwt_token,
)
from .reporting import send_alert, send_summary
from .sandbox import sandbox_run_command, sandbox_run_python

#: Every tool MAPTA ships. Agents receive a subset, chosen by their profile.
DEFAULT_TOOLS: tuple[Tool, ...] = (
    sandbox_run_command,
    sandbox_run_python,
    run_sandbox_agent,
    run_validator_agent,
    create_email_account,
    set_email_jwt_token,
    get_registered_emails,
    list_account_messages,
    get_message_by_id,
    send_alert,
    send_summary,
)


def default_registry() -> ToolRegistry:
    """Build a registry holding every bundled tool."""
    return ToolRegistry.of(DEFAULT_TOOLS)


__all__ = [
    "DEFAULT_TOOLS",
    "SubAgentRunner",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "default_registry",
    "tool",
]
