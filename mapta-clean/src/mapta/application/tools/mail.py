"""Disposable-mailbox tools, used to receive activation mails and credentials."""

import json

from .base import ToolContext, tool

__all__ = [
    "create_email_account",
    "get_message_by_id",
    "get_registered_emails",
    "list_account_messages",
    "set_email_jwt_token",
]


@tool
async def create_email_account(
    ctx: ToolContext, address: str | None = None, password: str | None = None
) -> str:
    """Create a disposable email account and store its token for later use.

    Use it to receive account activation emails, password resets or credentials.
    Returns JSON: {address, password}.

    Args:
        address: Optional full address to claim (e.g. "someone@domain.tld"). A random
            local part on an available domain is generated when omitted.
        password: Optional password. A random one is generated when omitted.
    """
    return json.dumps(await ctx.require_mailbox().create_account(address, password))


@tool
async def set_email_jwt_token(ctx: ToolContext, email: str, jwt_token: str) -> str:
    """Store a JWT for an email account you already own.

    The list_account_messages and get_message_by_id tools then work on it.

    Args:
        email: The email address the token belongs to.
        jwt_token: The mailbox JWT for that account.
    """
    if not email or not jwt_token:
        return "Both email and jwt_token are required."
    await ctx.require_mailbox().remember_token(email, jwt_token)
    return json.dumps({"success": True, "email": email})


@tool
async def get_registered_emails(ctx: ToolContext) -> str:
    """List the email accounts available to receive mail during this scan."""
    return json.dumps(await ctx.require_mailbox().known_addresses())


@tool
async def list_account_messages(ctx: ToolContext, email: str, limit: int = 50) -> str:
    """List recent messages for the given email account.

    Returns JSON list: [{id, subject, from, intro, seen, createdAt}].

    Args:
        email: The email account to fetch messages for.
        limit: Maximum number of messages to return (default: 50).
    """
    return json.dumps(await ctx.require_mailbox().list_messages(email, limit))


@tool
async def get_message_by_id(ctx: ToolContext, email: str, message_id: str) -> str:
    """Fetch one message by id for the given email account.

    Returns JSON: {id, subject, from, text, html}.

    Args:
        email: The email account to fetch the message from.
        message_id: The ID of the message to fetch.
    """
    return json.dumps(await ctx.require_mailbox().get_message(email, message_id))
