"""Disposable mailbox backed by the mail.tm API.

One instance per scan: the token store lives on the instance, so two targets
never see each other's inboxes.
"""

import logging
import secrets

import httpx2 as httpx

from ...domain import JSONObject
from ..http import HTTPAdapter

__all__ = ["MailTmMailbox", "MailTmMailboxFactory"]

logger = logging.getLogger(__name__)

API_ROOT = "https://api.mail.tm"


class MailTmMailbox(HTTPAdapter):
    """Implements the :class:`~mapta.domain.ports.Mailbox` port."""

    def __init__(self, client: httpx.AsyncClient | None = None, api_root: str = API_ROOT) -> None:
        super().__init__(client)
        self._api_root = api_root.rstrip("/")
        self._tokens: dict[str, str] = {}

    async def create_account(
        self, address: str | None = None, password: str | None = None
    ) -> JSONObject:
        """Register an account and remember its JWT."""
        password = password or secrets.token_urlsafe(16)

        if not address:
            response = await self._client.get(f"{self._api_root}/domains")
            if response.status_code != 200:
                return _error(f"Failed to list domains: {response.status_code} {response.text}")
            domains = [
                item["domain"]
                for item in response.json().get("hydra:member", [])
                if item.get("domain")
            ]
            if not domains:
                return _error("No mail.tm domains are currently available.")
            address = f"mapta{secrets.token_hex(6)}@{domains[0]}"

        credentials = {"address": address, "password": password}
        response = await self._client.post(f"{self._api_root}/accounts", json=credentials)
        if response.status_code not in (200, 201):
            return _error(
                f"Failed to create account {address}: {response.status_code} {response.text}"
            )

        response = await self._client.post(f"{self._api_root}/token", json=credentials)
        if response.status_code not in (200, 201):
            return _error(
                f"Account {address} created but the token request failed: "
                f"{response.status_code} {response.text}"
            )
        token = response.json().get("token")
        if not token:
            return _error(f"Account {address} created but no token was returned.")

        self._tokens[address] = token
        return {"address": address, "password": password}

    async def remember_token(self, address: str, token: str) -> None:
        self._tokens[address] = token

    async def known_addresses(self) -> list[str]:
        return list(self._tokens)

    async def list_messages(self, address: str, limit: int = 50) -> list[JSONObject]:
        headers = self._auth(address)
        if headers is None:
            return [_missing_token(address)]
        response = await self._client.get(f"{self._api_root}/messages", headers=headers)
        if response.status_code != 200:
            return [_error(f"Failed to fetch messages: {response.status_code} {response.text}")]
        return [
            {
                "id": message.get("id"),
                "subject": message.get("subject"),
                "from": _sender(message),
                "intro": message.get("intro", ""),
                "seen": message.get("seen", False),
                "createdAt": message.get("createdAt", ""),
            }
            for message in response.json().get("hydra:member", [])[:limit]
        ]

    async def get_message(self, address: str, message_id: str) -> JSONObject:
        headers = self._auth(address)
        if headers is None:
            return _missing_token(address)
        response = await self._client.get(
            f"{self._api_root}/messages/{message_id}", headers=headers
        )
        if response.status_code != 200:
            return _error(f"Failed to fetch message: {response.status_code} {response.text}")
        message = response.json()
        return {
            "id": message.get("id"),
            "subject": message.get("subject"),
            "from": _sender(message),
            "text": message.get("text", ""),
            "html": message.get("html", ""),
        }

    def _auth(self, address: str) -> dict[str, str] | None:
        token = self._tokens.get(address)
        return {"Authorization": f"Bearer {token}"} if token else None


def _sender(message: JSONObject) -> str:
    sender = message.get("from") or {}
    return sender.get("address") or sender.get("name") or ""


def _error(detail: str) -> JSONObject:
    logger.debug("mailbox error: %s", detail)
    return {"error": "mailbox_error", "detail": detail}


def _missing_token(address: str) -> JSONObject:
    return {
        "error": "no_token",
        "detail": (
            f"No token stored for {address}. Call set_email_jwt_token(email, jwt_token) "
            "or create the account with create_email_account first."
        ),
    }


class MailTmMailboxFactory:
    """Creates one mailbox per scan."""

    def __call__(self) -> MailTmMailbox:
        return MailTmMailbox()
