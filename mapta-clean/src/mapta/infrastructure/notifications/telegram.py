"""Telegram Bot API notifier.

Message building is pure and separate from the HTTP call, so payloads are
tested without a network. Telegram caps a message at 4096 characters and
rejects malformed markup, so every agent-supplied field is HTML-escaped and
clipped *before* assembly, rather than the finished message being cut in half.
"""

import html
import json
import logging

import httpx2 as httpx

from ...config import TelegramSettings
from ...domain import Finding, JSONObject, ScanTally, Severity, Target, utcnow
from ..http import HTTPAdapter

__all__ = ["TelegramNotifier", "build_finding_message", "build_summary_message"]

logger = logging.getLogger(__name__)

#: Telegram's hard limit for `sendMessage`.
MAX_MESSAGE_CHARS = 4096

# Per-field budgets, sized so an assembled message stays well under the cap.
_TYPE_BUDGET = 120
_URL_BUDGET = 400
_DESCRIPTION_BUDGET = 1400
_EVIDENCE_BUDGET = 800
_RECOMMENDATION_BUDGET = 800


def _field(text: str, budget: int) -> str:
    """HTML-escape ``text`` and clip it so the escaped result fits ``budget``.

    Clipping happens on the raw text and the result is re-escaped, so an entity
    like ``&amp;`` is never cut in half into markup Telegram would reject.
    """
    escaped = html.escape(text, quote=False)
    if len(escaped) <= budget:
        return escaped
    clipped = text[: max(1, int(len(text) * budget / len(escaped)))]
    while len(html.escape(clipped, quote=False)) > budget - 1 and clipped:
        clipped = clipped[:-1]
    return html.escape(clipped, quote=False) + "…"


def _link(url: str) -> str:
    """A safe anchor: the href is quote-escaped, the label is not a URL."""
    return f'<a href="{html.escape(url, quote=True)}">{_field(url, _URL_BUDGET)}</a>'


def _timestamp() -> str:
    return utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _cap(message: str) -> str:
    """Last-resort guard; the field budgets make this unreachable in practice."""
    if len(message) <= MAX_MESSAGE_CHARS:
        return message
    logger.warning("Telegram message exceeded %d chars, truncating", MAX_MESSAGE_CHARS)
    return message[: MAX_MESSAGE_CHARS - 1] + "…"


def build_finding_message(finding: Finding) -> str:
    """Render one finding as Telegram-flavoured HTML."""
    severity = finding.severity
    lines = [
        f"{severity.emoji} <b>{_field(finding.vulnerability_type, _TYPE_BUDGET)}</b>",
        "",
        f"<b>Severity:</b> {severity.value}",
        f"<b>Target:</b> {_link(finding.target_url)}",
        "",
        "<b>Description</b>",
        _field(finding.description, _DESCRIPTION_BUDGET),
    ]
    if finding.evidence:
        lines += [
            "",
            "<b>Evidence / PoC</b>",
            f"<pre><code>{_field(finding.evidence, _EVIDENCE_BUDGET)}</code></pre>",
        ]
    if finding.recommendation:
        lines += [
            "",
            "<b>Recommendation</b>",
            _field(finding.recommendation, _RECOMMENDATION_BUDGET),
        ]
    lines += ["", f"<i>Detected at {_timestamp()}</i>"]
    return _cap("\n".join(lines))


_STATUS: dict[Severity | None, tuple[str, str]] = {
    Severity.CRITICAL: ("🔴", "Critical issues found"),
    Severity.HIGH: ("🟠", "High risk issues found"),
    Severity.MEDIUM: ("🟡", "Medium risk issues found"),
    Severity.LOW: ("🟢", "Low risk issues found"),
    Severity.INFO: ("🔵", "Informational findings"),
    None: ("✅", "No issues found"),
}


def _status(tally: ScanTally) -> tuple[str, str]:
    """Never show a clean tick next to a non-zero finding count."""
    headline = tally.headline_severity
    if headline is None and tally.total > 0:
        headline = Severity.INFO
    return _STATUS[headline]


def build_summary_message(target: Target, tally: ScanTally, duration: str | None) -> str:
    """Render an end-of-scan summary as Telegram-flavoured HTML."""
    emoji, status = _status(tally)
    lines = [
        f"{emoji} <b>Security scan summary</b>",
        "",
        f"<b>Target:</b> {_link(target.url)}",
        f"<b>Status:</b> {status}",
        f"<b>Total findings:</b> {tally.total}",
    ]
    breakdown = [
        f"{severity.emoji} {severity.value}: {count}"
        for severity, count in (
            (Severity.CRITICAL, tally.critical),
            (Severity.HIGH, tally.high),
            (Severity.MEDIUM, tally.medium),
            (Severity.LOW, tally.low),
        )
        if count > 0
    ]
    if breakdown:
        lines += ["", "<b>Breakdown</b>", *breakdown]
    if duration:
        lines += ["", f"<i>Duration {html.escape(duration)} · completed {_timestamp()}</i>"]
    return _cap("\n".join(lines))


class TelegramNotifier(HTTPAdapter):
    """Posts findings through the Bot API. A no-op when unconfigured."""

    def __init__(
        self, settings: TelegramSettings, client: httpx.AsyncClient | None = None
    ) -> None:
        super().__init__(client)
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    async def send_finding(self, finding: Finding, *, thread_id: str | None = None) -> str:
        return await self._send(build_finding_message(finding), reply_to=thread_id)

    async def send_summary(
        self, target: Target, tally: ScanTally, *, duration: str | None = None
    ) -> str:
        return await self._send(build_summary_message(target, tally, duration))

    def _payload(self, text: str, reply_to: str | None) -> JSONObject:
        payload: JSONObject = {
            "chat_id": self._settings.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
            "disable_notification": self._settings.silent,
        }
        # Forum topic to post into, distinct from replying to one message.
        if self._settings.message_thread_id is not None:
            payload["message_thread_id"] = self._settings.message_thread_id
        if reply_to and (message_id := _as_message_id(reply_to)) is not None:
            payload["reply_parameters"] = {"message_id": message_id}
        return payload

    async def _send(self, text: str, *, reply_to: str | None = None) -> str:
        settings = self._settings
        if not settings.enabled:
            return json.dumps({"success": False, "error": "No Telegram bot configured."})

        url = f"{settings.api_root}/bot{settings.bot_token}/sendMessage"
        try:
            response = await self._client.post(url, json=self._payload(text, reply_to))
        except httpx.HTTPError as exc:
            logger.warning("Telegram call failed: %s", exc)
            return json.dumps({"success": False, "error": f"Request failed: {exc}"})

        try:
            body = response.json()
        except ValueError:
            body = {}

        if response.status_code == 200 and body.get("ok"):
            # Returning the message id lets the agent thread later alerts under it.
            return json.dumps(
                {
                    "success": True,
                    "message": "Sent to Telegram successfully",
                    "message_id": body.get("result", {}).get("message_id"),
                }
            )

        detail = body.get("description") or response.text
        logger.warning("Telegram rejected the message: %s", detail)
        return json.dumps({"success": False, "error": f"Failed to send to Telegram: {detail}"})


def _as_message_id(value: str) -> int | None:
    """Telegram message ids are integers; ignore anything else quietly."""
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("Ignoring non-numeric Telegram message id: %r", value)
        return None
