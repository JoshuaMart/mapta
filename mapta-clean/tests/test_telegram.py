"""Telegram payloads are built and validated without touching the network."""

import json

import httpx2 as httpx
import pytest

from mapta.config import TelegramSettings
from mapta.domain import Finding, ScanTally, Severity, Target
from mapta.infrastructure.notifications import (
    TelegramNotifier,
    build_finding_message,
    build_summary_message,
)
from mapta.infrastructure.notifications.telegram import MAX_MESSAGE_CHARS

TARGET = Target("https://a.test")


def finding(**overrides) -> Finding:
    base = {
        "vulnerability_type": "IDOR",
        "severity": Severity.HIGH,
        "target_url": TARGET.url,
        "description": "object ids are sequential",
    }
    return Finding(**(base | overrides))


# --- message building -----------------------------------------------------


def test_finding_message_leads_with_severity_and_type():
    message = build_finding_message(finding())
    assert message.startswith(f"{Severity.HIGH.emoji} <b>IDOR</b>")
    assert "<b>Severity:</b> High" in message
    assert '<a href="https://a.test">' in message


def test_operator_text_is_html_escaped():
    message = build_finding_message(finding(description='<script>alert("x")</script> & co'))
    assert "<script>" not in message
    assert "&lt;script&gt;" in message
    assert "&amp; co" in message


def test_evidence_goes_in_a_code_block():
    message = build_finding_message(finding(evidence="' OR 1=1 -- <injected>"))
    assert "<pre><code>' OR 1=1 -- &lt;injected&gt;</code></pre>" in message


def test_long_fields_are_clipped_without_breaking_entities():
    message = build_finding_message(finding(description="<" * 5000, evidence="&" * 5000))
    assert len(message) <= MAX_MESSAGE_CHARS
    # A clipped entity would leave a bare "&l" or "&am"; every one stays whole.
    for fragment in message.split("&")[1:]:
        assert fragment.split(";")[0] in ("lt", "gt", "amp", "quot", "#x27")


def test_a_message_never_exceeds_the_telegram_cap():
    message = build_finding_message(
        finding(
            vulnerability_type="X" * 500,
            description="D" * 9000,
            evidence="E" * 9000,
            recommendation="R" * 9000,
        )
    )
    assert len(message) <= MAX_MESSAGE_CHARS


def test_optional_sections_are_omitted():
    message = build_finding_message(finding())
    assert "Evidence" not in message
    assert "Recommendation" not in message


def test_clean_summary_reports_no_issues():
    message = build_summary_message(TARGET, ScanTally(total=0), None)
    assert "No issues found" in message
    assert "Breakdown" not in message


def test_summary_headline_follows_the_worst_finding():
    message = build_summary_message(TARGET, ScanTally(total=3, critical=1, low=2), "0:10:00")
    assert "Critical issues found" in message
    assert f"{Severity.CRITICAL.emoji} Critical: 1" in message
    assert "Duration 0:10:00" in message


# --- the HTTP call --------------------------------------------------------


def notifier(handler, **overrides) -> TelegramNotifier:
    settings = TelegramSettings(bot_token="123:abc", chat_id="-100", **overrides)
    transport = httpx.MockTransport(handler)
    return TelegramNotifier(settings, httpx.AsyncClient(transport=transport))


@pytest.fixture
def captured() -> list[httpx.Request]:
    return []


async def test_successful_send_returns_the_message_id(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 99}})

    result = json.loads(await notifier(handler).send_finding(finding()))
    assert result == {
        "success": True,
        "message": "Sent to Telegram successfully",
        "message_id": 99,
    }
    assert captured[0].url.path == "/bot123:abc/sendMessage"


async def test_payload_shape(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    await notifier(handler, message_thread_id=7, silent=True).send_finding(
        finding(), thread_id="55"
    )
    payload = json.loads(captured[0].content)
    assert payload["chat_id"] == "-100"
    assert payload["parse_mode"] == "HTML"
    assert payload["link_preview_options"] == {"is_disabled": True}
    assert payload["message_thread_id"] == 7
    assert payload["disable_notification"] is True
    assert payload["reply_parameters"] == {"message_id": 55}


async def test_non_numeric_reply_id_is_dropped(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    await notifier(handler).send_finding(finding(), thread_id="1690000000.123")
    assert "reply_parameters" not in json.loads(captured[0].content)


async def test_api_error_is_reported_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"ok": False, "description": "chat not found"})

    result = json.loads(await notifier(handler).send_summary(TARGET, ScanTally(total=0)))
    assert result["success"] is False
    assert "chat not found" in result["error"]


async def test_network_failure_is_reported_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    result = json.loads(await notifier(handler).send_finding(finding()))
    assert result["success"] is False
    assert "Request failed" in result["error"]


async def test_unconfigured_notifier_sends_nothing():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    quiet = TelegramNotifier(
        TelegramSettings(), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert quiet.enabled is False
    assert json.loads(await quiet.send_finding(finding()))["success"] is False
    assert calls == []
