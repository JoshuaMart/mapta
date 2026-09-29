"""Tools that push findings to the operator's chat channel."""

from ...domain import Finding, ScanTally, Severity, Target
from .base import ToolContext, tool

__all__ = ["send_alert", "send_summary"]


@tool(name="send_telegram_alert")
async def send_alert(
    ctx: ToolContext,
    vulnerability_type: str,
    severity: str,
    target_url: str,
    description: str,
    evidence: str | None = None,
    recommendation: str | None = None,
    reply_to_message_id: str | None = None,
) -> str:
    """Send a security vulnerability alert to the operator's Telegram chat.

    Returns JSON including the sent message_id, which you can pass back as
    reply_to_message_id to keep related alerts in one thread.

    Args:
        vulnerability_type: Type of vulnerability (e.g., "XSS", "SQL Injection", "IDOR").
        severity: Severity level ("Critical", "High", "Medium", "Low", "Info").
        target_url: The affected URL or endpoint.
        description: Detailed description of the vulnerability.
        evidence: Optional proof-of-concept or evidence details.
        recommendation: Optional remediation recommendation.
        reply_to_message_id: Optional id of a previous alert to reply to.
    """
    finding = Finding(
        vulnerability_type=vulnerability_type,
        severity=Severity.parse(severity),
        target_url=target_url,
        description=description,
        evidence=evidence,
        recommendation=recommendation,
    )
    return await ctx.require_notifier().send_finding(
        finding, thread_id=reply_to_message_id
    )


@tool(name="send_telegram_summary")
async def send_summary(
    ctx: ToolContext,
    target_url: str,
    total_findings: int,
    critical_count: int = 0,
    high_count: int = 0,
    medium_count: int = 0,
    low_count: int = 0,
    scan_duration: str | None = None,
) -> str:
    """Send a summary of the security scan to the operator's Telegram chat.

    Args:
        target_url: The target that was scanned.
        total_findings: Total number of vulnerabilities found.
        critical_count: Number of critical severity findings.
        high_count: Number of high severity findings.
        medium_count: Number of medium severity findings.
        low_count: Number of low severity findings.
        scan_duration: Optional duration of the scan.
    """
    tally = ScanTally(
        total=total_findings,
        critical=critical_count,
        high=high_count,
        medium=medium_count,
        low=low_count,
    )
    target = ctx.target if target_url == ctx.target.url else Target(target_url)
    return await ctx.require_notifier().send_summary(target, tally, duration=scan_duration)
