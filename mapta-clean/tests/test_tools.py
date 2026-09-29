"""Tool behaviour against fake collaborators."""

import json

from conftest import FakeNotifier
from mapta.application.tools import reporting
from mapta.application.tools import sandbox as sandbox_tools
from mapta.domain import CommandResult, Severity, ToolCall


async def test_run_command_renders_exit_code_and_streams(ctx, sandbox):
    sandbox.results["whoami"] = CommandResult(stdout="root\n", stderr="", exit_code=0)
    output = await sandbox_tools.sandbox_run_command.invoke(ctx, {"command": "whoami"})
    assert "Exit code: 0" in output
    assert "root" in output


async def test_run_command_truncates_long_output(ctx, sandbox):
    ctx.max_tool_output_chars = 100
    sandbox.results["big"] = CommandResult(stdout="x" * 500, stderr="", exit_code=0)
    output = await sandbox_tools.sandbox_run_command.invoke(ctx, {"command": "big"})
    assert output.endswith("[OUTPUT TRUNCATED - EXCEEDED 100 CHARACTERS]")


async def test_run_python_writes_a_script_then_executes_it(ctx, sandbox):
    await sandbox_tools.sandbox_run_python.invoke(ctx, {"python_code": "print(1)"})
    (path, content), = sandbox.files.items()
    assert content == "print(1)"
    assert path in sandbox.commands[0]


async def test_alert_without_a_notifier_says_so(ctx, registry):
    """Reported through the registry, which is what turns it into a result."""
    call = ToolCall(
        "1",
        "send_telegram_alert",
        json.dumps(
            {
                "vulnerability_type": "XSS",
                "severity": "High",
                "target_url": ctx.target.url,
                "description": "reflected",
            }
        ),
    )
    payload = json.loads((await registry.execute(ctx, call)).output)
    assert payload["error"] == "tool_unavailable"
    assert "TELEGRAM_BOT_TOKEN" in payload["detail"]


async def test_alert_reaches_the_notifier(ctx):
    ctx.notifier = FakeNotifier()
    await reporting.send_alert.invoke(
        ctx,
        {
            "vulnerability_type": "SQLi",
            "severity": "critical",
            "target_url": ctx.target.url,
            "description": "union based",
            "evidence": "' OR 1=1",
        },
    )
    finding = ctx.notifier.findings[0]
    assert finding.severity is Severity.CRITICAL
    assert finding.evidence == "' OR 1=1"


async def test_summary_tallies_are_forwarded(ctx):
    ctx.notifier = FakeNotifier()
    await reporting.send_summary.invoke(
        ctx,
        {"target_url": ctx.target.url, "total_findings": 2, "critical_count": 1, "low_count": 1},
    )
    _, tally = ctx.notifier.summaries[0]
    assert (tally.total, tally.critical, tally.low) == (2, 1, 1)


async def test_mail_tools_are_graceful_without_a_mailbox(ctx, registry):
    call = ToolCall("2", "get_registered_emails", "{}")
    payload = json.loads((await registry.execute(ctx, call)).output)
    assert payload["error"] == "tool_unavailable"


async def test_nested_agents_are_graceful_when_absent(ctx, registry):
    call = ToolCall("3", "sandbox_agent", '{"instruction": "look around"}')
    payload = json.loads((await registry.execute(ctx, call)).output)
    assert payload["error"] == "tool_unavailable"
