"""Shared fakes: every port has an in-memory double, so no test needs a network."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from mapta.application.tools import ToolContext, default_registry
from mapta.application.usage import UsageTracker
from mapta.domain import (
    CommandResult,
    Finding,
    JSONObject,
    ModelResponse,
    ScanTally,
    Target,
    ToolCall,
    ToolSpec,
    TranscriptItem,
    UsageSummary,
)


class FakeSandbox:
    """Records commands and replays canned results."""

    def __init__(self, results: dict[str, CommandResult] | None = None) -> None:
        self.results = results or {}
        self.commands: list[str] = []
        self.files: dict[str, str] = {}
        self.closed = False

    async def run(self, command, *, timeout=None, user="root") -> CommandResult:
        self.commands.append(command)
        return self.results.get(command, CommandResult(stdout="ok", stderr="", exit_code=0))

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def aclose(self) -> None:
        self.closed = True


class FakeSandboxFactory:
    def __init__(self) -> None:
        self.created: list[FakeSandbox] = []

    async def create(self) -> FakeSandbox:
        sandbox = FakeSandbox()
        self.created.append(sandbox)
        return sandbox


@dataclass
class FakeLLM:
    """Replays a scripted list of responses, recording what it was asked."""

    responses: list[ModelResponse] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def developer_message(self, text: str) -> TranscriptItem:
        return {"role": "developer", "text": text}

    def user_message(self, text: str) -> TranscriptItem:
        return {"role": "user", "text": text}

    def tool_output(self, call_id: str, output: str) -> TranscriptItem:
        return {"type": "function_call_output", "call_id": call_id, "output": output}

    async def respond(
        self,
        *,
        transcript: Sequence[TranscriptItem],
        tools: Sequence[ToolSpec],
        metadata: JSONObject | None = None,
    ) -> ModelResponse:
        self.requests.append(
            {"transcript": list(transcript), "tools": list(tools), "metadata": metadata}
        )
        if not self.responses:
            return ModelResponse(text="done", tool_calls=(), items=())
        return self.responses.pop(0)


class FakeNotifier:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self.findings: list[Finding] = []
        self.summaries: list[tuple[Target, ScanTally]] = []
        self.replies: list[str | None] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send_finding(self, finding: Finding, *, thread_id: str | None = None) -> str:
        self.findings.append(finding)
        self.replies.append(thread_id)
        return '{"success": true, "message_id": 42}'

    async def send_summary(self, target, tally, *, duration=None) -> str:
        self.summaries.append((target, tally))
        return '{"success": true}'


class FakeReportStore:
    def __init__(self) -> None:
        self.reports: dict[str, str] = {}
        self.usage: dict[str, UsageSummary] = {}

    def save_report(self, target: Target, report: str) -> str:
        self.reports[target.url] = report
        return f"/fake/{target.slug}.md"

    def save_usage(self, target: Target, summary: UsageSummary, *, failed: bool = False) -> str:
        self.usage[target.url] = summary
        return f"/fake/{target.slug}_usage.json"


def call(name: str, call_id: str = "c1", **arguments) -> ToolCall:
    """Build a tool call the way a provider would send one."""
    import json

    return ToolCall(call_id=call_id, name=name, raw_arguments=json.dumps(arguments))


def response(*tool_calls: ToolCall, text: str = "") -> ModelResponse:
    return ModelResponse(text=text, tool_calls=tool_calls, items=(f"item:{len(tool_calls)}",))


@pytest.fixture
def target() -> Target:
    return Target("https://example.test/app")


@pytest.fixture
def sandbox() -> FakeSandbox:
    return FakeSandbox()


@pytest.fixture
def registry():
    return default_registry()


@pytest.fixture
def ctx(target, sandbox) -> ToolContext:
    return ToolContext(target=target, sandbox=sandbox, usage=UsageTracker(target.url))
