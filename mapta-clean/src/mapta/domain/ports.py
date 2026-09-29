"""Ports: the interfaces the application layer depends on.

Each protocol describes a capability MAPTA needs, never how it is provided.
Concrete adapters live in ``mapta.infrastructure`` and are wired in
``mapta.interface.container``.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import (
    CommandResult,
    Finding,
    JSONObject,
    ModelResponse,
    ScanTally,
    Target,
    ToolSpec,
    TranscriptItem,
    UsageSummary,
)


@runtime_checkable
class Sandbox(Protocol):
    """An isolated execution environment the agents drive."""

    async def write_file(self, path: str, content: str) -> None:
        """Create or overwrite ``path`` inside the sandbox."""
        ...

    async def run(
        self, command: str, *, timeout: float | None = None, user: str = "root"
    ) -> CommandResult:
        """Execute a shell command and return its captured output."""
        ...

    async def aclose(self) -> None:
        """Release the sandbox. Must be safe to call more than once."""
        ...


class SandboxFactory(Protocol):
    """Creates one fresh sandbox per scan."""

    async def create(self) -> Sandbox:
        ...


class LLMClient(Protocol):
    """A tool-calling chat model.

    The adapter owns both the wire format and the transcript encoding, so the
    agent loop never learns which provider is answering.
    """

    async def respond(
        self,
        *,
        transcript: Sequence[TranscriptItem],
        tools: Sequence[ToolSpec],
        metadata: JSONObject | None = None,
    ) -> ModelResponse:
        """Issue one round-trip and return the normalised response."""
        ...

    def developer_message(self, text: str) -> TranscriptItem:
        """Build the provider-specific system/developer transcript item."""
        ...

    def user_message(self, text: str) -> TranscriptItem:
        """Build the provider-specific user transcript item."""
        ...

    def tool_output(self, call_id: str, output: str) -> TranscriptItem:
        """Build the provider-specific tool-result transcript item."""
        ...


class Notifier(Protocol):
    """Somewhere to push findings while a scan is running."""

    @property
    def enabled(self) -> bool:
        """False when the integration is not configured; tools say so politely."""
        ...

    async def send_finding(self, finding: Finding, *, thread_id: str | None = None) -> str:
        """Publish a single finding. Returns a human-readable status."""
        ...

    async def send_summary(
        self, target: Target, tally: ScanTally, *, duration: str | None = None
    ) -> str:
        """Publish an end-of-scan summary. Returns a human-readable status."""
        ...


class MailboxFactory(Protocol):
    """Creates one mailbox per scan, so tokens never leak between targets."""

    def __call__(self) -> Mailbox:
        ...


class Mailbox(Protocol):
    """Disposable inbox used to receive activation mails, resets, credentials."""

    async def create_account(
        self, address: str | None = None, password: str | None = None
    ) -> JSONObject:
        ...

    async def remember_token(self, address: str, token: str) -> None:
        ...

    async def known_addresses(self) -> list[str]:
        ...

    async def list_messages(self, address: str, limit: int = 50) -> list[JSONObject]:
        ...

    async def get_message(self, address: str, message_id: str) -> JSONObject:
        ...

    async def aclose(self) -> None:
        ...


class ReportStore(Protocol):
    """Where reports and usage logs are persisted."""

    def save_report(self, target: Target, report: str) -> str:
        """Write the Markdown report and return its path."""
        ...

    def save_usage(self, target: Target, summary: UsageSummary, *, failed: bool = False) -> str:
        """Write the usage log and return its path."""
        ...


class ProgressReporter(Protocol):
    """User-facing progress output, kept out of the application layer."""

    def scan_started(self, target: Target) -> None: ...
    def scan_finished(self, outcome: ScanOutcomeLike) -> None: ...
    def round_started(self, agent: str, target: Target, tool_calls: int) -> None: ...
    def note(self, message: str) -> None: ...


class ScanOutcomeLike(Protocol):
    """Structural view of a scan outcome, so the port stays import-light."""

    @property
    def target(self) -> Target: ...
    @property
    def succeeded(self) -> bool: ...
