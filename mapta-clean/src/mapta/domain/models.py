"""Pure data structures shared by every layer.

Nothing here imports from ``application`` or ``infrastructure``: these types are
the vocabulary the other layers speak.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

type JSONObject = dict[str, Any]

#: What a provider reports for one call: a token dict, a scalar, or nothing.
type UsageValue = JSONObject | str | int | float | None

#: One entry of a model conversation. The concrete shape belongs to the LLM
#: adapter; the rest of the code only ever appends items and passes them back.
type TranscriptItem = Any


class Severity(StrEnum):
    """Severity of a reported finding, ordered from worst to mildest."""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"

    @classmethod
    def parse(cls, value: str) -> Self:
        """Accept any casing, falling back to ``INFO`` for unknown values."""
        wanted = value.strip().casefold()
        for member in cls:
            if member.value.casefold() == wanted:
                return member  # type: ignore[return-value]
        return cls.INFO  # type: ignore[return-value]

    @property
    def colour(self) -> str:
        """Hex colour used by chat integrations."""
        return _SEVERITY_COLOURS[self]

    @property
    def emoji(self) -> str:
        """Emoji used by chat integrations."""
        return _SEVERITY_EMOJIS[self]


_SEVERITY_COLOURS: dict[Severity, str] = {
    Severity.CRITICAL: "#FF0000",
    Severity.HIGH: "#FF6600",
    Severity.MEDIUM: "#FFB84D",
    Severity.LOW: "#FFCC00",
    Severity.INFO: "#0099FF",
}

_SEVERITY_EMOJIS: dict[Severity, str] = {
    Severity.CRITICAL: "🚨",
    Severity.HIGH: "⚠️",
    Severity.MEDIUM: "⚡",
    Severity.LOW: "📝",
    Severity.INFO: "ℹ️",  # noqa: RUF001 - emoji, not a latin "i"
}


@dataclass(frozen=True, slots=True)
class Target:
    """A single authorised scan target."""

    url: str

    @property
    def host(self) -> str:
        """Hostname without scheme or path, e.g. ``example.com``."""
        return self.url.split("://", 1)[-1].split("/", 1)[0]

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier derived from the URL."""
        raw = self.url.split("://", 1)[-1]
        cleaned = "".join(c if c.isalnum() or c in "-._" else "_" for c in raw)
        return cleaned.strip("_") or "target"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of a command executed inside a sandbox."""

    stdout: str
    stderr: str
    exit_code: int

    def render(self, *, max_chars: int | None = None) -> str:
        """Format the result the way agents expect to read it."""
        text = f"Exit code: {self.exit_code}\n\nSTDOUT\n{self.stdout}\n\nSTDERR\n{self.stderr}"
        if max_chars is not None and len(text) > max_chars:
            return f"{text[:max_chars]}\n...[OUTPUT TRUNCATED - EXCEEDED {max_chars} CHARACTERS]"
        return text


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What a tool advertises, before any provider's wire format is applied.

    Chat Completions nests this under a ``function`` key, the Responses API
    keeps it flat; translating is the LLM adapter's job, not the registry's.
    """

    name: str
    description: str
    parameters: JSONObject
    strict: bool = True

    def as_dict(self) -> JSONObject:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    call_id: str
    name: str
    raw_arguments: str | JSONObject | None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The answer sent back to the model for one :class:`ToolCall`."""

    call_id: str
    output: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalised view of one provider response."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    items: tuple[TranscriptItem, ...]
    usage: JSONObject | None = None
    response_id: str | None = None
    #: Provider stop reason; "length" means the answer was cut off.
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """Whether the model ran out of output budget mid-answer."""
        return self.finish_reason == "length"


@dataclass(frozen=True, slots=True)
class Finding:
    """A vulnerability the agent claims to have confirmed."""

    vulnerability_type: str
    severity: Severity
    target_url: str
    description: str
    evidence: str | None = None
    recommendation: str | None = None


@dataclass(frozen=True, slots=True)
class ScanTally:
    """Counts reported at the end of a scan, per severity."""

    total: int
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0

    @property
    def headline_severity(self) -> Severity | None:
        """Worst severity present, or ``None`` when the scan was clean."""
        for severity, count in (
            (Severity.CRITICAL, self.critical),
            (Severity.HIGH, self.high),
            (Severity.MEDIUM, self.medium),
            (Severity.LOW, self.low),
        ):
            if count > 0:
                return severity
        return None


class ScanStatus(StrEnum):
    """Terminal state of a single-target scan."""

    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ScanOutcome:
    """Everything a caller needs to know about one finished scan."""

    target: Target
    status: ScanStatus
    report: str | None = None
    report_path: str | None = None
    usage_path: str | None = None
    error: str | None = None
    usage: UsageSummary | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ScanStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One billed model call."""

    timestamp: datetime
    agent: str
    target_url: str
    usage: UsageValue


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Aggregated usage for a scan."""

    duration: str
    records: tuple[UsageRecord, ...] = field(default_factory=tuple)

    def count_for(self, agent: str) -> int:
        return sum(1 for record in self.records if record.agent == agent)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    def as_dict(self) -> JSONObject:
        """JSON-serialisable view, written verbatim to the usage log."""
        by_agent: dict[str, list[JSONObject]] = {}
        for record in self.records:
            by_agent.setdefault(record.agent, []).append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "target_url": record.target_url,
                    "usage": record.usage,
                }
            )
        return {
            "scan_duration": self.duration,
            "total_calls": self.total_calls,
            "calls_per_agent": {agent: len(items) for agent, items in by_agent.items()},
            "usage_per_agent": by_agent,
        }


def utcnow() -> datetime:
    """Timezone-aware ``now``, isolated here so tests can monkeypatch it."""
    return datetime.now(UTC)


def targets_from_lines(lines: Sequence[str]) -> list[Target]:
    """Parse a targets file body: one URL per line, ``#`` starts a comment."""
    targets: list[Target] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            targets.append(Target(stripped))
    return targets
