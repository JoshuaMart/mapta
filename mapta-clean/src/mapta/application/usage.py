"""Per-scan token accounting.

The tracker is created by the scan service and handed to the agents through the
tool context, so parallel scans never share state.
"""

from dataclasses import dataclass, field
from typing import Any, cast

from ..domain import UsageRecord, UsageSummary, UsageValue, utcnow

__all__ = ["UsageTracker", "normalise_usage"]


def normalise_usage(usage: Any) -> UsageValue:
    """Turn an SDK usage object into plain JSON-serialisable data.

    Without this, usage objects fall through ``json.dump(default=str)`` and land
    in the log as Python reprs that no tool can read back.
    """
    if usage is None or isinstance(usage, dict | int | float | str):
        return usage
    for attribute in ("model_dump", "dict", "to_dict"):
        if callable(method := getattr(usage, attribute, None)):
            try:
                return cast(UsageValue, method())
            except Exception:
                continue
    return str(usage)


@dataclass(slots=True)
class UsageTracker:
    """Collects one :class:`UsageRecord` per model call."""

    target_url: str = ""
    _records: list[UsageRecord] = field(default_factory=list)
    _started_at: Any = field(default_factory=utcnow)

    def record(self, agent: str, usage: Any) -> None:
        self._records.append(
            UsageRecord(
                timestamp=utcnow(),
                agent=agent,
                target_url=self.target_url,
                usage=normalise_usage(usage),
            )
        )

    def summary(self) -> UsageSummary:
        return UsageSummary(
            duration=str(utcnow() - self._started_at),
            records=tuple(self._records),
        )
