"""A round budget shared by every agent taking part in one scan.

``max_rounds`` used to bound the main agent only, while the nested agents kept
their own hard-coded caps -- so a budget of 40 authorised 40 x 100 sandbox
rounds and bounded nothing. One counter, shared through the tool context, is
what makes the limit mean what an operator expects.
"""

from dataclasses import dataclass

__all__ = ["RoundBudget"]


@dataclass(slots=True)
class RoundBudget:
    """Model rounds a scan may still spend. ``total=0`` means unlimited."""

    total: int = 0
    used: int = 0

    def claim(self) -> bool:
        """Take one round, or return False when the scan has spent its budget."""
        if self.total and self.used >= self.total:
            return False
        self.used += 1
        return True

    @property
    def exhausted(self) -> bool:
        return bool(self.total) and self.used >= self.total

    @property
    def remaining(self) -> int | None:
        return None if not self.total else max(0, self.total - self.used)
