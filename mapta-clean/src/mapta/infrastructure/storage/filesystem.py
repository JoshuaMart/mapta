"""Filesystem report store: Markdown reports plus a JSON usage log per scan."""

import json
import logging
from pathlib import Path

from ...config import ReportSettings
from ...domain import Target, UsageSummary, utcnow

__all__ = ["FileReportStore"]

logger = logging.getLogger(__name__)


class FileReportStore:
    """Implements the :class:`~mapta.domain.ports.ReportStore` port."""

    def __init__(self, settings: ReportSettings) -> None:
        self._root = Path(settings.output_dir)
        self._claimed: set[str] = set()

    @property
    def root(self) -> Path:
        return self._root

    def save_report(self, target: Target, report: str) -> str:
        return self._write(self._claim(f"{target.slug}.md"), report)

    def save_usage(self, target: Target, summary: UsageSummary, *, failed: bool = False) -> str:
        stamp = utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = "_failed" if failed else ""
        name = f"{target.slug}{suffix}_usage_log_{stamp}.json"
        return self._write(self._claim(name), json.dumps(summary.as_dict(), indent=2, default=str))

    def _claim(self, name: str) -> Path:
        """Reserve a filename, suffixing it rather than overwriting a sibling.

        Distinct targets can share a slug -- ``http://x.com`` and
        ``https://x.com``, or ``x.com/a/b`` and ``x.com/a_b`` -- and they run
        concurrently, so one would otherwise silently replace the other.
        """
        stem, _, extension = name.rpartition(".")
        candidate = name
        counter = 2
        while candidate in self._claimed:
            candidate = f"{stem}-{counter}.{extension}"
            counter += 1
        self._claimed.add(candidate)
        return self._root / candidate

    def _write(self, path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote %s", path)
        return str(path)
