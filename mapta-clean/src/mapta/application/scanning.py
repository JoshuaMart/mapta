"""The scan use case: one target in, one report out.

This is the only place that knows the order of operations for a scan. It owns
no I/O: sandboxes, mailboxes, notifiers and storage all arrive as ports.
"""

import asyncio
import logging
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from ..domain import (
    LLMClient,
    Mailbox,
    MailboxFactory,
    Notifier,
    ReportStore,
    SandboxFactory,
    ScanOutcome,
    ScanStatus,
    Target,
)
from .agent import AgentLoop, AgentRunner
from .budget import RoundBudget
from .prompts import MAIN_AGENT, SANDBOX_AGENT, VALIDATOR_AGENT, AgentProfile
from .tools import ToolContext, ToolRegistry
from .usage import UsageTracker

__all__ = ["ScanRequest", "ScanService", "summarise"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScanRequest:
    """What the operator asked for."""

    targets: tuple[Target, ...]
    prompt_template: str
    max_concurrency: int = 0  # 0 = no limit

    def instruction_for(self, target: Target) -> str:
        return self.prompt_template.format(target_url=target.url)


@dataclass(slots=True)
class ScanService:
    """Runs scans, sequentially or in parallel, isolating one sandbox per target."""

    llm: LLMClient
    registry: ToolRegistry
    sandbox_factory: SandboxFactory
    reports: ReportStore
    #: Model rounds one scan may spend across all its agents (0 = unlimited).
    max_rounds: int = 0
    profiles: dict[str, AgentProfile] = field(
        default_factory=lambda: {
            MAIN_AGENT.name: MAIN_AGENT,
            SANDBOX_AGENT.name: SANDBOX_AGENT,
            VALIDATOR_AGENT.name: VALIDATOR_AGENT,
        }
    )
    notifier: Notifier | None = None
    mailbox_factory: MailboxFactory | None = None

    async def run(self, request: ScanRequest) -> list[ScanOutcome]:
        """Scan every target, never letting one failure cancel the others."""
        limit = asyncio.Semaphore(request.max_concurrency) if request.max_concurrency else None

        async def guarded(target: Target) -> ScanOutcome:
            try:
                if limit is None:
                    return await self.scan(target, request.instruction_for(target))
                async with limit:
                    return await self.scan(target, request.instruction_for(target))
            except Exception as exc:
                # A task that raised would cancel the whole TaskGroup and with it
                # every sibling scan, so nothing is allowed to escape here.
                logger.exception("Scan raised for %s", target.url)
                return ScanOutcome(target=target, status=ScanStatus.ERROR, error=str(exc))

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(guarded(target)) for target in request.targets]
        return [task.result() for task in tasks]

    async def scan(self, target: Target, instruction: str) -> ScanOutcome:
        """Run one target end to end and persist its report and usage log."""
        logger.info("Starting scan for %s", target.url)
        tracker = UsageTracker(target_url=target.url)

        try:
            report = await self._drive_agents(target, instruction, tracker)
        except Exception as exc:
            logger.exception("Scan failed for %s", target.url)
            # The tokens were already billed, so keep the usage log even on failure.
            return ScanOutcome(
                target=target,
                status=ScanStatus.ERROR,
                error=str(exc),
                usage_path=self._save_usage(target, tracker, failed=True),
                usage=tracker.summary(),
            )

        try:
            report_path = self.reports.save_report(
                target, report or "[no report produced by the agent]"
            )
        except Exception as exc:
            # The scan itself succeeded; say so, and keep the text in memory so
            # the caller can still do something with it.
            logger.exception("Could not write the report for %s", target.url)
            return ScanOutcome(
                target=target,
                status=ScanStatus.ERROR,
                report=report,
                error=f"the report could not be written: {exc}",
                usage_path=self._save_usage(target, tracker, failed=True),
                usage=tracker.summary(),
            )

        usage_path = self._save_usage(target, tracker)
        logger.info("Scan completed for %s -> %s", target.url, report_path)
        return ScanOutcome(
            target=target,
            status=ScanStatus.COMPLETED,
            report=report,
            report_path=report_path,
            usage_path=usage_path,
            usage=tracker.summary(),
        )

    async def _drive_agents(
        self, target: Target, instruction: str, tracker: UsageTracker
    ) -> str:
        """Run the main agent with a sandbox and mailbox scoped to this target."""
        stack = AsyncExitStack()
        sandbox = await self.sandbox_factory.create()
        stack.push_async_callback(sandbox.aclose)

        mailbox: Mailbox | None = None
        if self.mailbox_factory is not None:
            mailbox = self.mailbox_factory()
            stack.push_async_callback(mailbox.aclose)

        ctx = ToolContext(
            target=target,
            sandbox=sandbox,
            usage=tracker,
            budget=RoundBudget(total=self.max_rounds),
            mailbox=mailbox,
            notifier=self.notifier,
        )
        ctx.run_sub_agent = AgentRunner(
            llm=self.llm, registry=self.registry, profiles=self.profiles
        ).bound_to(ctx)
        loop = AgentLoop(
            llm=self.llm, registry=self.registry, profile=self.profiles[MAIN_AGENT.name]
        )

        try:
            return await loop.run(
                instruction,
                ctx,
                metadata={"site_name": target.host, "target_url": target.url},
            )
        finally:
            # A sandbox that fails to shut down must not discard a finished report.
            try:
                await stack.aclose()
            except Exception:
                logger.warning("Teardown failed for %s", target.url, exc_info=True)

    def _save_usage(
        self, target: Target, tracker: UsageTracker, *, failed: bool = False
    ) -> str | None:
        try:
            return self.reports.save_usage(target, tracker.summary(), failed=failed)
        except Exception as exc:  # never lose a report because the log could not be written
            logger.warning("Could not save usage data for %s: %s", target.url, exc)
            return None


def summarise(outcomes: Sequence[ScanOutcome]) -> dict[str, int]:
    """Aggregate counts used by the CLI's closing report."""
    completed = sum(1 for outcome in outcomes if outcome.succeeded)
    calls = sum(outcome.usage.total_calls for outcome in outcomes if outcome.usage)
    return {
        "targets": len(outcomes),
        "completed": completed,
        "failed": len(outcomes) - completed,
        "model_calls": calls,
    }
