"""Composition root: the single place where concrete adapters are chosen.

Nothing below this module knows which sandbox, model provider or chat
integration is in use; they all receive ports.
"""

from dataclasses import dataclass

from ..application import ScanService, default_registry
from ..application.prompts import MAIN_AGENT, SANDBOX_AGENT, VALIDATOR_AGENT, AgentProfile
from ..config import Settings
from ..infrastructure import (
    FileReportStore,
    MailTmMailboxFactory,
    OpenRouterClient,
    TelegramNotifier,
    build_sandbox_factory,
)

__all__ = ["Container", "build_scan_service"]


def build_profiles(settings: Settings) -> dict[str, AgentProfile]:
    """Apply the operator's prompt and round-budget overrides to each profile."""
    scan = settings.scan
    return {
        MAIN_AGENT.name: MAIN_AGENT.with_(system_prompt=scan.system_prompt),
        SANDBOX_AGENT.name: SANDBOX_AGENT.with_(system_prompt=scan.sandbox_prompt),
        VALIDATOR_AGENT.name: VALIDATOR_AGENT.with_(system_prompt=scan.validator_prompt),
    }


@dataclass(frozen=True, slots=True)
class Container:
    """Wired collaborators, kept together so the CLI can close them cleanly."""

    settings: Settings
    scans: ScanService
    llm: OpenRouterClient
    telegram: TelegramNotifier | None

    async def aclose(self) -> None:
        await self.llm.aclose()
        if self.telegram is not None:
            await self.telegram.aclose()


def build_scan_service(settings: Settings) -> Container:
    """Assemble every adapter the scan service needs."""
    telegram = TelegramNotifier(settings.telegram) if settings.telegram.enabled else None
    llm = OpenRouterClient(settings.llm)
    service = ScanService(
        llm=llm,
        registry=default_registry(),
        sandbox_factory=build_sandbox_factory(settings.sandbox),
        reports=FileReportStore(settings.reports),
        max_rounds=settings.scan.max_rounds,
        profiles=build_profiles(settings),
        notifier=telegram,
        mailbox_factory=MailTmMailboxFactory() if settings.scan.mailbox_enabled else None,
    )
    return Container(settings=settings, scans=service, llm=llm, telegram=telegram)
