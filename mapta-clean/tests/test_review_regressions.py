"""Regressions for the findings of the high-effort review of `src/`.

Each test names the failure it prevents, so a future change that reintroduces
one fails loudly rather than quietly.
"""


import pytest

from conftest import FakeLLM, FakeReportStore, FakeSandboxFactory
from mapta.application import ScanRequest, ScanService, decode_tool_arguments
from mapta.application.prompts import MAIN_AGENT, SANDBOX_AGENT, VALIDATOR_AGENT
from mapta.application.tools import default_registry
from mapta.config import ReportSettings, Settings
from mapta.domain import (
    ConfigurationError,
    ModelResponse,
    ScanStatus,
    ScanTally,
    Target,
)
from mapta.infrastructure.notifications import build_summary_message
from mapta.infrastructure.storage import FileReportStore

PROFILES = {
    MAIN_AGENT.name: MAIN_AGENT,
    SANDBOX_AGENT.name: SANDBOX_AGENT,
    VALIDATOR_AGENT.name: VALIDATOR_AGENT,
}


def service(llm, *, factory=None, reports=None) -> ScanService:
    return ScanService(
        llm=llm,
        registry=default_registry(),
        sandbox_factory=factory or FakeSandboxFactory(),
        reports=reports or FakeReportStore(),
        profiles=PROFILES,
    )


# --- 1: a failed write must not cancel the sibling scans -----------------


class ExplodingStore(FakeReportStore):
    def save_report(self, target, report):
        raise OSError("read-only file system")


async def test_a_failed_report_write_does_not_cancel_other_targets():
    llm = FakeLLM()
    request = ScanRequest(
        targets=(Target("https://a.test"), Target("https://b.test")),
        prompt_template="scan {target_url}",
    )
    outcomes = await service(llm, reports=ExplodingStore()).run(request)

    # Both scans finish and report the problem; neither is cancelled.
    assert len(outcomes) == 2
    assert {o.status for o in outcomes} == {ScanStatus.ERROR}
    assert all("read-only file system" in (o.error or "") for o in outcomes)


async def test_a_failed_report_write_keeps_the_text_in_memory():
    llm = FakeLLM([ModelResponse(text="# Report", tool_calls=(), items=())])
    outcome = await service(llm, reports=ExplodingStore()).scan(Target("https://a.test"), "go")
    assert outcome.report == "# Report"


# --- 2: teardown failures must not discard a finished report -------------


class UncloseableSandboxFactory(FakeSandboxFactory):
    async def create(self):
        sandbox = await super().create()

        async def boom() -> None:
            raise TimeoutError("docker rm timed out")

        sandbox.aclose = boom  # type: ignore[method-assign]
        return sandbox


async def test_a_sandbox_that_fails_to_close_does_not_lose_the_report():
    reports = FakeReportStore()
    llm = FakeLLM([ModelResponse(text="# Report", tool_calls=(), items=())])
    outcome = await service(
        llm, factory=UncloseableSandboxFactory(), reports=reports
    ).scan(Target("https://a.test"), "go")

    assert outcome.status is ScanStatus.COMPLETED
    assert outcome.report == "# Report"
    assert reports.reports["https://a.test"] == "# Report"


# --- 3: a truncated value is never completed -----------------------------


def test_a_truncated_command_is_refused_rather_than_shortened():
    with pytest.raises(ValueError):
        decode_tool_arguments('{"command": "rm -rf /tmp/scratch/sub')


# --- 8: a bad number in .env is a configuration error --------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_TOKENS", "16k"),
        ("MAX_ROUNDS", "many"),
        ("MAX_CONCURRENCY", "2.5"),
        ("SANDBOX_TIMEOUT", "two minutes"),
        ("SANDBOX_MAX_TIMEOUT", ""),
    ],
)
def test_an_unparseable_number_is_a_configuration_error(name, value):
    env = {"OPENROUTER_API_KEY": "sk-or", name: value}
    if value == "":  # an empty value falls back to the default instead
        assert Settings.from_env(env) is not None
        return
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env(env)


def test_zero_max_tokens_means_the_provider_default():
    settings = Settings.from_env({"OPENROUTER_API_KEY": "sk-or", "MAX_TOKENS": "0"})
    assert settings.llm.max_tokens is None


# --- 9: never a clean tick next to a non-zero finding count --------------


def test_findings_without_a_severity_are_not_reported_as_clean():
    message = build_summary_message(Target("https://a.test"), ScanTally(total=5), None)
    assert "No issues found" not in message
    assert "<b>Total findings:</b> 5" in message


def test_a_genuinely_clean_scan_still_reads_as_clean():
    message = build_summary_message(Target("https://a.test"), ScanTally(total=0), None)
    assert "No issues found" in message


def test_a_severity_still_wins_over_the_neutral_status():
    message = build_summary_message(Target("https://a.test"), ScanTally(total=2, high=2), None)
    assert "High risk issues found" in message


# --- 10: colliding slugs must not overwrite each other -------------------


def test_targets_that_slug_alike_get_distinct_reports(tmp_path):
    store = FileReportStore(ReportSettings(output_dir=str(tmp_path)))
    first = store.save_report(Target("http://example.com"), "first")
    second = store.save_report(Target("https://example.com"), "second")

    assert first != second
    assert (tmp_path / "example.com.md").read_text() == "first"
    assert (tmp_path / "example.com-2.md").read_text() == "second"


# --- 11: the SDK connection pool is released -----------------------------


async def test_the_container_closes_the_llm_client():
    """The SDK client owns a connection pool that nothing else releases."""
    from mapta.interface.container import build_scan_service

    container = build_scan_service(Settings.from_env({"OPENROUTER_API_KEY": "sk-or"}))
    assert container.llm.client is not None  # force the lazy client into existence

    await container.aclose()

    assert container.llm._client is None
    await container.aclose()  # and closing twice is safe
