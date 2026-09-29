"""The scan use case: isolation per target, persistence, failure handling."""

from conftest import FakeLLM, FakeReportStore, FakeSandboxFactory, call, response
from mapta.application import ScanRequest, ScanService
from mapta.application.prompts import MAIN_AGENT, SANDBOX_AGENT, VALIDATOR_AGENT
from mapta.application.tools import ToolRegistry, default_registry
from mapta.domain import ModelResponse, ScanStatus, Target

PROFILES = {
    MAIN_AGENT.name: MAIN_AGENT,
    SANDBOX_AGENT.name: SANDBOX_AGENT,
    VALIDATOR_AGENT.name: VALIDATOR_AGENT,
}


def service(llm, *, registry: ToolRegistry | None = None, factory=None, reports=None):
    return ScanService(
        llm=llm,
        registry=registry or default_registry(),
        sandbox_factory=factory or FakeSandboxFactory(),
        reports=reports or FakeReportStore(),
        profiles=PROFILES,
    )


async def test_report_is_persisted(target):
    reports = FakeReportStore()
    llm = FakeLLM([ModelResponse(text="# Report", tool_calls=(), items=())])
    outcome = await service(llm, reports=reports).scan(target, "scan it")

    assert outcome.status is ScanStatus.COMPLETED
    assert reports.reports[target.url] == "# Report"
    assert outcome.report_path.endswith(".md")
    assert outcome.usage_path is not None


async def test_sandbox_is_closed_even_on_failure(target):
    factory = FakeSandboxFactory()

    class Exploding(FakeLLM):
        async def respond(self, **kwargs):
            raise RuntimeError("provider down")

    outcome = await service(Exploding(), factory=factory).scan(target, "scan it")

    assert outcome.status is ScanStatus.ERROR
    assert outcome.error == "provider down"
    assert factory.created[0].closed is True
    # Billed tokens are kept even when the scan fails.
    assert outcome.usage_path is not None


async def test_each_target_gets_its_own_sandbox():
    factory = FakeSandboxFactory()
    llm = FakeLLM()
    request = ScanRequest(
        targets=(Target("https://a.test"), Target("https://b.test")),
        prompt_template="scan {target_url}",
    )
    outcomes = await service(llm, factory=factory).run(request)

    assert len(factory.created) == 2
    assert all(sandbox.closed for sandbox in factory.created)
    assert all(outcome.succeeded for outcome in outcomes)


async def test_one_failing_target_does_not_cancel_the_others():
    class Flaky(FakeLLM):
        async def respond(self, *, transcript, tools, metadata=None):
            if "b.test" in str(transcript):
                raise RuntimeError("nope")
            return ModelResponse(text="ok", tool_calls=(), items=())

    request = ScanRequest(
        targets=(Target("https://a.test"), Target("https://b.test")),
        prompt_template="scan {target_url}",
    )
    outcomes = await service(Flaky()).run(request)

    by_host = {outcome.target.host: outcome.status for outcome in outcomes}
    assert by_host == {"a.test": ScanStatus.COMPLETED, "b.test": ScanStatus.ERROR}


async def test_prompt_template_is_filled_per_target():
    llm = FakeLLM()
    request = ScanRequest(targets=(Target("https://a.test"),), prompt_template="scan {target_url}")
    await service(llm).run(request)
    assert llm.requests[0]["transcript"][1]["text"] == "scan https://a.test"


async def test_nested_agent_runs_through_the_runner(target):
    """The main agent delegates; the nested agent sees only sandbox tools."""
    llm = FakeLLM(
        [
            response(call("sandbox_agent", instruction="look around")),  # main agent
            ModelResponse(text="nested answer", tool_calls=(), items=()),  # nested agent
            ModelResponse(text="# Final", tool_calls=(), items=()),  # main agent again
        ]
    )
    outcome = await service(llm).scan(target, "scan it")

    assert outcome.report == "# Final"
    nested_tools = {t.name for t in llm.requests[1]["tools"]}
    assert nested_tools == {"sandbox_run_command", "sandbox_run_python"}


async def test_main_agent_cannot_reach_the_raw_sandbox_tools(target):
    llm = FakeLLM([ModelResponse(text="ok", tool_calls=(), items=())])
    await service(llm).scan(target, "scan it")
    advertised = {t.name for t in llm.requests[0]["tools"]}
    assert advertised.isdisjoint({"sandbox_run_command", "sandbox_run_python"})
    assert "sandbox_agent" in advertised
