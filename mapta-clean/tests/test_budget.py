"""The scan round budget, and the sandbox orphan reaper.

Both fix defects the XBOW benchmark run exposed: `MAX_ROUNDS` bounded only the
main agent, and a SIGKILLed scan left its container running.
"""


from conftest import FakeLLM, FakeReportStore, FakeSandboxFactory, call, response
from mapta.application import ScanService
from mapta.application.agent import AgentLoop
from mapta.application.budget import RoundBudget
from mapta.application.prompts import MAIN_AGENT, SANDBOX_AGENT, VALIDATOR_AGENT, AgentProfile
from mapta.application.tools import ToolContext, ToolRegistry, tool
from mapta.domain import ModelResponse, Target

PROFILES = {
    MAIN_AGENT.name: MAIN_AGENT,
    SANDBOX_AGENT.name: SANDBOX_AGENT,
    VALIDATOR_AGENT.name: VALIDATOR_AGENT,
}


# --- the counter itself ---------------------------------------------------


def test_a_budget_of_zero_is_unlimited():
    budget = RoundBudget(total=0)
    assert all(budget.claim() for _ in range(1000))
    assert budget.exhausted is False
    assert budget.remaining is None


def test_a_budget_runs_out_exactly_once_spent():
    budget = RoundBudget(total=3)
    assert [budget.claim() for _ in range(5)] == [True, True, True, False, False]
    assert budget.exhausted is True
    assert budget.remaining == 0


# --- the loop honours it --------------------------------------------------


@tool
async def ping(ctx: ToolContext) -> str:
    """Return pong."""
    return "pong"


PROFILE = AgentProfile(name="tester", system_prompt="", max_rounds=0)


async def test_the_loop_stops_when_the_scan_budget_is_spent(ctx):
    ctx.budget = RoundBudget(total=3)
    llm = FakeLLM([response(call("ping")) for _ in range(20)])
    loop = AgentLoop(llm=llm, registry=ToolRegistry.of([ping]), profile=PROFILE)

    result = await loop.run("go", ctx)

    assert "round budget of 3 is exhausted" in result
    assert len(llm.requests) == 3


async def test_an_unlimited_budget_leaves_the_profile_cap_in_charge(ctx):
    ctx.budget = RoundBudget(total=0)
    llm = FakeLLM([response(call("ping")) for _ in range(20)])
    profile = AgentProfile(name="tester", system_prompt="", max_rounds=2)
    loop = AgentLoop(llm=llm, registry=ToolRegistry.of([ping]), profile=profile)
    result = await loop.run("go", ctx)
    assert "max rounds limit: 2" in result


# --- the regression: nested agents used to be unbounded -------------------


async def test_the_budget_covers_nested_agents_too():
    """A budget of 5 used to authorise 5 x 100 nested rounds."""
    llm = FakeLLM(
        # main delegates, then the nested agent loops forever if allowed to.
        [response(call("sandbox_agent", instruction="dig"))]
        + [response(call("sandbox_run_command", command="ls")) for _ in range(50)]
    )
    from mapta.application.tools import default_registry

    service = ScanService(
        llm=llm,
        registry=default_registry(),
        sandbox_factory=FakeSandboxFactory(),
        reports=FakeReportStore(),
        max_rounds=5,
        profiles=PROFILES,
    )
    await service.scan(Target("https://a.test"), "go")

    # Five model calls in total, shared between the main and the nested agent.
    assert len(llm.requests) == 5


async def test_the_budget_is_per_scan_not_global():
    llm = FakeLLM([ModelResponse(text="done", tool_calls=(), items=()) for _ in range(10)])
    from mapta.application.tools import default_registry

    service = ScanService(
        llm=llm, registry=default_registry(), sandbox_factory=FakeSandboxFactory(),
        reports=FakeReportStore(), max_rounds=2, profiles=PROFILES,
    )
    first = await service.scan(Target("https://a.test"), "go")
    second = await service.scan(Target("https://b.test"), "go")
    assert first.succeeded and second.succeeded
