"""The agent loop: rounds, tool subsets, budgets and usage accounting."""

from conftest import FakeLLM, call, response
from mapta.application import AgentLoop, AgentProfile
from mapta.application.tools import ToolContext, ToolRegistry, tool
from mapta.domain import ModelResponse


@tool
async def ping(ctx: ToolContext) -> str:
    """Return pong."""
    return "pong"


@tool
async def secret(ctx: ToolContext) -> str:
    """A tool the profile hides."""
    return "nope"


def loop_for(llm, profile: AgentProfile) -> AgentLoop:
    return AgentLoop(llm=llm, registry=ToolRegistry.of([ping, secret]), profile=profile)


PROFILE = AgentProfile(name="tester", system_prompt="be brief", max_rounds=10)


async def test_returns_text_when_no_tools_are_requested(ctx):
    llm = FakeLLM([ModelResponse(text="all clear", tool_calls=(), items=())])
    assert await loop_for(llm, PROFILE).run("go", ctx) == "all clear"


async def test_runs_tools_then_returns_the_final_text(ctx):
    llm = FakeLLM(
        [
            response(call("ping")),
            ModelResponse(text="finished", tool_calls=(), items=()),
        ]
    )
    assert await loop_for(llm, PROFILE).run("go", ctx) == "finished"
    # The tool result was fed back before the second request.
    second = llm.requests[1]["transcript"]
    assert second[-1] == {"type": "function_call_output", "call_id": "c1", "output": "pong"}


async def test_parallel_calls_keep_their_order(ctx):
    llm = FakeLLM(
        [
            response(call("ping", "a"), call("ping", "b")),
            ModelResponse(text="done", tool_calls=(), items=()),
        ]
    )
    await loop_for(llm, PROFILE).run("go", ctx)
    outputs = [item["call_id"] for item in llm.requests[1]["transcript"] if isinstance(item, dict)
               and item.get("type") == "function_call_output"]
    assert outputs == ["a", "b"]


async def test_max_rounds_stops_the_loop(ctx):
    llm = FakeLLM([response(call("ping")) for _ in range(10)])
    profile = AgentProfile(name="tester", system_prompt="", max_rounds=2)
    result = await loop_for(llm, profile).run("go", ctx)
    assert result == "[tester] Reached max rounds limit: 2"
    assert len(llm.requests) == 2


async def test_denied_tools_are_not_advertised(ctx):
    llm = FakeLLM([ModelResponse(text="ok", tool_calls=(), items=())])
    profile = AgentProfile(name="tester", system_prompt="", max_rounds=1, denied_tools=("secret",))
    await loop_for(llm, profile).run("go", ctx)
    assert [t.name for t in llm.requests[0]["tools"]] == ["ping"]


async def test_allowed_tools_restrict_the_subset(ctx):
    llm = FakeLLM([ModelResponse(text="ok", tool_calls=(), items=())])
    profile = AgentProfile(name="tester", system_prompt="", max_rounds=1, allowed_tools=("secret",))
    await loop_for(llm, profile).run("go", ctx)
    assert [t.name for t in llm.requests[0]["tools"]] == ["secret"]


async def test_every_round_is_billed_to_the_tracker(ctx):
    llm = FakeLLM(
        [
            ModelResponse(text="", tool_calls=(call("ping"),), items=(), usage={"total_tokens": 5}),
            ModelResponse(text="done", tool_calls=(), items=(), usage={"total_tokens": 3}),
        ]
    )
    await loop_for(llm, PROFILE).run("go", ctx)
    summary = ctx.usage.summary()
    assert summary.total_calls == 2
    assert summary.count_for("tester") == 2


async def test_the_system_prompt_opens_the_transcript(ctx):
    llm = FakeLLM([ModelResponse(text="ok", tool_calls=(), items=())])
    await loop_for(llm, PROFILE).run("scan me", ctx)
    transcript = llm.requests[0]["transcript"]
    assert transcript[0] == {"role": "developer", "text": "be brief"}
    assert transcript[1] == {"role": "user", "text": "scan me"}
