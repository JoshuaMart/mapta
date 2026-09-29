"""A failing tool must produce a model-readable result, never an exception."""

import json

import pytest

from mapta.application.tools import ToolContext, ToolRegistry, tool
from mapta.domain import ToolCall


@tool
async def boom(ctx: ToolContext) -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


@tool
async def echo(ctx: ToolContext, value: str) -> str:
    """Echo a value.

    Args:
        value: What to echo.
    """
    return value


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry.of([boom, echo])


async def test_successful_call(registry, ctx):
    result = await registry.execute(ctx, ToolCall("1", "echo", '{"value": "hi"}'))
    assert result.output == "hi"
    assert result.call_id == "1"


async def test_tool_exception_becomes_a_result(registry, ctx):
    result = await registry.execute(ctx, ToolCall("2", "boom", "{}"))
    assert json.loads(result.output) == {
        "error": "RuntimeError",
        "detail": "kaboom",
        "args": {},
    }


async def test_unknown_tool_is_reported(registry, ctx):
    result = await registry.execute(ctx, ToolCall("3", "nope", "{}"))
    assert json.loads(result.output)["error"] == "unknown_tool"


async def test_malformed_arguments_are_reported(registry, ctx):
    result = await registry.execute(ctx, ToolCall("4", "echo", "value=hi"))
    payload = json.loads(result.output)
    assert payload["error"] == "invalid_tool_arguments"
    # The expected types travel with the error so one retry can fix the call.
    assert payload["expected_parameters"] == {"value": "string"}


async def test_a_truncated_payload_is_named_as_such(registry, ctx):
    """Calling this "invalid JSON" makes the model re-send the same long call."""
    # Cut inside the key itself: nothing complete is left to salvage.
    result = await registry.execute(ctx, ToolCall("4b", "echo", '{"val'))
    assert json.loads(result.output)["error"] == "truncated_tool_arguments"


async def test_a_truncated_optional_argument_is_recovered(ctx):
    """The failure seen in the wild: cut off at `"max_rounds": ` with no value."""

    @tool
    async def delegate(ctx: ToolContext, instruction: str, max_rounds: int = 100) -> str:
        """Delegate.

        Args:
            instruction: What to do.
            max_rounds: How many rounds.
        """
        return f"{instruction}|{max_rounds}"

    raw = '{"instruction": "map the attack surface", "max_rounds": '
    result = await ToolRegistry.of([delegate]).execute(ctx, ToolCall("4c", "delegate", raw))
    assert result.output == "map the attack surface|100"


async def test_a_truncated_required_argument_is_refused_clearly(ctx):
    @tool
    async def delegate(ctx: ToolContext, instruction: str, detail: str) -> str:
        """Delegate.

        Args:
            instruction: What to do.
            detail: Extra detail.
        """
        return instruction

    raw = '{"instruction": "map the surface", "detail": '
    result = await ToolRegistry.of([delegate]).execute(ctx, ToolCall("4d", "delegate", raw))
    payload = json.loads(result.output)
    assert payload["error"] == "invalid_tool_arguments"
    assert "`detail` is required but was not provided" in payload["detail"]


async def test_null_for_a_defaulted_argument_is_dropped(ctx):
    @tool
    async def greet(ctx: ToolContext, name: str = "world") -> str:
        """Greet.

        Args:
            name: Who to greet.
        """
        return f"hello {name}"

    result = await ToolRegistry.of([greet]).execute(ctx, ToolCall("5", "greet", '{"name": null}'))
    assert result.output == "hello world"


def test_duplicate_names_are_refused():
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry.of([echo, echo])


def test_specs_honour_include_and_exclude(registry):
    assert [s.name for s in registry.specs(include=["echo"])] == ["echo"]
    assert [s.name for s in registry.specs(exclude=["echo"])] == ["boom"]
