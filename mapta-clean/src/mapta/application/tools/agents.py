"""Tools that delegate work to a nested agent loop."""

from .base import ToolContext, tool

__all__ = ["run_sandbox_agent", "run_validator_agent"]


@tool(name="sandbox_agent")
async def run_sandbox_agent(ctx: ToolContext, instruction: str, max_rounds: int = 100) -> str:
    """Delegate hands-on work to a nested agent that only has sandbox tools.

    Returns its final answer once it stops requesting tools, or when max_rounds is hit.

    Args:
        instruction: The instruction for the sandbox agent to execute.
        max_rounds: Maximum number of execution rounds (default: 100).
    """
    return await ctx.require_sub_agent()("sandbox_agent", instruction, max_rounds)


@tool(name="validator_agent")
async def run_validator_agent(ctx: ToolContext, instruction: str, max_rounds: int = 50) -> str:
    """Ask a nested agent to reproduce a PoC in the sandbox and return a verdict.

    Use it to confirm every finding before reporting it.

    Args:
        instruction: Validation instruction that includes the PoC and expected outcome.
        max_rounds: Maximum number of execution rounds (default: 50).
    """
    return await ctx.require_sub_agent()("validator_agent", instruction, max_rounds)
