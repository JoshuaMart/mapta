"""The agent loop.

One implementation serves the main agent and both nested agents; they differ
only by their :class:`~mapta.application.prompts.AgentProfile` (system prompt,
round budget, tool subset).
"""

import asyncio
import logging
from dataclasses import dataclass

from ..domain import JSONObject, LLMClient, ModelResponse, ToolResult, TranscriptItem
from .prompts import AgentProfile
from .tools.base import ToolContext, ToolRegistry

__all__ = ["AgentLoop", "AgentRunner"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentLoop:
    """Prompt the model, run the tools it asks for, repeat until it stops."""

    llm: LLMClient
    registry: ToolRegistry
    profile: AgentProfile

    async def run(
        self,
        instruction: str,
        ctx: ToolContext,
        *,
        metadata: JSONObject | None = None,
    ) -> str:
        """Drive the loop and return the model's final text.

        The loop ends when the model answers without requesting a tool, or when
        the profile's round budget is exhausted (``max_rounds=0`` = unlimited).
        """
        transcript: list[TranscriptItem] = [
            self.llm.developer_message(self.profile.system_prompt),
            self.llm.user_message(instruction),
        ]
        tools = self.registry.specs(
            include=self.profile.allowed_tools, exclude=self.profile.denied_tools
        )
        request_metadata = {"name": self.profile.name} | (metadata or {})

        rounds = 0
        while True:
            if not ctx.budget.claim():
                logger.warning(
                    "[%s] scan round budget exhausted (%d rounds)",
                    self.profile.name, ctx.budget.total,
                )
                return (
                    f"[{self.profile.name}] The scan's round budget of "
                    f"{ctx.budget.total} is exhausted; stopping here."
                )

            response = await self.llm.respond(
                transcript=transcript, tools=tools, metadata=request_metadata
            )
            ctx.usage.record(self.profile.name, response.usage)

            if response.truncated:
                logger.warning(
                    "[%s] answer truncated by the model's output limit (round %d)",
                    self.profile.name, rounds + 1,
                )

            if not response.tool_calls:
                logger.debug("[%s] finished after %d round(s)", self.profile.name, rounds)
                return response.text

            transcript.extend(response.items)
            logger.info(
                "[%s] executing %d tool call(s)", self.profile.name, len(response.tool_calls)
            )
            results = await self._execute_calls(ctx, response)
            transcript.extend(
                self.llm.tool_output(result.call_id, result.output) for result in results
            )

            rounds += 1
            if self.profile.max_rounds and rounds >= self.profile.max_rounds:
                logger.warning(
                    "[%s] reached max rounds limit: %d", self.profile.name, self.profile.max_rounds
                )
                return f"[{self.profile.name}] Reached max rounds limit: {self.profile.max_rounds}"

    async def _execute_calls(
        self, ctx: ToolContext, response: ModelResponse
    ) -> list[ToolResult]:
        """Run every requested tool concurrently, preserving call order."""
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self.registry.execute(ctx, call))
                for call in response.tool_calls
            ]
        return [task.result() for task in tasks]


@dataclass(frozen=True, slots=True)
class AgentRunner:
    """Resolves a profile by name and runs it. Injected into the tool context.

    This is what lets the ``sandbox_agent`` / ``validator_agent`` tools start a
    nested loop without the tools package importing :class:`AgentLoop`.
    """

    llm: LLMClient
    registry: ToolRegistry
    profiles: dict[str, AgentProfile]
    #: Bound once per scan by :meth:`bound_to`, since the context outlives nothing.
    ctx: ToolContext | None = None

    def bound_to(self, ctx: ToolContext) -> AgentRunner:
        """Return a copy tied to one scan's tool context."""
        return AgentRunner(
            llm=self.llm, registry=self.registry, profiles=self.profiles, ctx=ctx
        )

    async def __call__(self, profile: str, instruction: str, max_rounds: int) -> str:
        if self.ctx is None:
            raise RuntimeError("AgentRunner used before bound_to(ctx) was called")
        base = self.profiles[profile]
        loop = AgentLoop(
            llm=self.llm,
            registry=self.registry,
            profile=base.with_(max_rounds=max_rounds) if max_rounds else base,
        )
        return await loop.run(instruction, self.ctx)
