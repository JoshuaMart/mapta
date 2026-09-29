"""The tool contract: context, decorator and registry.

Tools are plain async functions whose first parameter is a
:class:`ToolContext`, carrying every collaborator the tool may need. That is
what keeps the sandbox and the usage tracker out of global state, where two
concurrent scans would have shared them.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, overload

from ...domain import (
    JSONObject,
    Mailbox,
    Notifier,
    Sandbox,
    Target,
    ToolArgumentError,
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolUnavailable,
)
from ..arguments import (
    coerce_tool_arguments,
    decode_tool_arguments,
    describe_parameters,
    looks_truncated,
)
from ..budget import RoundBudget
from ..usage import UsageTracker
from .schema import build_tool_schema

__all__ = ["SubAgentRunner", "Tool", "ToolContext", "ToolRegistry", "tool"]

logger = logging.getLogger(__name__)

#: How much of a bad payload reaches the log.
_PAYLOAD_PREVIEW_CHARS = 2000


class SubAgentRunner(Protocol):
    """Runs a nested agent loop. Injected to avoid a tools -> agent import cycle."""

    async def __call__(self, profile: str, instruction: str, max_rounds: int) -> str:
        ...


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to touch, scoped to one scan."""

    target: Target
    sandbox: Sandbox
    usage: UsageTracker
    #: Shared by every agent of this scan, nested ones included.
    budget: RoundBudget = field(default_factory=RoundBudget)
    mailbox: Mailbox | None = None
    notifier: Notifier | None = None
    run_sub_agent: SubAgentRunner | None = None
    max_tool_output_chars: int = 30_000

    def require_mailbox(self) -> Mailbox:
        if self.mailbox is None:
            raise ToolUnavailable("No mailbox provider is configured for this run.")
        return self.mailbox

    def require_notifier(self) -> Notifier:
        if self.notifier is None or not self.notifier.enabled:
            raise ToolUnavailable(
                "No notifier is configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
                "to enable alerts."
            )
        return self.notifier

    def require_sub_agent(self) -> SubAgentRunner:
        if self.run_sub_agent is None:
            raise ToolUnavailable("Nested agents are not available in this run.")
        return self.run_sub_agent


type ToolFunc = Callable[..., Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class Tool:
    """An async function advertised to the model, plus its schema."""

    name: str
    description: str
    parameters: JSONObject
    strict: bool
    func: ToolFunc
    optional_params: frozenset[str] = frozenset()
    required_params: frozenset[str] = frozenset()

    def spec(self) -> ToolSpec:
        """The provider-neutral description handed to the LLM adapter."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            strict=self.strict,
        )

    def _clean(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Drop explicit nulls for arguments that have a Python default."""
        return {
            key: value
            for key, value in arguments.items()
            if not (value is None and key in self.optional_params)
        }

    async def invoke(self, ctx: ToolContext, arguments: Mapping[str, Any]) -> Any:
        return await self.func(ctx, **self._clean(arguments))


@overload
def tool(func: ToolFunc, /) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    strict_mode: bool = ...,
) -> Callable[[ToolFunc], Tool]: ...


def tool(
    func: ToolFunc | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    strict_mode: bool = True,
) -> Tool | Callable[[ToolFunc], Tool]:
    """Turn an async function into a :class:`Tool`.

    Usable bare (``@tool``) or with arguments (``@tool(name="short_name")``).
    The leading ``ctx`` parameter is injected at call time and never advertised.
    """

    def decorate(target: ToolFunc) -> Tool:
        schema = build_tool_schema(
            target, skip=frozenset({"self", "cls", "ctx"}), strict_mode=strict_mode
        )
        tool_name = name or target.__name__
        return Tool(
            name=tool_name,
            description=description or schema.description or f"Call {tool_name}.",
            parameters=schema.parameters,
            strict=schema.strict,
            func=target,
            optional_params=schema.optional_params,
            required_params=schema.required_params,
        )

    return decorate(func) if func is not None else decorate


@dataclass(slots=True)
class ToolRegistry:
    """Name -> :class:`Tool`, with the subset helpers the agents need."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    @classmethod
    def of(cls, tools: Iterable[Tool]) -> ToolRegistry:
        registry = cls()
        for item in tools:
            registry.register(item)
        return registry

    def register(self, item: Tool) -> None:
        if item.name in self._tools:
            raise ValueError(f"duplicate tool name: {item.name}")
        self._tools[item.name] = item

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def specs(
        self,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
    ) -> list[ToolSpec]:
        """Tool specs, optionally narrowed to an allow/deny list."""
        excluded = set(exclude)
        wanted = list(include) if include is not None else list(self._tools)
        return [
            self._tools[name].spec()
            for name in wanted
            if name in self._tools and name not in excluded
        ]

    async def execute(self, ctx: ToolContext, call: ToolCall) -> ToolResult:
        """Run one tool call, converting every failure into a model-readable result.

        A malformed call must never abort a scan: the model is told what went
        wrong so it can retry.
        """
        item = self._tools.get(call.name)
        if item is None:
            return _error(call, "unknown_tool", f"No tool named {call.name!r} is available.")

        try:
            arguments = decode_tool_arguments(call.raw_arguments)
        except ValueError as exc:
            _log_bad_arguments(call, exc)
            if looks_truncated(call.raw_arguments):
                # Telling the model its JSON was invalid here makes it re-send
                # the same over-long call; it has to be told it was cut off.
                return _error(
                    call,
                    "truncated_tool_arguments",
                    f"Your `arguments` for `{call.name}` were cut off mid-write and "
                    "could not be recovered, most likely because the answer hit the "
                    "model's output limit. Re-issue the call with a substantially "
                    "shorter value, or split the work across several calls.",
                    expected=item.parameters,
                )
            return _error(
                call,
                "invalid_tool_arguments",
                f"Your `arguments` for `{call.name}` were not valid JSON ({exc}). "
                "Re-issue the call with a single well-formed JSON object: double "
                "quotes around keys and strings, `null`/`true`/`false` rather than "
                "Python literals, and newlines escaped as \\n inside strings.",
                expected=item.parameters,
            )

        try:
            # Checked here rather than in the tool body, so a wrong type is
            # something the model can fix instead of a crash mid-scan.
            arguments = coerce_tool_arguments(
                item.parameters, arguments, required=item.required_params
            )
        except ToolArgumentError as exc:
            _log_bad_arguments(call, exc)
            return _error(
                call,
                "invalid_tool_arguments",
                f"Your `arguments` for `{call.name}` did not match the tool's "
                f"parameters: {exc}. Re-issue the call with the declared types.",
                expected=item.parameters,
            )

        try:
            output = await item.invoke(ctx, arguments)
        except ToolUnavailable as exc:
            return _error(call, "tool_unavailable", str(exc))
        except Exception as exc:  # a tool failure is data for the model, not a crash
            logger.warning("Tool %r failed: %s", call.name, exc, exc_info=True)
            return _error(call, type(exc).__name__, str(exc), arguments=arguments)

        return ToolResult(call_id=call.call_id, output=_stringify(output))


def _stringify(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, default=str)


def _log_bad_arguments(call: ToolCall, exc: Exception) -> None:
    """Log the payload itself: without it these failures are undiagnosable."""
    raw = call.raw_arguments
    preview = raw if isinstance(raw, str) else repr(raw)
    logger.error(
        "Malformed arguments for tool %r (%s). Length=%d. Raw payload: %r",
        call.name, exc, len(preview or ""), (preview or "")[:_PAYLOAD_PREVIEW_CHARS],
    )


def _error(
    call: ToolCall,
    code: str,
    detail: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    expected: JSONObject | None = None,
) -> ToolResult:
    payload: JSONObject = {"error": code, "detail": detail}
    if arguments is not None:
        payload["args"] = dict(arguments)
    if expected is not None:
        payload["expected_parameters"] = describe_parameters(expected)
    return ToolResult(call_id=call.call_id, output=json.dumps(payload, default=str))
