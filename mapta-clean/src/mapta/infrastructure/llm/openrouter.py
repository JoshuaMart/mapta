"""LLM adapter for OpenRouter.

Requests go to OpenRouter's **Chat Completions** endpoint, the surface every
model in its catalogue implements. The ``openai`` package is used only as a
typed HTTP client for it; no request reaches api.openai.com.

The wire format and the transcript shape are this module's business alone,
which is what keeps the agent loop free of provider details.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# httpx2 is httpx 2.x under its new name; it is what the openai SDK expects.
import httpx2 as httpx
from openai import AsyncOpenAI

from ...config import LLMSettings
from ...domain import JSONObject, LLMError, ModelResponse, ToolCall, ToolSpec, TranscriptItem

__all__ = ["OpenRouterClient"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenRouterClient:
    """Adapter implementing :class:`~mapta.domain.ports.LLMClient`."""

    settings: LLMSettings
    #: Optional transport seam, for proxies, custom timeouts and tests.
    http_client: httpx.AsyncClient | None = None
    _client: AsyncOpenAI | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        logger.info(
            "LLM: OpenRouter | model: %s | base_url: %s",
            self.settings.model,
            self.settings.base_url,
        )

    @property
    def client(self) -> AsyncOpenAI:
        """The SDK client, created on first use so imports need no API key."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                default_headers=self.settings.default_headers,
                http_client=self.http_client,
            )
        return self._client

    async def aclose(self) -> None:
        """Release the SDK's connection pool."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # --- transcript items -------------------------------------------------

    def developer_message(self, text: str) -> TranscriptItem:
        # OpenRouter normalises `system` for every vendor; `developer` is an
        # OpenAI-only role that other models behind OpenRouter reject.
        return {"role": "system", "content": text}

    def user_message(self, text: str) -> TranscriptItem:
        return {"role": "user", "content": text}

    def tool_output(self, call_id: str, output: str) -> TranscriptItem:
        return {"role": "tool", "tool_call_id": call_id, "content": output}

    # --- the call ---------------------------------------------------------

    async def respond(
        self,
        *,
        transcript: Sequence[TranscriptItem],
        tools: Sequence[ToolSpec],
        metadata: JSONObject | None = None,
    ) -> ModelResponse:
        if metadata:
            # OpenRouter has no per-request metadata field; keep it for tracing.
            logger.debug("request metadata: %s", metadata)

        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(transcript),
            "tools": [self._tool_definition(spec) for spec in tools],
        }
        if self.settings.max_tokens is not None:
            request["max_tokens"] = self.settings.max_tokens
        # Everything below is an OpenRouter extension the SDK does not model as a
        # named argument, so it travels in extra_body.
        extra_body: dict[str, Any] = {}
        if self.settings.sends_reasoning:
            # OpenRouter's unified reasoning parameter; ignored by models that
            # do not reason, so it is safe across the catalogue.
            extra_body["reasoning"] = {"effort": self.settings.reasoning_effort}
        if self.settings.track_cost:
            # Returns token counts *and* the generation's cost in `usage`.
            extra_body["usage"] = {"include": True}
        if order := self.settings.provider_order:
            extra_body["provider"] = {
                "order": list(order),
                "allow_fallbacks": self.settings.allow_fallbacks,
            }
        if extra_body:
            request["extra_body"] = extra_body

        try:
            completion = await self.client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMError(f"OpenRouter request failed: {exc}") from exc

        return _normalise(completion)

    def _tool_definition(self, spec: ToolSpec) -> JSONObject:
        """Chat Completions nests the tool under a ``function`` key."""
        function: JSONObject = {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
        # Not every vendor behind OpenRouter honours strict mode; sending it
        # only when it is both derivable and wanted keeps those models usable.
        if spec.strict and self.settings.strict_tools:
            function["strict"] = True
        return {"type": "function", "function": function}


def _normalise(completion: Any) -> ModelResponse:
    """Convert a Chat Completions answer into the domain's :class:`ModelResponse`."""
    choices = getattr(completion, "choices", None) or []
    if not choices:
        # OpenRouter reports some upstream failures with HTTP 200 and no choice.
        detail = getattr(completion, "error", None) or completion
        raise LLMError(f"OpenRouter returned no choices: {detail}")

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        logger.warning(
            "The model hit its output limit mid-answer; tool-call arguments may be "
            "truncated. Raise MAX_TOKENS, or pick a model with a larger output budget."
        )

    message = choice.message

    tool_calls = tuple(
        ToolCall(
            call_id=call.id,
            name=call.function.name,
            raw_arguments=call.function.arguments,
        )
        for call in (getattr(message, "tool_calls", None) or [])
    )

    # The assistant turn is echoed back verbatim on the next round; dumping it
    # keeps vendor extras (reasoning_details, ...) that some models need.
    item = message.model_dump(exclude_none=True)
    item.setdefault("role", "assistant")

    return ModelResponse(
        text=getattr(message, "content", None) or "",
        tool_calls=tool_calls,
        items=(item,),
        usage=getattr(completion, "usage", None),
        response_id=getattr(completion, "id", None),
        finish_reason=finish_reason,
    )
