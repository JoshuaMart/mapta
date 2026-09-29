"""The OpenRouter adapter owns the wire format; this pins it down."""

import json

import httpx2 as httpx
import pytest

from mapta.config import LLMSettings
from mapta.domain import LLMError, ToolSpec
from mapta.infrastructure.llm import OpenRouterClient

SPEC = ToolSpec(
    name="sandbox_run_command",
    description="Run a command.",
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    strict=True,
)


def completion(**message) -> dict:
    return {
        "id": "gen-1",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.0004},
    }


def client_for(body, captured: list[httpx.Request], **settings_overrides) -> OpenRouterClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=body() if callable(body) else body)

    settings = LLMSettings(api_key="sk-or", **settings_overrides)
    return OpenRouterClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.fixture
def captured() -> list[httpx.Request]:
    return []


async def test_requests_go_to_openrouter_chat_completions(captured):
    client = client_for(completion(content="hi"), captured)
    await client.respond(transcript=[], tools=[])
    assert str(captured[0].url) == "https://openrouter.ai/api/v1/chat/completions"


async def test_attribution_headers_are_sent(captured):
    client = client_for(completion(content="hi"), captured, app_name="recon", site_url="https://x.test")
    await client.respond(transcript=[], tools=[])
    assert captured[0].headers["x-title"] == "recon"
    assert captured[0].headers["http-referer"] == "https://x.test"


async def test_tools_are_nested_under_a_function_key(captured):
    client = client_for(completion(content="hi"), captured)
    await client.respond(transcript=[], tools=[SPEC])
    tools = json.loads(captured[0].content)["tools"]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "sandbox_run_command",
                "description": "Run a command.",
                "parameters": SPEC.parameters,
                "strict": True,
            },
        }
    ]


async def test_strict_can_be_turned_off_for_vendors_that_reject_it(captured):
    client = client_for(completion(content="hi"), captured, strict_tools=False)
    await client.respond(transcript=[], tools=[SPEC])
    assert "strict" not in json.loads(captured[0].content)["tools"][0]["function"]


async def test_reasoning_and_cost_accounting_are_requested(captured):
    client = client_for(completion(content="hi"), captured, reasoning_effort="low")
    await client.respond(transcript=[], tools=[])
    body = json.loads(captured[0].content)
    assert body["reasoning"] == {"effort": "low"}
    assert body["usage"] == {"include": True}


async def test_reasoning_is_omitted_when_disabled(captured):
    client = client_for(completion(content="hi"), captured, reasoning_effort="none")
    await client.respond(transcript=[], tools=[])
    assert "reasoning" not in json.loads(captured[0].content)


async def test_provider_routing_is_forwarded(captured):
    client = client_for(
        completion(content="hi"), captured, provider_order=("anthropic",), allow_fallbacks=False
    )
    await client.respond(transcript=[], tools=[])
    body = json.loads(captured[0].content)
    assert body["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}


def test_transcript_items_use_roles_every_vendor_accepts():
    client = OpenRouterClient(LLMSettings(api_key="sk-or"))
    # `developer` is OpenAI-only; OpenRouter normalises `system` for all vendors.
    assert client.developer_message("be brief") == {"role": "system", "content": "be brief"}
    assert client.user_message("go") == {"role": "user", "content": "go"}
    assert client.tool_output("call_1", "done") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "done",
    }


async def test_tool_calls_are_normalised(captured):
    body = completion(
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "sandbox_run_command", "arguments": '{"command": "ls"}'},
            }
        ],
    )
    response = await client_for(body, captured).respond(transcript=[], tools=[SPEC])

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert (call.call_id, call.name) == ("call_1", "sandbox_run_command")
    assert call.raw_arguments == '{"command": "ls"}'
    assert response.text == ""


async def test_the_assistant_turn_is_echoed_back_for_the_next_round(captured):
    body = completion(
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "sandbox_run_command", "arguments": "{}"},
            }
        ],
    )
    response = await client_for(body, captured).respond(transcript=[], tools=[SPEC])
    (item,) = response.items
    assert item["role"] == "assistant"
    assert item["tool_calls"][0]["id"] == "call_1"


async def test_usage_including_cost_is_carried_through(captured):
    response = await client_for(completion(content="hi"), captured).respond(transcript=[], tools=[])
    from mapta.application.usage import normalise_usage

    usage = normalise_usage(response.usage)
    assert usage["total_tokens"] == 12
    assert usage["cost"] == 0.0004


async def test_plain_text_answer(captured):
    response = await client_for(completion(content="all clear"), captured).respond(
        transcript=[], tools=[]
    )
    assert response.text == "all clear"
    assert response.tool_calls == ()
    assert response.response_id == "gen-1"


async def test_an_empty_choices_list_is_an_llm_error(captured):
    client = client_for({"id": "gen-1", "choices": []}, captured)
    with pytest.raises(LLMError, match="no choices"):
        await client.respond(transcript=[], tools=[])


async def test_transport_failures_become_llm_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    client = OpenRouterClient(
        LLMSettings(api_key="sk-or"),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError, match="OpenRouter request failed"):
        await client.respond(transcript=[], tools=[])


async def test_max_tokens_is_sent_when_configured(captured):
    client = client_for(completion(content="hi"), captured, max_tokens=8000)
    await client.respond(transcript=[], tools=[])
    assert json.loads(captured[0].content)["max_tokens"] == 8000


async def test_max_tokens_is_omitted_by_default(captured):
    client = client_for(completion(content="hi"), captured)
    await client.respond(transcript=[], tools=[])
    assert "max_tokens" not in json.loads(captured[0].content)


async def test_a_cut_off_answer_is_reported(captured):
    body = completion(content="partial")
    body["choices"][0]["finish_reason"] = "length"
    response = await client_for(body, captured).respond(transcript=[], tools=[])
    assert response.finish_reason == "length"
    assert response.truncated is True


async def test_a_complete_answer_is_not_flagged(captured):
    response = await client_for(completion(content="done"), captured).respond(
        transcript=[], tools=[]
    )
    assert response.truncated is False
