from __future__ import annotations

import asyncio
import sys
import types

import pytest

from afk.llms.clients.adapters.litellm import LiteLLMClient
from afk.llms.types import (
    EmbeddingRequest,
    LLMRequest,
    Message,
    StreamCompletedEvent,
    StreamTextDeltaEvent,
)


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture
def fake_litellm(monkeypatch):
    module = types.ModuleType("litellm")
    calls: list[dict] = []

    async def aresponses(**kwargs):
        calls.append(kwargs)

        if kwargs.get("stream"):
            async def _iter():
                yield {
                    "type": "response.output_text.delta",
                    "delta": "stream ",
                }
                yield {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "add_numbers",
                    },
                }
                yield {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": '{"numbers":[1,2,3]}',
                }
                yield {
                    "type": "response.completed",
                    "response": {
                        "model": kwargs.get("model"),
                        "status": "completed",
                        "usage": {
                            "input_tokens": 4,
                            "output_tokens": 5,
                            "total_tokens": 9,
                        },
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "stream done",
                                    }
                                ],
                            },
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "add_numbers",
                                "arguments": '{"numbers":[1,2,3]}',
                            },
                        ],
                    },
                }

            return _iter()

        return {
            "model": kwargs.get("model"),
            "status": "completed",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
            },
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "hello world",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "add_numbers",
                    "arguments": '{"numbers":[5,10,15]}',
                },
            ],
        }

    async def aembedding(**kwargs):
        calls.append(kwargs)
        return {
            "model": "embed-demo",
            "data": [{"embedding": [0.1, 0.2]}],
        }

    module.aresponses = aresponses
    module.aembedding = aembedding

    monkeypatch.setitem(sys.modules, "litellm", module)
    return calls


def _tool_def() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                    }
                },
                "required": ["numbers"],
                "additionalProperties": False,
            },
        },
    }


def test_litellm_chat_uses_responses_api(fake_litellm):
    llm = LiteLLMClient.from_env()
    req = LLMRequest(
        model="ollama_chat/gpt-oss:20b",
        messages=[Message(role="user", content="add numbers")],
        tools=[_tool_def()],
        tool_choice={"type": "function", "function": {"name": "add_numbers"}},
        max_tokens=128,
        request_id="req_litellm_1",
        idempotency_key="idem_litellm_1",
        stop=["<STOP>"],
    )

    out = run_async(llm.chat(req))

    assert out.text == "hello world"
    assert out.tool_calls
    assert out.tool_calls[0].tool_name == "add_numbers"
    assert out.tool_calls[0].arguments == {"numbers": [5, 10, 15]}

    assert len(fake_litellm) == 1
    payload = fake_litellm[0]
    assert payload["model"] == "ollama_chat/gpt-oss:20b"
    assert "input" in payload
    assert "messages" not in payload
    assert payload["max_output_tokens"] == 128
    assert payload["stop"] == ["<STOP>"]
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "add_numbers"
    assert payload["tool_choice"] == {"type": "function", "name": "add_numbers"}
    assert payload["metadata"]["afk_request_id"] == "req_litellm_1"
    assert payload["headers"]["Idempotency-Key"] == "idem_litellm_1"
    assert payload["headers"]["X-Request-Id"] == "req_litellm_1"


def test_litellm_stream_uses_responses_events(fake_litellm):
    llm = LiteLLMClient.from_env()
    req = LLMRequest(
        model="ollama_chat/gpt-oss:20b",
        messages=[Message(role="user", content="stream me")],
    )

    async def scenario():
        stream = await llm.chat_stream(req)
        return [event async for event in stream]

    events = run_async(scenario())

    text_deltas = [e for e in events if isinstance(e, StreamTextDeltaEvent)]
    completed = [e for e in events if isinstance(e, StreamCompletedEvent)]

    assert text_deltas
    assert len(completed) == 1
    assert completed[0].response.text == "stream done"
    assert completed[0].response.tool_calls
    assert completed[0].response.tool_calls[0].tool_name == "add_numbers"
    assert completed[0].response.tool_calls[0].arguments == {"numbers": [1, 2, 3]}

    assert len(fake_litellm) == 1
    assert fake_litellm[0]["stream"] is True
    assert "input" in fake_litellm[0]
    assert "messages" not in fake_litellm[0]


def test_litellm_embed_maps_metadata(fake_litellm):
    llm = LiteLLMClient.from_env()
    req = EmbeddingRequest(
        model="embed-demo",
        inputs=["hello"],
        metadata={"trace": "embed-litellm"},
    )

    out = run_async(llm.embed(req))

    assert out.embeddings == [[0.1, 0.2]]
    assert len(fake_litellm) == 1
    payload = fake_litellm[0]
    assert payload["model"] == "embed-demo"
    assert payload["input"] == ["hello"]
    assert payload["metadata"] == {"trace": "embed-litellm"}
