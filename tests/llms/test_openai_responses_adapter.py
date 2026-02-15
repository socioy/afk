from __future__ import annotations

import asyncio
import sys
import types

import pytest

from afk.llms.clients.adapters.openai import OpenAIClient
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
def fake_openai(monkeypatch):
    module = types.ModuleType("openai")
    calls: list[dict] = []

    class _ResponsesAPI:
        async def create(self, **kwargs):
            calls.append(kwargs)

            if kwargs.get("stream"):

                async def _iter():
                    yield {
                        "type": "response.output_text.delta",
                        "delta": "openai ",
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
                        "delta": '{"numbers":[2,4,6]}',
                    }
                    yield {
                        "type": "response.completed",
                        "response": {
                            "model": kwargs.get("model"),
                            "status": "completed",
                            "usage": {
                                "input_tokens": 4,
                                "output_tokens": 6,
                                "total_tokens": 10,
                            },
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "openai done",
                                        }
                                    ],
                                },
                                {
                                    "type": "function_call",
                                    "call_id": "call_1",
                                    "name": "add_numbers",
                                    "arguments": '{"numbers":[2,4,6]}',
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
                    "output_tokens": 5,
                    "total_tokens": 8,
                },
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello from responses",
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
            }

    class _EmbeddingsAPI:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return {
                "model": "text-embedding-3-small",
                "data": [{"embedding": [0.11, 0.22]}],
            }

    class _ChatTrap:
        def __getattr__(self, name):
            raise AssertionError(f"chat completions API should not be used: {name}")

    class AsyncOpenAI:
        def __init__(self, **kwargs):
            _ = kwargs
            self.responses = _ResponsesAPI()
            self.embeddings = _EmbeddingsAPI()
            self.chat = _ChatTrap()

    module.AsyncOpenAI = AsyncOpenAI

    monkeypatch.setitem(sys.modules, "openai", module)
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


def test_openai_chat_uses_responses_api(fake_openai):
    llm = OpenAIClient.from_env()
    req = LLMRequest(
        model="gpt-4.1-mini",
        messages=[Message(role="user", content="add numbers")],
        tools=[_tool_def()],
        tool_choice={"type": "function", "function": {"name": "add_numbers"}},
        max_tokens=96,
        thinking=True,
        thinking_effort="low",
        request_id="req_openai_1",
        idempotency_key="idem_openai_1",
        stop=["<END>"],
    )

    out = run_async(llm.chat(req))

    assert out.text == "hello from responses"
    assert out.tool_calls
    assert out.tool_calls[0].tool_name == "add_numbers"
    assert out.tool_calls[0].arguments == {"numbers": [1, 2, 3]}

    assert len(fake_openai) == 1
    payload = fake_openai[0]
    assert payload["model"] == "gpt-4.1-mini"
    assert "input" in payload
    assert "messages" not in payload
    assert payload["max_output_tokens"] == 96
    assert payload["stop"] == ["<END>"]
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "add_numbers"
    assert payload["tool_choice"] == {"type": "function", "name": "add_numbers"}
    assert payload["metadata"]["afk_request_id"] == "req_openai_1"
    assert payload["extra_headers"]["Idempotency-Key"] == "idem_openai_1"
    assert payload["extra_headers"]["X-Request-Id"] == "req_openai_1"


def test_openai_stream_uses_responses_events(fake_openai):
    llm = OpenAIClient.from_env()
    req = LLMRequest(
        model="gpt-4.1-mini",
        messages=[Message(role="user", content="stream")],
    )

    async def scenario():
        stream = await llm.chat_stream(req)
        return [event async for event in stream]

    events = run_async(scenario())

    text_deltas = [e for e in events if isinstance(e, StreamTextDeltaEvent)]
    completed = [e for e in events if isinstance(e, StreamCompletedEvent)]

    assert text_deltas
    assert len(completed) == 1
    assert completed[0].response.text == "openai done"
    assert completed[0].response.tool_calls
    assert completed[0].response.tool_calls[0].tool_name == "add_numbers"
    assert completed[0].response.tool_calls[0].arguments == {"numbers": [2, 4, 6]}

    assert len(fake_openai) == 1
    assert fake_openai[0]["stream"] is True
    assert "input" in fake_openai[0]
    assert "messages" not in fake_openai[0]


def test_openai_embed_maps_metadata(fake_openai):
    llm = OpenAIClient.from_env()
    req = EmbeddingRequest(
        model="text-embedding-3-small",
        inputs=["hello"],
        metadata={"trace": "embed-openai"},
    )

    out = run_async(llm.embed(req))

    assert out.embeddings == [[0.11, 0.22]]
    assert len(fake_openai) == 1
    payload = fake_openai[0]
    assert payload["model"] == "text-embedding-3-small"
    assert payload["input"] == ["hello"]
    assert payload["metadata"] == {"trace": "embed-openai"}
