from __future__ import annotations

import asyncio
import sys
import types

import pytest
from pydantic import BaseModel

from afk.llms.clients.adapters.anthropic_agent import AnthropicAgentClient
from afk.llms.errors import LLMCapabilityError
from afk.llms.types import LLMRequest, Message, StreamCompletedEvent, StreamTextDeltaEvent


def run_async(coro):
    return asyncio.run(coro)


class Out(BaseModel):
    value: int


class TextBlock:
    def __init__(self, text: str):
        self.text = text


class ToolUseBlock:
    def __init__(self, block_id: str, name: str, input_data: dict):
        self.id = block_id
        self.name = name
        self.input = input_data


class AssistantMessage:
    def __init__(self, content, model: str = "claude-test"):
        self.content = content
        self.model = model


class ResultMessage:
    def __init__(
        self,
        subtype: str = "success",
        usage: dict | None = None,
        structured_output=None,
        session_id: str | None = None,
        user_message_uuid: str | None = None,
    ):
        self.subtype = subtype
        self.usage = usage or {"input_tokens": 2, "output_tokens": 3}
        self.structured_output = structured_output
        self.session_id = session_id
        self.user_message_uuid = user_message_uuid


class StreamEvent:
    def __init__(self, event: dict):
        self.event = event


class _ClaudeAgentOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_claude_agent_sdk(monkeypatch):
    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = _ClaudeAgentOptions
    option_calls: list[dict] = []

    async def fake_query(*, prompt, options):
        assert isinstance(prompt, str)
        assert isinstance(options, _ClaudeAgentOptions)
        option_calls.append(options.kwargs)

        include_partials = bool(options.kwargs.get("include_partial_messages"))
        if include_partials:
            yield StreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "hello "},
                }
            )
            yield StreamEvent(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "math",
                    },
                }
            )
            yield AssistantMessage(content=[TextBlock("world")], model="claude-stream")
            yield ResultMessage(
                structured_output={"value": 9},
                session_id="sess_stream",
                user_message_uuid="chk_stream",
            )
            return

        yield AssistantMessage(
            content=[
                TextBlock("hello"),
                ToolUseBlock("tool_1", "math", {"x": 1}),
            ],
            model="claude-chat",
        )
        yield ResultMessage(
            structured_output={"value": 7},
            session_id="sess_chat",
            user_message_uuid="chk_chat",
        )

    module.query = fake_query

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return option_calls


def test_anthropic_adapter_chat_uses_claude_agent_sdk(fake_claude_agent_sdk):
    llm = AnthropicAgentClient.from_env()
    req = LLMRequest(
        model="claude-opus",
        request_id="req_anthropic_1",
        session_token="sess_prev",
        messages=[Message(role="user", content="hi")],
    )

    out = run_async(llm.chat(req, response_model=Out))

    assert out.text == "hello"
    assert out.model == "claude-chat"
    assert out.structured_response == {"value": 7}
    assert out.request_id == "req_anthropic_1"
    assert out.session_token == "sess_chat"
    assert out.checkpoint_token == "chk_chat"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].tool_name == "math"
    assert out.tool_calls[0].arguments == {"x": 1}
    assert fake_claude_agent_sdk[0]["resume"] == "sess_prev"
    assert fake_claude_agent_sdk[0]["continue_conversation"] is True
    assert fake_claude_agent_sdk[0]["user"] == "req_anthropic_1"


def test_anthropic_adapter_stream_maps_stream_events(fake_claude_agent_sdk):
    llm = AnthropicAgentClient.from_env()
    req = LLMRequest(
        model="claude-opus",
        request_id="req_anthropic_2",
        messages=[Message(role="user", content="hi")],
    )

    async def scenario():
        stream = await llm.chat_stream(req, response_model=Out)
        return [event async for event in stream]

    events = run_async(scenario())
    completed = [e for e in events if isinstance(e, StreamCompletedEvent)]
    text_deltas = [e for e in events if isinstance(e, StreamTextDeltaEvent)]

    assert len(text_deltas) >= 1
    assert len(completed) == 1
    assert completed[0].response.structured_response == {"value": 9}
    assert completed[0].response.request_id == "req_anthropic_2"
    assert completed[0].response.session_token == "sess_stream"
    assert completed[0].response.checkpoint_token == "chk_stream"


def test_anthropic_adapter_stop_is_unsupported(fake_claude_agent_sdk):
    llm = AnthropicAgentClient.from_env()
    req = LLMRequest(
        model="claude-opus",
        messages=[Message(role="user", content="hi")],
        stop=["END"],
    )

    with pytest.raises(LLMCapabilityError):
        run_async(llm.chat(req))
