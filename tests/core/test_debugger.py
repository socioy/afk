from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from afk.agents import Agent
from afk.core import Runner, RunnerConfig
from afk.debugger import Debugger, DebuggerConfig
from afk.tools import tool


class _SecretArgs(BaseModel):
    value: str


@tool(args_model=_SecretArgs, name="emit_secret")
def emit_secret(args: _SecretArgs) -> dict[str, str]:
    return {"token": args.value}


# ---------------------------------------------------------------------------
# DebuggerConfig tests
# ---------------------------------------------------------------------------


class TestDebuggerConfigDefaults:
    def test_default_values(self):
        config = DebuggerConfig()
        assert config.enabled is True
        assert config.verbosity == "detailed"
        assert config.include_content is True
        assert config.redact_secrets is True
        assert config.max_payload_chars == 4000
        assert config.emit_timestamps is True
        assert config.emit_step_snapshots is True

    def test_custom_values(self):
        config = DebuggerConfig(
            enabled=False,
            verbosity="basic",
            include_content=False,
            redact_secrets=False,
            max_payload_chars=100,
            emit_timestamps=False,
            emit_step_snapshots=False,
        )
        assert config.enabled is False
        assert config.verbosity == "basic"
        assert config.include_content is False
        assert config.redact_secrets is False
        assert config.max_payload_chars == 100
        assert config.emit_timestamps is False
        assert config.emit_step_snapshots is False

    def test_frozen(self):
        config = DebuggerConfig()
        with pytest.raises(AttributeError):
            config.enabled = False

    @pytest.mark.parametrize("verbosity", ["basic", "detailed", "trace"])
    def test_verbosity_values(self, verbosity):
        config = DebuggerConfig(verbosity=verbosity)
        assert config.verbosity == verbosity


# ---------------------------------------------------------------------------
# Debugger tests
# ---------------------------------------------------------------------------


class TestDebugger:
    def test_debugger_creates_runner_with_debug_enabled(self):
        debugger = Debugger()
        runner = debugger.runner()
        assert runner.config.debug is True

    def test_debugger_with_custom_config(self):
        config = DebuggerConfig(enabled=True, redact_secrets=True, verbosity="trace")
        debugger = Debugger(config)
        runner = debugger.runner()
        assert runner.config.debug is True


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_debugger_runner_enables_debug_and_redacts_secret_payloads():
    from afk.llms import LLM
    from afk.llms.types import (
        EmbeddingRequest,
        EmbeddingResponse,
        LLMCapabilities,
        LLMRequest,
        LLMResponse,
        ToolCall,
    )

    class _LLM(LLM):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        @property
        def provider_id(self) -> str:
            return "dbg"

        @property
        def capabilities(self) -> LLMCapabilities:
            return LLMCapabilities(chat=True, streaming=False, tool_calling=True)

        async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
            _ = response_model
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="tc_dbg_1",
                            tool_name="emit_secret",
                            arguments={"value": "top-secret-token"},
                        )
                    ],
                    model=req.model,
                )
            return LLMResponse(text="done", model=req.model)

        async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
            _ = req
            _ = response_model
            raise NotImplementedError

        async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
            _ = req
            raise NotImplementedError

    async def _scenario():
        dbg = Debugger(DebuggerConfig(redact_secrets=True))
        runner = dbg.runner(config=RunnerConfig(sanitize_tool_output=False))
        handle = await runner.run_handle(
            Agent(model=_LLM(), instructions="x", tools=[emit_secret]),
            user_message="go",
        )
        tool_event = None
        async for event in handle.events:
            if event.type == "tool_completed":
                tool_event = event
        _ = await handle.await_result()
        return tool_event

    event = asyncio.run(_scenario())
    assert event is not None
    assert isinstance(event.data.get("debug"), dict)
    payload = event.data["debug"].get("payload_preview")
    assert isinstance(payload, dict)
    output = payload.get("output")
    assert isinstance(output, dict)
    assert output.get("token") == "***REDACTED***"


def test_runner_config_debug_without_debugger_facade_emits_debug_payloads():
    from afk.llms import LLM
    from afk.llms.types import (
        EmbeddingRequest,
        EmbeddingResponse,
        LLMCapabilities,
        LLMRequest,
        LLMResponse,
    )

    class _LLM(LLM):
        @property
        def provider_id(self) -> str:
            return "dbg-config"

        @property
        def capabilities(self) -> LLMCapabilities:
            return LLMCapabilities(chat=True, streaming=False, tool_calling=True)

        async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
            _ = req
            _ = response_model
            return LLMResponse(text="ok")

        async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
            _ = req
            _ = response_model
            raise NotImplementedError

        async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
            _ = req
            raise NotImplementedError

    async def _scenario():
        runner = Runner(config=RunnerConfig(debug=True))
        handle = await runner.run_handle(Agent(model=_LLM(), instructions="x"))
        first = None
        async for event in handle.events:
            if first is None:
                first = event
        _ = await handle.await_result()
        return first

    event = asyncio.run(_scenario())
    assert event is not None
    assert "debug" in event.data


def test_debugger_disabled_emits_no_debug_payload():
    from afk.llms import LLM
    from afk.llms.types import (
        EmbeddingRequest,
        EmbeddingResponse,
        LLMCapabilities,
        LLMRequest,
        LLMResponse,
    )

    class _LLM(LLM):
        @property
        def provider_id(self) -> str:
            return "dbg-disabled"

        @property
        def capabilities(self) -> LLMCapabilities:
            return LLMCapabilities(chat=True, streaming=False, tool_calling=True)

        async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
            _ = req
            _ = response_model
            return LLMResponse(text="ok")

        async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
            _ = req
            _ = response_model
            raise NotImplementedError

        async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
            _ = req
            raise NotImplementedError

    async def _scenario():
        debugger = Debugger(DebuggerConfig(enabled=False))
        runner = debugger.runner()
        handle = await runner.run_handle(Agent(model=_LLM(), instructions="x"))
        events = []
        async for event in handle.events:
            events.append(event)
        _ = await handle.await_result()
        return events

    events = asyncio.run(_scenario())
    first = events[0]
    assert "debug" not in first.data
