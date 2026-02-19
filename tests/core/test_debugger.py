from __future__ import annotations

import asyncio

from pydantic import BaseModel

from afk.agents import Agent
from afk.core import RunnerConfig
from afk.debugger import Debugger, DebuggerConfig
from afk.tools import tool


class _SecretArgs(BaseModel):
    value: str


@tool(args_model=_SecretArgs, name="emit_secret")
def emit_secret(args: _SecretArgs) -> dict[str, str]:
    return {"token": args.value}


def test_debugger_runner_enables_debug_and_redacts_secret_payloads():
    from afk.llms import LLM
    from afk.llms.types import EmbeddingRequest, EmbeddingResponse, LLMCapabilities, LLMRequest, LLMResponse, ToolCall

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
    from afk.llms.types import EmbeddingRequest, EmbeddingResponse, LLMCapabilities, LLMRequest, LLMResponse

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
        from afk.core import Runner

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
