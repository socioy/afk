from __future__ import annotations

import asyncio

from pydantic import BaseModel

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner, RunnerConfig
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from afk.tools import ToolDeferredHandle, ToolResult, tool


class _NoArgs(BaseModel):
    pass


@tool(args_model=_NoArgs, name="write_docs")
def write_docs(args: _NoArgs) -> dict[str, str]:
    _ = args
    return {"status": "docs_written"}


@tool(args_model=_NoArgs, name="build_project")
async def build_project(args: _NoArgs) -> ToolResult[dict[str, str]]:
    _ = args
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, str]] = loop.create_future()

    async def _finish() -> None:
        await asyncio.sleep(0.001)
        if not future.done():
            future.set_result({"status": "ok", "artifact": "dist/app"})

    asyncio.create_task(_finish())
    return ToolResult(
        output=None,
        success=True,
        deferred=ToolDeferredHandle(
            ticket_id="build-ticket-1",
            tool_name="build_project",
            status="running",
            summary="Build started",
            resume_hint="Continue docs while build runs",
            poll_after_s=0.01,
        ),
        metadata={"background_future": future},
    )


class _BackgroundAwareLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "bg-llm"

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
                        id="tc_build_1",
                        tool_name="build_project",
                        arguments={},
                    )
                ],
                model=req.model,
            )

        has_build_result = False
        for message in req.messages:
            if message.role != "tool" or message.name != "build_project":
                continue
            if isinstance(message.content, str) and '"status": "ok"' in message.content:
                has_build_result = True
                break

        if has_build_result:
            return LLMResponse(text="Build done, applied fixes and finalized docs.")

        return LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id=f"tc_docs_{self.calls}",
                    tool_name="write_docs",
                    arguments={},
                )
            ],
            model=req.model,
        )

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def test_background_tool_defer_resolve_and_stream_events():
    async def _scenario():
        runner = Runner(
            config=RunnerConfig(
                sanitize_tool_output=False,
                background_tools_enabled=True,
                background_tool_poll_interval_s=0.01,
                background_tool_result_ttl_s=5.0,
                background_tool_interrupt_on_resolve=True,
            )
        )
        agent = Agent(
            model=_BackgroundAwareLLM(),
            instructions="x",
            tools=[build_project, write_docs],
            fail_safe=FailSafeConfig(max_steps=20),
        )

        stream = await runner.run_stream(agent, user_message="build and document")
        event_types: list[str] = []
        errors: list[str] = []
        async for event in stream:
            event_types.append(event.type)
            if event.type == "error" and event.error:
                errors.append(event.error)

        result = stream.result
        assert result is not None, f"stream errors={errors!r}, events={event_types!r}"
        return result, event_types

    result, event_types = asyncio.run(_scenario())
    assert "tool_deferred" in event_types
    assert "tool_background_resolved" in event_types
    assert any(row.tool_name == "build_project" for row in result.tool_executions)
    assert "finalized docs" in result.final_text
