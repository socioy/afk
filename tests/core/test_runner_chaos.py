from __future__ import annotations

import asyncio

from afk.agents import Agent, AgentBudgetExceededError, FailSafeConfig
from afk.core.runner import Runner
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from afk.memory import InMemoryMemoryStore
from afk.tools import tool
from pydantic import BaseModel


class _FastLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "fast"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=True, structured_output=True)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        return LLMResponse(text=f"ok:{req.request_id}", model=req.model)

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


class _SlowLLM(_FastLLM):
    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        await asyncio.sleep(0.05)
        return LLMResponse(text="slow")


class _NoopArgs(BaseModel):
    value: int = 1


@tool(args_model=_NoopArgs, name="noop_tool")
def noop_tool(args: _NoopArgs) -> dict[str, int]:
    return {"value": args.value}


class _SlowLoopLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "slow-loop"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=True, structured_output=True)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        self.calls += 1
        await asyncio.sleep(0.05)
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc_slow_loop_1",
                        tool_name="noop_tool",
                        arguments={"value": 1},
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


class _FailingThenRecoverLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "flaky"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=True, structured_output=True)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        self.calls += 1
        if self.calls <= 2:
            raise RuntimeError("transient")
        return LLMResponse(text="recovered")

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


class _FlakyMemoryStore(InMemoryMemoryStore):
    def __init__(self, fail_first_n_events: int) -> None:
        super().__init__()
        self._remaining_failures = fail_first_n_events

    async def append_event(self, event) -> None:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("transient append failure")
        await super().append_event(event)


def test_runner_handles_many_concurrent_runs():
    runner = Runner()

    async def scenario() -> list[str]:
        handles = []
        for idx in range(40):
            agent = Agent(model=_FastLLM(), instructions="fast")
            handles.append(
                await runner.run_handle(
                    agent,
                    user_message=f"hello-{idx}",
                    thread_id=f"thread-{idx}",
                )
            )

        results = await asyncio.gather(*[handle.await_result() for handle in handles])
        return [result.final_text for result in results if result is not None]

    outputs = asyncio.run(scenario())
    assert len(outputs) == 40
    assert all(text.startswith("ok:") for text in outputs)


def test_runner_fallback_chain_recovers_from_chaos_failures():
    flaky = _FailingThenRecoverLLM()
    backup = _FastLLM()

    def resolver(model: str):
        if model == "primary":
            return flaky
        if model == "backup":
            return backup
        raise ValueError(model)

    agent = Agent(
        model="primary",
        model_resolver=resolver,
        instructions="chaos",
        fail_safe=FailSafeConfig(fallback_model_chain=["backup"]),
    )

    result = asyncio.run(Runner().run(agent, user_message="go"))
    assert result.final_text.startswith("ok:")


def test_runner_survives_transient_memory_append_failures():
    memory = _FlakyMemoryStore(fail_first_n_events=3)
    runner = Runner(memory_store=memory)
    agent = Agent(model=_FastLLM(), instructions="memory resilience")
    result = asyncio.run(runner.run(agent, user_message="go"))
    assert result.final_text.startswith("ok:")


def test_runner_timeout_burst_produces_budget_errors_without_deadlock():
    async def scenario() -> list[Exception | str]:
        runner = Runner()
        handles = []
        for idx in range(12):
            agent = Agent(
                model=_SlowLoopLLM(),
                tools=[noop_tool],
                instructions="timeout burst",
                fail_safe=FailSafeConfig(max_wall_time_s=0.01),
            )
            handles.append(
                await runner.run_handle(
                    agent,
                    user_message=f"slow-{idx}",
                    thread_id=f"burst-{idx}",
                )
            )

        outputs: list[Exception | str] = []
        for handle in handles:
            try:
                result = await handle.await_result()
                outputs.append("cancelled" if result is None else result.final_text)
            except Exception as e:
                outputs.append(e)
        return outputs

    outputs = asyncio.run(scenario())
    assert len(outputs) == 12
    assert all(isinstance(item, AgentBudgetExceededError) for item in outputs)
