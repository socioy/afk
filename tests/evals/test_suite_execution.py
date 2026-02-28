from __future__ import annotations

from collections.abc import AsyncIterator

from afk.agents import Agent
from afk.core import Runner
from afk.evals import EvalCase, EvalSuiteConfig, arun_suite
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
)


def run_async(coro):
    import asyncio

    return asyncio.run(coro)


class _StaticLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "eval"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            chat=True,
            streaming=False,
            tool_calling=True,
            structured_output=True,
        )

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        return LLMResponse(text="ok")

    async def _chat_stream_core(
        self, req: LLMRequest, *, response_model=None
    ) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def _cases(count: int) -> list[EvalCase]:
    return [
        EvalCase(
            name=f"case-{i}",
            agent=Agent(model=_StaticLLM(), instructions="suite"),
            user_message="hello",
        )
        for i in range(count)
    ]


def test_adaptive_mode_resolves_to_sequential_for_small_case_count():
    out = run_async(
        arun_suite(
            runner_factory=Runner,
            cases=_cases(2),
            config=EvalSuiteConfig(execution_mode="adaptive", max_concurrency=8),
        )
    )
    assert out.execution_mode == "sequential"
    assert out.total == 2


def test_parallel_mode_preserves_input_order():
    cases = _cases(4)
    out = run_async(
        arun_suite(
            runner_factory=Runner,
            cases=cases,
            config=EvalSuiteConfig(execution_mode="parallel", max_concurrency=4),
        )
    )
    assert out.execution_mode == "parallel"
    assert [row.case for row in out.results] == [case.name for case in cases]
