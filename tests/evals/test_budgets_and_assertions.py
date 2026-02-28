from __future__ import annotations

from collections.abc import AsyncIterator

from afk.agents import Agent
from afk.core import Runner
from afk.evals import (
    EvalBudget,
    EvalCase,
    FinalTextContainsAssertion,
    arun_case,
)
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
        return LLMResponse(text="hello world")

    async def _chat_stream_core(
        self, req: LLMRequest, *, response_model=None
    ) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def test_case_budget_violation_marks_result_failed():
    case = EvalCase(
        name="budget-case",
        agent=Agent(model=_StaticLLM(), instructions="budget"),
        user_message="hello",
    )
    out = run_async(arun_case(Runner(), case, budget=EvalBudget(max_duration_s=0.0)))

    assert out.passed is False
    assert out.budget_violations


def test_assertion_failure_marks_result_failed():
    case = EvalCase(
        name="assertion-case",
        agent=Agent(model=_StaticLLM(), instructions="assert"),
        user_message="hello",
    )
    out = run_async(
        arun_case(
            Runner(),
            case,
            assertions=(FinalTextContainsAssertion("missing-substring"),),
        )
    )

    assert out.passed is False
    assert any(not row.passed for row in out.assertions)
