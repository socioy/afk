from __future__ import annotations

from collections.abc import AsyncIterator

from afk.agents import Agent
from afk.core import Runner
from afk.evals import EvalCase, EvalSuiteConfig, FinalTextContainsAssertion, run_suite
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
)


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
        return LLMResponse(text="always-ok")

    async def _chat_stream_core(
        self, req: LLMRequest, *, response_model=None
    ) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def _cases() -> list[EvalCase]:
    return [
        EvalCase(
            name="case-1",
            agent=Agent(model=_StaticLLM(), instructions="c1"),
            user_message="x",
        ),
        EvalCase(
            name="case-2",
            agent=Agent(model=_StaticLLM(), instructions="c2"),
            user_message="x",
        ),
    ]


def test_fail_fast_stops_after_first_failure():
    suite = run_suite(
        runner_factory=Runner,
        cases=_cases(),
        config=EvalSuiteConfig(
            execution_mode="sequential",
            fail_fast=True,
            assertions=(FinalTextContainsAssertion("not-present"),),
        ),
    )
    assert suite.total == 1


def test_non_fail_fast_executes_all_cases():
    suite = run_suite(
        runner_factory=Runner,
        cases=_cases(),
        config=EvalSuiteConfig(
            execution_mode="sequential",
            fail_fast=False,
            assertions=(FinalTextContainsAssertion("not-present"),),
        ),
    )
    assert suite.total == 2
