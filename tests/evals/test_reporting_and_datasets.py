from __future__ import annotations

import json
from collections.abc import AsyncIterator

from afk.agents import Agent
from afk.core import Runner
from afk.evals import (
    EvalCase,
    EvalSuiteConfig,
    load_eval_cases_json,
    run_suite,
    write_suite_report_json,
)
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


def test_dataset_loader_and_suite_report(tmp_path):
    agent = Agent(model=_StaticLLM(), instructions="dataset")
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "name": "case-1",
                    "agent": "simple",
                    "user_message": "hello",
                    "context": {"k": 1},
                    "tags": ["smoke"],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases_json(
        dataset_path,
        agent_resolver=lambda name: agent if name == "simple" else None,
    )
    assert len(cases) == 1
    assert isinstance(cases[0], EvalCase)

    suite = run_suite(
        runner_factory=Runner,
        cases=cases,
        config=EvalSuiteConfig(execution_mode="sequential"),
    )
    report_path = tmp_path / "suite-report.json"
    write_suite_report_json(report_path, suite)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "eval_suite.v1"
    assert payload["summary"]["total"] == 1
