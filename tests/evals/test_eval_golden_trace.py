from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator

from afk.agents import Agent
from afk.core import Runner
from afk.evals import EvalScenario, compare_event_types, run_scenario
from afk.llms import LLM
from afk.llms.types import EmbeddingRequest, EmbeddingResponse, LLMCapabilities, LLMRequest, LLMResponse


class _EvalLLM(LLM):
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
        return LLMResponse(text="golden-ok")

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def test_eval_event_types_match_golden_trace():
    golden_path = Path(__file__).parent / "golden" / "basic_event_types.json"
    agent = Agent(model=_EvalLLM(), instructions="golden eval")
    scenario = EvalScenario(name="golden-basic", agent=agent, user_message="hello")
    observed = asyncio.run(run_scenario(Runner(), scenario)).event_types

    if os.getenv("AFK_UPDATE_GOLDEN_TRACES") == "1":
        golden_path.write_text(
            json.dumps(observed, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    ok, message = compare_event_types(expected, observed)
    assert ok, message
