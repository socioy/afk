from __future__ import annotations

import asyncio

from afk.agents import Agent
from afk.agents.types import RouterDecision
from afk.core.runner import Runner
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
)
from afk.observability import contracts as obs_contracts
from afk.observability.backends import InMemoryTelemetrySink


def run_async(coro):
    return asyncio.run(coro)


class _StaticLLM(LLM):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    @property
    def provider_id(self) -> str:
        return "static"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            chat=True, streaming=False, tool_calling=True, structured_output=True
        )

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        return LLMResponse(text=self._text)

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def test_runner_emits_subagent_node_telemetry_metrics():
    child = Agent(model=_StaticLLM("child done"), instructions="child")
    parent = Agent(
        model=_StaticLLM("parent done"),
        instructions="parent",
        subagents=[child],
        subagent_router=lambda _: RouterDecision(targets=[child.name], parallel=False),
    )

    sink = InMemoryTelemetrySink()
    result = run_async(Runner(telemetry=sink).run(parent, user_message="go"))
    assert result.state == "completed"

    counter_names = [row["name"] for row in sink.counters()]
    histogram_names = [row["name"] for row in sink.histograms()]

    assert obs_contracts.METRIC_AGENT_SUBAGENT_NODES_TOTAL in counter_names
    assert obs_contracts.METRIC_AGENT_SUBAGENT_NODE_LATENCY_MS in histogram_names
