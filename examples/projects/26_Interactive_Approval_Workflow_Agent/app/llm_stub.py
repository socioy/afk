"""LLM stub that intentionally triggers a gated tool call."""

from collections.abc import AsyncIterator

from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    ToolCall,
)


class InputGatedLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "interactive-stub"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=True)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc_interactive_1",
                        tool_name="echo_change_plan",
                        arguments={},
                    )
                ],
            )
        return LLMResponse(text="Approved plan captured and logged.")

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError
