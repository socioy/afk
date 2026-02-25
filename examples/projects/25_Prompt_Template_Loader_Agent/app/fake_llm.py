"""Deterministic LLM used to make prompt loading behavior easy to inspect."""

from collections.abc import AsyncIterator

from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
)


class PromptAwareLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "prompt-aware"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=False)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        system_lines = [m.content for m in req.messages if m.role == "system"]
        user_lines = [m.content for m in req.messages if m.role == "user"]
        return LLMResponse(
            text=(
                f"prompt_lines={len(system_lines)} "
                f"user_lines={len(user_lines)} "
                f"prompt_size={len(str(system_lines[0])) if system_lines else 0}"
            )
        )

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError
