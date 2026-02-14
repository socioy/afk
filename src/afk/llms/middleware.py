from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

Module defining middleware protocols and stack for LLM.
"""

from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Protocol

from .types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)


LLMChatNext = Callable[[LLMRequest], Awaitable[LLMResponse]]
LLMEmbedNext = Callable[[EmbeddingRequest], Awaitable[EmbeddingResponse]]
LLMChatStreamNext = Callable[[LLMRequest], AsyncIterator[LLMStreamEvent]]


class LLMChatMiddleware(Protocol):
    async def __call__(
        self, call_next: LLMChatNext, req: LLMRequest
    ) -> LLMResponse: ...


class LLMEmbedMiddleware(Protocol):
    async def __call__(
        self, call_next: LLMEmbedNext, req: EmbeddingRequest
    ) -> EmbeddingResponse: ...


class LLMStreamMiddleware(Protocol):
    def __call__(
        self, call_next: LLMChatStreamNext, req: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]: ...


@dataclass
class MiddlewareStack:
    chat: list[LLMChatMiddleware]
    embed: list[LLMEmbedMiddleware]
    stream: list[LLMStreamMiddleware]

    def __init__(
        self,
        chat: list[LLMChatMiddleware] | None = None,
        embed: list[LLMEmbedMiddleware] | None = None,
        stream: list[LLMStreamMiddleware] | None = None,
    ) -> None:
        self.chat = chat or []
        self.embed = embed or []
        self.stream = stream or []
