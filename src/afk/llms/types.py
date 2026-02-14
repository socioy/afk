from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module defines common provider-agnostic types used in LLM interactions.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, NotRequired, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    from pydantic import BaseModel


JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
JSONSchema: TypeAlias = dict[str, JSONValue]

Role = Literal["user", "assistant", "system", "tool"]


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageURLRef(TypedDict):
    url: str


class ImageURLContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURLRef


class ToolUseContentPart(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: JSONObject


class ToolResultContentPart(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str
    is_error: NotRequired[bool]


MessagePart: TypeAlias = (
    TextContentPart
    | ImageURLContentPart
    | ToolUseContentPart
    | ToolResultContentPart
)
MessageContent: TypeAlias = str | list[MessagePart]


class ToolFunctionSpec(TypedDict):
    name: str
    parameters: JSONSchema
    description: NotRequired[str]


class ToolDefinition(TypedDict):
    type: Literal["function"]
    function: ToolFunctionSpec


class ToolChoiceFunction(TypedDict):
    name: str


class ToolChoiceNamed(TypedDict):
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice: TypeAlias = Literal["auto", "none", "required"] | ToolChoiceNamed
# Provider-specific effort labels (for example: "low", "medium", "high",
# "minimal", "balanced", etc.). Validation is adapter-driven in `LLM`.
ThinkingEffort: TypeAlias = str


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: MessageContent
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """
    Data-only representation of a model-returned tool call.
    The agent module decides if/when/how to execute this.
    """

    id: str | None = None
    tool_name: str = ""
    arguments: JSONObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    request_id: str | None = None
    provider_request_id: str | None = None
    session_token: str | None = None
    checkpoint_token: str | None = None
    structured_response: JSONObject | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    raw: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """
    Canonical request type used by middleware, the client and agents.
    """

    model: str
    request_id: str | None = None
    idempotency_key: str | None = None
    session_token: str | None = None
    checkpoint_token: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    thinking: bool | None = None
    thinking_effort: ThinkingEffort | None = None
    max_thinking_tokens: int | None = None
    timeout_s: float | None = None
    metadata: JSONObject = field(default_factory=dict)
    extra: JSONObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    model: str | None = None
    inputs: list[str] = field(default_factory=list)
    timeout_s: float | None = None
    metadata: JSONObject = field(default_factory=dict)
    extra: JSONObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMCapabilities:
    chat: bool = True
    streaming: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    embeddings: bool = False
    interrupt: bool = False
    session_control: bool = False
    checkpoint_resume: bool = False
    idempotency: bool = False


@dataclass(frozen=True, slots=True)
class ThinkingConfig:
    """
    Normalized thinking controls resolved by the base `LLM`.

    Adapters consume this shape so thinking behavior can be overridden per
    client instance or subclass while keeping request/response contracts stable.
    """

    enabled: bool | None = None
    effort: ThinkingEffort | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StreamMessageStartEvent:
    type: Literal["message_start"] = "message_start"
    model: str | None = None


@dataclass(frozen=True, slots=True)
class StreamTextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    delta: str = ""


@dataclass(frozen=True, slots=True)
class StreamToolCallDeltaEvent:
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int = 0
    call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True, slots=True)
class StreamMessageStopEvent:
    type: Literal["message_stop"] = "message_stop"
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StreamErrorEvent:
    error: str
    type: Literal["error"] = "error"


@dataclass(frozen=True, slots=True)
class StreamCompletedEvent:
    response: LLMResponse
    type: Literal["completed"] = "completed"


LLMStreamEvent: TypeAlias = (
    StreamMessageStartEvent
    | StreamTextDeltaEvent
    | StreamToolCallDeltaEvent
    | StreamMessageStopEvent
    | StreamErrorEvent
    | StreamCompletedEvent
)


@dataclass(frozen=True, slots=True)
class LLMSessionSnapshot:
    session_token: str | None = None
    checkpoint_token: str | None = None
    paused: bool = False
    closed: bool = False


class LLMStreamHandle(Protocol):
    """
    Control handle for a streaming chat request.

    `events` yields normalized stream events from the underlying provider call.
    `await_result` returns the final normalized response when completion occurs,
    or `None` when the stream was cancelled/interrupted before completion.
    """

    @property
    def events(self) -> AsyncIterator[LLMStreamEvent]:
        ...

    async def cancel(self) -> None:
        ...

    async def interrupt(self) -> None:
        ...

    async def await_result(self) -> LLMResponse | None:
        ...


class LLMSessionHandle(Protocol):
    """
    Provider/session continuity handle.

    The agent layer remains responsible for prompt/tool orchestration; this
    handle only manages transport-level session controls.
    """

    async def chat(
        self,
        req: LLMRequest,
        *,
        response_model: type["BaseModel"] | None = None,
    ) -> LLMResponse:
        ...

    async def stream(
        self,
        req: LLMRequest,
        *,
        response_model: type["BaseModel"] | None = None,
    ) -> LLMStreamHandle:
        ...

    async def pause(self) -> None:
        ...

    async def resume(self, session_token: str | None = None) -> None:
        ...

    async def interrupt(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def snapshot(self) -> LLMSessionSnapshot:
        ...
