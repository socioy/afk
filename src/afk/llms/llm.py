from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.
"""

import asyncio
import json
import socket
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from .config import LLMConfig
from .errors import (
    LLMCapabilityError,
    LLMError,
    LLMInvalidResponseError,
    LLMRetryableError,
)
from .middleware import LLMChatNext, LLMChatStreamNext, LLMEmbedNext, MiddlewareStack
from .structured import make_repair_prompt, parse_and_validate_json
from .types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    Message,
    StreamCompletedEvent,
    ThinkingConfig,
)
from .utils import backoff_delay, run_sync

ModelT = TypeVar("ModelT", bound=BaseModel)
ReturnT = TypeVar("ReturnT")


class LLM(ABC):
    """
    Base class for provider-agnostic LLM interactions.

    Public methods define a single stable contract for agents:
      - chat/chat_sync
      - chat_stream
      - embed/embed_sync

    Adapters only implement provider transport in the abstract *_core hooks.
    """

    def __init__(
        self,
        *,
        config: LLMConfig | None = None,
        middlewares: MiddlewareStack | None = None,
        thinking_effort_aliases: Mapping[str, str] | None = None,
        supported_thinking_efforts: set[str] | None = None,
        default_thinking_effort: str | None = None,
    ) -> None:
        """
        Create a base LLM client.

        Thinking controls are configurable at instance level so callers can
        customize provider-specific effort labels without subclassing.
        Subclasses can still override provider defaults via dedicated hooks.
        """
        self.config = config or LLMConfig.from_env()
        self.middlewares = middlewares or MiddlewareStack()
        self._thinking_effort_aliases_override = self._validate_thinking_effort_aliases(
            thinking_effort_aliases
        )
        self._supported_thinking_efforts_override = (
            self._validate_supported_thinking_efforts(supported_thinking_efforts)
        )
        self._default_thinking_effort_override = (
            self._validate_optional_thinking_effort_value(
                default_thinking_effort,
                field_name="default_thinking_effort",
            )
        )

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Stable provider id (e.g. 'litellm', 'anthropic_agent')."""

    @property
    @abstractmethod
    def capabilities(self) -> LLMCapabilities:
        """Capability flags for the concrete adapter."""

    @classmethod
    def from_env(
        cls,
        *,
        middlewares: MiddlewareStack | None = None,
        thinking_effort_aliases: Mapping[str, str] | None = None,
        supported_thinking_efforts: set[str] | None = None,
        default_thinking_effort: str | None = None,
    ) -> "LLM":
        """
        Build an LLM client from environment configuration.

        If called on the abstract base class, this delegates to the adapter
        factory (`AFK_LLM_ADAPTER`). Subclasses can still call this classmethod
        to construct themselves using `LLMConfig.from_env()`.
        """
        # When called on the abstract base, resolve via factory.
        if cls is LLM:
            from .factory import create_llm_from_env

            return create_llm_from_env(
                middlewares=middlewares,
                thinking_effort_aliases=thinking_effort_aliases,
                supported_thinking_efforts=supported_thinking_efforts,
                default_thinking_effort=default_thinking_effort,
            )

        return cls(
            config=LLMConfig.from_env(),
            middlewares=middlewares,
            thinking_effort_aliases=thinking_effort_aliases,
            supported_thinking_efforts=supported_thinking_efforts,
            default_thinking_effort=default_thinking_effort,
        )

    def _provider_thinking_effort_aliases(self) -> dict[str, str]:
        """
        Provider-default aliases for request thinking effort labels.

        Subclasses can override to map generic labels (for example `balanced`)
        into provider-native labels.
        """
        return {}

    def _provider_supported_thinking_efforts(self) -> set[str] | None:
        """
        Provider-default allowed thinking effort labels.

        Return `None` to allow arbitrary effort values.
        """
        return None

    def _provider_default_thinking_effort(self) -> str | None:
        """
        Provider-default effort used when `thinking=True` and no explicit effort
        is supplied.
        """
        return None

    def thinking_effort_aliases(self) -> dict[str, str]:
        """
        Effective alias map after combining provider defaults and instance
        overrides.
        """
        aliases = dict(self._provider_thinking_effort_aliases())
        aliases.update(self._thinking_effort_aliases_override)
        return aliases

    def supported_thinking_efforts(self) -> set[str] | None:
        """
        Effective allowed effort labels.

        Instance-level values take precedence over provider defaults.
        """
        if self._supported_thinking_efforts_override is not None:
            return set(self._supported_thinking_efforts_override)

        provider_supported = self._provider_supported_thinking_efforts()
        return set(provider_supported) if provider_supported is not None else None

    def default_thinking_effort(self) -> str | None:
        """
        Effective default effort when `thinking=True` and no effort is provided.
        """
        if self._default_thinking_effort_override is not None:
            return self._default_thinking_effort_override
        return self._provider_default_thinking_effort()

    def normalize_thinking_effort(self, effort: str | None) -> str | None:
        """
        Normalize and validate a thinking effort label.

        Applies alias mapping first, then enforces allowed labels when defined.
        """
        if effort is None:
            return None

        candidate = self._validate_optional_thinking_effort_value(
            effort,
            field_name="LLMRequest.thinking_effort",
        )
        if candidate is None:
            return None

        aliases = self.thinking_effort_aliases()
        normalized = aliases.get(candidate, candidate)

        allowed = self.supported_thinking_efforts()
        if allowed is not None and normalized not in allowed:
            allowed_values = ", ".join(sorted(allowed))
            raise LLMError(
                "LLMRequest.thinking_effort is not supported by this client. "
                f"Received '{normalized}', allowed values: {allowed_values}"
            )

        return normalized

    def resolve_thinking(self, req: LLMRequest) -> ThinkingConfig:
        """
        Resolve request thinking controls into a normalized provider-agnostic
        config consumed by adapters.
        """
        effort = self.normalize_thinking_effort(req.thinking_effort)
        if effort is None and req.thinking is True:
            effort = self.normalize_thinking_effort(self.default_thinking_effort())

        return ThinkingConfig(
            enabled=req.thinking,
            effort=effort,
            max_tokens=req.max_thinking_tokens,
        )

    async def chat(
        self,
        req: LLMRequest,
        *,
        response_model: type[ModelT] | None = None,
    ) -> LLMResponse:
        """
        Execute a non-streaming chat completion.

        Applies request validation, capability checks, middleware, retry/timeout
        behavior, and optional structured-output validation.
        """
        self._ensure_capability("chat", self.capabilities.chat)
        self._validate_chat_request(req)

        if req.tools is not None and not self.capabilities.tool_calling:
            raise LLMCapabilityError(
                f"Provider '{self.provider_id}' does not support tool calling"
            )

        if response_model is not None and not self.capabilities.structured_output:
            raise LLMCapabilityError(
                f"Provider '{self.provider_id}' does not support structured outputs"
            )

        async def _base_handler(current_req: LLMRequest) -> LLMResponse:
            return await self._chat_core_with_safety(
                current_req,
                response_model=response_model,
            )

        call_next: LLMChatNext = _base_handler
        for middleware in reversed(self.middlewares.chat):
            previous = call_next

            async def _wrapped(
                current_req: LLMRequest,
                *,
                _mw=middleware,
                _next=previous,
            ) -> LLMResponse:
                return await _mw(_next, current_req)

            call_next = _wrapped

        return await call_next(req)

    def chat_sync(
        self,
        req: LLMRequest,
        *,
        response_model: type[ModelT] | None = None,
    ) -> LLMResponse:
        """Synchronous wrapper around `chat`."""
        return run_sync(self.chat(req, response_model=response_model))

    async def chat_stream(
        self,
        req: LLMRequest,
        *,
        response_model: type[ModelT] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """
        Execute a streaming chat completion.

        Emits normalized stream events and always ends with a
        `StreamCompletedEvent` carrying the final `LLMResponse`.
        """
        self._ensure_capability("chat", self.capabilities.chat)
        self._ensure_capability("streaming", self.capabilities.streaming)
        self._validate_chat_request(req)

        if req.tools is not None and not self.capabilities.tool_calling:
            raise LLMCapabilityError(
                f"Provider '{self.provider_id}' does not support tool calling"
            )

        if response_model is not None and not self.capabilities.structured_output:
            raise LLMCapabilityError(
                f"Provider '{self.provider_id}' does not support structured outputs"
            )

        def _base_handler(current_req: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
            return self._chat_stream_with_safety(
                current_req,
                response_model=response_model,
            )

        call_next: LLMChatStreamNext = _base_handler
        for middleware in reversed(self.middlewares.stream):
            previous = call_next

            def _wrapped(
                current_req: LLMRequest,
                *,
                _mw=middleware,
                _next=previous,
            ) -> AsyncIterator[LLMStreamEvent]:
                return _mw(_next, current_req)

            call_next = _wrapped

        return call_next(req)

    async def embed(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for a batch of input strings.

        Applies request validation, capability checks, middleware, and
        retry/timeout behavior.
        """
        self._ensure_capability("embeddings", self.capabilities.embeddings)
        self._validate_embedding_request(req)

        async def _base_handler(current_req: EmbeddingRequest) -> EmbeddingResponse:
            return await self._embed_core_with_safety(current_req)

        call_next: LLMEmbedNext = _base_handler
        for middleware in reversed(self.middlewares.embed):
            previous = call_next

            async def _wrapped(
                current_req: EmbeddingRequest,
                *,
                _mw=middleware,
                _next=previous,
            ) -> EmbeddingResponse:
                return await _mw(_next, current_req)

            call_next = _wrapped

        return await call_next(req)

    def embed_sync(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Synchronous wrapper around `embed`."""
        return run_sync(self.embed(req))

    async def _chat_core_with_safety(
        self,
        req: LLMRequest,
        *,
        response_model: type[ModelT] | None,
    ) -> LLMResponse:
        """Run provider chat call under timeout/retry and structured checks."""
        timeout = req.timeout_s if req.timeout_s is not None else self.config.timeout_s

        async def _provider_call() -> LLMResponse:
            if timeout is None:
                return await self._chat_core(req, response_model=response_model)
            return await asyncio.wait_for(
                self._chat_core(req, response_model=response_model),
                timeout=timeout,
            )

        response = await self._call_with_retries(_provider_call)
        if response_model is None:
            return response

        return await self._ensure_structured_response(
            req,
            response,
            response_model=response_model,
        )

    def _chat_stream_with_safety(
        self,
        req: LLMRequest,
        *,
        response_model: type[ModelT] | None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Run provider streaming call with retry setup and completion validation."""
        async def _run() -> AsyncIterator[LLMStreamEvent]:
            timeout = (
                req.timeout_s if req.timeout_s is not None else self.config.timeout_s
            )

            async def _provider_call() -> AsyncIterator[LLMStreamEvent]:
                if timeout is None:
                    return await self._chat_stream_core(
                        req,
                        response_model=response_model,
                    )
                return await asyncio.wait_for(
                    self._chat_stream_core(
                        req,
                        response_model=response_model,
                    ),
                    timeout=timeout,
                )

            stream = await self._call_with_retries(_provider_call)

            async def _guarded() -> AsyncIterator[LLMStreamEvent]:
                try:
                    async for event in stream:
                        if isinstance(event, StreamCompletedEvent):
                            if response_model is None:
                                yield event
                                continue
                            validated = await self._ensure_structured_response(
                                req,
                                event.response,
                                response_model=response_model,
                            )
                            yield StreamCompletedEvent(response=validated)
                            continue

                        yield event
                except Exception as e:
                    classified = e if isinstance(e, LLMError) else self._classify_error(e)
                    raise classified from e

            return _guarded()

        async def _iter() -> AsyncIterator[LLMStreamEvent]:
            stream_iter = await _run()
            async for event in stream_iter:
                yield event

        return _iter()

    async def _embed_core_with_safety(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Run provider embedding call under timeout/retry policies."""
        timeout = req.timeout_s if req.timeout_s is not None else self.config.timeout_s

        async def _provider_call() -> EmbeddingResponse:
            if timeout is None:
                return await self._embed_core(req)
            return await asyncio.wait_for(
                self._embed_core(req),
                timeout=timeout,
            )

        return await self._call_with_retries(_provider_call)

    async def _ensure_structured_response(
        self,
        req: LLMRequest,
        initial_response: LLMResponse,
        *,
        response_model: type[ModelT],
    ) -> LLMResponse:
        """
        Ensure response matches a structured schema.

        Validates provider-native structured payloads when present. Otherwise it
        parses JSON from text and retries repair prompts on failure.
        """
        response = initial_response
        last_error: Exception | None = None

        for attempt in range(self.config.json_max_retries + 1):
            try:
                structured = self._validate_structured_payload(response, response_model)
                return replace(response, structured_response=structured)
            except LLMInvalidResponseError as e:
                last_error = e
                if attempt >= self.config.json_max_retries:
                    break

            repair_prompt = make_repair_prompt(response.text, response_model)
            repair_req = self._make_repair_request(req, repair_prompt)
            response = await self._call_with_retries(
                lambda: self._chat_core(repair_req, response_model=response_model)
            )

        raise LLMInvalidResponseError(
            f"Structured output remained invalid after {self.config.json_max_retries + 1} attempts."
        ) from last_error

    def _validate_structured_payload(
        self,
        response: LLMResponse,
        response_model: type[ModelT],
    ) -> dict[str, Any]:
        """Validate structured payload or parse validated JSON from response text."""
        if response.structured_response is not None:
            payload: Any = response.structured_response
            if isinstance(payload, BaseModel):
                payload = payload.model_dump(mode="json")

            if not isinstance(payload, dict):
                raise LLMInvalidResponseError(
                    "Provider structured payload must be a JSON object"
                )

            try:
                return response_model.model_validate(payload).model_dump(mode="json")
            except ValidationError as e:
                raise LLMInvalidResponseError(
                    f"Provider structured payload did not match schema: {e}"
                ) from e

        try:
            parsed = parse_and_validate_json(response.text, response_model)
            return parsed.model_dump(mode="json")
        except LLMInvalidResponseError:
            raise
        except ValidationError as e:
            raise LLMInvalidResponseError(
                f"Parsed JSON did not match schema: {e}"
            ) from e

    def _make_repair_request(self, req: LLMRequest, repair_prompt: str) -> LLMRequest:
        """Create a follow-up repair request for schema correction retries."""
        return replace(
            req,
            messages=[*req.messages, Message(role="user", content=repair_prompt)],
            tools=None,
            tool_choice=None,
        )

    def _ensure_capability(self, capability: str, enabled: bool) -> None:
        """Raise a capability error when an adapter lacks a required feature."""
        if enabled:
            return
        raise LLMCapabilityError(
            f"Provider '{self.provider_id}' does not support capability '{capability}'"
        )

    def _validate_chat_request(self, req: LLMRequest) -> None:
        """Validate chat request structure and enforce global input limits."""
        if not req.model or not req.model.strip():
            raise LLMError("LLMRequest.model must be a non-empty string")

        if not req.messages:
            raise LLMError("LLMRequest.messages must contain at least one message")

        if req.max_tokens is not None and req.max_tokens <= 0:
            raise LLMError("LLMRequest.max_tokens must be greater than 0")

        if req.timeout_s is not None and req.timeout_s <= 0:
            raise LLMError("LLMRequest.timeout_s must be greater than 0")

        if req.temperature is not None and req.temperature < 0:
            raise LLMError("LLMRequest.temperature must be >= 0")

        if req.top_p is not None and (req.top_p <= 0 or req.top_p > 1):
            raise LLMError("LLMRequest.top_p must be in (0, 1]")

        if req.max_thinking_tokens is not None and req.max_thinking_tokens <= 0:
            raise LLMError("LLMRequest.max_thinking_tokens must be greater than 0")

        if req.thinking is False and (
            req.thinking_effort is not None or req.max_thinking_tokens is not None
        ):
            raise LLMError(
                "LLMRequest.thinking=False cannot be combined with thinking_effort/max_thinking_tokens"
            )

        # Validate provider/instance-specific effort mapping and defaults.
        self.resolve_thinking(req)

        total_chars = 0
        for idx, message in enumerate(req.messages):
            self._validate_message(message, idx)
            total_chars += self._message_char_count(message)

        if total_chars > self.config.max_input_chars:
            raise LLMError(
                f"LLMRequest exceeds max input chars ({self.config.max_input_chars})"
            )

        if req.tools is not None:
            for idx, tool in enumerate(req.tools):
                self._validate_tool_definition(tool, idx)

        if isinstance(req.tool_choice, dict):
            choice_type = req.tool_choice.get("type")
            fn = req.tool_choice.get("function")
            if choice_type != "function" or not isinstance(fn, dict):
                raise LLMError("LLMRequest.tool_choice dict must be a function choice")
            fn_name = fn.get("name")
            if not isinstance(fn_name, str) or not fn_name.strip():
                raise LLMError("LLMRequest.tool_choice.function.name must be set")

    def _validate_embedding_request(self, req: EmbeddingRequest) -> None:
        """Validate embedding request structure and enforce global input limits."""
        if not req.model or not req.model.strip():
            raise LLMError("EmbeddingRequest.model must be a non-empty string")

        if not req.inputs:
            raise LLMError("EmbeddingRequest.inputs must contain at least one input")

        if req.timeout_s is not None and req.timeout_s <= 0:
            raise LLMError("EmbeddingRequest.timeout_s must be greater than 0")

        total_chars = 0
        for idx, value in enumerate(req.inputs):
            if not isinstance(value, str) or not value.strip():
                raise LLMError(
                    f"EmbeddingRequest.inputs[{idx}] must be a non-empty string"
                )
            total_chars += len(value)

        if total_chars > self.config.max_input_chars:
            raise LLMError(
                f"EmbeddingRequest exceeds max input chars ({self.config.max_input_chars})"
            )

    def _validate_message(self, message: Message, idx: int) -> None:
        """Validate one normalized message and its content parts."""
        if message.role not in ("user", "assistant", "system", "tool"):
            raise LLMError(f"LLMRequest.messages[{idx}] has unsupported role")

        content = message.content
        if isinstance(content, str):
            return

        if not isinstance(content, list):
            raise LLMError(
                f"LLMRequest.messages[{idx}].content must be a string or list of parts"
            )

        for p_idx, part in enumerate(content):
            if not isinstance(part, dict):
                raise LLMError(
                    f"LLMRequest.messages[{idx}].content[{p_idx}] must be an object"
                )

            p_type = part.get("type")
            if p_type == "text":
                if not isinstance(part.get("text"), str):
                    raise LLMError(
                        f"LLMRequest.messages[{idx}].content[{p_idx}].text must be a string"
                    )
                continue

            if p_type == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict) or not isinstance(
                    image_url.get("url"), str
                ):
                    raise LLMError(
                        f"LLMRequest.messages[{idx}].content[{p_idx}].image_url.url must be a string"
                    )
                continue

            if p_type == "tool_use":
                if not isinstance(part.get("id"), str) or not isinstance(
                    part.get("name"), str
                ):
                    raise LLMError(
                        f"LLMRequest.messages[{idx}].content[{p_idx}] invalid tool_use fields"
                    )
                if not isinstance(part.get("input"), dict):
                    raise LLMError(
                        f"LLMRequest.messages[{idx}].content[{p_idx}].input must be an object"
                    )
                continue

            if p_type == "tool_result":
                if not isinstance(part.get("tool_use_id"), str) or not isinstance(
                    part.get("content"), str
                ):
                    raise LLMError(
                        f"LLMRequest.messages[{idx}].content[{p_idx}] invalid tool_result fields"
                    )
                continue

            raise LLMError(
                f"LLMRequest.messages[{idx}].content[{p_idx}] has unsupported part type"
            )

    def _validate_tool_definition(self, tool: dict[str, Any], idx: int) -> None:
        """Validate one provider-neutral tool definition."""
        if tool.get("type") != "function":
            raise LLMError(f"LLMRequest.tools[{idx}] must have type='function'")

        function = tool.get("function")
        if not isinstance(function, dict):
            raise LLMError(f"LLMRequest.tools[{idx}].function must be an object")

        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise LLMError(
                f"LLMRequest.tools[{idx}].function.name must be a non-empty string"
            )

        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise LLMError(
                f"LLMRequest.tools[{idx}].function.parameters must be an object"
            )

    def _validate_thinking_effort_aliases(
        self,
        aliases: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """Validate and normalize instance-level effort alias mapping."""
        if aliases is None:
            return {}

        out: dict[str, str] = {}
        for key, value in aliases.items():
            normalized_key = self._validate_optional_thinking_effort_value(
                key,
                field_name="thinking_effort_aliases key",
            )
            normalized_value = self._validate_optional_thinking_effort_value(
                value,
                field_name="thinking_effort_aliases value",
            )
            if normalized_key is None or normalized_value is None:
                raise LLMError("thinking_effort_aliases cannot contain empty keys/values")
            out[normalized_key] = normalized_value
        return out

    def _validate_supported_thinking_efforts(
        self,
        supported: set[str] | None,
    ) -> set[str] | None:
        """Validate and normalize instance-level allowed effort labels."""
        if supported is None:
            return None

        out: set[str] = set()
        for effort in supported:
            normalized = self._validate_optional_thinking_effort_value(
                effort,
                field_name="supported_thinking_efforts item",
            )
            if normalized is None:
                raise LLMError("supported_thinking_efforts cannot contain empty values")
            out.add(normalized)
        return out

    def _validate_optional_thinking_effort_value(
        self,
        value: str | None,
        *,
        field_name: str,
    ) -> str | None:
        """Validate a single optional thinking effort label."""
        if value is None:
            return None

        if not isinstance(value, str):
            raise LLMError(f"{field_name} must be a string")

        normalized = value.strip()
        if not normalized:
            raise LLMError(f"{field_name} must be a non-empty string")
        return normalized

    def _message_char_count(self, message: Message) -> int:
        """Best-effort character count used for input size limiting."""
        content = message.content
        if isinstance(content, str):
            return len(content)

        total = 0
        for part in content:
            p_type = part.get("type")
            if p_type == "text":
                text = part.get("text")
                total += len(text) if isinstance(text, str) else 0
            elif p_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    total += len(url) if isinstance(url, str) else 0
            else:
                total += len(json.dumps(part, ensure_ascii=True, default=str))
        return total

    async def _call_with_retries(
        self,
        fn: Callable[[], Awaitable[ReturnT]],
    ) -> ReturnT:
        """Execute a callable with retry-on-transient-error semantics."""
        last: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await fn()
            except Exception as e:
                classified = e if isinstance(e, LLMError) else self._classify_error(e)
                last = classified

                retryable = isinstance(classified, LLMRetryableError)
                if (not retryable) or attempt >= self.config.max_retries:
                    raise classified from e

                await asyncio.sleep(
                    backoff_delay(
                        attempt,
                        self.config.backoff_base_s,
                        self.config.backoff_jitter_s,
                    )
                )

        raise LLMError(
            f"LLM call failed after {self.config.max_retries} retries"
        ) from last

    def _classify_error(self, e: Exception) -> LLMError:
        """Map arbitrary exceptions into retryable vs non-retryable LLM errors."""
        msg = ""
        try:
            msg = str(e) or ""
        except Exception:
            msg = repr(e)

        m = msg.lower()
        status = None

        for attr in ("status_code", "status", "code", "errno"):
            val = getattr(e, attr, None)
            if isinstance(val, int):
                status = val
                break
            if isinstance(val, str) and val.isdigit():
                status = int(val)
                break

        resp = getattr(e, "response", None) or getattr(e, "resp", None)
        if resp is not None:
            try:
                sc = getattr(resp, "status_code", None) or getattr(resp, "status", None)
                if isinstance(sc, int):
                    status = sc
                elif isinstance(sc, str) and sc.isdigit():
                    status = int(sc)
            except Exception:
                pass

        if status is not None:
            if status == 429 or status == 408 or 500 <= status < 600:
                return LLMRetryableError(msg)
            if 400 <= status < 500:
                return LLMError(msg)

        transient_types = (
            asyncio.TimeoutError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            OSError,
        )
        if isinstance(e, transient_types):
            return LLMRetryableError(msg)

        retry_phrases = (
            "rate limit",
            "rate_limit",
            "rate-limit",
            "quota exceeded",
            "temporarily unavailable",
            "overloaded",
            "service unavailable",
            "try again",
            "please retry",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection error",
            "dns",
            "name or service not known",
            "econnreset",
            "econnrefused",
            "ehostunreach",
            "eai_again",
            "502",
            "503",
            "504",
            "500",
            "408",
            "429",
        )
        if any(phrase in m for phrase in retry_phrases):
            return LLMRetryableError(msg)

        non_retry_phrases = (
            "invalid api key",
            "invalid_api_key",
            "unauthorized",
            "authentication failed",
            "forbidden",
            "permission denied",
            "not found",
            "bad request",
            "invalid request",
            "unsupported",
            "unprocessable entity",
            "billing",
        )
        if any(phrase in m for phrase in non_retry_phrases):
            return LLMError(msg)

        return LLMError(msg)

    @abstractmethod
    async def _chat_core(
        self,
        req: LLMRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """
        Provider-specific chat implementation.

        Providers that support native structured decoding can return
        `LLMResponse.structured_response` directly when `response_model` is set.
        """

    @abstractmethod
    async def _chat_stream_core(
        self,
        req: LLMRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Provider-specific streaming chat implementation."""

    @abstractmethod
    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Provider-specific embedding implementation."""
