from __future__ import annotations

"""
Shared base adapter for providers exposing OpenAI-style Responses APIs.

This class centralizes:
  - request -> Responses payload mapping
  - stream event normalization
  - response/tool-call extraction
  - embedding payload/response normalization

Concrete adapters only implement transport and provider-specific message/JSON
schema mapping.
"""

from abc import abstractmethod
from collections import defaultdict
from typing import Any, AsyncIterator

from pydantic import BaseModel

from ..shared.normalization import extract_usage, finalize_stream_tool_calls, to_plain_dict
from ...llm import LLM
from ...types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    Message,
    StreamCompletedEvent,
    StreamMessageStartEvent,
    StreamMessageStopEvent,
    StreamTextDeltaEvent,
    StreamToolCallDeltaEvent,
    ToolCall,
    Usage,
)
from ...utils import safe_json_loads


class ResponsesClientBase(LLM):
    """Provider-agnostic base for Responses-compatible clients."""

    _CAPABILITIES = LLMCapabilities(
        chat=True,
        streaming=True,
        tool_calling=True,
        structured_output=True,
        embeddings=True,
    )

    @property
    def capabilities(self) -> LLMCapabilities:
        """Responses adapters expose chat/stream/tool/structured/embed."""
        return self._CAPABILITIES

    async def _chat_core(
        self,
        req: LLMRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Execute non-streaming call using provider transport hook."""
        payload = self._build_responses_payload(
            req,
            response_model=response_model,
            stream=False,
        )
        raw = await self._responses_create(payload)
        return self._normalize_responses_response(raw)

    async def _chat_stream_core(
        self,
        req: LLMRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Execute streaming call and normalize provider stream events."""
        payload = self._build_responses_payload(
            req,
            response_model=response_model,
            stream=True,
        )
        raw_stream = await self._responses_create(payload)

        async def _iter() -> AsyncIterator[LLMStreamEvent]:
            yield StreamMessageStartEvent(model=req.model)

            text_chunks: list[str] = []
            tool_buffers: dict[int, dict[str, Any]] = defaultdict(
                lambda: {"id": None, "name": None, "args_parts": []}
            )
            completed_response_payload: dict[str, Any] | None = None

            async for event in raw_stream:
                event_dict = to_plain_dict(event)
                event_type = event_dict.get("type")

                if event_type == "response.output_text.delta":
                    delta = event_dict.get("delta")
                    if isinstance(delta, str) and delta:
                        text_chunks.append(delta)
                        yield StreamTextDeltaEvent(delta=delta)
                    continue

                if event_type in ("response.output_item.added", "response.output_item.done"):
                    output_index = event_dict.get("output_index")
                    item = event_dict.get("item")
                    if isinstance(output_index, int) and isinstance(item, dict):
                        self._update_stream_tool_buffer_from_item(
                            output_index=output_index,
                            item=item,
                            tool_buffers=tool_buffers,
                        )
                    continue

                if event_type == "response.function_call_arguments.delta":
                    output_index = event_dict.get("output_index")
                    if not isinstance(output_index, int):
                        output_index = 0

                    delta = event_dict.get("delta")
                    if isinstance(delta, str) and delta:
                        buf = tool_buffers[output_index]
                        buf["args_parts"].append(delta)
                        yield StreamToolCallDeltaEvent(
                            index=output_index,
                            call_id=buf.get("id") if isinstance(buf.get("id"), str) else None,
                            tool_name=buf.get("name") if isinstance(buf.get("name"), str) else None,
                            arguments_delta=delta,
                        )
                    continue

                if event_type == "response.completed":
                    response_obj = event_dict.get("response")
                    if isinstance(response_obj, dict):
                        completed_response_payload = response_obj
                    continue

            if completed_response_payload is not None:
                response = self._normalize_responses_response(completed_response_payload)
            else:
                response = LLMResponse(
                    text="".join(text_chunks),
                    tool_calls=finalize_stream_tool_calls(tool_buffers),
                    finish_reason=None,
                    usage=Usage(),
                    raw={"provider": self._stream_provider_label()},
                    model=req.model,
                )

            yield StreamMessageStopEvent(finish_reason=response.finish_reason)
            yield StreamCompletedEvent(response=response)

        return _iter()

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Execute embedding call using provider transport hook."""
        payload: dict[str, Any] = {
            "model": req.model,
            "input": req.inputs,
        }
        if req.timeout_s is not None:
            payload["timeout"] = req.timeout_s

        payload.update(req.extra)

        raw = await self._embedding_create(payload)
        raw_dict = to_plain_dict(raw)
        data = raw_dict.get("data")

        embeddings: list[list[float]] = []
        if isinstance(data, list):
            for row in data:
                item = to_plain_dict(row)
                emb = item.get("embedding")
                if isinstance(emb, list):
                    embeddings.append([float(v) for v in emb])

        return EmbeddingResponse(
            embeddings=embeddings,
            raw=raw_dict,
            model=raw_dict.get("model") if isinstance(raw_dict.get("model"), str) else req.model,
        )

    def _build_responses_payload(
        self,
        req: LLMRequest,
        *,
        response_model: type[BaseModel] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Map normalized `LLMRequest` into a Responses API payload."""
        thinking = self.resolve_thinking(req)
        payload: dict[str, Any] = {
            "model": req.model,
            "input": self._messages_to_responses_input(req.messages),
            "stream": stream,
        }

        if req.max_tokens is not None:
            payload["max_output_tokens"] = req.max_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p

        reasoning: dict[str, Any] = {}
        if thinking.effort is not None:
            reasoning["effort"] = thinking.effort
        elif thinking.enabled is False:
            reasoning["effort"] = "none"
        if reasoning:
            payload["reasoning"] = reasoning

        if req.timeout_s is not None:
            payload["timeout"] = req.timeout_s

        if req.tools is not None:
            payload["tools"] = [self._tool_to_responses_tool(t) for t in req.tools]

        if req.tool_choice is not None:
            payload["tool_choice"] = self._tool_choice_to_responses_tool_choice(req.tool_choice)

        if req.metadata:
            payload["metadata"] = req.metadata

        if response_model is not None:
            payload.update(self._structured_output_payload(response_model))

        payload.update(req.extra)
        return payload

    def _messages_to_responses_input(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert normalized messages to Responses API input items."""
        out: list[dict[str, Any]] = []
        for message in messages:
            out.extend(self._message_to_responses_input_items(message))
        return out

    def _tool_to_responses_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert normalized function tool definition to Responses tool schema."""
        if tool.get("type") != "function":
            return dict(tool)

        function = tool.get("function")
        if not isinstance(function, dict):
            return dict(tool)

        out: dict[str, Any] = {
            "type": "function",
            "name": function.get("name"),
            "parameters": function.get("parameters"),
        }

        description = function.get("description")
        if isinstance(description, str):
            out["description"] = description

        return out

    def _tool_choice_to_responses_tool_choice(self, tool_choice: Any) -> Any:
        """Convert normalized tool-choice into Responses `tool_choice` shape."""
        if isinstance(tool_choice, str):
            return tool_choice

        if not isinstance(tool_choice, dict):
            return tool_choice

        if tool_choice.get("type") != "function":
            return tool_choice

        function = tool_choice.get("function")
        if not isinstance(function, dict):
            return tool_choice

        name = function.get("name")
        if not isinstance(name, str):
            return tool_choice

        return {
            "type": "function",
            "name": name,
        }

    def _normalize_responses_response(self, raw: Any) -> LLMResponse:
        """Normalize raw Responses payload into `LLMResponse`."""
        raw_dict = to_plain_dict(raw)
        model = raw_dict.get("model") if isinstance(raw_dict.get("model"), str) else None
        usage = extract_usage(raw_dict)
        output = raw_dict.get("output")

        output_items = output if isinstance(output, list) else []
        text = self._extract_text_from_responses_output(output_items)
        tool_calls = self._extract_tool_calls_from_responses_output(output_items)

        structured_response = None
        parsed = raw_dict.get("output_parsed")
        if isinstance(parsed, BaseModel):
            as_dict = parsed.model_dump(mode="json")
            if isinstance(as_dict, dict):
                structured_response = as_dict
        elif isinstance(parsed, dict):
            structured_response = parsed

        if structured_response is None:
            structured_response = self._extract_structured_from_responses_output(output_items)

        finish_reason = raw_dict.get("status")
        if not isinstance(finish_reason, str):
            finish_reason = None

        return LLMResponse(
            text=text,
            structured_response=structured_response,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=raw_dict,
            model=model,
        )

    def _extract_text_from_responses_output(self, output_items: list[Any]) -> str:
        """Extract assistant text from Responses output message items."""
        chunks: list[str] = []
        for item in output_items:
            row = to_plain_dict(item)
            if row.get("type") != "message":
                continue

            content = row.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue

            if not isinstance(content, list):
                continue

            for part in content:
                block = to_plain_dict(part)
                p_type = block.get("type")
                if p_type in ("output_text", "text", "input_text"):
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)

        return "".join(chunks)

    def _extract_structured_from_responses_output(
        self,
        output_items: list[Any],
    ) -> dict[str, Any] | None:
        """Extract provider-native parsed JSON payload when available."""
        for item in output_items:
            row = to_plain_dict(item)
            if row.get("type") != "message":
                continue

            content = row.get("content")
            if not isinstance(content, list):
                continue

            for part in content:
                block = to_plain_dict(part)
                parsed = block.get("parsed")
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, BaseModel):
                    as_dict = parsed.model_dump(mode="json")
                    if isinstance(as_dict, dict):
                        return as_dict

        return None

    def _extract_tool_calls_from_responses_output(
        self,
        output_items: list[Any],
    ) -> list[ToolCall]:
        """Extract normalized function tool calls from Responses output items."""
        out: list[ToolCall] = []
        for item in output_items:
            row = to_plain_dict(item)
            if row.get("type") != "function_call":
                continue

            name = row.get("name") if isinstance(row.get("name"), str) else ""
            call_id = row.get("call_id") if isinstance(row.get("call_id"), str) else None
            if call_id is None and isinstance(row.get("id"), str):
                call_id = row.get("id")

            arguments: dict[str, Any] = {}
            raw_args = row.get("arguments")
            if isinstance(raw_args, dict):
                arguments = raw_args
            elif isinstance(raw_args, str):
                parsed = safe_json_loads(raw_args)
                if isinstance(parsed, dict):
                    arguments = parsed

            out.append(ToolCall(id=call_id, tool_name=name, arguments=arguments))

        return out

    def _update_stream_tool_buffer_from_item(
        self,
        *,
        output_index: int,
        item: dict[str, Any],
        tool_buffers: dict[int, dict[str, Any]],
    ) -> None:
        """Update stream tool-call buffer from `response.output_item.*` event."""
        if item.get("type") != "function_call":
            return

        buf = tool_buffers[output_index]

        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            maybe_id = item.get("id")
            if isinstance(maybe_id, str):
                call_id = maybe_id

        if isinstance(call_id, str):
            buf["id"] = call_id

        name = item.get("name")
        if isinstance(name, str):
            buf["name"] = name

        arguments = item.get("arguments")
        if isinstance(arguments, str) and arguments and not buf["args_parts"]:
            buf["args_parts"].append(arguments)

    def _stream_provider_label(self) -> str:
        """Raw-provider label attached to synthesized stream fallback payloads."""
        return f"{self.provider_id}_responses_stream"

    @abstractmethod
    async def _responses_create(self, payload: dict[str, Any]) -> Any:
        """Provider transport hook for Responses API calls."""

    @abstractmethod
    async def _embedding_create(self, payload: dict[str, Any]) -> Any:
        """Provider transport hook for embedding API calls."""

    @abstractmethod
    def _message_to_responses_input_items(self, message: Message) -> list[dict[str, Any]]:
        """Provider-specific mapping from one normalized message to input items."""

    @abstractmethod
    def _structured_output_payload(
        self,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        """Provider-specific structured-output payload fragment."""
