from __future__ import annotations

"""
Claude Agent SDK-backed Anthropic adapter.

This adapter uses `claude_agent_sdk.query`/`ClaudeAgentOptions` and normalizes
SDK messages/events to AFK's provider-agnostic LLM response and stream types.
"""

import json
from typing import Any, AsyncIterator, Iterable

from pydantic import BaseModel

from ..shared.normalization import get_attr, get_attr_str, to_jsonable, to_plain_dict
from ...errors import LLMCapabilityError, LLMConfigurationError
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


class AnthropicAgentClient(LLM):
    """Concrete adapter that integrates with `claude-agent-sdk`."""
    _CAPABILITIES = LLMCapabilities(
        chat=True,
        streaming=True,
        tool_calling=True,
        structured_output=True,
        embeddings=False,
        interrupt=True,
        session_control=True,
        checkpoint_resume=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active_stream_clients: dict[str, Any] = {}

    @property
    def provider_id(self) -> str:
        return "anthropic_agent"

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._CAPABILITIES

    async def _chat_core(
        self,
        req: LLMRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Execute one non-streaming SDK query and normalize final response."""
        if req.stop is not None:
            raise LLMCapabilityError(
                "AnthropicAgentClient does not support request-level `stop` in this transport."
            )

        query_fn, options_type, _ = self._load_sdk_api()

        prompt, system_prompt = self._build_prompt(req.messages)
        options = self._build_sdk_options(
            req,
            options_type=options_type,
            system_prompt=system_prompt,
            response_model=response_model,
            include_partial_messages=False,
        )

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None
        structured: dict[str, Any] | None = None
        usage = Usage()
        model: str | None = None
        session_token: str | None = None
        checkpoint_token: str | None = None
        raw_messages: list[dict[str, Any]] = []

        async for message in query_fn(prompt=prompt, options=options):
            raw_messages.append(self._serialize_sdk_message(message))

            msg_type = type(message).__name__
            if msg_type == "AssistantMessage":
                model = get_attr_str(message, "model") or model
                blocks = self._iter_content_blocks(message)
                text_chunks.extend(self._extract_text_blocks(blocks))
                tool_calls.extend(self._extract_tool_blocks(blocks))

            elif msg_type == "ResultMessage":
                finish_reason = get_attr_str(message, "subtype") or finish_reason
                usage = self._usage_from_obj(get_attr(message, "usage"))
                result_structured = self._extract_result_structured(message)
                if isinstance(result_structured, dict):
                    structured = result_structured
                session_token = self._extract_result_session_token(message) or session_token
                checkpoint_token = (
                    self._extract_result_checkpoint_token(message) or checkpoint_token
                )

        return LLMResponse(
            text="".join(text_chunks),
            request_id=req.request_id,
            session_token=session_token or req.session_token,
            checkpoint_token=checkpoint_token or req.checkpoint_token,
            structured_response=structured,
            tool_calls=self._dedupe_tool_calls(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
            raw={"provider": "claude_agent_sdk", "messages": raw_messages},
            model=model or req.model,
        )

    async def _chat_stream_core(
        self,
        req: LLMRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Execute streaming SDK query and emit normalized stream events."""
        if req.stop is not None:
            raise LLMCapabilityError(
                "AnthropicAgentClient does not support request-level `stop` in this transport."
            )

        query_fn, options_type, sdk_client_type = self._load_sdk_api()

        prompt, system_prompt = self._build_prompt(req.messages)
        options = self._build_sdk_options(
            req,
            options_type=options_type,
            system_prompt=system_prompt,
            response_model=response_model,
            include_partial_messages=True,
        )

        async def _iter() -> AsyncIterator[LLMStreamEvent]:
            yield StreamMessageStartEvent(model=req.model)

            text_chunks: list[str] = []
            tool_calls: list[ToolCall] = []
            finish_reason: str | None = None
            structured: dict[str, Any] | None = None
            usage = Usage()
            model: str | None = None
            session_token: str | None = None
            checkpoint_token: str | None = None
            raw_messages: list[dict[str, Any]] = []
            saw_text_delta = False

            request_id = req.request_id or ""
            if sdk_client_type is not None and request_id:
                client = sdk_client_type(options=options)
                self._active_stream_clients[request_id] = client
                try:
                    await client.connect()
                    await client.query(prompt)
                    message_iter = client.receive_response()
                    async for message in message_iter:
                        raw_messages.append(self._serialize_sdk_message(message))

                        msg_type = type(message).__name__
                        if msg_type == "AssistantMessage":
                            model = get_attr_str(message, "model") or model
                            blocks = self._iter_content_blocks(message)
                            block_texts = self._extract_text_blocks(blocks)
                            if block_texts:
                                chunk = "".join(block_texts)
                                text_chunks.extend(block_texts)
                                yield StreamTextDeltaEvent(delta=chunk)
                            tool_calls.extend(self._extract_tool_blocks(blocks))
                            continue

                        if msg_type == "ResultMessage":
                            finish_reason = get_attr_str(message, "subtype") or finish_reason
                            usage = self._usage_from_obj(get_attr(message, "usage"))
                            result_structured = self._extract_result_structured(message)
                            if isinstance(result_structured, dict):
                                structured = result_structured
                            session_token = (
                                self._extract_result_session_token(message) or session_token
                            )
                            checkpoint_token = (
                                self._extract_result_checkpoint_token(message)
                                or checkpoint_token
                            )
                finally:
                    self._active_stream_clients.pop(request_id, None)
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
            else:
                async for message in query_fn(prompt=prompt, options=options):
                    raw_messages.append(self._serialize_sdk_message(message))

                    msg_type = type(message).__name__

                    if msg_type == "StreamEvent":
                        event = get_attr(message, "event")
                        if isinstance(event, dict):
                            for stream_event in self._stream_events_from_raw(event):
                                if isinstance(stream_event, StreamTextDeltaEvent):
                                    saw_text_delta = True
                                    text_chunks.append(stream_event.delta)
                                yield stream_event
                        continue

                    if msg_type == "AssistantMessage":
                        model = get_attr_str(message, "model") or model
                        blocks = self._iter_content_blocks(message)
                        if not saw_text_delta:
                            block_texts = self._extract_text_blocks(blocks)
                            if block_texts:
                                chunk = "".join(block_texts)
                                text_chunks.extend(block_texts)
                                yield StreamTextDeltaEvent(delta=chunk)
                        tool_calls.extend(self._extract_tool_blocks(blocks))
                        continue

                    if msg_type == "ResultMessage":
                        finish_reason = get_attr_str(message, "subtype") or finish_reason
                        usage = self._usage_from_obj(get_attr(message, "usage"))
                        result_structured = self._extract_result_structured(message)
                        if isinstance(result_structured, dict):
                            structured = result_structured
                        session_token = (
                            self._extract_result_session_token(message) or session_token
                        )
                        checkpoint_token = (
                            self._extract_result_checkpoint_token(message)
                            or checkpoint_token
                        )

            response = LLMResponse(
                text="".join(text_chunks),
                request_id=req.request_id,
                session_token=session_token or req.session_token,
                checkpoint_token=checkpoint_token or req.checkpoint_token,
                structured_response=structured,
                tool_calls=self._dedupe_tool_calls(tool_calls),
                finish_reason=finish_reason,
                usage=usage,
                raw={"provider": "claude_agent_sdk", "messages": raw_messages},
                model=model or req.model,
            )

            yield StreamMessageStopEvent(finish_reason=finish_reason)
            yield StreamCompletedEvent(response=response)

        return _iter()

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Embeddings are intentionally unsupported for this adapter."""
        _ = req
        raise LLMCapabilityError(
            "AnthropicAgentClient does not support embeddings in claude-agent-sdk. "
            "Use a separate embeddings adapter (e.g., LiteLLM)."
        )

    async def _interrupt_request(self, req: LLMRequest) -> None:
        """Interrupt active SDK streaming request for the given request id."""
        request_id = req.request_id
        if not isinstance(request_id, str) or not request_id:
            raise LLMCapabilityError("Interrupt requires a request_id for AnthropicAgentClient")

        client = self._active_stream_clients.get(request_id)
        if client is None:
            raise LLMCapabilityError(
                "No active ClaudeSDKClient stream found for this request_id"
            )
        await client.interrupt()

    def _load_sdk_api(self) -> tuple[Any, Any, Any | None]:
        """Import and return required SDK symbols with a clear config error."""
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except Exception as e:  # pragma: no cover - environment dependent
            raise LLMConfigurationError(
                "claude-agent-sdk is not installed. Install it with: pip install claude-agent-sdk"
            ) from e

        try:  # pragma: no cover - optional symbol in tests
            from claude_agent_sdk import ClaudeSDKClient
        except Exception:
            ClaudeSDKClient = None

        return query, ClaudeAgentOptions, ClaudeSDKClient

    def _build_sdk_options(
        self,
        req: LLMRequest,
        *,
        options_type: Any,
        system_prompt: str | None,
        response_model: type[BaseModel] | None,
        include_partial_messages: bool,
    ) -> Any:
        """Map normalized request fields into `ClaudeAgentOptions`."""
        thinking = self.resolve_thinking(req)
        option_kwargs: dict[str, Any] = {}

        if req.model:
            option_kwargs["model"] = req.model

        if system_prompt:
            option_kwargs["system_prompt"] = system_prompt

        allowed_tools = self._allowed_tools(req)
        if allowed_tools:
            option_kwargs["allowed_tools"] = allowed_tools

        if include_partial_messages:
            option_kwargs["include_partial_messages"] = True

        if req.session_token:
            option_kwargs.setdefault("resume", req.session_token)
            option_kwargs.setdefault("continue_conversation", True)
        elif req.checkpoint_token:
            option_kwargs.setdefault("resume", req.checkpoint_token)

        if req.request_id:
            option_kwargs.setdefault("user", req.request_id)

        if thinking.max_tokens is not None:
            option_kwargs["max_thinking_tokens"] = thinking.max_tokens
        elif thinking.enabled is False:
            option_kwargs["max_thinking_tokens"] = 0

        if response_model is not None:
            option_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": response_model.model_json_schema(),
            }

        # Pass selected request extras into ClaudeAgentOptions.
        extras = dict(req.extra)
        passthrough_keys = (
            "tools",
            "permission_mode",
            "mcp_servers",
            "continue_conversation",
            "resume",
            "max_turns",
            "max_budget_usd",
            "disallowed_tools",
            "fallback_model",
            "betas",
            "permission_prompt_tool_name",
            "cwd",
            "cli_path",
            "settings",
            "add_dirs",
            "env",
            "extra_args",
            "max_buffer_size",
            "hooks",
            "user",
            "fork_session",
            "agents",
            "setting_sources",
            "max_thinking_tokens",
            "sandbox",
        )

        for key in passthrough_keys:
            if key in extras:
                option_kwargs[key] = extras[key]

        options_override = extras.get("claude_options")
        if isinstance(options_override, dict):
            option_kwargs.update(options_override)

        # Ensure partial streaming remains enabled for stream method.
        if include_partial_messages:
            option_kwargs["include_partial_messages"] = True

        if self.config.api_key:
            env = option_kwargs.get("env")
            if not isinstance(env, dict):
                env = {}
            env = dict(env)
            env.setdefault("ANTHROPIC_API_KEY", self.config.api_key)
            option_kwargs["env"] = env

        return options_type(**option_kwargs)

    def _allowed_tools(self, req: LLMRequest) -> list[str]:
        """Derive `allowed_tools` from normalized tools/tool_choice fields."""
        names: list[str] = []
        for tool in req.tools or []:
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    names.append(name)

        if isinstance(req.tool_choice, dict):
            function = req.tool_choice.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    names = [name]

        if req.tool_choice == "none":
            return []

        # Preserve order but dedupe.
        out: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _build_prompt(self, messages: Iterable[Message]) -> tuple[str, str | None]:
        """
        Build SDK prompt/system_prompt from normalized messages.

        Since `query()` accepts one prompt input, non-system messages are
        flattened into a conversation transcript string.
        """
        system_chunks: list[str] = []
        conversation: list[str] = []

        for msg in messages:
            text = self._content_to_text(msg.content).strip()
            if not text:
                continue

            if msg.role == "system":
                system_chunks.append(text)
                continue

            label = msg.role.upper()
            conversation.append(f"{label}:\n{text}")

        prompt = "\n\n".join(conversation).strip()
        if not prompt and system_chunks:
            prompt = "Please follow the system instructions."

        system_prompt = "\n\n".join(system_chunks).strip() or None
        return prompt, system_prompt

    def _content_to_text(self, content: Any) -> str:
        """Convert normalized message content into plain text transcript form."""
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content)

        out: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                out.append(str(part))
                continue

            p_type = part.get("type")
            if p_type == "text":
                text = part.get("text")
                if isinstance(text, str):
                    out.append(text)
                continue

            if p_type == "tool_use":
                name = part.get("name")
                inp = part.get("input")
                out.append(
                    f"[tool_use:{name}] {json.dumps(inp, ensure_ascii=True, default=str)}"
                )
                continue

            if p_type == "tool_result":
                tool_use_id = part.get("tool_use_id")
                result_content = part.get("content")
                out.append(f"[tool_result:{tool_use_id}] {result_content}")
                continue

            if p_type == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                out.append(f"[image_url] {url}" if isinstance(url, str) else "[image_url]")
                continue

            out.append(json.dumps(part, ensure_ascii=True, default=str))

        return "\n".join(out)

    def _iter_content_blocks(self, message: Any) -> list[Any]:
        """Return assistant content blocks when present."""
        content = get_attr(message, "content")
        if isinstance(content, list):
            return content
        return []

    def _extract_text_blocks(self, blocks: Iterable[Any]) -> list[str]:
        """Extract text chunks from SDK/content dict block variants."""
        out: list[str] = []
        for block in blocks:
            b_name = type(block).__name__
            if b_name == "TextBlock":
                text = get_attr(block, "text")
                if isinstance(text, str):
                    out.append(text)
                continue

            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    out.append(block["text"])
        return out

    def _extract_tool_blocks(self, blocks: Iterable[Any]) -> list[ToolCall]:
        """Extract normalized tool calls from SDK/content block variants."""
        out: list[ToolCall] = []
        for block in blocks:
            b_name = type(block).__name__
            if b_name == "ToolUseBlock":
                name = get_attr(block, "name")
                args = get_attr(block, "input")
                out.append(
                    ToolCall(
                        id=get_attr_str(block, "id"),
                        tool_name=name if isinstance(name, str) else "",
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
                continue

            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name")
                args = block.get("input")
                out.append(
                    ToolCall(
                        id=block.get("id") if isinstance(block.get("id"), str) else None,
                        tool_name=name if isinstance(name, str) else "",
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
        return out

    def _stream_events_from_raw(self, event: dict[str, Any]) -> list[LLMStreamEvent]:
        """Map raw SDK partial stream events into normalized stream events."""
        out: list[LLMStreamEvent] = []
        event_type = event.get("type")

        if event_type == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                return out

            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    out.append(StreamTextDeltaEvent(delta=text))

            elif delta_type == "input_json_delta":
                partial_json = delta.get("partial_json")
                if isinstance(partial_json, str) and partial_json:
                    out.append(StreamToolCallDeltaEvent(arguments_delta=partial_json))

            return out

        if event_type == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                index = event.get("index")
                if not isinstance(index, int):
                    index = 0
                out.append(
                    StreamToolCallDeltaEvent(
                        index=index,
                        call_id=block.get("id") if isinstance(block.get("id"), str) else None,
                        tool_name=block.get("name")
                        if isinstance(block.get("name"), str)
                        else None,
                        arguments_delta="",
                    )
                )

        return out

    def _extract_result_structured(self, message: Any) -> dict[str, Any] | None:
        """Extract structured result payload from SDK `ResultMessage`."""
        structured = get_attr(message, "structured_output")
        if isinstance(structured, dict):
            return structured
        return None

    def _extract_result_session_token(self, message: Any) -> str | None:
        """Extract session continuation token from SDK `ResultMessage`."""
        return get_attr_str(message, "session_id")

    def _extract_result_checkpoint_token(self, message: Any) -> str | None:
        """Extract checkpoint token from SDK `ResultMessage` when available."""
        return get_attr_str(message, "user_message_uuid")

    def _usage_from_obj(self, usage_obj: Any) -> Usage:
        """Normalize usage counters from SDK result usage object/dict."""
        usage_dict = usage_obj if isinstance(usage_obj, dict) else to_plain_dict(usage_obj)

        input_tokens = usage_dict.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage_dict.get("prompt_tokens")

        output_tokens = usage_dict.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage_dict.get("completion_tokens")

        total_tokens = usage_dict.get("total_tokens")
        if (
            total_tokens is None
            and isinstance(input_tokens, int)
            and isinstance(output_tokens, int)
        ):
            total_tokens = input_tokens + output_tokens

        return Usage(
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        )

    def _dedupe_tool_calls(self, calls: list[ToolCall]) -> list[ToolCall]:
        """Deduplicate tool calls emitted across mixed SDK events/messages."""
        out: list[ToolCall] = []
        seen: set[tuple[str | None, str, str]] = set()

        for call in calls:
            args_key = json.dumps(call.arguments, sort_keys=True, ensure_ascii=True)
            key = (call.id, call.tool_name, args_key)
            if key in seen:
                continue
            seen.add(key)
            out.append(call)

        return out

    def _serialize_sdk_message(self, message: Any) -> dict[str, Any]:
        """Serialize SDK message objects into raw debug-safe payloads."""
        payload = {
            "type": type(message).__name__,
            "payload": to_jsonable(to_plain_dict(message) or message),
        }
        return payload
