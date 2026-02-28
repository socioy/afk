"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Google A2A SDK adapter implementing AFK agent communication protocol.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..contracts import (
    AgentCommunicationProtocol,
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentProtocolEvent,
)


class GoogleA2AAdapterError(RuntimeError):
    """Raised when Google A2A integration is misconfigured or unavailable."""


class GoogleA2AProtocolAdapter(AgentCommunicationProtocol):
    """Wrap a Google A2A SDK client behind AFK's communication protocol contract."""

    protocol_id = "google.a2a.v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._client = client
        self._client_factory = client_factory

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        try:
            # The exact import path may vary between SDK versions.
            from a2a.client import A2AClient  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise GoogleA2AAdapterError(
                "Google A2A SDK is required. Install and provide a configured client."
            ) from exc
        self._client = A2AClient()
        return self._client

    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        client = self._resolve_client()
        response_payload = await self._call_client(client, "send_message", request)
        return self._to_response(request, response_payload)

    async def invoke_stream(
        self,
        request: AgentInvocationRequest,
    ) -> AsyncIterator[AgentProtocolEvent]:
        client = self._resolve_client()
        stream_payload = await self._call_client(
            client, "send_message_streaming", request
        )
        if hasattr(stream_payload, "__aiter__"):
            async for item in stream_payload:
                maybe_response = self._to_response(request, item)
                event_type = "completed" if maybe_response.success else "failed"
                yield AgentProtocolEvent(
                    type=event_type,
                    request=request,
                    response=maybe_response,
                    details={"provider": "google_a2a"},
                )
            return

        response = self._to_response(request, stream_payload)
        yield AgentProtocolEvent(
            type="completed" if response.success else "failed",
            request=request,
            response=response,
            details={"provider": "google_a2a"},
        )

    async def get_task(self, task_id: str) -> dict[str, Any]:
        client = self._resolve_client()
        payload = await self._call_client(client, "get_task", task_id)
        if isinstance(payload, dict):
            return payload
        return {"task": payload}

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        client = self._resolve_client()
        payload = await self._call_client(client, "cancel_task", task_id)
        if isinstance(payload, dict):
            return payload
        return {"task": payload}

    async def _call_client(self, client: Any, method: str, payload: Any) -> Any:
        fn = getattr(client, method, None)
        if fn is None:
            raise GoogleA2AAdapterError(
                f"Configured Google A2A client does not implement '{method}'"
            )
        out = fn(payload)
        if inspect.isawaitable(out):
            return await out
        return out

    def _to_response(
        self,
        request: AgentInvocationRequest,
        payload: Any,
    ) -> AgentInvocationResponse:
        if isinstance(payload, AgentInvocationResponse):
            return payload

        if isinstance(payload, dict):
            success = bool(payload.get("success", True))
            output = payload.get("output", payload)
            error = payload.get("error")
            metadata = payload.get("metadata")
            return AgentInvocationResponse(
                run_id=request.run_id,
                thread_id=request.thread_id,
                conversation_id=request.conversation_id,
                correlation_id=request.correlation_id,
                idempotency_key=request.idempotency_key,
                source_agent=request.target_agent,
                target_agent=request.source_agent,
                success=success,
                output=output,
                error=str(error) if error is not None else None,
                metadata=metadata if isinstance(metadata, dict) else {},
            )

        return AgentInvocationResponse(
            run_id=request.run_id,
            thread_id=request.thread_id,
            conversation_id=request.conversation_id,
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            source_agent=request.target_agent,
            target_agent=request.source_agent,
            success=True,
            output=payload,
        )
