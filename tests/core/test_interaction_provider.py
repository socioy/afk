from __future__ import annotations

import asyncio

from afk.agents.types import (
    AgentRunEvent,
    ApprovalDecision,
    ApprovalRequest,
    UserInputRequest,
)
from afk.core.interaction import HeadlessInteractionProvider, InMemoryInteractiveProvider


def run_async(coro):
    return asyncio.run(coro)


def test_headless_provider_uses_fallback_decisions():
    provider = HeadlessInteractionProvider(approval_fallback="deny", input_fallback="deny")
    approval = run_async(
        provider.request_approval(
            ApprovalRequest(
                run_id="r1",
                thread_id="t1",
                step=1,
                reason="approve?",
            )
        )
    )
    user_input = run_async(
        provider.request_user_input(
            UserInputRequest(
                run_id="r1",
                thread_id="t1",
                step=1,
                prompt="value?",
            )
        )
    )
    assert approval.kind == "deny"
    assert user_input.kind == "deny"


def test_inmemory_provider_supports_deferred_resolution():
    provider = InMemoryInteractiveProvider()
    deferred = run_async(
        provider.request_approval(
            ApprovalRequest(
                run_id="run_1",
                thread_id="thread_1",
                step=2,
                reason="approve",
            )
        )
    )
    token = deferred.token
    provider.set_deferred_result(token, ApprovalDecision(kind="allow"))
    resolved = run_async(provider.await_deferred(token, timeout_s=0.01))
    assert isinstance(resolved, ApprovalDecision)
    assert resolved.kind == "allow"


def test_provider_collects_notifications():
    provider = InMemoryInteractiveProvider()
    event = AgentRunEvent(
        type="run_started",
        run_id="r1",
        thread_id="t1",
        state="running",
    )
    run_async(provider.notify(event))
    notes = provider.notifications()
    assert notes and notes[0].type == "run_started"

