"""Request factories and dispatch handlers for internal A2A flows."""

from afk.agents.contracts import AgentInvocationRequest, AgentInvocationResponse


def build_request() -> AgentInvocationRequest:
    return AgentInvocationRequest(
        run_id="run-32",
        thread_id="thread-32",
        conversation_id="conv-32",
        correlation_id="corr-32",
        idempotency_key="idem-32",
        causation_id="cause-32",
        source_agent="ops-supervisor",
        target_agent="capacity-analyst",
        payload={"window": "next-14-days"},
    )


async def dispatch_capacity(request: AgentInvocationRequest) -> AgentInvocationResponse:
    return AgentInvocationResponse(
        run_id=request.run_id,
        thread_id=request.thread_id,
        conversation_id=request.conversation_id,
        correlation_id=request.correlation_id,
        idempotency_key=request.idempotency_key,
        source_agent=request.target_agent,
        target_agent=request.source_agent,
        success=True,
        output={"risk": "medium", "action": "increase headroom by 12%"},
    )
