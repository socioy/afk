"""Workflow entry for internal A2A protocol execution."""

import asyncio

from afk.agents import InternalA2AProtocol

from .complexity_chain import run_chain
from .contracts import build_request, dispatch_capacity
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario


async def _run() -> None:
    scenario = build_scenario("a2a-internal-protocol")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    protocol = InternalA2AProtocol(dispatch=dispatch_capacity)
    request = build_request()

    first = await protocol.invoke(request)
    second = await protocol.invoke(request)

    stream_event_types: list[str] = []
    async for event in protocol.invoke_stream(request):
        stream_event_types.append(event.type)

    await protocol.record_dead_letter(request, error="timeout", attempts=2)

    feature_payload: dict[str, object] = {
        "kind": "a2a_internal_protocol",
        "status": "ok" if first.success and second.success else "error",
        "stream_events": len(stream_event_types),
        "dead_letters": len(protocol.dead_letters()),
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[a2a-internal] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
