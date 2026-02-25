"""Workflow entry for hooks and middleware example."""

import asyncio

from .complexity_chain import run_chain
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .registry_ops import run_once
from .report_builder import build_report
from .scenario import build_scenario


async def _run() -> None:
    channel = input("channel > ").strip() or "email"
    message = input("message > ").strip() or "Please confirm incident closure ETA."

    scenario = build_scenario(f"{channel}:{message}")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    result, records = await run_once(channel, message)
    feature_payload: dict[str, object] = {
        "kind": "tool_hooks_registry_middleware",
        "status": "ok" if result.success else "error",
        "result_success": result.success,
        "record_count": len(records),
        "metadata": result.metadata,
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[hooks] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
