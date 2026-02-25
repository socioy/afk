"""Workflow entry for prompt directory and template resolution."""

import asyncio
from pathlib import Path

from afk.core import Runner

from .agents import build_agent
from .complexity_chain import run_chain
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario


async def _run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    agent = build_agent(project_root)

    context = {
        "customer_name": "Northwind Retail",
        "segment": "enterprise",
        "region": "us-east",
    }

    scenario = build_scenario("prompt-loader-template-context")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    resolved_instructions = await agent.resolve_instructions(context)
    result = await Runner().run(
        agent,
        user_message="Generate a concise operational briefing.",
        context=context,
    )

    feature_payload: dict[str, object] = {
        "kind": "prompt_loader",
        "status": "ok",
        "instruction_size": len(resolved_instructions or ""),
        "final_text": result.final_text,
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[prompt-loader] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
