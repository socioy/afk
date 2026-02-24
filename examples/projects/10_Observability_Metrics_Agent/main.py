"""
---
name: Observability Metrics Agent
description: Project run metrics from telemetry and export structured observability artifacts.
tags: [agent, runner, telemetry, observability, analytics]
---
---
This example demonstrates observability-first AFK usage.
It captures runtime telemetry, projects metrics, and exports JSON/JSONL artifacts.
---
"""

from pathlib import Path

from afk.agents import Agent
from afk.core import Runner
from afk.observability import (
    JSONLRunMetricsExporter,
    JSONRunMetricsExporter,
    RuntimeTelemetryCollector,
    project_run_metrics_from_collector,
    project_run_metrics_from_result,
)
from afk.tools import tool
from pydantic import BaseModel, Field

MODEL = "ollama_chat/gpt-oss:20b"
ARTIFACT_DIR = Path("examples/projects/10_Observability_Metrics_Agent/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class QueryArgs(BaseModel):
    metric_name: str = Field(description="Metric to fetch, for example conversion_rate.")


@tool(
    args_model=QueryArgs,
    name="fetch_business_metric",
    description="Fetch mocked business metric snapshots for reporting workflows.",
)
def fetch_business_metric(args: QueryArgs) -> dict:
    """Mock metric fetch tool to generate telemetry-rich runs."""
    key = args.metric_name.lower()
    values = {
        "conversion_rate": {"value": 0.042, "trend": "up"},
        "churn_rate": {"value": 0.018, "trend": "down"},
        "activation_rate": {"value": 0.63, "trend": "up"},
    }
    row = values.get(key, {"value": 0.0, "trend": "unknown"})
    return {"metric_name": args.metric_name, **row}


observability_agent = Agent(
    name="observability_metrics_agent",
    model=MODEL,
    instructions="""
    You are an analytics operations assistant.
    Always call fetch_business_metric before writing your recommendation.
    Return an executive summary with KPI interpretation and one action item.
    """,
    tools=[fetch_business_metric],
)

collector = RuntimeTelemetryCollector()
runner = Runner(telemetry=collector)

if __name__ == "__main__":
    user_input = input("[] > ")

    result = runner.run_sync(
        observability_agent,
        user_message=user_input,
    )

    collector_metrics = project_run_metrics_from_collector(collector)
    result_metrics = project_run_metrics_from_result(result)

    json_exporter = JSONRunMetricsExporter(path=ARTIFACT_DIR / "run_metrics.json")
    jsonl_exporter = JSONLRunMetricsExporter(path=ARTIFACT_DIR / "run_metrics.jsonl")

    json_exporter.export(collector_metrics)
    jsonl_exporter.export(collector_metrics)

    print(f"[observability_metrics_agent] > {result.final_text}")

    print("\n--- Observability Analytics ---")
    print(f"collector_llm_calls: {collector_metrics.llm_calls}")
    print(f"collector_tool_calls: {collector_metrics.tool_calls}")
    print(f"collector_total_tokens: {collector_metrics.total_tokens}")
    print(f"result_tool_calls: {result_metrics.tool_calls}")
    print(f"result_total_tokens: {result_metrics.total_tokens}")
    print(f"result_estimated_cost_usd: {result_metrics.estimated_cost_usd}")
    print(f"json_export: {ARTIFACT_DIR / 'run_metrics.json'}")
    print(f"jsonl_export: {ARTIFACT_DIR / 'run_metrics.jsonl'}")


"""
---
Tl;dr: This example captures telemetry during execution, projects run metrics from two sources, and exports observability artifacts for dashboards.
---
---
What's next?
- Ship JSONL metrics to your data lake for historical trend analysis.
- Alert on error spikes or latency regressions from projected metrics.
- Compare metrics across models to optimize cost/performance.
---
"""
