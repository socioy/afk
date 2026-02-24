"""
---
name: Portfolio Risk Analytics Agent
description: Run portfolio risk workflows with progressive, aggregate analytics across many runs.
tags: [agent, runner, tools, subagents, batch-analytics]
---
---
This example demonstrates a portfolio risk desk workflow at batch scale.
It combines tools, subagents, fail-safe controls, and aggregate analytics across multiple runs.
---
"""

import asyncio

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner
from afk.observability import project_run_metrics_from_result
from afk.tools import tool
from pydantic import BaseModel, Field

MODEL = "ollama_chat/gpt-oss:20b"


class SnapshotArgs(BaseModel):
    symbol: str = Field(description="Ticker symbol to analyze.")


@tool(
    args_model=SnapshotArgs,
    name="get_market_snapshot",
    description="Get synthetic market volatility and beta signals for a symbol.",
)
def get_market_snapshot(args: SnapshotArgs) -> dict:
    """Return deterministic market factors for risk analysis."""
    symbol = args.symbol.upper()
    base = (sum(ord(char) for char in symbol) % 30) / 100
    return {
        "symbol": symbol,
        "volatility": round(0.15 + base, 3),
        "beta": round(0.8 + (base / 2), 3),
        "liquidity_bucket": "high" if base < 0.2 else "medium",
    }


class StressArgs(BaseModel):
    symbol: str
    notional_musd: float = Field(gt=0)
    volatility: float = Field(gt=0)
    beta: float = Field(gt=0)


@tool(
    args_model=StressArgs,
    name="run_stress_test",
    description="Compute a synthetic 95% one-day VaR estimate under stress.",
)
def run_stress_test(args: StressArgs) -> dict:
    """Simple stress approximation for demonstration purposes."""
    var_95_musd = round(args.notional_musd * args.volatility * args.beta * 1.65, 3)
    stressed_drawdown_pct = round(min(55.0, args.volatility * 100 * 1.8), 2)
    return {
        "symbol": args.symbol,
        "notional_musd": args.notional_musd,
        "var_95_musd": var_95_musd,
        "stressed_drawdown_pct": stressed_drawdown_pct,
    }


macro_risk_specialist = Agent(
    name="macro_risk_specialist",
    model=MODEL,
    instructions="""
    You evaluate macro exposures (rates, FX, inflation, geopolitics).
    Return top macro risks with confidence and a hedging direction.
    """,
)

concentration_specialist = Agent(
    name="concentration_risk_specialist",
    model=MODEL,
    instructions="""
    You evaluate concentration by sector, factor, and single-name exposure.
    Return the top concentration vulnerabilities and mitigation options.
    """,
)

portfolio_agent = Agent(
    name="portfolio_risk_analytics_agent",
    model=MODEL,
    instructions="""
    You are the head of portfolio risk.

    For every request:
    - Call get_market_snapshot.
    - Call run_stress_test.
    - Delegate once to macro_risk_specialist.
    - Delegate once to concentration_risk_specialist.

    Final response format:
    1) Risk score (1-10)
    2) Key risk drivers
    3) Hedging and rebalancing actions
    4) Monitoring plan for next trading session
    """,
    tools=[get_market_snapshot, run_stress_test],
    subagents=[macro_risk_specialist, concentration_specialist],
    fail_safe=FailSafeConfig(
        max_steps=30,
        max_total_cost_usd=1.50,
        max_llm_calls=40,
    ),
)


async def main() -> None:
    runner = Runner()

    portfolio_requests = [
        "Assess risk for AAPL with notional 12.5 musd.",
        "Assess risk for NVDA with notional 9.0 musd and highlight macro concerns.",
        "Assess risk for TSLA with notional 7.2 musd and propose hedge actions.",
    ]

    run_results = []
    run_metrics = []

    for idx, request in enumerate(portfolio_requests, start=1):
        result = await runner.run(
            portfolio_agent,
            user_message=request,
            thread_id=f"portfolio-risk-thread-{idx}",
        )
        metrics = project_run_metrics_from_result(result)

        run_results.append(result)
        run_metrics.append(metrics)

        print(f"\nRun {idx} output:\n{result.final_text}\n")

    completed_runs = sum(1 for row in run_results if row.state == "completed")
    total_runs = len(run_results)
    completion_rate = round((completed_runs / total_runs) * 100, 2) if total_runs else 0.0

    total_tokens = sum(row.usage_aggregate.total_tokens for row in run_results)
    total_tool_calls = sum(len(row.tool_executions) for row in run_results)
    total_subagent_calls = sum(len(row.subagent_executions) for row in run_results)
    avg_tokens = round(total_tokens / total_runs, 2) if total_runs else 0.0

    all_tool_latencies = [
        record.latency_ms
        for result in run_results
        for record in result.tool_executions
        if record.latency_ms is not None
    ]
    peak_tool_latency = max(all_tool_latencies) if all_tool_latencies else None

    print("--- Portfolio Analytics ---")
    print(f"runs: {total_runs}")
    print(f"completion_rate_pct: {completion_rate}")
    print(f"total_tokens: {total_tokens}")
    print(f"avg_tokens_per_run: {avg_tokens}")
    print(f"total_tool_calls: {total_tool_calls}")
    print(f"total_subagent_calls: {total_subagent_calls}")
    print(f"peak_tool_latency_ms: {peak_tool_latency}")

    print("\nPer-run metrics snapshot:")
    for idx, metrics in enumerate(run_metrics, start=1):
        print(
            "- "
            f"run={idx} | state={metrics.state} | llm_calls={metrics.llm_calls} | "
            f"tool_calls={metrics.tool_calls} | total_tokens={metrics.total_tokens} | "
            f"estimated_cost_usd={metrics.estimated_cost_usd}"
        )


if __name__ == "__main__":
    asyncio.run(main())


"""
---
Tl;dr: This is a portfolio-scale AFK program that combines tools, subagents, fail-safe constraints, and aggregate analytics across multiple production-like runs.
---
---
What's next?
- Feed per-run metrics into a risk control dashboard.
- Add policy gates for high-risk rebalancing recommendations.
- Compare outputs and costs across multiple model configurations.
---
"""
