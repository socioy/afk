"""
---
name: Lead Intelligence Agent
description: Qualify inbound B2B leads with tool-assisted enrichment and basic run analytics.
tags: [agent, runner, tool, analytics]
---
---
This example demonstrates how to build a practical lead qualification assistant.
The agent uses a typed tool to enrich lead context, then returns a sales-ready summary.
---
"""

from afk.agents import Agent
from afk.core import Runner
from afk.tools import tool
from pydantic import BaseModel, Field

MODEL = "ollama_chat/gpt-oss:20b"


class LeadEnrichmentArgs(BaseModel):
    company: str = Field(description="Company name from the inbound lead form.")
    annual_revenue_musd: float = Field(
        ge=0, description="Estimated annual revenue in millions of USD."
    )
    employee_count: int = Field(ge=1, description="Estimated headcount.")
    region: str = Field(description="Primary sales region, for example NA or EMEA.")


@tool(
    args_model=LeadEnrichmentArgs,
    name="enrich_lead_profile",
    description="Enrich and score a lead profile for B2B qualification.",
)
def enrich_lead_profile(args: LeadEnrichmentArgs) -> dict:
    """Return deterministic lead-scoring signals for the model to reason over."""
    growth_signal = "high" if args.annual_revenue_musd >= 100 else "moderate"
    size_signal = "enterprise" if args.employee_count >= 1000 else "mid-market"
    score = min(
        100,
        int(args.annual_revenue_musd * 0.35)
        + min(args.employee_count // 20, 50)
        + (15 if args.region.upper() in {"NA", "EMEA"} else 5),
    )

    next_step = (
        "book enterprise demo with solutions architect"
        if score >= 80
        else "schedule discovery call with account executive"
    )

    return {
        "company": args.company,
        "segment": size_signal,
        "growth_signal": growth_signal,
        "qualification_score": score,
        "recommended_next_step": next_step,
    }


lead_intelligence_agent = Agent(
    name="lead_intelligence_agent",
    model=MODEL,
    instructions="""
    You are an enterprise SDR assistant.
    Always call `enrich_lead_profile` before giving a final recommendation.

    Response format:
    1) One-line lead decision (Qualified, Nurture, or Disqualify)
    2) Short rationale with 3 bullets
    3) Suggested next outbound message (2-3 sentences)

    Keep outputs actionable and realistic for a sales team.
    """,
    tools=[enrich_lead_profile],
)

runner = Runner()

if __name__ == "__main__":
    user_input = input(
        "[] > "
    )  # <- Ask for a real inbound lead prompt from the terminal.

    response = runner.run_sync(
        lead_intelligence_agent,
        user_message=user_input,
    )  # <- Run the agent synchronously for a script-friendly workflow.

    print(f"[lead_intelligence_agent] > {response.final_text}")

    print("\n--- Run Analytics ---")
    print(f"state: {response.state}")
    print(f"thread_id: {response.thread_id}")
    print(f"total_tokens: {response.usage_aggregate.total_tokens}")
    print(f"tool_calls: {len(response.tool_executions)}")
    print(f"estimated_cost_usd: {response.total_cost_usd}")


"""
---
Tl;dr: This example uses one typed enrichment tool to qualify an inbound B2B lead and then prints basic runtime analytics from AgentResult.
---
---
What's next?
- Add CRM write-back as another tool and gate it with policy rules.
- Track qualification outcomes over many runs and compute win-rate by segment.
- Compare token usage by prompt style to reduce per-lead cost.
---
"""
