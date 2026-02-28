"""
---
name: Data Pipeline
description: A data pipeline orchestrator agent that uses DelegationPlan for DAG-based multi-agent execution.
tags: [agent, runner, delegation, delegation-plan, dag, async]
---
---
This example demonstrates AFK's DelegationPlan system for orchestrating complex multi-agent
workflows as a directed acyclic graph (DAG). Instead of simple sequential or parallel subagent
calls, you define nodes (agents), edges (dependencies), and execution constraints. The delegation
engine handles scheduling, parallelism, retries, and result collection. This pattern is ideal for
data pipelines, CI/CD workflows, document processing, and any task where agents have dependencies.

The project is split into three files:
- stages.py: Stage agents with their tools and simulated data
- pipeline.py: DelegationPlan DAG definition with nodes, edges, and policies
- main.py: Orchestrator agent and entry point
---
"""

import asyncio  # <- Async required for delegation engine.
from afk.core import Runner  # <- Runner orchestrates agent execution.
from afk.agents import Agent  # <- Agent defines the orchestrator.

from stages import (  # <- Import all stage agents from stages.py.
    extractor_agent,
    validator_agent,
    transformer_agent,
    reporter_agent,
)
from pipeline import pipeline_plan  # <- Import the DelegationPlan DAG from pipeline.py.


# ===========================================================================
# Orchestrator agent that owns the pipeline
# ===========================================================================

orchestrator = Agent(
    name="pipeline-orchestrator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data pipeline orchestrator. You manage a 4-stage data pipeline:
    1. Extract: Pull raw employee data
    2. Validate: Check data quality
    3. Transform: Aggregate by department
    4. Report: Generate executive summary

    When the user asks to run the pipeline, delegate to your subagents.
    Explain what each stage does and present results as they complete.
    """,
    subagents=[extractor_agent, validator_agent, transformer_agent, reporter_agent],  # <- All pipeline stages are registered as subagents.
)

runner = Runner()


# ===========================================================================
# Main entry point — runs the pipeline
# ===========================================================================

async def main():
    print("Data Pipeline Orchestrator")
    print("=" * 40)
    print("This pipeline runs 4 stages: Extract -> Validate -> Transform -> Report")
    print("Extract and Validate run in parallel; Transform and Report are sequential.\n")

    user_input = input("[] > Type 'run' to start the pipeline (or 'quit' to exit): ").strip().lower()

    if user_input in ("quit", "exit", "q"):
        print("Goodbye!")
        return

    print("\nStarting pipeline...\n")

    # --- Run the orchestrator agent (which delegates to subagents via the plan) ---
    response = await runner.run(
        orchestrator,
        user_message="Run the full data pipeline: extract, validate, transform, and generate report.",
    )

    print(f"[orchestrator] > {response.final_text}")
    print(f"\n--- Pipeline Complete ---")
    print(f"Success: {response.success}")
    print(f"Subagent calls: {len(response.subagent_calls)}")
    for sub in response.subagent_calls:
        print(f"  - {sub.target_agent}: {'ok' if sub.success else 'failed'}")


if __name__ == "__main__":
    asyncio.run(main())



"""
---
Tl;dr: This example creates a data pipeline with 4 agent stages (extract, validate, transform, report)
orchestrated via a DelegationPlan DAG. Nodes define agent invocations with timeouts and retry policies.
Edges define dependencies — extract and validate run in parallel, then transform runs when both complete,
then report runs last. The join_policy="all_required" ensures all stages must succeed. The project is
split into stages.py (agents + tools), pipeline.py (DAG definition), and main.py (orchestrator + entry).
---
---
What's next?
- Try setting a node's required=False to make it optional (the pipeline continues even if it fails).
- Experiment with join_policy="first_success" to use the first stage that completes.
- Add error injection (make a tool raise an exception) to see retry behavior.
- Use output_key_map on edges to pass specific outputs between stages.
- Scale max_parallelism to run more stages concurrently.
- Check out the Travel Planner example for quorum-based delegation with multiple perspectives!
---
"""
