"""
---
name: Runtime File Audit Agent
description: Audit policy and compliance files using prebuilt runtime tools and tool-level analytics.
tags: [agent, runner, prebuilt-tools, filesystem, analytics]
---
---
This example demonstrates secure filesystem analysis with AFK prebuilt runtime tools.
The agent audits local policy docs and reports compliance findings plus tool analytics.
---
"""

from pathlib import Path

from afk.agents import Agent
from afk.core import Runner
from afk.tools import build_runtime_tools

MODEL = "ollama_chat/gpt-oss:20b"
WORKSPACE_DIR = Path("examples/projects/09_Runtime_File_Audit_Agent/workspace")


def prepare_workspace() -> None:
    """Create sample files so the example is runnable out of the box."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    policies = {
        "security_policy.md": """
        # Security Policy
        - MFA required for all admins.
        - Password rotation every 90 days.
        - Incident escalation within 15 minutes.
        """,
        "access_control.md": """
        # Access Control
        - Least privilege model.
        - Quarterly role review.
        - Temporary credentials expire in 24 hours.
        """,
        "incident_playbook.md": """
        # Incident Playbook
        - P1 incident bridge starts within 5 minutes.
        - Customer update cadence: every 30 minutes.
        - Final postmortem in 5 business days.
        """,
    }

    for file_name, content in policies.items():
        (WORKSPACE_DIR / file_name).write_text(content.strip() + "\n", encoding="utf-8")


prepare_workspace()
runtime_tools = build_runtime_tools(root_dir=WORKSPACE_DIR)

audit_agent = Agent(
    name="runtime_file_audit_agent",
    model=MODEL,
    instructions="""
    You are a compliance auditor.
    Always use list_directory first, then read relevant markdown files.

    Return:
    1) Controls found
    2) Missing controls
    3) Priority remediation list
    """,
    tools=runtime_tools,
)

runner = Runner()

if __name__ == "__main__":
    user_input = input("[] > ")

    result = runner.run_sync(
        audit_agent,
        user_message=user_input,
    )

    print(f"[runtime_file_audit_agent] > {result.final_text}")

    stats: dict[str, list[float]] = {}
    for record in result.tool_executions:
        bucket = stats.setdefault(record.tool_name, [])
        if record.latency_ms is not None:
            bucket.append(record.latency_ms)

    print("\n--- File Audit Analytics ---")
    print(f"state: {result.state}")
    print(f"tool_calls: {len(result.tool_executions)}")
    print(f"total_tokens: {result.usage_aggregate.total_tokens}")

    for tool_name, values in stats.items():
        average = round(sum(values) / len(values), 2) if values else None
        print(f"- {tool_name}: calls={len(values)} avg_latency_ms={average}")


"""
---
Tl;dr: This example uses AFK runtime-safe filesystem tools to audit local policy documents and then reports per-tool usage and latency analytics.
---
---
What's next?
- Add policy rules to require approval before reading sensitive files.
- Pipe compliance findings into your ticketing system.
- Track trendlines of missing controls across weekly scans.
---
"""
