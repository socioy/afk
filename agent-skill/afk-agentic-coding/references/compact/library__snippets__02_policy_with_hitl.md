# 02_policy_with_hitl

Compact reference for 02_policy_with_hitl.

Source: `docs/library/snippets/02_policy_with_hitl.mdx`

````python 02_policy_with_hitl.py
"""
Example 02: Policy with deferred human approval.

Run:
    python 02_policy_with_hitl.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from afk.agents import (
    Agent,
    ApprovalDecision,
    PolicyEngine,
    PolicyRule,
    PolicyRuleCondition,
)
from afk.core import InMemoryInteractiveProvider, Runner, RunnerConfig
from afk.tools import tool

class RiskyArgs(BaseModel):
    reason: str = Field(min_length=1)

@tool(
    args_model=RiskyArgs,
    name="risky_action",
    description="Execute a side-effecting action that requires explicit approval.",
)
def risky_action(args: RiskyArgs) -> dict[str, str]:
    return {
        "status": "executed",
        "reason": args.reason,
    }

````

> Code block truncated to 40 lines. Source: `docs/library/snippets/02_policy_with_hitl.mdx`
