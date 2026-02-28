"""
---
name: Content Moderator — Policy
description: PolicyEngine rules and PolicyRole callbacks for dynamic content moderation decisions.
tags: [policy-engine, policy-rule, policy-role]
---
---
This module demonstrates two layers of policy enforcement in AFK:

1. **PolicyEngine + PolicyRule** (declarative, static): Rules defined at construction time with
   conditions like tool_name and context_equals. Evaluated deterministically by priority.

2. **PolicyRole** (dynamic, callback): A protocol callback invoked at runtime for every policy
   event. Receives the full PolicyEvent (tool_name, tool_args, context, etc.) and returns a
   PolicyDecision. This allows Python logic the static rules can't express — like scanning
   the actual tool arguments for keywords or rate-limiting per author.

Both are attached to the Agent. The Runner evaluates PolicyEngine rules first, then calls
each PolicyRole. If ANY policy layer denies the action, the tool call is blocked.
---
"""

from afk.agents.policy.engine import (  # <- Static policy subsystem.
    PolicyEngine,
    PolicyRule,
    PolicyRuleCondition,
)

from tools import SENSITIVE_KEYWORDS, PUBLISHED_CONTENT  # <- Import data for dynamic policy checks.


# ===========================================================================
# PolicyEngine — static declarative rules
# ===========================================================================
# These rules are evaluated by priority. The highest-priority matching rule wins.
# They use context_equals to match on values set in the runtime context dict.

policy_engine = PolicyEngine(
    rules=[
        PolicyRule(  # <- DENY publish when context signals content is flagged.
            rule_id="deny-publish-flagged",
            action="deny",
            priority=200,
            enabled=True,
            subjects=["tool_call"],
            reason="Content has been flagged for review — publishing is blocked until a human moderator approves it.",
            condition=PolicyRuleCondition(
                tool_name="publish_post",
                context_equals={"content_status": "flagged"},
            ),
        ),
        PolicyRule(  # <- DENY publish when context signals content is rejected. Highest priority.
            rule_id="deny-publish-rejected",
            action="deny",
            priority=300,
            enabled=True,
            subjects=["tool_call"],
            reason="Content has been rejected — publishing is permanently blocked for this post.",
            condition=PolicyRuleCondition(
                tool_name="publish_post",
                context_equals={"content_status": "rejected"},
            ),
        ),
        PolicyRule(  # <- DEFER publish for opinion-category posts (require human approval).
            rule_id="defer-publish-sensitive-category",
            action="defer",
            priority=150,
            enabled=True,
            subjects=["tool_call"],
            reason="Posts in sensitive categories require moderator approval before publishing.",
            condition=PolicyRuleCondition(
                tool_name="publish_post",
                context_equals={"content_category": "opinion"},
            ),
        ),
        PolicyRule(  # <- Default ALLOW — fallback for all actions.
            rule_id="default-allow",
            action="allow",
            priority=50,
            enabled=True,
            subjects=["any"],
            reason="Default policy: allow all actions unless a higher-priority rule blocks them.",
            condition=PolicyRuleCondition(),
        ),
    ],
)


# ===========================================================================
# PolicyRole callbacks (dynamic, runtime policy decisions)
# ===========================================================================
# PolicyRole is a protocol: (event: PolicyEvent) -> PolicyDecision
#
# PolicyEvent fields:
#   event_type: str (e.g., "tool_before_execute")
#   run_id, thread_id, step: execution context
#   tool_name: str | None — which tool is being called
#   tool_args: dict | None — the arguments passed to the tool
#   context: dict — the runtime context from runner.run(context={...})
#   metadata: dict — additional event metadata
#
# PolicyDecision fields:
#   action: "allow" | "deny" | "defer" | "rewrite"
#   reason: str | None — explanation for the decision
#   updated_tool_args: dict | None — replacement args (for "rewrite" action)
#   policy_id: str | None — identifier for audit logging


def content_scan_role(event) -> object:  # <- PolicyRole callback. Called for every policy event during agent execution. The event object has tool_name, tool_args, context, etc.
    """Dynamically scan tool arguments for sensitive keywords.

    Unlike static PolicyRules (which can only check context_equals), this role
    inspects the actual tool_args content at runtime. If the publish_post tool
    is called with content containing sensitive keywords, it denies the call
    even if the context didn't flag the content.

    This catches cases where the LLM decides to publish without analyzing first.
    """
    if event.tool_name != "publish_post":
        return _allow()  # <- Only inspect publish_post calls. Other tools pass through.

    tool_args = event.tool_args or {}
    content = tool_args.get("content", "").lower()

    # --- Scan for sensitive keywords in the actual content ---
    found = [kw for kw in SENSITIVE_KEYWORDS if kw in content]
    if found:
        return _deny(  # <- Block publish if sensitive keywords are found in the content itself.
            reason=f"Content scan detected policy-violating keywords: {', '.join(found)}. Publishing denied.",
            policy_id="content-scan-role",
        )

    return _allow()


def author_rate_limit_role(event) -> object:  # <- PolicyRole callback for rate-limiting. Prevents any single author from publishing too many posts in one session.
    """Rate-limit publishing per author.

    This role tracks how many posts an author has published (using the
    shared PUBLISHED_CONTENT list) and denies further publishing if they
    exceed the limit. This is dynamic logic that cannot be expressed as
    a static PolicyRule because it depends on runtime state.
    """
    if event.tool_name != "publish_post":
        return _allow()

    tool_args = event.tool_args or {}
    author = tool_args.get("author", "").lower()
    max_posts_per_author = 5  # <- Configurable rate limit per author per session.

    author_count = sum(1 for p in PUBLISHED_CONTENT if p.get("author", "").lower() == author)
    if author_count >= max_posts_per_author:
        return _deny(
            reason=f"Author '{author}' has reached the publishing limit ({max_posts_per_author} posts). Further publishing is denied.",
            policy_id="author-rate-limit-role",
        )

    return _allow()


def category_rewrite_role(event) -> object:  # <- PolicyRole callback that demonstrates the "rewrite" action. It can modify tool arguments before the tool executes.
    """Auto-assign default category for publish_post if none provided.

    The "rewrite" action lets a PolicyRole modify the tool arguments before
    the tool actually executes. This is useful for normalizing inputs,
    applying defaults, or sanitizing data without denying the call.
    """
    if event.tool_name != "publish_post":
        return _allow()

    tool_args = event.tool_args or {}
    category = tool_args.get("category", "").strip().lower()

    valid_categories = {"general", "news", "opinion", "creative"}
    if category not in valid_categories:
        updated_args = dict(tool_args)
        updated_args["category"] = "general"  # <- Default to "general" if the category is invalid.
        return _rewrite(
            updated_tool_args=updated_args,
            reason=f"Invalid category '{category}' replaced with 'general'.",
            policy_id="category-rewrite-role",
        )

    return _allow()


# ===========================================================================
# Helper functions for PolicyDecision construction
# ===========================================================================
# These build PolicyDecision-compatible objects. We use simple namespace
# objects since the Runner only reads the .action, .reason, .policy_id,
# and .updated_tool_args attributes from the returned decision.

class _PolicyDecision:
    """Minimal PolicyDecision-compatible object for PolicyRole returns."""
    def __init__(self, action="allow", reason=None, policy_id=None, updated_tool_args=None, matched_rules=None, request_payload=None):
        self.action = action
        self.reason = reason
        self.policy_id = policy_id
        self.updated_tool_args = updated_tool_args
        self.matched_rules = matched_rules or []
        self.request_payload = request_payload or {}


def _allow():
    return _PolicyDecision(action="allow")


def _deny(reason: str, policy_id: str):
    return _PolicyDecision(action="deny", reason=reason, policy_id=policy_id)


def _rewrite(updated_tool_args: dict, reason: str, policy_id: str):
    return _PolicyDecision(action="rewrite", reason=reason, policy_id=policy_id, updated_tool_args=updated_tool_args)
