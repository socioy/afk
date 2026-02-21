# Failure Policies

Failure handling matrix across llm, tools, subagents, approvals, and runtime budgets.

Source: `docs/library/failure-policy-matrix.mdx`

This document explains how `FailSafeConfig` policy strings are normalized into runtime outcomes.

## TL;DR

- Policy strings are normalized to runtime actions (`fail`, `degrade`, `continue`).
- LLM failures are stricter: most values end up as `fail` or `degrade`.
- Tool/subagent/approval failures can continue depending on policy.
- Choosing policy values changes user-visible behavior significantly.

## When to Use

- You are setting `FailSafeConfig` for production.
- You need predictable behavior during outages/tool failures.
- You are tuning reliability vs strictness tradeoffs.

Primary code path:

- [afk/core/runner/internals.py](https://github.com/socioy/afk/blob/main/src/afk/core/runner/internals.py)

Policy source type:

- [afk/agents/types/__init__.py](https://github.com/socioy/afk/blob/main/src/afk/agents/types/__init__.py) (`FailurePolicy`, `FailSafeConfig`)

## Required Capabilities

- explicit `FailSafeConfig` values per failure domain
- consistent policy strategy across llm/tools/subagents/approvals
- alerting and telemetry on degraded/continued failure paths
- test coverage for expected fail/degrade/continue behavior

## Failure Policy Values

Allowed `FailurePolicy` values:

- `retry_then_fail`
- `retry_then_degrade`
- `fail_fast`
- `continue_with_error`
- `retry_then_continue`
- `continue`
- `fail_run`
- `skip_action`

## LLM Failure Mapping

Function: `_apply_llm_failure_policy(...)`

| Policy value | Runtime action |
| --- | --- |
| `retry_then_degrade` | `degrade` |
| `continue` | `degrade` |
| `continue_with_error` | `degrade` |
| `retry_then_continue` | `degrade` |
| `skip_action` | `degrade` |
| any other configured value | `fail` |

Practical effect:

- `degrade`: run transitions to degraded terminal path
- `fail`: error is raised and run enters failed terminal path

## Tool Failure Mapping

Function: `_apply_tool_failure_policy(...)`

| Policy value | Runtime action |
| --- | --- |
| `retry_then_degrade` | `degrade` |
| `retry_then_fail` | `fail` |
| `fail_fast` | `fail` |
| `fail_run` | `fail` |
| all others | `continue` |

## Subagent Failure Mapping

Function: `_apply_subagent_failure_policy(...)`

| Policy value | Runtime action |
| --- | --- |
| `retry_then_degrade` | `degrade` |
| `retry_then_fail` | `fail` |
| `fail_fast` | `fail` |
| `fail_run` | `fail` |
| all others | `continue` |

## Approval Denial Mapping

Function: `_apply_approval_denial_policy(...)`

| Policy value | Runtime action |
| --- | --- |
| `retry_then_degrade` | `degrade` |
| `retry_then_fail` | `fail` |
| `fail_fast` | `fail` |
| `fail_run` | `fail` |
| all others | `continue` |

## Where These Outcomes Are Applied

LLM stage:

- policy deny for `llm_before_execute`
- all candidate model attempts failed
- approval/input gating denied in LLM stage

Tool stage:

- tool execution errors
- sandbox violations
- approval/input denial for tool gates

Subagent stage:

- one or more subagent failures in selected batch

Approval/user-input stage:

- denied decisions from interaction provider
- deferred timeouts falling back to deny when configured

## Failure Modes

- permissive policies can hide operationally critical errors
- overly strict policies can reduce resilience during transient outages
- inconsistent policy combinations cause surprising user-visible outcomes
- missing telemetry makes degrade/continue behavior hard to detect

## Recommendations

For strict safety-critical flows:

- Use `fail_fast` or `fail_run` for tool and approval policies

For resilient exploratory flows:

- Use `continue_with_error` or `continue`
- pair with clear operator-visible warnings and telemetry

For controlled degradation:

- Use `retry_then_degrade` to terminate gracefully with partial results

## Practical Profiles

- strict operations:
  - llm/tool/subagent/approval policies: `fail_fast` or `fail_run`
- balanced operations:
  - llm: `retry_then_degrade`
  - tool/subagent: `retry_then_continue`
  - approval: `retry_then_fail`
- exploratory operations:
  - llm: `retry_then_degrade`
  - tool/subagent/approval: `continue_with_error` or `continue`

## Examples

- [examples/02_policy_with_hitl.py](/library/examples/index#02-policy-with-hitl)
- [examples/03_subagents_with_router.py](/library/examples/index#03-subagents-with-router)

## Continue Reading

1. [Tool Lifecycle](/library/tool-call-lifecycle)
2. [Run Event Contract](/library/run-event-contract)
