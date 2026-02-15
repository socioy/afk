# Security Model

This document explains AFK's security boundaries for prompt safety, tool execution, and secret propagation.

## TL;DR

- Tool output is untrusted by default and can be sanitized.
- Sandbox policies validate command, network, and path constraints pre-execution.
- Secret scope is opt-in and injected via tool context metadata.
- Production safety depends on policy + sandbox + operator approvals together.

## When To Read This

- You are enabling tools that read files, run commands, or call external systems.
- You are defining production defaults for safe agent execution.
- You need a checklist for secure-by-default rollout.

Primary code paths:

- [afk/agents/security.py](../../src/afk/agents/security.py)
- [afk/core/runner_execution.py](../../src/afk/core/runner_execution.py)
- [afk/tools/security.py](../../src/afk/tools/security.py)
- [afk/core/runner_types.py](../../src/afk/core/runner_types.py)

## Trust Boundaries

AFK separates trusted and untrusted content channels:

- trusted system channel: framework/runtime-authored instructions
- untrusted tool channel: raw tool output returned to LLM context

Runtime can inject a preamble reminding model not to obey instructions embedded inside tool output.

## Tool Output Sanitization

When enabled (`RunnerConfig.sanitize_tool_output=True`):

- suspicious prompt-injection markers are redacted
- payloads are recursively sanitized
- output is length-bounded before appending to transcript
- content is wrapped with untrusted channel marker

Relevant functions:

- `sanitize_text(...)`
- `sanitize_json_value(...)`
- `render_untrusted_tool_message(...)`

## Sandbox Policy Enforcement

Sandbox policies are evaluated per tool call.

Profile fields include:

- network permission
- command execution permission
- command allowlist prefixes
- shell-operator restrictions
- path allowlist/denylist
- command timeout and output limits

Validation checks include:

- URL arguments when network is disabled
- command and args against allowlist/operator rules
- path-like fields against allowed/denied roots

## Output Limiting

After execution, tool output can be truncated according to sandbox profile limits.

Used in two places:

- registry middleware level (`build_registry_output_limit_middleware`)
- runner post-call normalization (`apply_tool_output_limits`)

## Secret Scope Provider

Runner optionally calls `secret_scope_provider.resolve(...)` for each tool call.

Resolved secrets are attached to tool context metadata for tool-level consumption.

Failure to resolve secret scope raises an execution error.

## Command Execution Risk Controls

For skill/runtime command-style tools, controls include:

- command allowlist
- shell-operator blocking
- cwd/root boundary enforcement
- timeout control
- stdout/stderr truncation

## Threats Covered

- prompt injection via tool output text
- unrestricted filesystem path traversal via tool args
- dangerous command composition through shell operators
- unintended network access from tool arguments
- oversized tool payloads destabilizing context windows

## Residual Risks to Consider

- allowlisted commands can still be dangerous in specific environments
- secret leakage depends on downstream tool implementations
- policy role code quality directly impacts enforcement guarantees
- external interaction providers must be authenticated and authorized

## Hardening Checklist

1. keep sanitization enabled
2. configure a strict default sandbox profile
3. provide per-tool sandbox overrides only when required
4. use approval gates for side-effecting tools
5. define command allowlists explicitly
6. audit secret scope provider output and tool usage
7. emit and monitor policy/tool failure telemetry

## Starter Secure Configuration

Use this baseline and tighten per environment:

```python
from afk.core import Runner, RunnerConfig
from afk.tools import SandboxProfile

runner = Runner(
    config=RunnerConfig(
        sanitize_tool_output=True,
        default_sandbox_profile=SandboxProfile(
            profile_id="default-secure",
            allow_network=False,
            allow_command_execution=False,
            deny_shell_operators=True,
            max_output_chars=20_000,
        ),
    )
)
```

## Related Example

- [examples/06_tool_registry_security.py](./examples/06_tool_registry_security.py)
