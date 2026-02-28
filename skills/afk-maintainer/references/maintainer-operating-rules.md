# AFK Maintainer Operating Rules

These rules govern how AFK maintainers evaluate, review, and merge contributions.
They apply to human contributors and agentic coding tools equally.

---

## Core Principles

1. **Reliability over novelty.** A boring fix that works is better than a clever refactor that might break.
2. **Deterministic runner semantics over convenience shortcuts.** The step loop must be predictable.
3. **Minimal safe change over broad refactor.** Fix the root cause, nothing more.
4. **Explicit behavior contracts over implicit assumptions.** Every module boundary is typed.
5. **Clean docs and examples are product quality, not optional polish.** Ship docs with code.
6. **Extensible, composable code over tightly coupled patches.** Middleware and hooks over subclassing.
7. **DX is a feature.** Respect the developer's time with sensible defaults and actionable errors.

---

## PR Standards

Every PR must satisfy these requirements before review:

### Required for All PRs

- [ ] **Clear scope**: One concern per PR. Bug fix OR refactor OR feature -- never mixed.
- [ ] **PR template completed**: Scope selected, summary written, release note present.
- [ ] **Lint passing**: `ruff check src tests` and `ruff format --check src tests` green.
- [ ] **Tests passing**: `pytest` green with no new warnings.
- [ ] **No internal imports in examples**: All code samples use `from afk...` public paths.

### Required for Bug Fixes

- [ ] Minimal reproducible example provided or linked.
- [ ] Root cause identified and documented in PR description.
- [ ] Regression test added that fails before the fix and passes after.
- [ ] CHANGELOG entry under `### Fixed`.

### Required for Features

- [ ] Acceptance criteria defined in PR description.
- [ ] Happy-path and failure-path tests included.
- [ ] Documentation updated (relevant `.mdx` page + examples if user-visible).
- [ ] CHANGELOG entry under `### Added`.
- [ ] Public API reviewed for consistency with existing patterns.

### Required for Breaking Changes

- [ ] Migration notes in CHANGELOG and affected doc pages.
- [ ] Deprecation warning added (if possible to maintain temp backward compat).
- [ ] All examples updated to reflect new behavior.
- [ ] CHANGELOG entry under `### Changed` or `### Removed`.

---

## High-Risk Change Protocol

Apply this protocol for changes in **any** of these areas:
- `src/afk/core/runner/` -- Execution loop, checkpointing, budget enforcement
- `src/afk/core/streaming.py` -- Stream bridge, event emission
- `src/afk/tools/core/base.py` -- Tool calling pipeline, hooks, middleware
- `src/afk/llms/runtime/` -- Circuit breakers, retry logic, fallback chains
- `src/afk/memory/` -- Store lifecycle, concurrent access, data integrity
- `src/afk/agents/a2a/` -- Authentication, protocol correctness
- `src/afk/agents/lifecycle/runtime.py` -- Circuit breaker, dependency health

### High-Risk Checklist

- [ ] **Event-loop safety**: Does the change affect `run_sync`, cancellation, or shutdown paths?
- [ ] **Concurrent access**: Are shared data structures protected by locks? Is the lock acquisition order consistent?
- [ ] **Resource lifecycle**: Are tasks stored (not fire-and-forget)? Are connections/handles closed in finally blocks?
- [ ] **Error propagation**: Do exception handlers always resolve futures/handles? Is the error chain preserved with `from`?
- [ ] **Budget enforcement**: Are threshold checks correct (>= vs >, off-by-one)?
- [ ] **Retry behavior**: Does backoff include actual delay? Are retries bounded? Is the retry classification correct?
- [ ] **Circuit breaker behavior**: Does the breaker auto-reset after cooldown? Is `record_success` thread-safe?
- [ ] **Checkpoint integrity**: Do checkpoint writes complete before the run handle resolves? Can a failing write leave dirty state?
- [ ] **Backward compatibility**: Does this change the behavior of existing code without an explicit opt-in?
- [ ] **Stream consumers**: Are error events deduplicated? Is there a single-consumer guard?

---

## Review Red Flags

Reject or request changes when you see any of these:

### Critical (Block merge)
- Silent behavior changes without docs/changelog update.
- Blanket `except Exception: pass` in critical paths (runner loop, tool execution, LLM calls).
- Missing tests for bug-fix paths.
- Fire-and-forget `asyncio.create_task()` without storing the reference.
- `assert` used for runtime invariants (stripped with `-O`).
- Blocking synchronous I/O (`Path.read_text()`, `open().read()`) inside `async def` functions.
- New defaults that increase cost or risk without explicit opt-in.

### Serious (Request changes)
- Unscoped PRs mixing unrelated concerns.
- Outdated examples or docs that no longer match runtime behavior.
- New coupling between modules that makes extension/reuse harder.
- Substring matching for error classification (e.g., `"500" in message`).
- `asyncio.sleep(0)` used as retry backoff (provides no actual delay).
- Unbounded data structures that grow over time without eviction.
- Provider-specific logic outside of adapter modules.

### Minor (Leave comment)
- Missing type annotations on public methods.
- Inconsistent naming with existing patterns.
- Redundant code that could be simplified.
- Missing docstring on complex non-obvious logic.

---

## Maintainer Workflow

### When Reviewing a PR

1. **Read the full diff** -- don't skim. Understand every line changed.
2. **Check scope** -- is this PR doing one thing? If it mixes concerns, request a split.
3. **Verify tests** -- do they test the actual behavior change? Are failure paths covered?
4. **Check docs** -- if behavior is user-visible, are docs updated?
5. **Apply the relevant checklist** -- standard, bug fix, feature, breaking change, or high-risk.
6. **Run locally if uncertain** -- `ruff check src tests && pytest` should pass.

### When Triaging Issues

1. **Confirm reproducibility** -- ask for minimal repro if not provided.
2. **Classify area** -- runner, tools, memory, queues, LLM adapters, A2A, MCP, docs.
3. **Set severity** using the P0-P3 scale (see release-and-triage-playbook.md).
4. **Assign labels** -- `bug`, `feature`, `docs`, `security`, `breaking`.
5. **Tag high-risk area** if applicable.

### When Approving a Merge

- All required checks pass (lint, tests, PR template).
- The change is the smallest safe change that addresses the root cause.
- No unrelated modifications are included.
- CHANGELOG entry is present and accurate.
- Documentation is updated if behavior is user-visible.
