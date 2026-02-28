# Release and Triage Playbook

This playbook defines how AFK maintainers handle issue triage, PR review flow,
release hygiene, and backport decisions.

---

## Issue Triage

For every incoming issue, follow this sequence:

### Step 1: Validate

- Confirm the issue is reproducible (or request a minimal repro).
- Check if it's a duplicate of an existing issue.
- Verify the reporter is using a supported Python version (3.13+ baseline).

### Step 2: Classify Area

| Area | Scope | Key Files |
|------|-------|-----------|
| Runner | Execution loop, checkpointing, budget | `src/afk/core/runner/` |
| Streaming | Stream bridge, events, handles | `src/afk/core/streaming.py` |
| Tools | Execution pipeline, hooks, middleware, security | `src/afk/tools/` |
| Memory | Store lifecycle, adapters, retention | `src/afk/memory/` |
| LLM | Provider adapters, runtime client, retry/fallback | `src/afk/llms/` |
| Queues | Task queues, workers, dead-letter | `src/afk/queues/` |
| A2A | Agent-to-agent, auth, protocol | `src/afk/agents/a2a/` |
| MCP | MCP server/client, store, transport | `src/afk/mcp/` |
| Agents | Configuration, lifecycle, policies, skills | `src/afk/agents/` |
| Observability | Telemetry, exporters, collectors | `src/afk/observability/` |
| Debugger | Debug instrumentation, redaction | `src/afk/debugger/` |
| Evals | Evaluation suite, assertions, datasets | `src/afk/evals/` |
| Docs | Documentation, examples | `docs/` |

### Step 3: Set Severity

| Severity | Criteria | Response Target |
|----------|----------|-----------------|
| `P0` | Data loss, security vulnerability, hard crash in common path | Immediate patch |
| `P1` | Major user-facing breakage with workaround available | Next patch release |
| `P2` | Moderate defect or quality regression | Next minor release |
| `P3` | Enhancement, polish, or nice-to-have improvement | Backlog |

### Step 4: Label and Assign

- Apply area label: `area:runner`, `area:tools`, `area:memory`, etc.
- Apply type label: `bug`, `feature`, `docs`, `security`, `breaking`.
- Apply severity label: `P0`, `P1`, `P2`, `P3`.
- Tag high-risk areas if the issue touches any subsystem in the high-risk list.
- Assign to the maintainer with most context in that area.

---

## PR Triage

### On PR Open

1. **Verify template**: Scope selected, summary written, release note present.
2. **Check CI**: Lint and tests must be green before human review begins.
3. **Assess scope**: Is the PR focused on one concern? If it mixes unrelated changes, request a split.
4. **Check impact surface**: How many subsystems does this touch?

### On PR Review

1. Apply the appropriate checklist from `code-review-checklist.md`.
2. For high-risk areas, require a second reviewer.
3. Verify CHANGELOG entry matches the actual change.
4. Verify docs are updated for user-visible changes.

### When to Request Splitting

Request a PR split when:
- It mixes a bug fix with a refactor.
- It touches more than 3 unrelated subsystems.
- The diff is large enough that reviewing individual concerns is difficult.
- It introduces a new feature AND changes existing behavior.

---

## Release Hygiene

### Pre-Release Checklist

- [ ] `ruff check src tests` and `ruff format --check src tests` green.
- [ ] Full test suite green: `pytest`.
- [ ] All CHANGELOG entries grouped under correct headings (`Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`, `Security`).
- [ ] Breaking changes include migration notes in CHANGELOG and affected doc pages.
- [ ] Provider adapter changes include integration sanity checks.
- [ ] All doc pages referenced in CHANGELOG are updated and accurate.
- [ ] Version bump in `pyproject.toml` follows semver.

### Release Note Quality

Release notes describe **what changed for users**, not implementation details.

```markdown
## [0.2.0] - 2026-03-01

### Added
- Vector memory search with cosine similarity via `MemoryStore.search_long_term_memory_vector()`.
- `SandboxProfile` enforcement for runtime tools with configurable command allowlists.

### Changed
- `FailSafeConfig.max_steps` default changed from 50 to 25 for safer default behavior.

### Fixed
- `run_sync` no longer raises `RuntimeError` when called from a CLI script on Python 3.13.
- Circuit breaker in `LLMClient` now correctly auto-resets after cooldown window expires.

### Security
- API key hashing in A2A auth now uses salted SHA-256 instead of plain SHA-256.
```

### Versioning Rules

- **Patch** (0.1.x): Bug fixes only, no behavior changes.
- **Minor** (0.x.0): New features, backward-compatible changes, deprecations.
- **Major** (x.0.0): Breaking changes, API removals, behavior changes.

---

## Backport Guidance

Backport to maintenance branches ONLY for:

- **Correctness fixes**: Bug that produces wrong results or data corruption.
- **Safety fixes**: Security vulnerabilities, crash bugs in common paths.
- **High-severity regressions**: P0/P1 issues introduced in a recent release.
- **Narrowly scoped patches**: Low merge risk, minimal code surface.

Do NOT backport:
- Refactors, performance improvements, or new features.
- Changes that touch more than 3 files.
- Changes that require modifying tests significantly.

---

## Emergency Response

For P0 issues (data loss, security, hard crash):

1. **Acknowledge within 4 hours** -- assign a maintainer, set `P0` label.
2. **Isolate the root cause** -- create a minimal repro.
3. **Ship the hotfix** -- smallest safe patch, skip non-critical review gates if needed.
4. **Write regression test** -- can be in a follow-up PR if under time pressure.
5. **Post-mortem** -- document root cause and prevention measures in the issue.
6. **Communicate** -- update CHANGELOG, notify users if the fix changes behavior.
