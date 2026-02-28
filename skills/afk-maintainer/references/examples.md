# Maintainer Examples

Concrete examples of triage notes, release notes, PR comments, and review
decisions that demonstrate AFK maintainer standards.

---

## Example: Issue Triage Note

```
Title: run_sync fails with RuntimeError on Python 3.13 in CLI scripts
Area: Core Runner lifecycle (src/afk/core/runner/)
Severity: P1
Reproducible: Yes (minimal repro provided)

Summary:
  Calling `Runner().run_sync(agent, input="hello")` from a standard CLI
  script raises RuntimeError due to event-loop bridge edge case where
  asyncio.run() is called inside an existing loop context.

Root Cause:
  The sync bridge in runner/api.py does not detect when called from
  within an active event loop (e.g., Jupyter, IPython, some CLI frameworks).

Required Actions:
  1. Add loop detection in run_sync (check for running loop before calling asyncio.run)
  2. Add regression test: test_run_sync_from_cli_context
  3. Add regression test: test_run_sync_from_active_loop_raises_clear_error
  4. Release note under ### Fixed
  5. Update core-runner.mdx if error message changes
```

---

## Example: Release Note Entry

```markdown
## [0.1.1] - 2026-03-01

### Fixed
- `run_sync` no longer raises `RuntimeError` when called from a CLI script on Python 3.13.
  The sync bridge now correctly detects active event loops and provides an actionable error
  message suggesting `await runner.run()` instead.

### Changed
- Circuit breaker in `LLMClient` now auto-resets after the configured cooldown window
  expires, instead of staying permanently open after reaching the failure threshold.

### Added
- `SandboxProfile.allowed_commands` now supports glob patterns for more flexible
  command allowlisting (e.g., `"python*"` to allow `python3`, `python3.13`, etc.).

### Security
- API key hashing in A2A `APIKeyAuthProvider` now uses HMAC-SHA256 with a server-side
  secret instead of plain SHA-256. Existing deployments should rotate keys after updating.
```

---

## Example: PR Comment -- Request Split

```
Thanks for this contribution!

The diff covers two distinct concerns:
1. A correctness fix for the circuit breaker cooldown reset (runtime.py)
2. A refactor renaming internal methods in the same file

Please split this into two PRs:
1. The behavior fix + regression test + release note
2. The rename refactor (no behavior change)

This keeps rollback risk low and release traceability clean. The behavior fix
can ship as a patch release, while the refactor can be batched with other cleanup.
```

---

## Example: PR Comment -- Blocking Issue

```
This PR introduces a potential issue that must be addressed before merge:

**File**: src/afk/core/runner/internals.py, line 472
**Issue**: `assert queue is not None` is used for a runtime invariant.

`assert` statements are stripped when Python runs with the `-O` flag. If this
invariant is ever violated in production, the code would proceed with `queue = None`
and crash with an unhelpful `AttributeError` instead of the intended error.

Please replace with:
```python
if queue is None:
    raise RuntimeError("Checkpoint writer queue not initialized")
```

This is a critical review flag per our maintainer operating rules (see "assert for
runtime invariants" in the red flags list).
```

---

## Example: PR Comment -- Approving with Notes

```
LGTM -- clean implementation, well-scoped, good test coverage.

Minor notes (non-blocking):
- Consider adding a brief docstring on `_infer_call_style` explaining the three
  supported signatures (args-only, args-ctx, ctx-args). The inspection logic is
  non-obvious to new contributors.
- The test name `test_tool_call_with_context` could be more specific:
  `test_tool_receives_context_when_signature_includes_ctx_param`

Approving as-is. The notes above can be addressed in a follow-up if you'd like.
```

---

## Example: Code Review -- Identifying Async Safety Issue

```
**File**: src/afk/tools/prebuilts/runtime.py, line 95
**Issue**: Blocking I/O on the event loop

`target.read_text(encoding="utf-8")` is a synchronous blocking call inside an
`async def` function. This blocks the event loop thread during execution.

Fix:
```python
content = await asyncio.to_thread(target.read_text, encoding="utf-8")
```

This is flagged in our coding principles as a critical anti-pattern (see
"Blocking I/O in async functions" in coding-principles-and-patterns.md).
```

---

## Example: Code Review -- Identifying Resource Leak

```
**File**: src/afk/core/runner/api.py, line 683
**Issue**: Fire-and-forget asyncio task

```python
asyncio.create_task(_bridge())
return stream
```

The task reference is not stored anywhere. Python's GC can collect the task
mid-execution. The task should be stored on the stream handle:

```python
task = asyncio.create_task(_bridge())
stream._bridge_task = task  # Prevent GC collection
return stream
```

See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
"Save a reference to the result of this function, to avoid a task disappearing
mid-execution."
```

---

## Example: Triage Decision -- Deferring Enhancement

```
Thanks for the feature request.

This is a reasonable enhancement, but it doesn't align with our current priorities:
- The existing API handles the common case well
- The proposed change would add complexity to the tool execution pipeline
- We don't have evidence of user demand beyond this single request

Labeling as `P3 / enhancement` and leaving open for community interest.
If more users request this or someone wants to submit a focused PR, we're
open to revisiting.
```

---

## Example: Emergency Hotfix Process

```
## P0: Data loss in SQLite memory adapter during compaction

### Timeline
- 2026-02-28 09:00 UTC -- Issue reported by user
- 2026-02-28 09:30 UTC -- Reproduced locally
- 2026-02-28 10:00 UTC -- Root cause identified: replace_thread_events has no
  rollback on INSERT failure, leaving DELETE committed
- 2026-02-28 11:00 UTC -- Hotfix PR opened with transaction wrapper + regression test
- 2026-02-28 12:00 UTC -- Patch release 0.1.2 shipped

### Root Cause
The `replace_thread_events` method in `sqlite.py` executes DELETE then INSERT
without an explicit transaction boundary. If any INSERT fails, the DELETE is
already committed, losing all thread events.

### Fix
Wrapped the DELETE + INSERT sequence in an explicit `BEGIN/COMMIT` transaction
with `ROLLBACK` on failure.

### Prevention
- Added to code-review-checklist.md: "SQLite operations that DELETE then INSERT
  must use explicit transactions with rollback on failure"
- Added integration test for partial INSERT failure during compaction
```
