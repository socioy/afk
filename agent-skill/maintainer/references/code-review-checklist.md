# AFK Code Review Checklist

Use this checklist when reviewing any PR to the AFK repository. Check the
applicable sections based on the PR type and affected areas.

---

## Universal Checks (Every PR)

### Scope and Hygiene
- [ ] PR addresses a single concern (bug fix, feature, refactor, or docs -- not a mix).
- [ ] PR template is filled out: scope selected, summary written, release note present.
- [ ] Lint passes: `ruff check src tests` and `ruff format --check src tests`.
- [ ] Test suite passes: `pytest` green with no new warnings.
- [ ] No unrelated formatting changes or drive-by refactors.

### Code Quality
- [ ] Type annotations on all new/modified public functions and methods.
- [ ] No raw `dict` passing across module boundaries (use Pydantic models or dataclasses).
- [ ] No provider-specific logic outside of `llms/clients/adapters/` or `llms/providers/`.
- [ ] No hidden mutable global state introduced.
- [ ] Naming follows existing conventions (see repo-design-and-quality-standards.md).

### Error Handling
- [ ] No bare `except Exception: pass` in critical paths.
- [ ] Exceptions use `raise ... from e` to preserve chains.
- [ ] Error messages are actionable (explain what, why, and how to fix).
- [ ] No `assert` statements used for runtime invariants.

### Async Safety
- [ ] No blocking I/O (`Path.read_text`, `open().read`, `requests.get`) in `async def` functions.
- [ ] `asyncio.create_task()` results are stored (not fire-and-forget).
- [ ] `asyncio.wait_for()` used with timeouts on external calls.
- [ ] No `asyncio.sleep(0)` as retry backoff (use actual delay).
- [ ] Asyncio primitives (Lock, Semaphore, Event) created in async context.

---

## Bug Fix Checks

- [ ] Minimal reproducible example provided or linked in PR description.
- [ ] Root cause identified and explained.
- [ ] Regression test added that fails before fix, passes after.
- [ ] CHANGELOG entry under `### Fixed`.
- [ ] Fix is minimal -- no unrelated changes bundled.

---

## Feature Checks

- [ ] Acceptance criteria defined in PR description.
- [ ] API design follows progressive disclosure (simple default, advanced opt-in).
- [ ] Configuration has sensible defaults (zero-config works).
- [ ] Happy-path tests included.
- [ ] Failure-path tests included (timeout, invalid input, unavailable dependency).
- [ ] Documentation updated for user-visible behavior.
- [ ] CHANGELOG entry under `### Added`.
- [ ] Examples use `from afk...` public imports.

---

## Runner / Core Execution Checks

Apply when PR touches `src/afk/core/runner/`, `src/afk/core/streaming.py`.

### Step Loop Integrity
- [ ] Budget enforcement checks use `>=` (not `>`) to prevent off-by-one.
- [ ] LLM call and tool call counters are incremented at the correct point.
- [ ] Step loop terminates on all terminal states (completed, failed, interrupted, cancelled).

### Checkpoint Safety
- [ ] Checkpoint writes complete before run handle resolves.
- [ ] Checkpoint writer errors are logged (not silently dropped).
- [ ] Failing checkpoint writes cannot leave memory store in dirty state.

### Stream Bridge
- [ ] Bridge task stored as reference on handle (not fire-and-forget).
- [ ] Error events are not emitted twice for the same failure.
- [ ] Single-consumer guard present on stream handles.

### Run Handle Lifecycle
- [ ] Exception handlers ALWAYS resolve the result future (use inner `finally`).
- [ ] `set_result` / `set_exception` is guaranteed to run even if cleanup awaits fail.

---

## Tool System Checks

Apply when PR touches `src/afk/tools/`.

### Tool Execution Pipeline
- [ ] Prehooks always return a dict (validated at hook registration).
- [ ] Posthooks are not silently skipped when tool returns ToolResult directly.
- [ ] Tool timeouts cover the full execution (including middleware).
- [ ] `raise_on_error` preserves exception chain with `from`.

### Registry
- [ ] `_records` list has a bounded size or eviction policy.
- [ ] Registry-level middleware timeout covers the full middleware chain.
- [ ] Sync middleware bridge has a timeout on `fut.result()`.

### Security
- [ ] Sandbox profile checks use proper escaping (not naive substring match).
- [ ] Shell operator detection doesn't produce false positives on legitimate args.
- [ ] Network access detection covers common parameter names beyond `url`/`uri`.
- [ ] `_truncate_json_like` and `sanitize_json_value` have recursion depth limits.

---

## LLM Client Checks

Apply when PR touches `src/afk/llms/`.

### Error Classification
- [ ] Retry phrases don't use bare numeric substring matching (avoid `"500" in message`).
- [ ] Status code ranges are used for HTTP error classification, not string matching.
- [ ] Error classification logic is unit-tested with edge cases.

### Retry and Fallback
- [ ] Retry backoff uses actual delay (not `sleep(0)`).
- [ ] Retry is bounded (max attempts configured).
- [ ] Fallback responses are not cached under the primary provider's cache key.
- [ ] All per-provider errors are preserved for diagnostic reporting when all providers fail.

### Streaming
- [ ] Stream body has idle timeout protection (not just initial connection timeout).
- [ ] `RuntimeStreamHandle.interrupt()` validates callback before setting `_interrupted` flag.
- [ ] Cancel/interrupt state is not conflated for error classification.

---

## Memory System Checks

Apply when PR touches `src/afk/memory/`.

### Store Lifecycle
- [ ] `setup()` and `close()` are coordinated under a lock to prevent concurrent setup/close races.
- [ ] Memory store is not used after `close()` is called.
- [ ] Memory fallback reason is set and logged when store becomes unavailable.

### Data Integrity
- [ ] SQLite operations that DELETE then INSERT use explicit transactions with rollback on failure.
- [ ] LIKE queries escape `%` and `_` wildcards in user-provided prefixes.
- [ ] Vector search has a LIMIT clause to prevent loading all rows into memory.
- [ ] Retention policy handles `max_events_per_thread=0` correctly (empty, not all events).

### Thread Safety
- [ ] All shared data structures accessed under asyncio.Lock.
- [ ] `record_success` is consistent with `record_failure` locking discipline.
- [ ] Lock-per-key maps have eviction to prevent unbounded growth.

---

## A2A / Communication Checks

Apply when PR touches `src/afk/agents/a2a/`, `src/afk/messaging/`.

### Protocol Correctness
- [ ] Task status is updated on ALL paths (success, failure, cancellation).
- [ ] Event log has bounded size or eviction policy.
- [ ] Streaming endpoints actually stream (not buffer-then-return).

### Authentication
- [ ] API key hashing uses salt (not plain SHA-256).
- [ ] JWT validation explicitly rejects `"none"` algorithm.
- [ ] Auth errors return proper HTTP status codes (401/403, not 500).

### Server
- [ ] Protocol errors are caught and returned as proper HTTP error responses.
- [ ] `get_task` / `cancel_task` handle missing tasks with 404 (not 500).
- [ ] Lazy client initialization is synchronized to prevent races.

---

## Documentation Checks

Apply when PR touches `docs/` or includes user-visible behavior changes.

- [ ] Affected doc pages updated to reflect new behavior.
- [ ] Examples use public `from afk...` imports.
- [ ] Examples are runnable (or obviously near-runnable).
- [ ] No duplicate explanations -- link to canonical pages instead.
- [ ] Import paths in docs match actual package exports.
- [ ] Module name references match code (`afk.observability` not `afk.telemetry`).
- [ ] CHANGELOG entry describes what changed for users (not implementation details).
