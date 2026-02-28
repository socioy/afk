"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Memory lifecycle helpers for retention and compaction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .store import MemoryStore
from .types import JsonValue, MemoryEvent


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """
    Policy controlling how many events are retained for a single thread.

    Fields
    - max_events_per_thread: maximum number of events to keep per thread.
    - keep_event_types: event `type` values that should always be preserved
      (these count toward `max_events_per_thread` but are never evicted
      preferentially).
    - scan_limit: an upper bound on how many recent events to fetch from the
      store when performing compaction (protects stores that would otherwise
      scan large histories).
    """

    max_events_per_thread: int = 5000
    keep_event_types: list[str] = field(default_factory=lambda: ["trace"])
    scan_limit: int = 20_000


@dataclass(frozen=True, slots=True)
class StateRetentionPolicy:
    """
    Retention rules for checkpoint and runtime state entries stored per thread.

    Behavior summary
    - Keeps metadata for up to `max_runs` most-recent runs (based on
      `checkpoint:<run_id>:latest`).
    - For each retained run the policy preserves:
      * the `checkpoint:<run_id>:latest` row,
      * the checkpoint row for the latest step/phase,
      * the most-recent `runtime_state` rows (up to
        `max_runtime_states_per_run`),
      * the most-recent `effect` entries (up to
        `max_effect_entries_per_run`),
      * any checkpoint rows whose `phase` is listed in
        `always_keep_phases`.
    - Any state keys that start with a prefix in `keep_state_prefixes` are
      always retained.

    The retention function attempts to keep the most-relevant state while
    bounding storage growth.
    """

    max_runs: int = 100
    max_runtime_states_per_run: int = 3
    max_effect_entries_per_run: int = 3000
    always_keep_phases: list[str] = field(
        default_factory=lambda: [
            "run_terminal",
            "runtime_state",
            "pre_tool_batch",
            "post_tool_batch",
            "pre_llm",
            "post_llm",
            "run_started",
            "paused",
            "resumed",
        ]
    )
    keep_state_prefixes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MemoryCompactionResult:
    """
    Result summary returned by `compact_thread_memory`.

    Attributes
    - events_before / events_after / events_removed: counts describing how
      many event records existed before and after compaction and how many
      were removed.
    - state_keys_before / state_keys_after / state_keys_removed: counts for
      state keys considered and selected for removal by the retention policy.
    - state_keys_removed_effective: number of keys actually deleted from the
      underlying `MemoryStore` (the store may not support deletion and may
      raise NotImplementedError).
    """

    events_before: int
    events_after: int
    events_removed: int
    state_keys_before: int
    state_keys_after: int
    state_keys_removed: int
    state_keys_removed_effective: int


def apply_event_retention(
    events: list[MemoryEvent],
    *,
    policy: RetentionPolicy,
) -> list[MemoryEvent]:
    """
    Produce a bounded, prioritized list of events to retain for a thread.

    Rules
    - Events whose `type` is listed in `policy.keep_event_types` are always
      preserved (they still count toward `max_events_per_thread`).
    - After preserving those types, the most-recent remaining events are
      included until `max_events_per_thread` is reached.
    - If the preserved set alone exceeds `max_events_per_thread`, the most
      recent preserved events are kept and older preserved events are
      dropped.
    - The returned list is sorted by `timestamp` (oldest → newest).

    Parameters
    - events: chronological list of `MemoryEvent` (older first) returned by
      the backing store's `get_recent_events`.
    - policy: retention parameters that control which events are protected
      and the maximum number to retain.

    Returns
    - A filtered list of `MemoryEvent` suitable for replacing the thread's
      stored events (no side effects).
    """
    if not events:
        return []
    if policy.max_events_per_thread <= 0:
        return []

    keep_types = set(policy.keep_event_types)
    preserved: list[MemoryEvent] = [
        event for event in events if event.type in keep_types
    ]
    if len(preserved) >= policy.max_events_per_thread:
        return preserved[-policy.max_events_per_thread :]

    remainder = [
        event for event in events if event.type not in keep_types
    ]
    budget = policy.max_events_per_thread - len(preserved)
    return sorted([*preserved, *remainder[-budget:]], key=lambda event: event.timestamp)


def apply_state_retention(
    state: dict[str, JsonValue],
    *,
    policy: StateRetentionPolicy,
) -> dict[str, JsonValue]:
    """
    Filter checkpoint/runtime state keys according to `policy` and return
    the subset that should be kept.

    Expected state key patterns understood by this function:
    - `checkpoint:<run_id>:latest` — latest metadata for a run
    - `checkpoint:<run_id>:<step>:<phase>` — checkpoint rows (phase-valued)
    - `effect:<run_id>:<step>:<id>` — effect log entries

    Retention strategy (high level):
    - Identify the most-recent `max_runs` by inspecting `checkpoint:<id>:latest`.
    - For each retained run, keep the `latest` row, the checkpoint row for the
      latest step/phase, recent `runtime_state` rows (up to
      `max_runtime_states_per_run`), `effect` entries (up to
      `max_effect_entries_per_run`), and any rows whose phase is in
      `always_keep_phases`.
    - Keys that match any prefix in `keep_state_prefixes` are always kept.
    - Any keys that do not match recognized patterns are treated as
      passthrough and preserved by default.

    Parameters
    - state: mapping of state keys → JSON-like payloads (as returned by the
      store's `list_state`).
    - policy: rules that determine which keys to keep.

    Returns
    - A new dict containing only the keys that should be retained.
    """
    if not state:
        return {}

    always_keep = set(policy.always_keep_phases)
    latest_rows: list[tuple[int, str, dict[str, Any]]] = []
    checkpoint_rows: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    effect_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    keep_keys: set[str] = set()
    passthrough_keys: set[str] = set()

    for key, value in state.items():
        latest_run = _parse_checkpoint_latest_key(key)
        if latest_run is not None:
            ts = _extract_timestamp_ms(value)
            latest_rows.append(
                (ts, latest_run, value if isinstance(value, dict) else {})
            )
            continue

        checkpoint_row = _parse_checkpoint_state_key(key)
        if checkpoint_row is not None:
            run_id, step, phase = checkpoint_row
            checkpoint_rows[run_id].append((step, phase, key))
            continue

        effect_row = _parse_effect_key(key)
        if effect_row is not None:
            run_id, step = effect_row
            effect_rows[run_id].append((step, key))
            continue

        passthrough_keys.add(key)

    latest_rows.sort(key=lambda row: (-row[0], row[1]))
    kept_runs = {row[1] for row in latest_rows[: max(policy.max_runs, 1)]}

    for key in passthrough_keys:
        keep_keys.add(key)

    for prefix in policy.keep_state_prefixes:
        for key in state:
            if key.startswith(prefix):
                keep_keys.add(key)

    for _ts, run_id, latest_payload in latest_rows:
        if run_id not in kept_runs:
            continue
        latest_key = f"checkpoint:{run_id}:latest"
        if latest_key in state:
            keep_keys.add(latest_key)

        latest_step = _safe_int(latest_payload.get("step"))
        latest_phase_raw = latest_payload.get("phase")
        latest_phase = latest_phase_raw if isinstance(latest_phase_raw, str) else ""
        if latest_step is not None and latest_phase:
            boundary_key = f"checkpoint:{run_id}:{latest_step}:{latest_phase}"
            if boundary_key in state:
                keep_keys.add(boundary_key)

        run_checkpoint_rows = checkpoint_rows.get(run_id, [])
        if not run_checkpoint_rows:
            continue

        runtime_rows = sorted(
            [row for row in run_checkpoint_rows if row[1] == "runtime_state"],
            key=lambda row: (-row[0], row[2]),
        )
        for step, phase, key in runtime_rows[
            : max(policy.max_runtime_states_per_run, 1)
        ]:
            _ = step
            _ = phase
            keep_keys.add(key)

        for step, phase, key in run_checkpoint_rows:
            if phase == "run_terminal":
                keep_keys.add(key)
            if phase in always_keep:
                keep_keys.add(key)
            if latest_step is not None and step == latest_step:
                keep_keys.add(key)

    for run_id in kept_runs:
        rows = sorted(
            effect_rows.get(run_id, []),
            key=lambda row: (-row[0], row[1]),
        )
        for _step, key in rows[: max(policy.max_effect_entries_per_run, 1)]:
            keep_keys.add(key)

    return {key: state[key] for key in sorted(keep_keys) if key in state}


async def compact_thread_memory(
    memory: MemoryStore,
    *,
    thread_id: str,
    event_policy: RetentionPolicy | None = None,
    state_policy: StateRetentionPolicy | None = None,
) -> MemoryCompactionResult:
    """
    Run compaction for a single thread: prune events and state according to
    the provided retention policies and persist changes to `memory` where
    supported.

    Steps performed
    1. Read recent events (bounded by `event_policy.scan_limit`) and apply
       `apply_event_retention`.
    2. If the trimmed event list differs, call `memory.replace_thread_events`.
       Stores that don't implement replacement may raise `NotImplementedError`
       and will be skipped.
    3. Read the thread's state, compute the retained keys via
       `apply_state_retention`, and delete removed keys with
       `memory.delete_state` where implemented.

    Parameters
    - memory: concrete `MemoryStore` instance used for reads/writes.
    - thread_id: identifier of the thread to compact.
    - event_policy / state_policy: optional policy overrides (defaults used if
      not provided).

    Returns
    - MemoryCompactionResult summarizing before/after counts and how many
      deletes were actually applied to the store.
    """
    event_policy = event_policy or RetentionPolicy()
    state_policy = state_policy or StateRetentionPolicy()

    events = await memory.get_recent_events(thread_id, limit=event_policy.scan_limit)
    retained_events = apply_event_retention(events, policy=event_policy)

    try:
        if _event_ids(events) != _event_ids(retained_events):
            await memory.replace_thread_events(thread_id, retained_events)
    except NotImplementedError:
        pass

    state = await memory.list_state(thread_id)
    retained_state = apply_state_retention(state, policy=state_policy)
    removed_state_keys = [key for key in state if key not in retained_state]

    removed_effective = 0
    for key in removed_state_keys:
        try:
            await memory.delete_state(thread_id, key)
            removed_effective += 1
        except NotImplementedError:
            continue

    return MemoryCompactionResult(
        events_before=len(events),
        events_after=len(retained_events),
        events_removed=max(0, len(events) - len(retained_events)),
        state_keys_before=len(state),
        state_keys_after=len(retained_state),
        state_keys_removed=max(0, len(state) - len(retained_state)),
        state_keys_removed_effective=removed_effective,
    )


def _event_ids(events: list[MemoryEvent]) -> list[str]:
    """Return a list of `id` values for the given events (preserves order)."""
    return [event.id for event in events]


def _safe_int(value: Any) -> int | None:
    """Safely convert `value` to an `int` when possible, otherwise return
    `None`.

    Accepts ints, floats and numeric strings (including negatives).
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _extract_timestamp_ms(value: JsonValue) -> int:
    """Extract `timestamp_ms` from a JSON-like `value` and return it as an
    integer number of milliseconds. Returns `0` when missing or invalid.
    """
    if not isinstance(value, dict):
        return 0
    ts = _safe_int(value.get("timestamp_ms"))
    return ts if ts is not None else 0


def _parse_checkpoint_latest_key(key: str) -> str | None:
    """Parse keys of the form `checkpoint:<run_id>:latest`.

    Returns the `run_id` when the key matches, otherwise `None`.
    """
    parts = key.split(":")
    if len(parts) == 3 and parts[0] == "checkpoint" and parts[2] == "latest":
        run_id = parts[1].strip()
        return run_id if run_id else None
    return None


def _parse_checkpoint_state_key(key: str) -> tuple[str, int, str] | None:
    """Parse checkpoint state keys of the form
    `checkpoint:<run_id>:<step>:<phase>`.

    Returns a tuple `(run_id, step, phase)` when the key is valid, otherwise
    `None`.
    """
    parts = key.split(":", 3)
    if len(parts) != 4 or parts[0] != "checkpoint":
        return None
    run_id = parts[1].strip()
    if not run_id:
        return None
    step = _safe_int(parts[2])
    phase = parts[3].strip()
    if step is None or not phase:
        return None
    return run_id, step, phase


def _parse_effect_key(key: str) -> tuple[str, int] | None:
    """Parse `effect` keys of the form `effect:<run_id>:<step>:<id>` and
    return `(run_id, step)` when valid. The trailing identifier is ignored by
    the parser.
    """
    parts = key.split(":", 3)
    if len(parts) != 4 or parts[0] != "effect":
        return None
    run_id = parts[1].strip()
    if not run_id:
        return None
    step = _safe_int(parts[2])
    if step is None:
        return None
    return run_id, step
