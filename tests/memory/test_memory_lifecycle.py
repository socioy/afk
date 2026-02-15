from __future__ import annotations

import asyncio

from afk.memory import (
    InMemoryMemoryStore,
    MemoryEvent,
    RetentionPolicy,
    StateRetentionPolicy,
    apply_event_retention,
    apply_state_retention,
    compact_thread_memory,
)


def _event(idx: int, event_type: str = "trace") -> MemoryEvent:
    return MemoryEvent(
        id=f"evt_{idx}",
        thread_id="thread_1",
        user_id="u1",
        type=event_type,
        timestamp=idx,
        payload={"idx": idx},
    )


def test_apply_event_retention_keeps_latest_with_priority_types():
    events = [
        _event(1, "trace"),
        _event(2, "debug"),
        _event(3, "trace"),
        _event(4, "debug"),
        _event(5, "debug"),
    ]
    policy = RetentionPolicy(max_events_per_thread=3, keep_event_types=["trace"])
    kept = apply_event_retention(events, policy=policy)

    assert len(kept) == 3
    assert {event.id for event in kept}.issuperset({"evt_1", "evt_3"})
    assert kept[-1].id == "evt_5"


def test_apply_state_retention_keeps_resume_safe_checkpoint_keys():
    state = {
        "checkpoint:run_new:latest": {
            "schema_version": "v1",
            "run_id": "run_new",
            "step": 4,
            "phase": "pre_tool_batch",
            "timestamp_ms": 200,
            "payload": {},
        },
        "checkpoint:run_new:4:pre_tool_batch": {
            "schema_version": "v1",
            "run_id": "run_new",
            "step": 4,
            "phase": "pre_tool_batch",
            "timestamp_ms": 200,
            "payload": {},
        },
        "checkpoint:run_new:4:runtime_state": {
            "schema_version": "v1",
            "run_id": "run_new",
            "step": 4,
            "phase": "runtime_state",
            "timestamp_ms": 200,
            "payload": {"messages": []},
        },
        "checkpoint:run_old:latest": {
            "schema_version": "v1",
            "run_id": "run_old",
            "step": 1,
            "phase": "runtime_state",
            "timestamp_ms": 10,
            "payload": {},
        },
        "checkpoint:run_old:1:runtime_state": {
            "schema_version": "v1",
            "run_id": "run_old",
            "step": 1,
            "phase": "runtime_state",
            "timestamp_ms": 10,
            "payload": {"messages": []},
        },
        "effect:run_new:4:tc_1": {"input_hash": "a", "output_hash": "b", "success": True},
        "effect:run_old:1:tc_1": {"input_hash": "a", "output_hash": "b", "success": True},
        "custom:key": {"value": 1},
    }
    compacted = apply_state_retention(
        state,
        policy=StateRetentionPolicy(
            max_runs=1,
            max_runtime_states_per_run=1,
            max_effect_entries_per_run=5,
        ),
    )
    assert "checkpoint:run_new:latest" in compacted
    assert "checkpoint:run_new:4:runtime_state" in compacted
    assert "checkpoint:run_old:latest" not in compacted
    assert "effect:run_new:4:tc_1" in compacted
    assert "effect:run_old:1:tc_1" not in compacted
    assert compacted["custom:key"] == {"value": 1}


def test_compact_thread_memory_applies_event_and_state_retention():
    async def _scenario() -> tuple[int, int]:
        store = InMemoryMemoryStore()
        await store.setup()
        for idx in range(6):
            await store.append_event(_event(idx, "trace" if idx % 2 == 0 else "debug"))
        await store.put_state(
            "thread_1",
            "checkpoint:run_1:latest",
            {
                "schema_version": "v1",
                "run_id": "run_1",
                "step": 1,
                "phase": "runtime_state",
                "timestamp_ms": 100,
                "payload": {},
            },
        )
        await store.put_state(
            "thread_1",
            "checkpoint:run_1:1:runtime_state",
            {
                "schema_version": "v1",
                "run_id": "run_1",
                "step": 1,
                "phase": "runtime_state",
                "timestamp_ms": 100,
                "payload": {"messages": []},
            },
        )
        await store.put_state(
            "thread_1",
            "checkpoint:run_0:latest",
            {
                "schema_version": "v1",
                "run_id": "run_0",
                "step": 1,
                "phase": "runtime_state",
                "timestamp_ms": 10,
                "payload": {},
            },
        )
        result = await compact_thread_memory(
            store,
            thread_id="thread_1",
            event_policy=RetentionPolicy(max_events_per_thread=3, keep_event_types=["trace"]),
            state_policy=StateRetentionPolicy(max_runs=1),
        )
        remaining_state = await store.list_state("thread_1")
        await store.close()
        assert result.events_after <= 3
        assert "checkpoint:run_1:latest" in remaining_state
        assert "checkpoint:run_0:latest" not in remaining_state
        return result.events_removed, result.state_keys_removed_effective

    removed_events, removed_state = asyncio.run(_scenario())
    assert removed_events >= 1
    assert removed_state >= 1
