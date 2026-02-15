from __future__ import annotations

import pytest

from afk.agents.versioning import (
    AGENT_EVENT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    check_checkpoint_schema_version,
    check_event_schema_version,
    migrate_checkpoint_record,
    migrate_event_record,
)


def test_check_event_schema_version_reports_compatibility():
    ok = check_event_schema_version(AGENT_EVENT_SCHEMA_VERSION)
    bad = check_event_schema_version("v999")
    assert ok.compatible is True
    assert bad.compatible is False


def test_check_checkpoint_schema_version_reports_compatibility():
    ok = check_checkpoint_schema_version(CHECKPOINT_SCHEMA_VERSION)
    bad = check_checkpoint_schema_version("legacy")
    assert ok.compatible is True
    assert bad.compatible is False


def test_migrate_event_record_from_legacy_aliases():
    result = migrate_event_record(
        {
            "schemaVersion": "v1",
            "eventType": "run_started",
            "run_id": "r1",
        }
    )
    assert result.to_version == AGENT_EVENT_SCHEMA_VERSION
    assert result.migrated["schema_version"] == "v1"
    assert result.migrated["type"] == "run_started"
    assert "schemaVersion->schema_version" in result.applied


def test_migrate_checkpoint_record_from_legacy_aliases():
    result = migrate_checkpoint_record(
        {
            "schemaVersion": "v1",
            "runId": "run_1",
            "threadId": "thread_1",
            "step": 2,
            "phase": "runtime_state",
            "data": {"messages": []},
        }
    )
    assert result.to_version == CHECKPOINT_SCHEMA_VERSION
    assert result.migrated["run_id"] == "run_1"
    assert result.migrated["thread_id"] == "thread_1"
    assert result.migrated["payload"] == {"messages": []}
    assert "data->payload" in result.applied


def test_migrate_checkpoint_record_rejects_unknown_schema():
    with pytest.raises(ValueError):
        migrate_checkpoint_record({"schema_version": "v999"})
