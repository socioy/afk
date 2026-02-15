"""
Schema versioning helpers for agent events/checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


AGENT_EVENT_SCHEMA_VERSION = "v1"
CHECKPOINT_SCHEMA_VERSION = "v1"
SUPPORTED_EVENT_SCHEMA_VERSIONS = frozenset({AGENT_EVENT_SCHEMA_VERSION})
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({CHECKPOINT_SCHEMA_VERSION})


@dataclass(frozen=True, slots=True)
class VersionCheckResult:
    """
    Compatibility result for a schema version check.

    Attributes:
        compatible: Whether supplied version is supported.
        expected: Current schema version expected by runtime.
        received: Supplied schema version value.
        message: Human-readable status.
    """

    compatible: bool
    expected: str
    received: str | None
    message: str


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """
    Migration result payload for legacy event/checkpoint records.

    Attributes:
        migrated: Migrated normalized record payload.
        from_version: Source schema version, when known.
        to_version: Target schema version.
        applied: Ordered list of migration transforms applied.
    """

    migrated: dict[str, Any]
    from_version: str | None
    to_version: str
    applied: list[str]


def check_event_schema_version(version: str | None) -> VersionCheckResult:
    """
    Validate event schema version compatibility.

    Args:
        version: Version value from stored event record.

    Returns:
        Compatibility result for event schema.
    """
    if version in SUPPORTED_EVENT_SCHEMA_VERSIONS:
        return VersionCheckResult(
            compatible=True,
            expected=AGENT_EVENT_SCHEMA_VERSION,
            received=version,
            message="ok",
        )
    return VersionCheckResult(
        compatible=False,
        expected=AGENT_EVENT_SCHEMA_VERSION,
        received=version,
        message="Event schema version mismatch",
    )


def check_checkpoint_schema_version(version: str | None) -> VersionCheckResult:
    """
    Validate checkpoint schema version compatibility.

    Args:
        version: Version value from stored checkpoint record.

    Returns:
        Compatibility result for checkpoint schema.
    """
    if version in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        return VersionCheckResult(
            compatible=True,
            expected=CHECKPOINT_SCHEMA_VERSION,
            received=version,
            message="ok",
        )
    return VersionCheckResult(
        compatible=False,
        expected=CHECKPOINT_SCHEMA_VERSION,
        received=version,
        message="Checkpoint schema version mismatch",
    )


def migrate_event_record(record: dict[str, Any]) -> MigrationResult:
    """
    Migrate legacy event records into current event schema.

    Args:
        record: Raw persisted event record from storage.

    Returns:
        Migration result containing normalized event payload.

    Raises:
        ValueError: If resulting schema version is unsupported.
    """
    migrated = dict(record)
    applied: list[str] = []

    if "schema_version" not in migrated:
        legacy_version = migrated.get("schemaVersion")
        if isinstance(legacy_version, str):
            migrated["schema_version"] = legacy_version
            applied.append("schemaVersion->schema_version")
        else:
            migrated["schema_version"] = AGENT_EVENT_SCHEMA_VERSION
            applied.append("default_event_schema_version")

    if "type" not in migrated and isinstance(migrated.get("eventType"), str):
        migrated["type"] = migrated["eventType"]
        applied.append("eventType->type")

    version = migrated.get("schema_version")
    check = check_event_schema_version(version if isinstance(version, str) else None)
    if not check.compatible:
        raise ValueError(check.message)

    return MigrationResult(
        migrated=migrated,
        from_version=record.get("schema_version")
        if isinstance(record.get("schema_version"), str)
        else (
            record.get("schemaVersion")
            if isinstance(record.get("schemaVersion"), str)
            else None
        ),
        to_version=AGENT_EVENT_SCHEMA_VERSION,
        applied=applied,
    )


def migrate_checkpoint_record(record: dict[str, Any]) -> MigrationResult:
    """
    Migrate legacy checkpoint records into current checkpoint schema.

    Args:
        record: Raw persisted checkpoint record from storage.

    Returns:
        Migration result containing normalized checkpoint payload.

    Raises:
        ValueError: If resulting schema version is unsupported.
    """
    migrated = dict(record)
    applied: list[str] = []

    if "schema_version" not in migrated:
        legacy_version = migrated.get("schemaVersion")
        if isinstance(legacy_version, str):
            migrated["schema_version"] = legacy_version
            applied.append("schemaVersion->schema_version")
        elif isinstance(migrated.get("version"), str):
            migrated["schema_version"] = migrated["version"]
            applied.append("version->schema_version")
        else:
            migrated["schema_version"] = CHECKPOINT_SCHEMA_VERSION
            applied.append("default_checkpoint_schema_version")

    if "run_id" not in migrated and isinstance(migrated.get("runId"), str):
        migrated["run_id"] = migrated["runId"]
        applied.append("runId->run_id")
    if "thread_id" not in migrated and isinstance(migrated.get("threadId"), str):
        migrated["thread_id"] = migrated["threadId"]
        applied.append("threadId->thread_id")
    if "payload" not in migrated and isinstance(migrated.get("data"), dict):
        migrated["payload"] = migrated["data"]
        applied.append("data->payload")

    version = migrated.get("schema_version")
    check = check_checkpoint_schema_version(version if isinstance(version, str) else None)
    if not check.compatible:
        raise ValueError(check.message)

    return MigrationResult(
        migrated=migrated,
        from_version=record.get("schema_version")
        if isinstance(record.get("schema_version"), str)
        else (
            record.get("schemaVersion")
            if isinstance(record.get("schemaVersion"), str)
            else None
        ),
        to_version=CHECKPOINT_SCHEMA_VERSION,
        applied=applied,
    )
