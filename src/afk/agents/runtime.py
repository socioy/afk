"""
Runtime helpers used by the core runner.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llms.types import JSONValue
from .errors import AgentCircuitOpenError, AgentExecutionError, SkillResolutionError
from .types import (
    AgentState,
    FailSafeConfig,
    SkillRef,
    SkillResolutionResult,
)


def resolve_skills(
    *,
    skill_names: list[str],
    skills_dir: Path,
    cwd: Path,
) -> SkillResolutionResult:
    """
    Resolve requested skill names to concrete SKILL.md paths.
    """
    base = skills_dir if skills_dir.is_absolute() else (cwd / skills_dir)
    base = base.resolve()

    resolved: list[SkillRef] = []
    missing: list[str] = []

    for name in skill_names:
        normalized = name.strip()
        if not normalized:
            continue
        skill_file = (base / normalized / "SKILL.md").resolve()
        if not _is_under(skill_file, base) or not skill_file.exists():
            missing.append(normalized)
            continue
        checksum = _sha256(skill_file.read_bytes())
        resolved.append(
            SkillRef(
                name=normalized,
                root_dir=str(base),
                skill_md_path=str(skill_file),
                checksum=checksum,
            )
        )

    if missing:
        raise SkillResolutionError(
            f"Missing requested skills under '{base}': {', '.join(missing)}"
        )

    return SkillResolutionResult(resolved_skills=resolved, missing_skills=missing)


def build_skill_manifest_prompt(skills: list[SkillRef]) -> str:
    """
    Build compact skill manifest text to place in system instructions.
    """
    if not skills:
        return ""

    lines = [
        "Skills are enabled for this run.",
        "Use skill tools to inspect detailed skill docs before acting.",
        "Enabled skills:",
    ]
    for skill in skills:
        lines.append(
            f"- {skill.name} (path: {skill.skill_md_path}, checksum: {skill.checksum or 'n/a'})"
        )
    return "\n".join(lines)


def to_message_payload(role: str, text: str) -> dict[str, JSONValue]:
    """Build a JSON-safe message payload used by router inputs."""
    return {"role": role, "text": text}


def _is_under(path: Path, root: Path) -> bool:
    """Return `True` when `path` is within `root`."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(blob: bytes) -> str:
    """Return SHA-256 hex digest for bytes."""
    return hashlib.sha256(blob).hexdigest()


def json_hash(payload: dict[str, Any]) -> str:
    """Build a stable hash for JSON-like payloads."""
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def checkpoint_state_key(run_id: str, step: int, phase: str) -> str:
    """Return namespaced key for step-level checkpoint state."""
    return f"checkpoint:{run_id}:{step}:{phase}"


def checkpoint_latest_key(run_id: str) -> str:
    """Return namespaced key for latest checkpoint pointer."""
    return f"checkpoint:{run_id}:latest"


def effect_state_key(run_id: str, step: int, tool_call_id: str) -> str:
    """Return namespaced key for idempotent tool-effect state."""
    return f"effect:{run_id}:{step}:{tool_call_id}"


_ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    "pending": {"running", "cancelled", "failed"},
    "running": {"paused", "cancelling", "cancelled", "degraded", "failed", "completed"},
    "paused": {"running", "cancelling", "cancelled", "failed"},
    "cancelling": {"cancelled", "failed"},
    "cancelled": set(),
    "degraded": {"failed", "completed"},
    "failed": set(),
    "completed": set(),
}


def validate_state_transition(current: AgentState, target: AgentState) -> AgentState:
    """Validate and return a legal state transition target."""
    if current == target:
        return target
    allowed = _ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise AgentExecutionError(
            f"Invalid state transition: {current} -> {target}"
        )
    return target


@dataclass(slots=True)
class EffectJournal:
    """
    Idempotency journal for tool-side effects.
    """

    _rows: dict[tuple[str, int, str], dict[str, Any]] = field(default_factory=dict)

    def get(self, run_id: str, step: int, tool_call_id: str) -> dict[str, Any] | None:
        """Fetch a previously journaled tool-effect record."""
        return self._rows.get((run_id, step, tool_call_id))

    def put(
        self,
        run_id: str,
        step: int,
        tool_call_id: str,
        input_hash: str,
        output_hash: str,
        *,
        output: JSONValue | None,
        success: bool,
    ) -> None:
        """Store idempotency metadata for a side-effecting tool call."""
        self._rows[(run_id, step, tool_call_id)] = {
            "input_hash": input_hash,
            "output_hash": output_hash,
            "output": output,
            "success": success,
        }


@dataclass(slots=True)
class CircuitBreaker:
    """
    Minimal breaker for runtime dependencies.
    """

    config: FailSafeConfig
    failures: dict[str, list[float]] = field(default_factory=dict)

    def record_success(self, key: str) -> None:
        """Reset failure history for a dependency after success."""
        self.failures.pop(key, None)

    def record_failure(self, key: str) -> None:
        """Record a dependency failure within cooldown window."""
        now = time.time()
        rows = self.failures.setdefault(key, [])
        rows.append(now)
        cutoff = now - self.config.breaker_cooldown_s
        self.failures[key] = [ts for ts in rows if ts >= cutoff]

    def ensure_closed(self, key: str) -> None:
        """Raise when failure threshold is exceeded for dependency key."""
        rows = self.failures.get(key, [])
        if len(rows) >= self.config.breaker_failure_threshold:
            raise AgentCircuitOpenError(
                f"Circuit open for dependency '{key}' "
                f"(failures={len(rows)} threshold={self.config.breaker_failure_threshold})."
            )


def state_snapshot(
    *,
    state: AgentState,
    step: int,
    llm_calls: int,
    tool_calls: int,
    started_at_s: float,
    requested_model: str | None = None,
    normalized_model: str | None = None,
    provider_adapter: str | None = None,
    total_cost_usd: float | None = None,
    replayed_effect_count: int = 0,
) -> dict[str, JSONValue]:
    """Return normalized runtime snapshot payload for checkpoint persistence."""
    return {
        "state": state,
        "step": step,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "started_at_s": started_at_s,
        "elapsed_s": time.time() - started_at_s,
        "requested_model": requested_model,
        "normalized_model": normalized_model,
        "provider_adapter": provider_adapter,
        "total_cost_usd": total_cost_usd,
        "replayed_effect_count": replayed_effect_count,
    }
