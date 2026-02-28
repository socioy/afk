"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Dataset loaders for eval case definitions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..agents import BaseAgent
from ..agents.types import JSONValue
from .models import EvalCase


def load_eval_cases_json(
    path: str | Path,
    *,
    agent_resolver: Callable[[str], BaseAgent],
) -> list[EvalCase]:
    """Load eval cases from JSON list and resolve agent references."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Eval dataset JSON must be a list")

    out: list[EvalCase] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Each eval dataset row must be an object")

        name = row.get("name")
        agent_name = row.get("agent")
        if not isinstance(name, str) or not name:
            raise ValueError("Eval dataset row missing non-empty 'name'")
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError("Eval dataset row missing non-empty 'agent'")

        user_message = row.get("user_message")
        context = row.get("context")
        thread_id = row.get("thread_id")
        tags = row.get("tags")

        out.append(
            EvalCase(
                name=name,
                agent=agent_resolver(agent_name),
                user_message=user_message if isinstance(user_message, str) else None,
                context=_as_json_obj(context),
                thread_id=thread_id if isinstance(thread_id, str) else None,
                tags=_as_tags(tags),
            )
        )
    return out


def _as_json_obj(value: Any) -> dict[str, JSONValue]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Eval dataset field 'context' must be an object")
    out: dict[str, JSONValue] = {}
    for key, item in value.items():
        out[str(key)] = _json_cast(item)
    return out


def _as_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Eval dataset field 'tags' must be a list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Eval dataset tags must be strings")
        out.append(item)
    return tuple(out)


def _json_cast(value: Any) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_cast(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_cast(v) for k, v in value.items()}
    return str(value)
