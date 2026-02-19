"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Debugger-facing configuration types.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DebuggerConfig:
    """Configuration for debugger formatting and payload redaction."""

    enabled: bool = True
    verbosity: str = "detailed"
    include_content: bool = True
    redact_secrets: bool = True
    max_payload_chars: int = 4000
    emit_timestamps: bool = True
    emit_step_snapshots: bool = True
