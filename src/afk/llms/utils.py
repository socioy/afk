from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

Utility functions for LLM interactions, including JSON extraction and backoff strategies.
"""
import asyncio
import json
import random
from typing import Any, Dict, Optional


def clamp_str(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"


def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _strip_fenced_code_block(text: str) -> str:
    """
    If `text` starts with a fenced code block (``` or ~~~), return the content inside.
    Otherwise return the original text stripped.
    """
    t = (text or "").strip()
    if not t:
        return t

    def _is_fence_line(line: str) -> Optional[str]:
        ls = line.lstrip()
        if ls.startswith("```"):
            return "```"
        if ls.startswith("~~~"):
            return "~~~"
        return None

    lines = t.splitlines()
    if not lines:
        return t

    fence = _is_fence_line(lines[0])
    if not fence:
        return t

    # Remove opening fence line (may include language tag, e.g. ```json)
    body_lines = lines[1:]

    # Find closing fence line
    close_idx = None
    for i, line in enumerate(body_lines):
        if line.lstrip().startswith(fence):
            close_idx = i
            break

    if close_idx is None:
        # No closing fence found; best-effort: treat remaining as body
        inner = "\n".join(body_lines).strip()
    else:
        inner = "\n".join(body_lines[:close_idx]).strip()

    # Sometimes people put a bare language tag on the first line inside the fence
    # (e.g., "json" or "javascript"). Strip it.
    inner_lines = inner.splitlines()
    if inner_lines:
        first = inner_lines[0].strip().lower()
        if first in ("json", "javascript", "js", "typescript", "ts"):
            inner = "\n".join(inner_lines[1:]).strip()

    return inner


def extract_json_object(text: str) -> Optional[str]:
    """
    Best-effort extraction of the first JSON object/array from a larger text blob.

    Handles:
      - Markdown fenced blocks (``` / ~~~) with optional language tags
      - Nested braces/brackets using a stack
      - Braces/brackets inside quoted strings (with escape handling)
      - Optional single-quote string handling for brace matching (LLM "JSON-ish" outputs)

    Returns:
      - The substring containing the first complete JSON object/array, or None.
    """
    if not text:
        return None

    t = _strip_fenced_code_block(text)

    if not t:
        return None

    # Find earliest start of a JSON container: '{' or '['
    obj_start = t.find("{")
    arr_start = t.find("[")

    if obj_start == -1 and arr_start == -1:
        return None

    if obj_start == -1:
        start = arr_start
    elif arr_start == -1:
        start = obj_start
    else:
        start = min(obj_start, arr_start)

    opener = t[start]
    if opener not in "{[":
        return None

    # Stack-based balanced matcher with string/escape handling.
    stack = [opener]
    in_string = False
    quote_char: Optional[str] = None
    escape = False

    # Best-effort support for single-quoted "strings" for matching only
    # (not valid JSON, but helps us avoid counting braces inside them).
    allow_single_quotes_for_matching = True

    for i in range(start + 1, len(t)):
        ch = t[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if quote_char is not None and ch == quote_char:
                in_string = False
                quote_char = None
            continue

        # Not inside a string
        if ch == '"':
            in_string = True
            quote_char = '"'
            continue

        if allow_single_quotes_for_matching and ch == "'":
            in_string = True
            quote_char = "'"
            continue

        if ch == "{":
            stack.append("{")
            continue
        if ch == "[":
            stack.append("[")
            continue

        if ch == "}" or ch == "]":
            if not stack:
                return None

            top = stack[-1]
            if (ch == "}" and top == "{") or (ch == "]" and top == "["):
                stack.pop()
                if not stack:
                    return t[start : i + 1]
                continue

            # Mismatched closer. Best-effort behavior:
            # Ignore it and keep scanning rather than failing hard.
            continue

    return None


def backoff_delay(attempt: int, base_s: float, jitter_s: float) -> float:
    """
    Exponential backoff with jitter.
    attempt=0 => base, attempt=1 => 2*base, etc.
    """
    exp = base_s * (2 ** attempt)
    jitter = random.uniform(0.0, jitter_s)
    return exp + jitter


def run_sync(coro):
    """
    Run an async coroutine from sync context.
    If already inside a running event loop, raise a clear error.
    """
    try:
        # If this succeeds, we're inside a running event loop context.
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread => safe to use asyncio.run
        return asyncio.run(coro)

    # If we got here, we are in a running loop (can't nest asyncio.run).
    raise RuntimeError(
        "Cannot use *_sync methods inside a running event loop. Use async methods instead."
    )