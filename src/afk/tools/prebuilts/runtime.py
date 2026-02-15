from __future__ import annotations

"""
General runtime-safe filesystem tools.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.base import Tool
from ..core.decorator import tool
from ...agents.errors import SkillAccessError


class _ListDirectoryArgs(BaseModel):
    path: str = "."
    max_entries: int = Field(default=200, ge=1, le=5000)


class _ReadFileArgs(BaseModel):
    path: str = Field(min_length=1)
    max_chars: int = Field(default=20_000, ge=1, le=500_000)


def build_runtime_tools(*, root_dir: Path) -> list[Tool[Any, Any]]:
    root = root_dir.resolve()

    @tool(args_model=_ListDirectoryArgs, name="list_directory")
    async def list_directory(args: _ListDirectoryArgs) -> dict[str, Any]:
        target = (root / args.path).resolve()
        _ensure_inside(target, root)
        if not target.exists() or not target.is_dir():
            raise SkillAccessError(f"Directory not found: {args.path}")

        entries = []
        for row in sorted(target.iterdir()):
            entries.append(
                {
                    "name": row.name,
                    "path": str(row),
                    "is_dir": row.is_dir(),
                    "is_file": row.is_file(),
                }
            )
            if len(entries) >= args.max_entries:
                break
        return {"root": str(root), "path": str(target), "entries": entries}

    @tool(args_model=_ReadFileArgs, name="read_file")
    async def read_file(args: _ReadFileArgs) -> dict[str, Any]:
        target = (root / args.path).resolve()
        _ensure_inside(target, root)
        if not target.exists() or not target.is_file():
            raise SkillAccessError(f"File not found: {args.path}")
        text = target.read_text(encoding="utf-8")
        truncated = len(text) > args.max_chars
        if truncated:
            text = text[: args.max_chars]
        return {
            "root": str(root),
            "path": str(target),
            "content": text,
            "truncated": truncated,
        }

    return [list_directory, read_file]


def _ensure_inside(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as e:
        raise SkillAccessError(f"Path '{path}' escapes root '{root}'") from e

