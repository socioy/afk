from __future__ import annotations

"""
Prebuilt tools for skill discovery, reading, and constrained command execution.
"""

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.base import Tool
from ..core.decorator import tool
from ...agents.errors import SkillAccessError, SkillCommandDeniedError
from ...agents.types import SkillRef, SkillToolPolicy


class _EmptyArgs(BaseModel):
    pass


class _ReadSkillArgs(BaseModel):
    skill_name: str = Field(min_length=1)


class _ReadSkillFileArgs(BaseModel):
    skill_name: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    max_chars: int = Field(default=20_000, ge=1, le=500_000)


class _RunSkillCommandArgs(BaseModel):
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    timeout_s: float | None = Field(default=None, gt=0)


def build_skill_tools(
    *,
    skills: list[SkillRef],
    policy: SkillToolPolicy,
) -> list[Tool[Any, Any]]:
    """
    Build runtime-bound skill tools for one run.
    """
    skills_by_name = {skill.name: skill for skill in skills}
    roots = {skill.name: Path(skill.skill_md_path).parent.resolve() for skill in skills}

    @tool(args_model=_EmptyArgs, name="list_skills")
    async def list_skills(args: _EmptyArgs) -> dict[str, Any]:
        _ = args
        return {
            "skills": [
                {
                    "name": s.name,
                    "skill_md_path": s.skill_md_path,
                    "checksum": s.checksum,
                }
                for s in skills
            ]
        }

    @tool(args_model=_ReadSkillArgs, name="read_skill_md")
    async def read_skill_md(args: _ReadSkillArgs) -> dict[str, Any]:
        ref = skills_by_name.get(args.skill_name)
        if ref is None:
            raise SkillAccessError(f"Unknown skill '{args.skill_name}'")
        path = Path(ref.skill_md_path).resolve()
        _ensure_inside(path, roots[args.skill_name])
        text = path.read_text(encoding="utf-8")
        return {
            "skill_name": ref.name,
            "path": str(path),
            "checksum": ref.checksum,
            "content": text,
        }

    @tool(args_model=_ReadSkillFileArgs, name="read_skill_file")
    async def read_skill_file(args: _ReadSkillFileArgs) -> dict[str, Any]:
        root = roots.get(args.skill_name)
        if root is None:
            raise SkillAccessError(f"Unknown skill '{args.skill_name}'")
        target = (root / args.relative_path).resolve()
        _ensure_inside(target, root)
        if not target.exists() or not target.is_file():
            raise SkillAccessError(f"Skill file not found: {args.relative_path}")
        text = target.read_text(encoding="utf-8")
        if len(text) > args.max_chars:
            text = text[: args.max_chars]
        return {
            "skill_name": args.skill_name,
            "path": str(target),
            "content": text,
            "truncated": len(text) == args.max_chars,
        }

    @tool(args_model=_RunSkillCommandArgs, name="run_skill_command")
    async def run_skill_command(args: _RunSkillCommandArgs) -> dict[str, Any]:
        command = args.command.strip()
        if not command:
            raise SkillCommandDeniedError("Command cannot be empty")
        if not _is_command_allowed(command, policy.command_allowlist):
            raise SkillCommandDeniedError(
                f"Command '{command}' is not allowlisted for skill execution"
            )

        all_parts = [command, *args.args]
        if policy.deny_shell_operators:
            forbidden = {"&&", "||", ";", "|", "`", "$("}
            if any(part in forbidden for part in all_parts):
                raise SkillCommandDeniedError(
                    "Command denied due to shell operator restrictions"
                )

        cwd = None
        if args.cwd:
            maybe = Path(args.cwd).expanduser().resolve()
            if not any(_is_inside(maybe, root) for root in roots.values()):
                raise SkillAccessError("Command cwd must be inside one of the skill roots")
            cwd = str(maybe)

        timeout_s = args.timeout_s or policy.command_timeout_s
        proc = await asyncio.create_subprocess_exec(
            command,
            *args.args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise SkillCommandDeniedError(
                f"Command timed out after {timeout_s} seconds: {command}"
            )

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if len(stdout) > policy.max_stdout_chars:
            stdout = stdout[: policy.max_stdout_chars]
        if len(stderr) > policy.max_stderr_chars:
            stderr = stderr[: policy.max_stderr_chars]

        return {
            "command": [command, *args.args],
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    return [list_skills, read_skill_md, read_skill_file, run_skill_command]


def _is_command_allowed(command: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return False
    for allowed in allowlist:
        prefix = allowed.strip()
        if not prefix:
            continue
        if command == prefix:
            return True
        if command.startswith(prefix + "/"):
            return True
    return False


def _ensure_inside(path: Path, root: Path) -> None:
    if not _is_inside(path, root):
        raise SkillAccessError(f"Path '{path}' escapes skill root '{root}'")


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
