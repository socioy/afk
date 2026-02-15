"""
Example 08: Prebuilt runtime filesystem tools.

Run:
    uv run python docs/library/examples/08_prebuilt_runtime_tools.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from afk.tools import ToolRegistry, build_runtime_tools


async def main() -> None:
    root = (Path.cwd() / "docs" / "library").resolve()
    runtime_tools = build_runtime_tools(root_dir=root)

    registry = ToolRegistry()
    registry.register_many(runtime_tools)

    list_result = await registry.call(
        "list_directory",
        {
            "path": ".",
            "max_entries": 10,
        },
    )

    read_result = await registry.call(
        "read_file",
        {
            "path": "index.md",
            "max_chars": 500,
        },
    )

    print("root:", root)
    print("list_success:", list_result.success)
    print("read_success:", read_result.success)
    print("read_truncated:", (read_result.output or {}).get("truncated"))


if __name__ == "__main__":
    asyncio.run(main())
