#!/usr/bin/env python3
"""
Search AFK coder skill reference docs.

Usage:
    python search_afk_docs.py "query"
    python search_afk_docs.py "query" --format json
    python search_afk_docs.py --help
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REFERENCES_DIR = Path(__file__).resolve().parent.parent / "references"
SKILL_FILE = Path(__file__).resolve().parent.parent / "SKILL.md"
LLMS_TXT = Path(__file__).resolve().parent.parent / "llms.txt"


def find_matches(
    query: str, files: list[Path], *, context_lines: int = 2
) -> list[dict]:
    """Search files for query, returning matches with context."""
    results: list[dict] = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    for filepath in files:
        if not filepath.exists():
            continue
        try:
            lines = filepath.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            if pattern.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context = "\n".join(lines[start:end])
                results.append(
                    {
                        "file": str(filepath.relative_to(filepath.parent.parent)),
                        "line": i + 1,
                        "match": line.strip(),
                        "context": context,
                    }
                )
    return results


def collect_files() -> list[Path]:
    """Collect all searchable files."""
    files: list[Path] = []

    if REFERENCES_DIR.exists():
        files.extend(sorted(REFERENCES_DIR.glob("*.md")))

    if SKILL_FILE.exists():
        files.append(SKILL_FILE)
    if LLMS_TXT.exists():
        files.append(LLMS_TXT)

    return files


def format_text(results: list[dict], query: str) -> str:
    """Format results as human-readable text."""
    if not results:
        return f"No matches found for '{query}'."

    parts: list[str] = [f"Found {len(results)} match(es) for '{query}':\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"--- Match {i}: {r['file']}:{r['line']} ---")
        parts.append(r["context"])
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search AFK coder skill reference documentation.",
        epilog="Examples:\n"
        '  python search_afk_docs.py "RunnerConfig"\n'
        '  python search_afk_docs.py "@tool" --format json\n'
        '  python search_afk_docs.py "MemoryStore" --context 5\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Search query string")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Number of context lines around each match (default: 2)",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List all searchable files and exit",
    )

    args = parser.parse_args()

    if args.list_files:
        files = collect_files()
        for f in files:
            print(f.relative_to(f.parent.parent))
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    files = collect_files()
    results = find_matches(args.query, files, context_lines=args.context)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(format_text(results, args.query))


if __name__ == "__main__":
    main()
