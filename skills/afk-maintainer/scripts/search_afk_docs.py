#!/usr/bin/env python3
"""Search bundled AFK docs index for the maintainer skill.

Usage:
    python scripts/search_afk_docs.py "query terms"
    python scripts/search_afk_docs.py --format json "circuit breaker"
    python scripts/search_afk_docs.py --top-k 5 "memory compaction"

Examples:
    scripts/search_afk_docs.py "event loop run_sync"
    scripts/search_afk_docs.py --format json "tool middleware"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score(query_terms: list[str], doc: dict) -> int:
    text = (
        doc.get("title", "")
        + " "
        + doc.get("description", "")
        + " "
        + doc.get("content", "")
    ).lower()
    return sum(text.count(term) for term in query_terms)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search bundled AFK docs index for the maintainer skill.",
        epilog="Examples:\n"
        '  %(prog)s "event loop run_sync"\n'
        '  %(prog)s --format json "tool middleware"\n'
        '  %(prog)s --top-k 5 "memory compaction"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Search query (space-separated terms)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of results to return (default: 8)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (human-readable) or json (structured)",
    )
    parser.add_argument(
        "--index",
        default=str(
            Path(__file__).resolve().parent.parent
            / "references"
            / "afk-docs"
            / "docs-index.jsonl"
        ),
        help="Path to docs-index.jsonl",
    )
    args = parser.parse_args()

    query_terms = [term.lower() for term in args.query.split() if term.strip()]
    if not query_terms:
        print("Error: Empty query. Provide one or more search terms.", file=sys.stderr)
        return 1

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"Error: Index not found: {idx_path}", file=sys.stderr)
        print("Run the index builder first or check the --index path.", file=sys.stderr)
        return 1

    rows = load_records(idx_path)
    ranked = sorted(
        ((score(query_terms, row), row) for row in rows),
        key=lambda item: item[0],
        reverse=True,
    )

    results = []
    for score_value, row in ranked:
        if score_value <= 0:
            continue
        results.append(
            {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "path": row.get("path", ""),
                "description": row.get("description", ""),
                "score": score_value,
            }
        )
        if len(results) >= args.top_k:
            break

    if not results:
        if args.format == "json":
            print(json.dumps({"query": args.query, "results": [], "count": 0}))
        else:
            print("No matches.")
        return 0

    if args.format == "json":
        print(
            json.dumps(
                {"query": args.query, "results": results, "count": len(results)},
                ensure_ascii=False,
            )
        )
    else:
        for row in results:
            print(f"[{row['id']}] {row['title']}")
            print(f"  url: {row['url']}")
            print(f"  path: {row['path']}")
            if row["description"]:
                print(f"  desc: {row['description']}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
