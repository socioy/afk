#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="${AFK_DOCS_DIR:-$ROOT_DIR/docs}"
SKILLS_DIR="${AFK_AGENT_SKILLS_DIR:-$ROOT_DIR/agent-skill}"
OUT_DIR="${AFK_AI_INDEX_DIR:-$ROOT_DIR/ai-index}"
BUNDLE_SKILL_DOCS=true
TARGET_SKILL="${AFK_TARGET_SKILL:-}"

usage() {
  cat <<'EOF'
Build AFK AI assets:
1) creates searchable index files from docs/
2) creates/refreshes skill metadata index in agent-skill/
3) bundles docs assets into each skill folder (optional)

Usage:
  scripts/build_agentic_ai_assets.sh [--no-bundle-skill-docs] [--skill <skill-slug>]

Environment overrides:
  AFK_DOCS_DIR
  AFK_AGENT_SKILLS_DIR
  AFK_AI_INDEX_DIR
  AFK_TARGET_SKILL
EOF
}

while (($#)); do
  case "$1" in
    --no-bundle-skill-docs)
      BUNDLE_SKILL_DOCS=false
      ;;
    --skill)
      if (($# < 2)); then
        echo "--skill requires a value" >&2
        usage
        exit 1
      fi
      TARGET_SKILL="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ ! -d "$DOCS_DIR" ]]; then
  echo "Docs directory not found: $DOCS_DIR" >&2
  exit 1
fi

if [[ ! -d "$SKILLS_DIR" ]]; then
  echo "Skills directory not found: $SKILLS_DIR" >&2
  echo "Create skill folders first under: $SKILLS_DIR" >&2
  exit 1
fi

if [[ -n "$TARGET_SKILL" ]] && [[ ! -f "$SKILLS_DIR/$TARGET_SKILL/SKILL.md" ]]; then
  echo "Target skill not found: $SKILLS_DIR/$TARGET_SKILL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/text" "$OUT_DIR/records"

python3 - "$DOCS_DIR" "$OUT_DIR" "$SKILLS_DIR" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

docs_dir = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
skills_dir = Path(sys.argv[3]).resolve()
snippets_dir = docs_dir / "library" / "snippets"

stopwords = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "if", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "to", "with", "this", "these", "those", "you", "your", "we", "our",
}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}, text
    raw = parts[1]
    body = parts[2]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def to_doc_url(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/")
    if rel == "index.mdx":
        return "/"
    if rel.endswith("/index.mdx"):
        return "/" + rel[:-10] + "/"
    if rel.endswith(".mdx"):
        return "/" + rel[:-4]
    if rel.endswith(".md"):
        return "/" + rel[:-3]
    if rel.endswith(".json"):
        return "/" + rel
    return "/" + rel


def plain_text(content: str) -> str:
    content = re.sub(r"```[\s\S]*?```", " ", content)
    content = re.sub(r"`([^`]*)`", r"\1", content)
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"^#{1,6}\s*", "", content, flags=re.M)
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def tokenize(content: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,40}", content.lower())
    return [w for w in words if w not in stopwords]


def read_skill_meta(skill_md_path: Path) -> tuple[str, str]:
    text = skill_md_path.read_text(encoding="utf-8")
    meta, _ = parse_frontmatter(text)
    name = meta.get("name", skill_md_path.parent.name)
    description = meta.get("description", "")
    return name, description


def parse_snippet(path: Path) -> tuple[str, str]:
    """
    Parse snippet files formatted as:
    ````python file_name.py
    <code>
    ````
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return path.stem, ""

    first = lines[0].strip()
    file_name = path.stem
    if first.startswith("````"):
        header = first[4:].strip()
        if header:
            parts = header.split(maxsplit=1)
            if len(parts) == 2:
                file_name = parts[1].strip()

    end_idx = None
    for idx in range(len(lines) - 1, 0, -1):
        if lines[idx].strip() == "````":
            end_idx = idx
            break

    if end_idx is None:
        body = "\n".join(lines)
    else:
        body = "\n".join(lines[1:end_idx])

    return file_name, body.strip("\n")


def compact_filename(rel_path: str) -> str:
    base = rel_path.replace("\\", "/")
    if base.endswith(".mdx"):
        base = base[:-4]
    base = re.sub(r"[^a-zA-Z0-9/_-]", "-", base).strip("/")
    if not base:
        base = "index"
    return base.replace("/", "__") + ".md"


def infer_description(text: str, fallback_topic: str) -> str:
    plain = plain_text(text).strip()
    if not plain:
        return f"Compact reference for {fallback_topic}."
    sentence = plain.split(". ", 1)[0].strip()
    sentence = sentence.strip("`*_>- ")
    if len(re.findall(r"[A-Za-z]", sentence)) < 3:
        return f"Compact reference for {fallback_topic}."
    if len(sentence) > 180:
        sentence = sentence[:177].rstrip() + "..."
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def truncate_fenced_code_blocks(content: str, source_rel_path: str, max_lines: int = 40) -> str:
    lines = content.splitlines()
    output: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        match = re.match(r"^(\s*)(`{3,})(.*)$", line)
        if not match:
            output.append(line)
            idx += 1
            continue

        indent = match.group(1)
        fence = match.group(2)
        opening = line
        block = [line]
        idx += 1
        closed = False

        while idx < len(lines):
            block.append(lines[idx])
            if re.match(rf"^{re.escape(indent)}{re.escape(fence)}\s*$", lines[idx]):
                closed = True
                idx += 1
                break
            idx += 1

        if not closed:
            output.extend(block)
            continue

        code_lines = block[1:-1]
        if len(code_lines) <= max_lines:
            output.extend(block)
            continue

        output.append(opening)
        output.extend(code_lines[:max_lines])
        output.append(f"{indent}{fence}")
        output.append("")
        output.append(
            f"> Code block truncated to {max_lines} lines. Source: `docs/{source_rel_path}`"
        )

    return "\n".join(output)


def normalize_mdx_body(body: str, source_rel_path: str) -> str:
    content = body.replace("\r\n", "\n")
    content = re.sub(r"^\s*(import|export)\s+.+$", "", content, flags=re.M)
    content = re.sub(r"<[A-Za-z][^>\n]*/>", "", content)
    content = re.sub(r"</?[^>]+>", "", content)
    content = truncate_fenced_code_blocks(content, source_rel_path, max_lines=40)
    content = re.sub(r"[ \t]+\n", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


doc_candidates = sorted(
    [
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix in {".mdx", ".md", ".json"}
    ]
)

documents = []
inverted: dict[str, set[str]] = defaultdict(set)
path_to_id: dict[str, str] = {}
id_to_path: dict[str, str] = {}

for path in doc_candidates:
    rel = path.relative_to(docs_dir).as_posix()
    if rel.startswith(("logo", "favicon")) and path.suffix == ".svg":
        continue

    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    title = meta.get("title") or path.stem
    description = meta.get("description", "")
    headings = re.findall(r"^#{1,6}\s+(.+)$", body, flags=re.M)
    text = plain_text(body if path.suffix != ".json" else raw)
    doc_id = "doc_" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    url = to_doc_url(rel)
    tokens = tokenize(text)

    for token in set(tokens):
        inverted[token].add(doc_id)

    payload = {
        "id": doc_id,
        "path": f"docs/{rel}",
        "url": url,
        "title": title,
        "description": description,
        "headings": headings,
        "content_sha256": sha,
        "content": text,
        "token_count": len(tokens),
    }
    documents.append(payload)
    path_to_id[payload["path"]] = doc_id
    id_to_path[doc_id] = payload["path"]

    (out_dir / "text" / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "records" / f"{doc_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

documents.sort(key=lambda x: x["path"])

index_payload = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "docs_root": str(docs_dir),
    "document_count": len(documents),
    "documents": documents,
}

(out_dir / "docs-index.json").write_text(
    json.dumps(index_payload, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

with (out_dir / "docs-index.jsonl").open("w", encoding="utf-8") as f:
    for doc in documents:
        f.write(json.dumps(doc, ensure_ascii=True) + "\n")

inverted_payload = {
    token: sorted(ids)
    for token, ids in sorted(inverted.items(), key=lambda item: item[0])
}
(out_dir / "inverted-index.json").write_text(
    json.dumps(inverted_payload, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

(out_dir / "path-to-id.json").write_text(
    json.dumps(path_to_id, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

(out_dir / "id-to-path.json").write_text(
    json.dumps(id_to_path, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

compact_dir = out_dir / "compact-mdx"
compact_dir.mkdir(parents=True, exist_ok=True)
for stale in compact_dir.glob("*.md"):
    stale.unlink()

mdx_candidates = sorted(p for p in docs_dir.rglob("*.mdx") if p.is_file())
compact_entries: list[dict[str, str]] = []

for mdx_path in mdx_candidates:
    rel = mdx_path.relative_to(docs_dir).as_posix()
    raw = mdx_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    topic = (meta.get("title") or mdx_path.stem).strip()
    normalized_body = normalize_mdx_body(body, rel)
    description = meta.get("description", "").strip() or infer_description(body, topic)
    file_name = compact_filename(rel)
    compact_lines = [f"# {topic}", "", description, "", f"Source: `docs/{rel}`", ""]
    if normalized_body:
        compact_lines.append(normalized_body)
    compact_text = "\n".join(compact_lines).strip() + "\n"
    content_sha = hashlib.sha256(compact_text.encode("utf-8")).hexdigest()
    (compact_dir / file_name).write_text(compact_text, encoding="utf-8")
    compact_entries.append(
        {
            "topic": topic,
            "description": description,
            "compact_filename": file_name,
            "source_mdx_path": f"docs/{rel}",
            "url": to_doc_url(rel),
            "content_sha256": content_sha,
        }
    )

compact_entries.sort(key=lambda entry: entry["source_mdx_path"])

(out_dir / "compact-mdx-index.json").write_text(
    json.dumps(
        {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "docs_root": str(docs_dir),
            "entries": compact_entries,
        },
        indent=2,
        ensure_ascii=True,
    )
    + "\n",
    encoding="utf-8",
)

skill_entries = []
for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
    name, description = read_skill_meta(skill_md)
    rel_path = skill_md.relative_to(skills_dir).as_posix()
    skill_id = "skill_" + hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
    skill_entries.append(
        {
            "id": skill_id,
            "name": name,
            "description": description,
            "path": f"agent-skill/{rel_path}",
            "github_url": f"https://github.com/socioy/afk/tree/main/agent-skill/{skill_md.parent.name}",
        }
    )

(skills_dir / "index.json").write_text(
    json.dumps(
        {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "skills": skill_entries,
        },
        indent=2,
        ensure_ascii=True,
    )
    + "\n",
    encoding="utf-8",
)

manifest = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "document_count": len(documents),
    "skill_count": len(skill_entries),
    "files": [
        "docs-index.json",
        "docs-index.jsonl",
        "inverted-index.json",
        "path-to-id.json",
        "id-to-path.json",
        "examples.md",
        "records/",
        "text/",
        "../agent-skill/index.json",
    ],
}
(out_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

snippet_paths = sorted(snippets_dir.glob("*.mdx"))
examples_lines = [
    "# AFK Examples",
    "",
    "Merged from `docs/library/snippets/*.mdx` for agent-friendly context loading.",
    "",
    f"Generated at: {dt.datetime.now(dt.timezone.utc).isoformat()}",
    "",
]

for snippet_path in snippet_paths:
    file_name, code = parse_snippet(snippet_path)
    rel = snippet_path.relative_to(docs_dir).as_posix()
    examples_lines.append(f"## {file_name}")
    examples_lines.append("")
    examples_lines.append(f"Source: `docs/{rel}`")
    examples_lines.append("")
    examples_lines.append("```python")
    examples_lines.append(code)
    examples_lines.append("```")
    examples_lines.append("")

examples_md = "\n".join(examples_lines).strip() + "\n"
(out_dir / "examples.md").write_text(examples_md, encoding="utf-8")
(skills_dir / "examples.md").write_text(examples_md, encoding="utf-8")

print(f"Indexed {len(documents)} docs files into: {out_dir}")
print(f"Indexed {len(skill_entries)} skills into: {skills_dir / 'index.json'}")
print(f"Merged {len(snippet_paths)} snippets into: {out_dir / 'examples.md'}")
print(f"Generated {len(compact_entries)} compact markdown docs into: {compact_dir}")
PY

to_title_case() {
  # shellcheck disable=SC2001
  echo "$1" | sed 's/-/ /g' | awk '{for (i=1;i<=NF;i++){$i=toupper(substr($i,1,1)) tolower(substr($i,2))} print}'
}

write_search_script() {
  local skill_dir="$1"
  local script_path="$skill_dir/scripts/search_afk_docs.py"
  cat > "$script_path" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    text = (doc.get("title", "") + " " + doc.get("description", "") + " " + doc.get("content", "")).lower()
    return sum(text.count(term) for term in query_terms)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search bundled AFK docs index for this skill.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--index",
        default=str(Path(__file__).resolve().parent.parent / "references" / "afk-docs" / "docs-index.jsonl"),
        help="Path to docs-index.jsonl",
    )
    args = parser.parse_args()

    query_terms = [t.lower() for t in args.query.split() if t.strip()]
    if not query_terms:
        print("Empty query.")
        return 1

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"Index not found: {idx_path}")
        return 1

    rows = load_records(idx_path)
    ranked = sorted(
        ((score(query_terms, r), r) for r in rows),
        key=lambda x: x[0],
        reverse=True,
    )
    shown = 0
    for s, row in ranked:
        if s <= 0:
            continue
        print(f"[{row['id']}] {row.get('title', '')}")
        print(f"  url: {row.get('url', '')}")
        print(f"  path: {row.get('path', '')}")
        desc = row.get("description", "")
        if desc:
            print(f"  desc: {desc}")
        print()
        shown += 1
        if shown >= args.top_k:
            break

    if shown == 0:
        print("No matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
  chmod +x "$script_path"
}

write_openai_yaml() {
  local skill_dir="$1"
  local skill_slug="$2"
  local title
  title="$(to_title_case "$skill_slug")"
  cat > "$skill_dir/agents/openai.yaml" <<EOF
interface:
  display_name: "${title}"
  short_description: "AFK skill for ${title} workflows."
  default_prompt: "Use \$${skill_slug} to complete this AFK task with the packaged guidance."
EOF
}

upsert_skill_indexed_references_section() {
  local skill_md_path="$1"
  python3 - "$skill_md_path" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

skill_md = Path(sys.argv[1]).resolve()
text = skill_md.read_text(encoding="utf-8")
section_body = """## Indexed References

- `references/index.json`: generated manifest of indexed references for this skill.
- `references/compact/*.md`: compact markdown generated from `docs/**/*.mdx`.
- `scripts/search_afk_docs.py "your query"`: quick lookup across bundled docs index.
"""

pattern = re.compile(r"\n## Indexed References\n(?:.*?)(?=\n## |\Z)", flags=re.S)
if pattern.search(text):
    updated = pattern.sub("\n" + section_body.strip() + "\n", text)
else:
    updated = text.rstrip() + "\n\n" + section_body.strip() + "\n"

if updated != text:
    skill_md.write_text(updated, encoding="utf-8")
PY
}

write_skill_references_index() {
  local skill_dir="$1"
  local skill_slug="$2"
  local compact_index_path="$3"
  python3 - "$skill_dir" "$skill_slug" "$compact_index_path" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

skill_dir = Path(sys.argv[1]).resolve()
skill_slug = sys.argv[2]
compact_index_path = Path(sys.argv[3]).resolve()
references_dir = skill_dir / "references"
compact_dir = references_dir / "compact"

compact_lookup: dict[str, dict[str, str | None]] = {}
if compact_index_path.exists():
    payload = json.loads(compact_index_path.read_text(encoding="utf-8"))
    for entry in payload.get("entries", []):
        file_name = entry.get("compact_filename")
        if isinstance(file_name, str) and file_name:
            compact_lookup[file_name] = entry


def heading_and_description(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    topic = path.stem.replace("-", " ").replace("_", " ").strip()
    topic = " ".join(part.capitalize() for part in topic.split()) or path.stem
    description = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                topic = heading
            continue
        if line.startswith("Source:"):
            continue
        description = line
        break
    if not description:
        description = f"Reference document from {path.name}."
    return topic, description


entries: list[dict[str, str | None]] = []
for ref_path in sorted(references_dir.glob("*.md")):
    if ref_path.name in {"README.md", "examples.md"}:
        continue
    topic, description = heading_and_description(ref_path)
    content_sha = hashlib.sha256(ref_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    entries.append(
        {
            "topic": topic,
            "description": description,
            "filepath": f"references/{ref_path.name}",
            "source_mdx_path": None,
            "url": None,
            "content_sha256": content_sha,
        }
    )

for ref_path in sorted(compact_dir.glob("*.md")):
    metadata = compact_lookup.get(ref_path.name, {})
    fallback_topic, fallback_description = heading_and_description(ref_path)
    topic = str(metadata.get("topic") or fallback_topic).strip() or fallback_topic
    description = str(metadata.get("description") or fallback_description).strip() or fallback_description
    content_sha = hashlib.sha256(ref_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    entries.append(
        {
            "topic": topic,
            "description": description,
            "filepath": f"references/compact/{ref_path.name}",
            "source_mdx_path": metadata.get("source_mdx_path"),
            "url": metadata.get("url"),
            "content_sha256": content_sha,
        }
    )

index_payload = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "skill": skill_slug,
    "entries": entries,
}
(references_dir / "index.json").write_text(
    json.dumps(index_payload, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)
PY
}

bundle_docs_into_skills() {
  for skill_dir in "$SKILLS_DIR"/*; do
    [[ -d "$skill_dir" ]] || continue
    [[ -f "$skill_dir/SKILL.md" ]] || continue

    local skill_slug
    skill_slug="$(basename "$skill_dir")"
    if [[ -n "$TARGET_SKILL" ]] && [[ "$skill_slug" != "$TARGET_SKILL" ]]; then
      continue
    fi

    local refs_dir="$skill_dir/references/afk-docs"
    local refs_root_dir="$skill_dir/references"
    local compact_refs_dir="$refs_root_dir/compact"
    mkdir -p "$refs_dir" "$refs_root_dir" "$compact_refs_dir" "$skill_dir/scripts" "$skill_dir/agents"

    cp "$OUT_DIR/docs-index.json" "$refs_dir/docs-index.json"
    cp "$OUT_DIR/docs-index.jsonl" "$refs_dir/docs-index.jsonl"
    cp "$OUT_DIR/inverted-index.json" "$refs_dir/inverted-index.json"
    cp "$OUT_DIR/id-to-path.json" "$refs_dir/id-to-path.json"
    cp "$OUT_DIR/path-to-id.json" "$refs_dir/path-to-id.json"
    cp "$OUT_DIR/manifest.json" "$refs_dir/manifest.json"
    cp "$OUT_DIR/examples.md" "$refs_root_dir/examples.md"

    if compgen -G "$compact_refs_dir/*.md" > /dev/null; then
      rm -f "$compact_refs_dir"/*.md
    fi
    cp "$OUT_DIR"/compact-mdx/*.md "$compact_refs_dir"/

    cat > "$refs_root_dir/README.md" <<EOF
# Bundled AFK Docs Index

This folder is generated by:

\`\`\`bash
./scripts/build_agentic_ai_assets.sh
\`\`\`

It keeps machine-readable AFK docs inside this skill package for agent runtimes that expect docs to live with \`SKILL.md\`.

Files:
- \`afk-docs/docs-index.jsonl\`: searchable doc records
- \`afk-docs/inverted-index.json\`: token -> doc id map
- \`afk-docs/id-to-path.json\`: doc id -> docs path
- \`examples.md\`: merged runnable examples from snippets
- \`compact/*.md\`: compact markdown generated from docs \`.mdx\` files
- \`index.json\`: indexed references manifest (\`topic\`, \`description\`, \`filepath\`, source metadata)

Quick search:

\`\`\`bash
python scripts/search_afk_docs.py "system prompts"
\`\`\`
EOF

    write_search_script "$skill_dir"
    write_skill_references_index "$skill_dir" "$skill_slug" "$OUT_DIR/compact-mdx-index.json"
    upsert_skill_indexed_references_section "$skill_dir/SKILL.md"

    if [[ ! -f "$skill_dir/agents/openai.yaml" ]]; then
      write_openai_yaml "$skill_dir" "$skill_slug"
    fi
  done
}

if $BUNDLE_SKILL_DOCS; then
  bundle_docs_into_skills
  if [[ -n "$TARGET_SKILL" ]]; then
    echo "Bundled docs index into skill folder: $SKILLS_DIR/$TARGET_SKILL"
  else
    echo "Bundled docs index into skill folders under: $SKILLS_DIR"
  fi
fi

echo "Done."
if [[ -n "$TARGET_SKILL" ]]; then
  echo "Target skill: $TARGET_SKILL"
fi
echo "Docs index: $OUT_DIR/docs-index.json"
echo "Fast search file set: $OUT_DIR/inverted-index.json and $OUT_DIR/text/*.txt"
echo "Merged examples: $OUT_DIR/examples.md"
echo "Compact docs metadata: $OUT_DIR/compact-mdx-index.json"
