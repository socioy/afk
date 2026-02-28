# Maintainer Skill References

Reference documents for the AFK maintainer skill. These are loaded on demand
when the agent needs detailed guidance on a specific topic.

---

## Reference Documents

| # | File | Purpose |
|---|------|---------|
| 1 | `coding-principles-and-patterns.md` | Core coding philosophy, design patterns, anti-patterns |
| 2 | `maintainer-operating-rules.md` | PR standards, review protocol, red flags |
| 3 | `repo-design-and-quality-standards.md` | DX, docs, examples, extensibility, code style |
| 4 | `code-review-checklist.md` | Concrete per-PR-type checklists |
| 5 | `release-and-triage-playbook.md` | Issue triage, release hygiene, backport, emergency response |
| 6 | `dependency-and-compatibility-rules.md` | Versioning, compatibility matrix, supply chain security |
| 7 | `claude-agent-sdk-playbook.md` | Claude Agent SDK integration guidelines |
| 8 | `litellm-playbook.md` | LiteLLM transport/adapter guidelines |
| 9 | `examples.md` | Concrete examples of triage notes, PR comments, release notes |

---

## Bundled Docs Search

If the bundled docs index is present, search it:

```bash
python scripts/search_afk_docs.py "event loop run_sync"
python scripts/search_afk_docs.py --format json "tool middleware"
python scripts/search_afk_docs.py --top-k 5 "memory compaction"
```

### Index Files

| File | Content |
|------|---------|
| `afk-docs/docs-index.jsonl` | Line-delimited JSON, one doc per line |
| `afk-docs/inverted-index.json` | Term-to-document mapping for search |
| `afk-docs/id-to-path.json` | Document ID to file path mapping |
| `afk-docs/path-to-id.json` | File path to document ID mapping |
| `afk-docs/manifest.json` | Index metadata (doc count, generation time) |

---

## Quick Reference Links

- Documentation: https://afk.arpan.sh
- API Reference: https://afk.arpan.sh/library/api-reference
- Developer Guide: https://afk.arpan.sh/library/developer-guide
- Configuration Reference: https://afk.arpan.sh/library/configuration-reference
- Security Model: https://afk.arpan.sh/library/security-model
- Tested Behaviors: https://afk.arpan.sh/library/tested-behaviors
