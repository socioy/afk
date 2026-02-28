## Summary

Describe what changed and why.

Fixes #

## Change Type

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor / internal improvement
- [ ] Documentation update
- [ ] Breaking change

## Scope

Which AFK subsystems are affected?

- [ ] Core Runner lifecycle
- [ ] LLM client/providers
- [ ] Tool registry / tool execution
- [ ] Background tools
- [ ] Debugger / observability
- [ ] Memory backend(s)
- [ ] Queues / worker contracts
- [ ] MCP integration
- [ ] Docs / examples

## Behavioral Notes

List any behavioral or API changes, including migration notes.

## Release Note (for `CHANGELOG.md`)

Add a changelog-ready note for `[Unreleased]` (Keep a Changelog):

- Section: `Added` | `Changed` | `Fixed` | `Deprecated` | `Removed` | `Security`
- Entry:

```markdown
- ...
```

If no changelog entry is needed, write: `No user-visible change`.

## Validation

Commands run locally:

```bash
uvx ruff check src tests
PYTHONPATH=src pytest -q
```

Additional targeted tests (if any):

```bash
# e.g. PYTHONPATH=src pytest -q tests/agents/test_background_tools.py
```

## Checklist

- [ ] I scoped this PR to one coherent concern.
- [ ] I added/updated tests for behavior changes.
- [ ] I updated docs/examples when user-facing behavior changed.
- [ ] I added a changelog-ready release note (or stated why not needed).
- [ ] I reviewed failure modes (timeouts, retries, background tasks, event loop interactions) where relevant.
- [ ] CI is expected to pass (`ruff` + tests).
