# Contributing to AFK

Thanks for contributing to AFK (Agent Forge Kit).

## Development Status

> **Note:** AFK is in **fast-paced development mode**.
> Internals and public APIs are still evolving. Expect frequent changes and keep PRs focused.

## Prerequisites

- Python `3.13`
- `pip` (or `uv`, optional)
- Git

## Local Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

## Run Tests

Use the same command as CI:

```bash
PYTHONPATH=src pytest -q
```

Run a specific test file:

```bash
PYTHONPATH=src pytest -q tests/agents/test_agent_runtime.py
```

## Docs Workflow

- Docs live under `docs/`
- Mintlify config is `docs/docs.json`
- Main landing page is `docs/index.mdx`

Local docs preview:

```bash
cd docs
bunx mintlify dev
```

## Contribution Guidelines

- Use public imports (`afk.*`) in examples and docs.
- Keep changes scoped to one concern when possible.
- Update docs for behavior changes, especially runtime, tools, and policy semantics.
- Add or update tests for bug fixes and behavior changes.
- Avoid destructive git operations in shared branches.

## Pull Request Checklist

- Code builds and tests pass locally.
- New behavior is covered by tests.
- Relevant docs are updated.
- PR description explains:
  - what changed
  - why it changed
  - any migration impact

## Reporting Issues

When filing an issue, include:

- environment (OS, Python version)
- minimal reproducible example
- expected behavior
- actual behavior
- logs or traceback

## Security

If you discover a security issue, please report it privately to maintainers first rather than posting full details publicly.

## Maintainer Contact

- GitHub: `arpan404@github` (handle: `@arpan404`)
- LinkedIn: `arpanbhandari`
- Email: `contact@arpan.sh`
- Docs: `https://afk.arpan.sh`
