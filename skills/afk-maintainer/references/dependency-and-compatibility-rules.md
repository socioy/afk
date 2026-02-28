# Dependency and Compatibility Rules

These rules govern how AFK manages its dependencies, ensures compatibility,
and mitigates supply chain risk.

---

## Versioning Policy

### Pinning Strategy

| Dependency Type | Strategy | Example |
|----------------|----------|---------|
| Core framework deps (FastAPI, Pydantic) | Bounded range with known-good floor | `fastapi>=0.129.0,<1.0` |
| LLM provider SDKs (openai, claude-agent-sdk) | Pin exact version | `openai==2.21.0` |
| Data/storage deps (redis, asyncpg) | Pin exact or bounded range | `redis==7.2.0` |
| Dev/test deps (pytest, ruff) | Bounded range | `pytest>=9.0.2` |

### Rules

- **Pin unstable dependencies** (pre-1.0, rapidly changing APIs) to exact versions for reproducibility.
- **Use bounded ranges** for stable semver libraries where minor/patch updates are safe.
- **Re-evaluate pins** on scheduled maintenance windows (not ad-hoc).
- **Never float to latest** for production-critical dependencies.
- **Lock file** (`uv.lock`) is committed and kept up to date.

---

## Compatibility Matrix

### Python Version

- **Baseline**: Python 3.13 (minimum supported version).
- **Target features**: Use Python 3.13+ features freely (type syntax, etc.).
- **CI validation**: Full test suite runs against Python 3.13.
- **Forward compat**: Test against Python 3.14 pre-releases when available.

### Provider SDK Compatibility

| Provider | SDK | Pinned Version | Notes |
|----------|-----|---------------|-------|
| OpenAI | `openai` | `2.21.0` | Responses API, streaming, structured output |
| Anthropic | `claude-agent-sdk` | `0.1.37` | Agent SDK with tool use, thinking |
| LiteLLM | `litellm` | `1.81.13` | Multi-provider routing, completion/chat |

### Validation on Dependency Update

When updating any dependency:

1. Run full test suite: `pytest`.
2. Run lint: `ruff check src tests`.
3. Verify no implicit API contract drift in wrappers/adapters.
4. Check for behavior changes in the dependency's changelog.
5. Test affected integration paths (LLM chat, streaming, tool calls, memory ops).

---

## Dependency Update Checklist

Before merging a dependency update PR:

- [ ] CHANGELOG entry documents the update and any user-visible impact.
- [ ] PR description mentions risk level (low/medium/high) and rollback strategy.
- [ ] No hidden API contract drift in AFK adapter/wrapper code.
- [ ] Test suite passes with no new warnings or failures.
- [ ] Lock file (`uv.lock`) is regenerated and committed.
- [ ] If updating an LLM provider SDK, adapter-specific regression tests pass:
  - Non-streaming chat
  - Streaming event normalization
  - Structured output parsing
  - Tool call extraction
  - Error wrapping/classification

---

## Security and Supply Chain

### Dependency Selection Criteria

- **Prefer official, actively maintained packages**. Check PyPI download counts, GitHub activity, and maintainer reputation.
- **Avoid unmaintained packages** for non-essential features. If the last release is >12 months old, evaluate alternatives.
- **Review transitive dependencies** for critical runtime paths. A vulnerability in a transitive dep of `litellm` affects AFK users.
- **Minimal dependency footprint**: Don't add a dependency for something that can be implemented in <50 lines.

### Audit Process

- Run `pip audit` (or equivalent) before releases to catch known vulnerabilities.
- Review transitive dependency tree for unexpected packages.
- If a critical vulnerability is found in a dependency:
  1. Assess impact on AFK users.
  2. If exploitable through AFK's usage of the dep, treat as P0.
  3. If not directly exploitable, pin to patched version in next release.

### Optional Dependencies

- Use `[extras]` groups for optional features that introduce heavy dependencies.
- Example: `pip install afk-py[a2a]` for JWT auth support (`PyJWT`).
- Core functionality must work without any optional dependencies installed.
- Always guard optional imports with try/except ImportError.

```python
# CORRECT: Guarded optional import
try:
    import jwt
except ImportError:
    jwt = None  # type: ignore[assignment]

class JWTAuthProvider:
    def __init__(self, ...):
        if jwt is None:
            raise ImportError(
                "JWTAuthProvider requires PyJWT. Install with: pip install afk-py[a2a]"
            )
```
