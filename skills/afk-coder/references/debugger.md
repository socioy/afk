# Debugger

AFK debugger module: Debug instrumentation, payload redaction, and run event enrichment.

- Doc page: https://afk.arpan.sh/library/debugger
- Source: `src/afk/debugger/`
- Cross-refs: `agents-and-runner.md`

---

## Overview

The `Debugger` module provides debug instrumentation for agent runs, including:
- Debug metadata enrichment and formatting hooks
- Payload redaction for sensitive data
- Verbosity levels for different detail amounts

Key public imports:

```python
from afk.debugger import Debugger, DebuggerConfig
```

---

## DebuggerConfig

Configuration for debugger formatting and payload redaction.

```python
from afk.debugger import DebuggerConfig

config = DebuggerConfig(
    enabled=True,
    verbosity="detailed",
    include_content=True,
    redact_secrets=True,
    max_payload_chars=4000,
    emit_timestamps=True,
    emit_step_snapshots=True,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable debug metadata enrichment |
| `verbosity` | `str` | `"detailed"` | Detail level: `"basic"`, `"detailed"`, or `"trace"` |
| `include_content` | `bool` | `True` | Include event/tool/message content in debug payloads |
| `redact_secrets` | `bool` | `True` | Redact sensitive keys from debug payload |
| `max_payload_chars` | `int` | `4000` | Truncate debug payload fields to this length |
| `emit_timestamps` | `bool` | `True` | Attach timestamp metadata to debug payloads |
| `emit_step_snapshots` | `bool` | `True` | Emit summarized per-step snapshot metadata |

---

## Debugger

The `Debugger` class provides a facade for creating configured `Runner` instances with debug instrumentation.

```python
from afk.debugger import Debugger, DebuggerConfig

# Create debugger with custom config
debugger = Debugger(
    DebuggerConfig(
        enabled=True,
        verbosity="detailed",
        redact_secrets=True,
    )
)

# Get a debug-configured runner
runner = debugger.runner()

# Or access the RunnerConfig directly
config = debugger.config()
```

### Debugger Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `DebuggerConfig \| None` | `None` | Debug configuration |

### Debugger Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `runner(config=None)` | `Runner` | Create a runner with debug instrumentation |
| `config()` | `RunnerConfig` | Get the debug-configured RunnerConfig |

---

## Usage with Runner

The debugger can be used to wrap runner creation with debug settings:

```python
from afk.debugger import Debugger, DebuggerConfig
from afk.agents import Agent

debugger = Debugger(
    DebuggerConfig(
        enabled=True,
        verbosity="detailed",
        redact_secrets=True,
    )
)

runner = debugger.runner()
agent = Agent(model="gpt-4.1-mini", instructions="You are helpful.")

result = runner.run_sync(
    agent,
    user_message="Hello",
    thread_id="debug-thread",
)
```

---

## Source Files

| File | Purpose |
|------|---------|
| `src/afk/debugger/__init__.py` | Public API exports |
| `src/afk/debugger/core.py` | `Debugger` implementation |
| `src/afk/debugger/types.py` | `DebuggerConfig` dataclass |

---

## Cross-References

- **Runner API**: See [agents-and-runner.md](./agents-and-runner.md) for `RunnerConfig.debug` and `RunnerConfig.debug_config`
- **Tools**: See [tools-system.md](./tools-system.md) for debugging tools