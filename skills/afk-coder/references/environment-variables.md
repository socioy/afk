# Environment Variable Reference

> Complete reference for all AFK environment variables.

---

## LLM Settings

| Variable | Default | Purpose |
|----------|---------|---------|
| `AFK_LLM_PROVIDER` | `litellm` | Primary provider |
| `AFK_LLM_PROVIDER_ORDER` | -- | Fallback order (comma-separated): `openai,anthropic,litellm` |
| `AFK_MCP_PORT` | `8000` | Bind port |
| `AFK_MCP_INSTRUCTIONS` | -- | Server instructions |
| `AFK_MCP_PATH` | `/mcp` | JSON-RPC endpoint |
| `AFK_MCP_SSE_PATH` | `/mcp/sse` | SSE endpoint |
| `AFK_MCP_HEALTH_PATH` | `/health` | Health check endpoint |
| `AFK_MCP_ENABLE_SSE` | `true` | Enable SSE transport |
| `AFK_MCP_ENABLE_HEALTH` | `true` | Enable health endpoint |
| `AFK_MCP_ALLOW_BATCH` | `true` | Allow batch requests |

---

## Runner / Safety

| Variable | Default | Purpose |
|----------|---------|---------|
| `AFK_ALLOWED_COMMANDS` | `ls,cat,head,tail,rg,pwd,echo` | Allowed shell commands (comma-separated) |

---

## Observability

| Variable | Default | Purpose |
|----------|---------|---------|
| `AFK_OTEL_ENDPOINT` | -- | OpenTelemetry collector endpoint |
| `AFK_OTEL_INSECURE` | `false` | Use insecure OTEL connection |

---

## Multi-Provider Routing

AFK supports multiple providers with automatic fallback via `RoutePolicy`:

```python
from afk.llms import LLMClient, RoutePolicy

# Client with fallback chain
client = LLMClient(
    provider="openai",  # primary
    settings=LLMSettings(
        default_model="gpt-4.1",
    ),
    route_policy=RoutePolicy(
        provider_order=("openai", "anthropic", "litellm"),
    ),
)
```

The router tries providers in order until one succeeds:

```python
# Use LLMBuilder for cleaner API
from afk.llms import LLMBuilder, RoutePolicy

client = (
    LLMBuilder()
    .provider("openai")
    .model("gpt-4.1")
    .with_router(RoutePolicy(provider_order=("openai", "anthropic")))
    .build()
)
```

Environment variable for default provider order:

```bash
# Primary
export AFK_LLM_PROVIDER="openai"  # primary

# Router tries in order: openai → anthropic → litellm
```

---

## Per-Provider Settings

Configure each provider with its own API key and base URL:

```python
from afk.llms import LLMClient, LLMSettings, RoutePolicy

# Per-provider settings (keyed by provider name)
provider_settings = {
    "openai": {
        "api_key": "sk-openai-...",
        "api_base_url": "https://api.openai.com/v1",
    },
    "anthropic": {
        "api_key": "sk-ant-...",
    },
    "litellm": {
        "api_key": "sk-litellm-...",
        "api_base_url": "https://my-proxy.com/v1",
    },
}

client = LLMClient(
    provider="openai",
    settings=LLMSettings(default_model="gpt-4.1"),
    provider_settings=provider_settings,
    route_policy=RoutePolicy(
        provider_order=("openai", "anthropic", "litellm"),
    ),
)
```

Or using LLMBuilder:

```python
from afk.llms import LLMBuilder, RoutePolicy

client = (
    LLMBuilder()
    .provider("openai")
    .model("gpt-4.1")
    .with_provider_settings("openai", {
        "api_key": "sk-openai-...",
    })
    .with_provider_settings("anthropic", {
        "api_key": "sk-ant-...",
    })
    .with_provider_settings("litellm", {
        "api_key": "sk-litellm-...",
        "api_base_url": "https://my-proxy.com/v1",
    })
    .with_router(RoutePolicy(
        provider_order=("openai", "anthropic", "litellm"),
    ))
    .build()
)
```

### Environment Variable Pattern

For env vars, prefix with provider name:

```bash
# Primary
export AFK_LLM_PROVIDER="openai"
export AFK_LLM_MODEL="gpt-4.1"

# OpenAI
export AFK_OPENAI_API_KEY="sk-openai-..."

# Anthropic  
export AFK_ANTHROPIC_API_KEY="sk-ant-..."

# LiteLLM (custom proxy)
export AFK_LITELLM_API_KEY="sk-litellm-..."
export AFK_LITELLM_API_BASE_URL="https://my-proxy.com/v1"

# Fallback order
export AFK_LLM_PROVIDER_ORDER="openai,anthropic,litellm"
```

Common provider env var prefixes:

| Provider | API Key Var | API Base URL Var |
|----------|------------|----------------|
| `openai` | `AFK_OPENAI_API_KEY` | `AFK_OPENAI_API_BASE_URL` |
| `anthropic` | `AFK_ANTHROPIC_API_KEY` | -- |
| `litellm` | `AFK_LITELLM_API_KEY` | `AFK_LITELLM_API_BASE_URL` |

---

## Multiple API Keys for Same Provider

You can create separate clients for different API keys:

```python
from afk.llms import LLMBuilder

# Create separate clients per API key
client_a = (
    LLMBuilder()
    .provider("openai")
    .with_provider_settings("openai", {"api_key": "sk-proj-a-..."})
    .build()
)

client_b = (
    LLMBuilder()
    .provider("openai")
    .with_provider_settings("openai", {"api_key": "sk-proj-b-"})
    .build()
)

# Use whichever you need
result = await client_a.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Or with Agents - pass the client explicitly:

```python
from afk.agents import Agent
from afk.core import Runner
from afk.llms import LLMBuilder

# Different agents with different API keys
agent_a = Agent(
    model="gpt-4.1",
    instructions="You are helpful.",
    llm=LLMBuilder()
        .provider("openai")
        .with_provider_settings("openai", {"api_key": "sk-proj-a-..."})
        .build(),
)

agent_b = Agent(
    model="gpt-4.1", 
    instructions="You are helpful.",
    llm=LLMBuilder()
        .provider("openai")
        .with_provider_settings("openai", {"api_key": "sk-proj-b-..."})
        .build(),
)
```

Or use LiteLLM as a proxy with multiple keys:

```bash
# LiteLLM can route to different keys via its config
export AFK_LITELLM_API_KEY="sk-litellm-master"
export AFK_LITELLM_API_BASE_URL="https://your-proxy.com/v1"

# In LiteLLM config.yaml:
model_list:
  - model_name: gpt-4.1
    litellm_params:
      api_key: os.environ/OPENAI_PROJECT_A_KEY
  - model_name: gpt-4.1-a
    litellm_params:
      api_key: os.environ/OPENAI_PROJECT_B_KEY
```

---

## Quick Setup

```bash
# LLM Configuration
export AFK_LLM_API_KEY="sk-your-key-here"
export AFK_LLM_PROVIDER="openai"  # or litellm, anthropic
export AFK_LLM_MODEL="gpt-4.1"

# Memory (optional: defaults to SQLite)
export AFK_MEMORY_BACKEND="sqlite"
# or Redis:
export AFK_MEMORY_BACKEND="redis"
export AFK_REDIS_URL="redis://localhost:6379/0"

# MCP Server (optional)
export AFK_MCP_PORT=8080
```

---

## Loading from Code

Most modules automatically load settings from environment:

```python
# Automatic - reads AFK_* env vars
from afk.llms import LLMBuilder
client = LLMBuilder().build()

# Manual override
from afk.llms import LLMSettings
settings = LLMSettings.from_env()
custom = settings.model_dump(update={"timeout_s": 60.0})
```

For direct env access:

```python
from afk.config import LLMEnv, MemoryEnv, MCPEnv

llm_cfg = LLMEnv.from_env()
memory_cfg = MemoryEnv.from_env()
mcp_cfg = MCPEnv.from_env()
```