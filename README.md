# AFK (Agent Forge Kit)

AFK is an agent runtime and SDK for building production-oriented agent systems with:

- policy-aware execution
- tool orchestration with security boundaries
- checkpoint/resume workflows
- provider-agnostic LLM integrations
- eval harness support

## Development Status

> **Note:** AFK is in **fast-paced development mode**.
> APIs, behavior, and docs may change quickly. Pin versions and test upgrades carefully.

## Installation

### From source (this repository)

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### SDK usage

Import from public package paths:

```python
from afk.agents import Agent
from afk.core import Runner, RunnerConfig
from afk.tools import tool
from afk.llms import create_llm
```

## Quick Start

```python
from pydantic import BaseModel, Field
from afk.agents import Agent
from afk.tools import tool


class SumArgs(BaseModel):
    numbers: list[float] = Field(min_length=1)


@tool(
    args_model=SumArgs,
    name="sum_numbers",
    description="Add all numbers and return their sum.",
)
def sum_numbers(args: SumArgs) -> dict[str, float]:
    return {"sum": float(sum(args.numbers))}


agent = Agent(
    model="gpt-4.1-mini",
    instructions="Use sum_numbers for arithmetic.",
    tools=[sum_numbers],
)
```

Configure environment for your adapter/model before running full examples:

```bash
export AFK_LLM_ADAPTER=openai
export AFK_LLM_MODEL=gpt-4.1-mini
export AFK_LLM_API_KEY=your_key_here
```

## Documentation

- Public docs: `https://afk.arpan.sh`
- Docs source: `docs/`
- Main docs entry: `docs/index.mdx`
- Mintlify config: `docs/docs.json`

Run docs locally:

```bash
cd docs
bunx mintlify dev
```

## Running Tests

```bash
PYTHONPATH=src pytest -q
```

CI currently runs Python `3.13`.

## Contributing

See `CONTRIBUTING.md` for setup, workflow, and pull request expectations.

## Maintainer Contact

- GitHub: `arpan404@github` (handle: `@arpan404`)
- LinkedIn: `arpanbhandari`
- Email: `contact@arpan.sh`

## License

MIT. See `LICENSE`.
