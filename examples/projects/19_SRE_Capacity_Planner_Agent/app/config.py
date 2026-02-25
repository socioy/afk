"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 19
EXAMPLE_TITLE = "SRE Capacity Planner Agent"
EXAMPLE_SUMMARY = "Plan SRE capacity under governance controls and aggregate portfolio metrics."
DOMAIN = "sre capacity planning"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 5

DEFAULT_PROMPT = (
    "Run sre capacity planning analysis with concrete recommendations and operational next steps."
)
