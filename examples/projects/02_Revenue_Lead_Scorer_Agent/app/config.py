"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 2
EXAMPLE_TITLE = "Revenue Lead Scorer Agent"
EXAMPLE_SUMMARY = "Score inbound pipeline opportunities with operationally grounded analytics."
DOMAIN = "revenue operations"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 1

DEFAULT_PROMPT = (
    "Run revenue operations analysis with concrete recommendations and operational next steps."
)
