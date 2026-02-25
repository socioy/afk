"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 12
EXAMPLE_TITLE = "FinOps Budget Guard Agent"
EXAMPLE_SUMMARY = "Track spend-risk tradeoffs with policy and telemetry-backed execution analytics."
DOMAIN = "finops governance"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 3

DEFAULT_PROMPT = (
    "Run finops governance analysis with concrete recommendations and operational next steps."
)
