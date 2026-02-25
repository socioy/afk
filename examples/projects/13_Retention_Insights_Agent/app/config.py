"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 13
EXAMPLE_TITLE = "Retention Insights Agent"
EXAMPLE_SUMMARY = "Generate retention action plans with policy-aware analytics and controls."
DOMAIN = "customer retention"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 3

DEFAULT_PROMPT = (
    "Run customer retention analysis with concrete recommendations and operational next steps."
)
