"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 14
EXAMPLE_TITLE = "Sales Forecast Planner Agent"
EXAMPLE_SUMMARY = "Run batch forecast workflows with sqlite memory and compaction analytics."
DOMAIN = "sales forecasting"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 4

DEFAULT_PROMPT = (
    "Run sales forecasting analysis with concrete recommendations and operational next steps."
)
