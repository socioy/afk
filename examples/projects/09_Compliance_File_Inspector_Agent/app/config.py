"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 9
EXAMPLE_TITLE = "Compliance File Inspector Agent"
EXAMPLE_SUMMARY = "Inspect compliance workflows with richer multi-turn analytics patterns."
DOMAIN = "compliance monitoring"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 2

DEFAULT_PROMPT = (
    "Run compliance monitoring analysis with concrete recommendations and operational next steps."
)
