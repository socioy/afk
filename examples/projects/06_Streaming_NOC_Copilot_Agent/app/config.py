"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 6
EXAMPLE_TITLE = "Streaming NOC Copilot Agent"
EXAMPLE_SUMMARY = "Stream network operations guidance with real-time event analytics."
DOMAIN = "network operations center"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 2

DEFAULT_PROMPT = (
    "Run network operations center analysis with concrete recommendations and operational next steps."
)
