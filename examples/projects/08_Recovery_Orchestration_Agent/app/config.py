"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 8
EXAMPLE_TITLE = "Recovery Orchestration Agent"
EXAMPLE_SUMMARY = "Drive recovery orchestration with streaming and thread continuity analytics."
DOMAIN = "service recovery orchestration"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 2

DEFAULT_PROMPT = (
    "Run service recovery orchestration analysis with concrete recommendations and operational next steps."
)
