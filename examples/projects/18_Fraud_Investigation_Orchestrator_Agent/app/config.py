"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 18
EXAMPLE_TITLE = "Fraud Investigation Orchestrator Agent"
EXAMPLE_SUMMARY = "Run governance-grade fraud investigations with portfolio-level execution analytics."
DOMAIN = "fraud operations"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 5

DEFAULT_PROMPT = (
    "Run fraud operations analysis with concrete recommendations and operational next steps."
)
