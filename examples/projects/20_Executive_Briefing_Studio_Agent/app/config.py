"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 20
EXAMPLE_TITLE = "Executive Briefing Studio Agent"
EXAMPLE_SUMMARY = "Produce executive-ready strategy briefings with governed analytics outputs."
DOMAIN = "executive operations"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 5

DEFAULT_PROMPT = (
    "Run executive operations analysis with concrete recommendations and operational next steps."
)
