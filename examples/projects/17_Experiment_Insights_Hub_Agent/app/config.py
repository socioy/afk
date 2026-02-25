"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 17
EXAMPLE_TITLE = "Experiment Insights Hub Agent"
EXAMPLE_SUMMARY = "Synthesize experiment outcomes with scenario simulation and batch analytics."
DOMAIN = "product experimentation"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 4

DEFAULT_PROMPT = (
    "Run product experimentation analysis with concrete recommendations and operational next steps."
)
