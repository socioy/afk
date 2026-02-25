"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 3
EXAMPLE_TITLE = "Support SLA Triage Agent"
EXAMPLE_SUMMARY = "Prioritize support workload using SLA-aware triage analytics."
DOMAIN = "support operations"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 1

DEFAULT_PROMPT = (
    "Run support operations analysis with concrete recommendations and operational next steps."
)
