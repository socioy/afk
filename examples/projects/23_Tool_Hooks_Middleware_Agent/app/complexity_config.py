"""Complexity configuration for progressive example tier 23."""

EXAMPLE_NUMBER = 23
COMPLEXITY_LEVEL = 2
STAGE_COUNT = 24

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
