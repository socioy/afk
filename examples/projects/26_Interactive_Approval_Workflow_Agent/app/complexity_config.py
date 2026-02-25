"""Complexity configuration for progressive example tier 26."""

EXAMPLE_NUMBER = 26
COMPLEXITY_LEVEL = 5
STAGE_COUNT = 27

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
