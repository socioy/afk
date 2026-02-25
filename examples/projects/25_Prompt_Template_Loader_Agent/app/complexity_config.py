"""Complexity configuration for progressive example tier 25."""

EXAMPLE_NUMBER = 25
COMPLEXITY_LEVEL = 4
STAGE_COUNT = 26

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
