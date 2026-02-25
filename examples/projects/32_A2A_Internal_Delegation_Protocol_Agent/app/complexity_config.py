"""Complexity configuration for progressive example tier 32."""

EXAMPLE_NUMBER = 32
COMPLEXITY_LEVEL = 11
STAGE_COUNT = 33

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
