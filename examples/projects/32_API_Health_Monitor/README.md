
# API Health Monitor

An API health monitoring agent that demonstrates LLMClient configuration with production-grade resilience policies: RetryPolicy, TimeoutPolicy, and CircuitBreakerPolicy. Shows how to configure the LLM layer for reliable agent execution.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/32_API_Health_Monitor

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/32_API_Health_Monitor

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/32_API_Health_Monitor

Expected interaction
User: Check the health of all our API endpoints
Agent: (calls ping_endpoint, check_response_time, check_error_rate tools)
Agent: Here's the health status of your API endpoints...

The script also prints the LLM client configuration showing all active resilience policies.

