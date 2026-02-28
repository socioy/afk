"""
---
name: API Health Monitor
description: An API health monitor that demonstrates LLMClient with retry, timeout, and circuit breaker policies for production-grade resilience.
tags: [agent, runner, llm-client, retry, timeout, circuit-breaker, resilience, llm-config]
---
---
This example demonstrates AFK's LLMClient runtime with production resilience policies. The
LLMClient is the layer between your agent and the LLM provider -- it handles retries, timeouts,
circuit breaking, rate limiting, caching, and request coalescing. By configuring RetryPolicy
(max_retries, backoff_base_s), TimeoutPolicy (request_timeout_s, stream_idle_timeout_s), and
CircuitBreakerPolicy (failure_threshold, cooldown_s), you make your agent robust against
transient failures, slow responses, and provider outages. This example shows how to create
an LLMClient with these policies, wire it into an agent, and run an API health monitoring
workflow. The focus is on the LLM configuration pattern rather than the agent conversation.
---
"""

import asyncio  # <- Async required for runner.run() and LLMClient operations.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner orchestrates agent execution.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- @tool decorator for creating agent-callable tools.
from afk.llms import (  # <- LLM runtime module: client, policies, and factory function.
    LLMClient,  # <- The runtime client that wraps LLM providers with enterprise policies. Handles retries, timeouts, circuit breaking, rate limiting, caching, and more.
    create_llm_client,  # <- Factory function for creating an LLMClient with explicit provider selection and settings.
    RetryPolicy,  # <- Retry configuration: max_retries, backoff_base_s, backoff_jitter_s. Controls how many times a failed LLM request is retried and the delay between attempts.
    TimeoutPolicy,  # <- Timeout configuration: request_timeout_s (max time for one request), stream_idle_timeout_s (max gap between stream chunks). Prevents hanging on slow providers.
    CircuitBreakerPolicy,  # <- Circuit breaker configuration: failure_threshold (consecutive failures to trip), cooldown_s (recovery wait time), half_open_max_calls (probe calls during recovery). Prevents cascading failures when a provider is down.
    LLMSettings,  # <- Settings object for LLM provider configuration: model, API keys, base URLs, etc.
)


# ===========================================================================
# Step 1: Configure LLM resilience policies
# ===========================================================================
# These policies control how the LLMClient handles failures at the provider
# level. They protect your agent from transient errors, slow responses, and
# complete provider outages.

# --- Retry Policy ---
# When an LLM request fails (network error, 500, rate limit), retry up to
# max_retries times with exponential backoff + jitter.
retry_policy = RetryPolicy(
    max_retries=3,  # <- Retry up to 3 times on failure. Total attempts = 1 original + 3 retries = 4.
    backoff_base_s=1.0,  # <- Base delay between retries in seconds. With exponential backoff: 1s, 2s, 4s. (The actual formula is base * 2^attempt + jitter.)
    backoff_jitter_s=0.25,  # <- Random jitter added to each retry delay. Prevents many clients from retrying at the exact same time ("thundering herd" problem).
)

# --- Timeout Policy ---
# Prevents the client from waiting forever on a slow or hung provider.
timeout_policy = TimeoutPolicy(
    request_timeout_s=10.0,  # <- Max time in seconds for a single LLM request (non-streaming). If the provider doesn't respond within 10s, the request is cancelled and may be retried.
    stream_idle_timeout_s=15.0,  # <- Max gap between streaming chunks. If no data arrives for 15s during a streaming response, the stream is aborted. Catches hung stream connections.
)

# --- Circuit Breaker Policy ---
# When a provider fails repeatedly, the circuit breaker "trips" and stops
# sending requests for a cooldown period. This prevents wasting time and
# resources on a provider that's clearly down.
circuit_breaker_policy = CircuitBreakerPolicy(
    failure_threshold=5,  # <- Trip the breaker after 5 consecutive failures. After this, all requests fail immediately without contacting the provider.
    cooldown_s=30.0,  # <- Wait 30 seconds before trying the provider again (half-open state). During cooldown, all requests fail fast.
    half_open_max_calls=1,  # <- During half-open state, allow 1 probe call to test if the provider has recovered. If it succeeds, the breaker closes (normal operation). If it fails, back to cooldown.
)


# ===========================================================================
# Step 2: Create the LLMClient with all policies
# ===========================================================================
# The LLMClient wraps the LLM provider with all configured policies. Every
# request the agent makes goes through this client, which applies retries,
# timeouts, and circuit breaking automatically.

llm_settings = LLMSettings(  # <- Provider-level settings. These configure which LLM provider to use and how to connect to it.
    default_provider="litellm",  # <- Provider backend. "litellm" is a universal adapter that supports 100+ LLM providers.
    default_model="ollama_chat/gpt-oss:20b",  # <- The default model to use. This can be overridden per-agent via the agent's model parameter.
    timeout_s=10.0,  # <- Default timeout (overridden by our TimeoutPolicy above, which is more specific).
    max_retries=3,  # <- Default retries (overridden by our RetryPolicy above).
)

llm_client = LLMClient(  # <- Create the client with all policies. This is the enterprise runtime layer between agents and LLM providers.
    provider="litellm",  # <- Which provider to use. "litellm" routes to the actual model specified in settings.
    settings=llm_settings,  # <- Connection settings (model, API base, keys, etc.).
    retry_policy=retry_policy,  # <- Apply our retry policy. All LLM requests will retry up to 3 times on failure.
    timeout_policy=timeout_policy,  # <- Apply our timeout policy. Requests are cancelled after 10s, streams after 15s idle.
    circuit_breaker_policy=circuit_breaker_policy,  # <- Apply our circuit breaker. After 5 failures, requests fail fast for 30s.
)


# ===========================================================================
# Step 3: Define the API monitoring tools
# ===========================================================================
# The agent monitors API endpoints using these tools. All data is simulated.
# In production, these would make actual HTTP requests to your API endpoints.

class EndpointArgs(BaseModel):  # <- Schema for tools that operate on a specific endpoint.
    endpoint: str = Field(description="The API endpoint URL or name to check, e.g., '/api/v1/users' or 'auth-service'")


class EmptyArgs(BaseModel):  # <- Schema for tools that take no arguments.
    pass


# --- Simulated API endpoint data ---
ENDPOINT_STATUS: dict[str, dict] = {  # <- Simulated health data for various API endpoints.
    "/api/v1/users": {
        "status": "healthy",
        "status_code": 200,
        "response_time_ms": 45,
        "error_rate": 0.2,
        "requests_per_minute": 1250,
        "last_error": None,
        "uptime": "99.97%",
    },
    "/api/v1/orders": {
        "status": "degraded",
        "status_code": 200,
        "response_time_ms": 850,
        "error_rate": 3.8,
        "requests_per_minute": 430,
        "last_error": "Timeout on database query (order_history)",
        "uptime": "99.12%",
    },
    "/api/v1/payments": {
        "status": "healthy",
        "status_code": 200,
        "response_time_ms": 120,
        "error_rate": 0.1,
        "requests_per_minute": 890,
        "last_error": None,
        "uptime": "99.99%",
    },
    "/api/v1/auth": {
        "status": "critical",
        "status_code": 503,
        "response_time_ms": 5200,
        "error_rate": 15.4,
        "requests_per_minute": 2100,
        "last_error": "Connection refused: auth-db-replica-2 unreachable",
        "uptime": "97.85%",
    },
    "/api/v1/search": {
        "status": "healthy",
        "status_code": 200,
        "response_time_ms": 210,
        "error_rate": 0.5,
        "requests_per_minute": 670,
        "last_error": None,
        "uptime": "99.94%",
    },
}


@tool(  # <- Ping tool. Checks basic reachability and status of an endpoint.
    args_model=EndpointArgs,
    name="ping_endpoint",
    description="Ping an API endpoint to check its basic health. Returns status code, reachability, and current status (healthy/degraded/critical).",
)
def ping_endpoint(args: EndpointArgs) -> str:
    # --- Find matching endpoint ---
    endpoint_key = None
    for key in ENDPOINT_STATUS:
        if args.endpoint.lower() in key.lower() or key.lower() in args.endpoint.lower():  # <- Fuzzy matching so "users" matches "/api/v1/users".
            endpoint_key = key
            break

    if endpoint_key is None:
        available = ", ".join(ENDPOINT_STATUS.keys())
        return f"Endpoint '{args.endpoint}' not found. Available endpoints: {available}"

    data = ENDPOINT_STATUS[endpoint_key]
    status_icon = {"healthy": "OK", "degraded": "WARN", "critical": "CRIT"}.get(data["status"], "?")

    return (
        f"Ping: {endpoint_key}\n"
        f"  Status: [{status_icon}] {data['status'].upper()}\n"
        f"  HTTP Status Code: {data['status_code']}\n"
        f"  Uptime: {data['uptime']}\n"
        f"  Requests/min: {data['requests_per_minute']}"
    )


@tool(  # <- Response time tool. Shows detailed latency metrics for an endpoint.
    args_model=EndpointArgs,
    name="check_response_time",
    description="Check the response time and latency metrics for an API endpoint. Returns current response time, percentiles, and trend.",
)
def check_response_time(args: EndpointArgs) -> str:
    endpoint_key = None
    for key in ENDPOINT_STATUS:
        if args.endpoint.lower() in key.lower() or key.lower() in args.endpoint.lower():
            endpoint_key = key
            break

    if endpoint_key is None:
        return f"Endpoint '{args.endpoint}' not found."

    data = ENDPOINT_STATUS[endpoint_key]
    base_ms = data["response_time_ms"]

    # --- Simulate percentile distribution ---
    p50 = base_ms  # <- Median response time.
    p90 = int(base_ms * 1.8)  # <- 90th percentile (tail latency).
    p99 = int(base_ms * 3.2)  # <- 99th percentile (worst case).

    trend = "stable" if base_ms < 200 else ("increasing" if base_ms < 1000 else "critical")

    return (
        f"Response Time: {endpoint_key}\n"
        f"  Current: {base_ms}ms\n"
        f"  p50: {p50}ms | p90: {p90}ms | p99: {p99}ms\n"
        f"  Trend: {trend}\n"
        f"  {'WARNING: Response time exceeds 500ms SLA threshold!' if base_ms > 500 else 'Within SLA thresholds.'}"
    )


@tool(  # <- Error rate tool. Shows error frequency and recent error details for an endpoint.
    args_model=EndpointArgs,
    name="check_error_rate",
    description="Check the error rate and recent errors for an API endpoint. Returns error percentage, volume, and last error details.",
)
def check_error_rate(args: EndpointArgs) -> str:
    endpoint_key = None
    for key in ENDPOINT_STATUS:
        if args.endpoint.lower() in key.lower() or key.lower() in args.endpoint.lower():
            endpoint_key = key
            break

    if endpoint_key is None:
        return f"Endpoint '{args.endpoint}' not found."

    data = ENDPOINT_STATUS[endpoint_key]
    error_rate = data["error_rate"]
    rpm = data["requests_per_minute"]
    errors_per_min = int(rpm * error_rate / 100)

    severity = "low" if error_rate < 1 else ("medium" if error_rate < 5 else "high")
    last_error = data["last_error"] or "No recent errors"

    return (
        f"Error Rate: {endpoint_key}\n"
        f"  Error rate: {error_rate}% [{severity.upper()}]\n"
        f"  Errors/min: ~{errors_per_min} (of {rpm} requests/min)\n"
        f"  Last error: {last_error}\n"
        f"  {'ALERT: Error rate exceeds 5% critical threshold!' if error_rate > 5 else 'Within acceptable thresholds.'}"
    )


@tool(  # <- Overview tool. Lists all endpoints and their current status at a glance.
    args_model=EmptyArgs,
    name="list_all_endpoints",
    description="List all monitored API endpoints with their current health status for a quick overview.",
)
def list_all_endpoints(args: EmptyArgs) -> str:
    lines = ["API Endpoint Overview:", "=" * 55]

    status_counts = {"healthy": 0, "degraded": 0, "critical": 0}

    for endpoint, data in ENDPOINT_STATUS.items():
        icon = {"healthy": "OK", "degraded": "WARN", "critical": "CRIT"}.get(data["status"], "?")
        status_counts[data["status"]] = status_counts.get(data["status"], 0) + 1
        lines.append(
            f"  [{icon:>4}] {endpoint:<25} {data['response_time_ms']:>5}ms  {data['error_rate']:>5.1f}% err  {data['uptime']}"
        )

    lines.append("")
    lines.append(f"  Summary: {status_counts['healthy']} healthy, {status_counts['degraded']} degraded, {status_counts['critical']} critical")
    lines.append(f"  Total endpoints monitored: {len(ENDPOINT_STATUS)}")

    return "\n".join(lines)


# ===========================================================================
# Step 4: Create the monitoring agent
# ===========================================================================

monitor_agent = Agent(
    name="api-monitor",  # <- The agent's display name.
    model="ollama_chat/gpt-oss:20b",  # <- The LLM model. The LLMClient handles all resilience policies for requests to this model.
    instructions="""
    You are an API health monitoring agent. You monitor API endpoints and provide
    health status reports.

    When asked to check API health:
    1. First use list_all_endpoints to get an overview of all endpoints.
    2. For any degraded or critical endpoints, run detailed checks:
       - ping_endpoint for basic reachability
       - check_response_time for latency analysis
       - check_error_rate for error details
    3. Provide a clear health report with:
       - Overall system status
       - Per-endpoint health details
       - Specific issues and their severity
       - Recommended actions for any problems found

    Prioritize critical issues first, then degraded, then healthy.
    """,  # <- Instructions guide the agent to use all monitoring tools systematically.
    tools=[ping_endpoint, check_response_time, check_error_rate, list_all_endpoints],  # <- Four monitoring tools for comprehensive health checking.
)

runner = Runner()  # <- The Runner executes the agent. The LLMClient policies (retry, timeout, circuit breaker) operate at the LLM request level underneath.


# ===========================================================================
# Step 5: Print the LLM configuration and run the monitor
# ===========================================================================

async def main():
    # --- Print the LLM client configuration ---
    print("API Health Monitor Agent")
    print("=" * 60)
    print()
    print("  LLM Client Configuration (Production Resilience)")
    print("  " + "-" * 56)
    print()
    print("  RetryPolicy:")
    print(f"    max_retries:      {retry_policy.max_retries}")  # <- How many times to retry failed LLM requests.
    print(f"    backoff_base_s:   {retry_policy.backoff_base_s}s")  # <- Base delay between retries (exponential backoff).
    print(f"    backoff_jitter_s: {retry_policy.backoff_jitter_s}s")  # <- Random jitter to prevent thundering herd.
    print()
    print("  TimeoutPolicy:")
    print(f"    request_timeout_s:      {timeout_policy.request_timeout_s}s")  # <- Max time for one non-streaming request.
    print(f"    stream_idle_timeout_s:  {timeout_policy.stream_idle_timeout_s}s")  # <- Max gap between streaming chunks.
    print()
    print("  CircuitBreakerPolicy:")
    print(f"    failure_threshold:   {circuit_breaker_policy.failure_threshold} consecutive failures")  # <- Failures before the breaker trips.
    print(f"    cooldown_s:          {circuit_breaker_policy.cooldown_s}s")  # <- Recovery wait time after tripping.
    print(f"    half_open_max_calls: {circuit_breaker_policy.half_open_max_calls}")  # <- Probe calls during half-open state.
    print()
    print("  How it works:")
    print("    1. Request fails -> RetryPolicy retries up to 3 times with backoff")
    print("    2. If still failing -> CircuitBreaker counts consecutive failures")
    print("    3. After 5 consecutive failures -> breaker TRIPS (all requests fail fast)")
    print("    4. After 30s cooldown -> breaker enters HALF-OPEN (1 probe call)")
    print("    5. If probe succeeds -> breaker CLOSES (back to normal)")
    print("    6. At any point, TimeoutPolicy cancels hung requests after 10s")
    print()
    print("  " + "-" * 56)
    print()

    print("Type 'quit' to exit.\n")

    while True:  # <- Interactive loop for monitoring queries.
        user_input = input("[] > ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Monitor shutting down.")
            break

        if not user_input:
            user_input = "Check the health of all our API endpoints. Alert me to any issues."  # <- Default prompt that exercises all monitoring tools.

        print(f"\nRunning health checks...\n")

        response = await runner.run(  # <- Run the agent. Underneath, every LLM request goes through the LLMClient with retry, timeout, and circuit breaker policies applied automatically.
            monitor_agent,
            user_message=user_input,
        )

        print(f"[api-monitor] > {response.final_text}\n")


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() starts the event loop for our async main function.



"""
---
Tl;dr: This example creates an API health monitoring agent and demonstrates LLMClient
configuration with production resilience policies. RetryPolicy(max_retries=3, backoff_base_s=1.0)
retries failed LLM requests with exponential backoff and jitter. TimeoutPolicy(request_timeout_s=
10.0) cancels hung requests after 10 seconds. CircuitBreakerPolicy(failure_threshold=5,
cooldown_s=30.0) trips after 5 consecutive failures, making all requests fail fast for 30
seconds to prevent cascading failures, then enters half-open state to probe for recovery. The
LLMClient is created with all three policies and wraps the LLM provider, so every request the
agent makes automatically benefits from retry, timeout, and circuit breaking. The agent itself
monitors API endpoints using four tools (ping_endpoint, check_response_time, check_error_rate,
list_all_endpoints) with simulated data, demonstrating how LLM reliability configuration works
alongside agent functionality.
---
---
What's next?
- Try adding a RateLimitPolicy(requests_per_second=5.0) to throttle LLM requests and prevent hitting provider rate limits.
- Add CachePolicy(enabled=True, ttl_s=60.0) to cache LLM responses and reduce redundant calls for repeated queries.
- Use HedgingPolicy(enabled=True, delay_s=0.5) to send speculative requests to a secondary provider for lower tail latency.
- Replace simulated endpoint data with real HTTP calls using httpx or aiohttp to build a production monitoring tool.
- Add a CoalescingPolicy to deduplicate identical in-flight LLM requests when multiple users ask the same question.
- Combine LLM resilience with telemetry (see the System Monitor example) to track retry rates, timeouts, and circuit breaker trips!
---
"""
