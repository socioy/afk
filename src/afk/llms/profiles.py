"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: profiles.py.
"""

from __future__ import annotations

from .runtime.contracts import (
    CachePolicy,
    CircuitBreakerPolicy,
    CoalescingPolicy,
    HedgingPolicy,
    RateLimitPolicy,
    RetryPolicy,
    TimeoutPolicy,
)

PROFILES = {
    "development": {
        "retry": RetryPolicy(max_retries=1, backoff_base_s=0.2, backoff_jitter_s=0.05),
        "timeout": TimeoutPolicy(request_timeout_s=30.0, stream_idle_timeout_s=90.0),
        "rate_limit": RateLimitPolicy(requests_per_second=50.0, burst=100),
        "breaker": CircuitBreakerPolicy(
            failure_threshold=8, cooldown_s=10.0, half_open_max_calls=2
        ),
        "hedging": HedgingPolicy(enabled=False, delay_s=0.2),
        "cache": CachePolicy(enabled=False, ttl_s=15.0),
        "coalescing": CoalescingPolicy(enabled=True),
    },
    "production": {
        "retry": RetryPolicy(max_retries=3, backoff_base_s=0.5, backoff_jitter_s=0.15),
        "timeout": TimeoutPolicy(request_timeout_s=30.0, stream_idle_timeout_s=45.0),
        "rate_limit": RateLimitPolicy(requests_per_second=20.0, burst=40),
        "breaker": CircuitBreakerPolicy(
            failure_threshold=5, cooldown_s=30.0, half_open_max_calls=1
        ),
        "hedging": HedgingPolicy(enabled=False, delay_s=0.2),
        "cache": CachePolicy(enabled=False, ttl_s=30.0),
        "coalescing": CoalescingPolicy(enabled=True),
    },
    "high_throughput": {
        "retry": RetryPolicy(max_retries=2, backoff_base_s=0.3, backoff_jitter_s=0.1),
        "timeout": TimeoutPolicy(request_timeout_s=20.0, stream_idle_timeout_s=40.0),
        "rate_limit": RateLimitPolicy(requests_per_second=120.0, burst=200),
        "breaker": CircuitBreakerPolicy(
            failure_threshold=10, cooldown_s=20.0, half_open_max_calls=3
        ),
        "hedging": HedgingPolicy(enabled=False, delay_s=0.2),
        "cache": CachePolicy(enabled=True, ttl_s=20.0),
        "coalescing": CoalescingPolicy(enabled=True),
    },
    "low_latency": {
        "retry": RetryPolicy(max_retries=1, backoff_base_s=0.2, backoff_jitter_s=0.05),
        "timeout": TimeoutPolicy(request_timeout_s=10.0, stream_idle_timeout_s=20.0),
        "rate_limit": RateLimitPolicy(requests_per_second=30.0, burst=60),
        "breaker": CircuitBreakerPolicy(
            failure_threshold=3, cooldown_s=15.0, half_open_max_calls=1
        ),
        "hedging": HedgingPolicy(enabled=True, delay_s=0.08),
        "cache": CachePolicy(enabled=True, ttl_s=10.0),
        "coalescing": CoalescingPolicy(enabled=True),
    },
}
