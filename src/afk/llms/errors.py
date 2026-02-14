from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module defines custom exceptions for error handling in the llm package.
"""


class LLMError(Exception):
    """Base exception for all AFK LLM-related errors."""

    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRetryableError(LLMError):
    """
    Transient failures: reate limits, timeouts, provider issues, etc.
    These errors may be retried with backoff.
    """

    pass


class LLMInvalidResponseError(LLMError):
    """
    The LLM returned a response that we couldn't parse or validate.
    This may indicate a schema mismatch, provider issue, or unexpected content.
    """

    pass


class LLMConfigurationError(LLMError):
    pass


class LLMCapabilityError(LLMError):
    """
    Raised when the selected provider adapter does not support a requested
    capability (e.g., embeddings or streaming).
    """

    pass
