"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module defines custom exceptions for error handling in the tools.
"""

from __future__ import annotations


class AFKToolError(Exception):
    """Base exception for all AFK tool-related errors."""

    pass


class ToolValidationError(AFKToolError):
    pass


class ToolAlreadyRegisteredError(AFKToolError):
    pass


class ToolValidationError(AFKToolError):
    pass


class ToolExecutionError(AFKToolError):
    pass


class ToolTimeoutError(AFKToolError):
    pass


class ToolPolicyError(AFKToolError):
    pass


class ToolNotFoundError(AFKToolError):
    pass


class ToolPermissionError(AFKToolError):
    pass
