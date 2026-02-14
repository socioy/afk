from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

Module for structured llm outputs
Note: currently we are using litellm which handles the structured output parsing and validation. In future, we might move away from litellm and implement using major llm providers' function calling / tool calling features. In that case, we can implement the output parsing and validation logic in this module.
"""
import json
from typing import TypeVar
from pydantic import BaseModel, ValidationError

from .errors import LLMInvalidResponseError
from .utils import extract_json_object, safe_json_loads

T = TypeVar("T", bound=BaseModel)


def json_system_prompt(schema: type[T]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=True)
    return (
        "You must respond with exactly one valid JSON value that strictly conforms to the Pydantic schema below. "
        "Do not add any explanations, markdown, code fences, or surrounding text—only the JSON. "
        f"{schema_json}\n"
        "Rules:\n"
        "- Do not include properties not defined in the schema.\n"
        "- Use double quotes for strings and proper JSON types for all fields.\n"
        "- Provide all required fields; use null only if the schema allows it.\n"
        "The output will be parsed and validated against the schema; ensure it is valid JSON and fully compliant."
    )


def parse_and_validate_json(text: str, schema: type[T]) -> T:
    json_str = extract_json_object(text) or text.strip()
    obj = safe_json_loads(json_str)
    if obj is None:
        raise LLMInvalidResponseError(
            f"Failed to extract valid JSON object from response: {text}"
        )
    try:
        return schema.model_validate(obj)
    except ValidationError as e:
        raise LLMInvalidResponseError(
            f"JSON does not conform to schema: {e}\nOriginal response: {text}"
        ) from e


def make_repair_prompt(invalid_response: str, schema: type[T]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=True)
    return (
        "The previous output was invalid and did not conform to the schema. "
        "Fix the JSON below so it exactly matches the Pydantic schema. "
        "Return only the corrected JSON—no explanations, markdown, or code fences.\n"
        f"{schema_json}\n"
        "Invalid response:\n"
        f"{invalid_response}"
    )
