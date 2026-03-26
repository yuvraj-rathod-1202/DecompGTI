from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .schemas import ToolCall
from .tool_catalog import TOOL_SPECS

_TOOL_NAMES = {t.name for t in TOOL_SPECS}
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"(\{[\s\S]*\})")


class ToolRoutingError(ValueError):
    pass


def _extract_json_blob(text: str) -> str:
    code_match = _CODE_BLOCK_RE.search(text)
    if code_match:
        return code_match.group(1)

    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        return obj_match.group(1)

    raise ToolRoutingError("No JSON object found in model output.")


def extract_tool_call(model_output: str) -> ToolCall:
    """Parse model output and return normalized tool call payload."""

    try:
        payload = json.loads(_extract_json_blob(model_output))
    except json.JSONDecodeError as exc:
        raise ToolRoutingError(f"Invalid JSON from model output: {exc}") from exc

    try:
        call = ToolCall.model_validate(payload)
    except ValidationError as exc:
        raise ToolRoutingError(f"Invalid tool payload schema: {exc}") from exc

    if call.tool_name not in _TOOL_NAMES:
        raise ToolRoutingError(
            f"Unsupported tool '{call.tool_name}'. Allowed: {sorted(_TOOL_NAMES)}"
        )

    return call


def extract_tool_name(model_output: str) -> str:
    """Extract only tool name from model output text."""

    return extract_tool_call(model_output).tool_name


def build_tool_call_from_model_output(model_output: str) -> tuple[str, dict[str, Any]]:
    """Convenience helper for inference pipelines."""

    parsed = extract_tool_call(model_output)
    return parsed.tool_name, parsed.arguments
