"""Helpers for history rows, reply objects, row digests, and size estimates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)


def attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return a key of a mapping or an attribute of an object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def tool_calls_of(row: Any) -> list[tuple[str, str, Any]]:
    """Return (id, name, arguments) for each tool call of a row or a reply object."""
    calls = []
    for call in attr(row, "tool_calls") or ():
        function = attr(call, "function") or {}
        calls.append((str(attr(call, "id") or ""), str(attr(function, "name") or ""), attr(function, "arguments")))
    return calls


def plain_text(content: Any) -> str:
    """Return the text of a content value. For a list of parts, join the text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def strip_think(text: str) -> str:
    """Remove one leading <think>...</think> block and the white space around it."""
    return THINK_BLOCK.sub("", text, count=1)


def reply_text(content: Any) -> str:
    """Return the text of a reply without a leading think block and without outer white space."""
    return strip_think(plain_text(content)).strip()


def compact_json(value: Any) -> str:
    """Return compact JSON text. A value that JSON cannot show becomes a string."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def estimate_tokens(value: Any) -> int:
    """Estimate tokens: the UTF-8 bytes of the compact JSON form, divided by 4."""
    return math.ceil(len(compact_json(value).encode("utf-8", "surrogatepass")) / 4)


def row_digest(row: Any) -> str:
    """Return the SHA-256 of the canonical JSON of the row fields that the model sees."""
    canonical = {
        "role": attr(row, "role"),
        "content": attr(row, "content"),
        "tool_call_id": attr(row, "tool_call_id"),
        "name": attr(row, "name"),
        "tool_calls": [list(call) for call in tool_calls_of(row)],
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("ascii")).hexdigest()
