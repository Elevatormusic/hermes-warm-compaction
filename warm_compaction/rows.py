"""Helpers for history rows, reply objects, row digests, and size estimates."""

from __future__ import annotations

import hashlib
import importlib
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


def hermes_value(module: str, name: str, default: Any) -> Any:
    """Read one Hermes value. Return the default when the read fails or the type is not the default type."""
    try:
        value = getattr(importlib.import_module(module), name)
    except Exception:
        return default
    return value if isinstance(value, type(default)) else default


API_CONTENT_ROLES = ("user", "assistant")


def api_content(row: Any) -> Any:
    """Return the content that Hermes sends for a stored row. A user or assistant row can carry an api_content
    sidecar: the exact text of the earlier request, which Hermes sends in place of content."""
    sidecar = attr(row, "api_content")
    if isinstance(sidecar, str) and sidecar and attr(row, "role") in API_CONTENT_ROLES:
        return sidecar
    return attr(row, "content")


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


ATTACHMENT_KINDS = {"image_url": "image", "input_audio": "audio", "file": "file"}
REFERENCE_CHARS = 200


def _attachment_mark(part: dict) -> str:
    kind = str(part.get("type") or "unknown")
    label = ATTACHMENT_KINDS.get(kind, kind)
    body = part.get(kind) if isinstance(part.get(kind), dict) else {}
    reference = body.get("filename") if kind == "file" else body.get("url") if kind == "image_url" else None
    if isinstance(reference, str) and reference and not reference.startswith("data:"):
        return f"[{label} attachment: {reference[:REFERENCE_CHARS]}]"
    return f"[{label} attachment]"


def visible_text(content: Any) -> str:
    """Return the text of a content value and one mark for each image, audio, or file part. A data URL is not
    copied; a web URL or a file name is."""
    if not isinstance(content, list):
        return plain_text(content)
    lines = []
    for part in content:
        if isinstance(part, str):
            lines.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str) and part.get("type", "text") == "text":
            lines.append(part["text"])
        elif isinstance(part, dict):
            lines.append(_attachment_mark(part))
    return "\n".join(lines)


MIDDLE_MARK = " [cut] "


def cut_bounds(text: str, limit: int) -> tuple[int, int]:
    """Return (end of the kept start, start of the kept end) of cut_middle: text[first:second] is the removed
    part."""
    if len(text) <= limit:
        return len(text), len(text)
    keep = limit - len(MIDDLE_MARK)
    if keep < 2:
        return max(limit, 0), len(text)
    head = keep * 2 // 3
    return head, len(text) - (keep - head)


def cut_middle(text: str, limit: int) -> str:
    """Return at most limit characters: the start (two thirds) and the end of the text, with a mark between.
    The end of a long text often has the result, the question, or the output rules."""
    if len(text) <= limit:
        return text
    head, end = cut_bounds(text, limit)
    if end == len(text):
        return text[:head]
    return text[:head] + MIDDLE_MARK + text[end:]


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
    """Estimate tokens of the compact JSON form: the ASCII characters divided by 4, plus one token for each
    other character. CJK text and emoji have about one token or more for each character."""
    text = compact_json(value)
    ascii_count = len(text.encode("ascii", "ignore"))
    return math.ceil(ascii_count / 4) + len(text) - ascii_count


def row_digest(row: Any) -> str:
    """Return the SHA-256 of the canonical JSON of the row fields that the model sees, with the api_content
    sidecar of a user or assistant row."""
    canonical = {
        "role": attr(row, "role"),
        "content": attr(row, "content"),
        "tool_call_id": attr(row, "tool_call_id"),
        "name": attr(row, "name"),
        "tool_calls": [list(call) for call in tool_calls_of(row)],
        # The text that the provider saw, when Hermes keeps it in the api_content sidecar.
        "api_content": api_content(row) if api_content(row) is not attr(row, "content") else None,
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("ascii")).hexdigest()
