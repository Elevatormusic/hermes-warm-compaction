"""Helpers for history rows, reply objects, row digests, and size estimates."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from typing import Any, NamedTuple

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


def attachment_mark(part: dict) -> str:
    """One short mark for an image, audio, or file part: a web URL or a file name (at most REFERENCE_CHARS
    characters) stays, a data URL does not."""
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
            lines.append(attachment_mark(part))
    return "\n".join(lines)


MIDDLE_MARK = " [cut] "


# The fields of a row that Hermes sends (warm.wire_row): a stored reasoning field and metadata are not sent.
SENT_FIELDS = ("role", "content", "name", "tool_call_id", "reasoning_content")


class SendPolicy(NamedTuple):
    """The route-dependent fields of warm.wire_row: reasoning_details (a route that replays them), the tool-call
    thought signature, extra_content (a model that reads it), and reasoning_content (a route that needs it back,
    apply_reasoning_content_policy). The default counts all of them. cut_reasoning: a cut can take reasoning
    (layout.bound_tail) only when a capture of the route shows that the route sends it; else stored reasoning can
    be of an earlier route, and a cut would give it to the fallback model. native_type: the private
    <provider>.native_assistant carrier that the provider profile declares (native_reasoning_details_type); Hermes
    replays it, and only it of the native carriers."""
    details: bool = True
    signatures: bool = True
    echo: bool = True
    cut_reasoning: bool = True
    native_type: str | None = None
    native_mode: str = ""


def reasoning_policy(source: dict, wire: dict, needs_pad: bool) -> None:
    """The reasoning_content rule of Hermes 45871e10 (agent.message_sanitization.apply_reasoning_content_policy):
    a thinking-mode route (DeepSeek, Kimi, MiMo) needs the field on every assistant row; other routes reject it."""
    if source.get("role") != "assistant":
        return
    if not needs_pad:
        wire.pop("reasoning_content", None)
        return
    existing, reasoning = source.get("reasoning_content"), source.get("reasoning")
    if isinstance(existing, str):
        wire["reasoning_content"] = existing or " "
    elif isinstance(reasoning, str) and reasoning and not source.get("tool_calls"):
        wire["reasoning_content"] = reasoning
    else:
        wire["reasoning_content"] = " "


def sent_reasoning_key(row: Any, policy: SendPolicy) -> str | None:
    """The stored field that Hermes sends as reasoning_content (reasoning_policy), or None: then the row sends no
    reasoning of its own."""
    if not policy.echo or not isinstance(row, dict) or row.get("role") != "assistant":
        return None
    if isinstance(row.get("reasoning_content"), str):
        return "reasoning_content"
    reasoning = row.get("reasoning")
    return "reasoning" if isinstance(reasoning, str) and reasoning and not row.get("tool_calls") else None


def has_thought_signature(extra: Any) -> bool:
    """True when a tool-call extra_content has a usable thought signature (Hermes sends only such a value)."""
    if not isinstance(extra, dict):
        return False
    candidate = extra.get("thought_signature")
    google = extra.get("google")
    if candidate is None and isinstance(google, dict):
        candidate = google.get("thought_signature")
    return isinstance(candidate, str) and bool(candidate.strip())


def replay_details(details: Any, native_type: str | None = None) -> list | None:
    """Return reasoning_details without private native-assistant carriers, except the carrier of the provider
    profile (native_type), or None when nothing is left. The items are not copied."""
    if not isinstance(details, list):
        return None
    kept = [item for item in details if not (
        isinstance(item, dict) and isinstance(item.get("type"), str) and item["type"].endswith(".native_assistant")
        and item["type"] != native_type)]
    return kept or None


def sent_rows(messages: list, policy: SendPolicy = SendPolicy()) -> list:
    """Return the rows as Hermes sends them, for an estimate, by the rules of warm.wire_row: the api_content
    sidecar in place of the content, no name on a tool row, each tool call as id, type, and function name and
    arguments (with a usable thought signature when the model reads it), replayed reasoning_details, and
    reasoning_content by the reasoning policy."""
    reasoning = hermes_value("agent.message_sanitization", "apply_reasoning_content_policy", reasoning_policy)
    out = []
    for row in messages:
        if isinstance(row, dict):
            role = row.get("role")
            sent = {key: row[key] for key in SENT_FIELDS if key in row and not (key == "name" and role == "tool")}
            sent["content"] = api_content(row)
            calls = []
            for (call_id, name, arguments), source in zip(tool_calls_of(row), row.get("tool_calls") or ()):
                call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": (
                    arguments if isinstance(arguments, str) else compact_json(
                        arguments if arguments is not None else {}))}}
                extra = attr(source, "extra_content")
                if policy.signatures and has_thought_signature(extra):
                    call["extra_content"] = extra
                calls.append(call)
            if calls:
                sent["tool_calls"] = calls
            if policy.details and role == "assistant":
                details = replay_details(row.get("reasoning_details"), policy.native_type)
                if details is not None:
                    sent["reasoning_details"] = details
            if policy.native_mode in ("codex_responses", "anthropic_messages"):
                # Count opaque replay too. Canonical text is also counted, which gives a safe upper estimate.
                for field in ("codex_message_items", "codex_reasoning_items", "anthropic_content_blocks"):
                    if row.get(field):
                        sent[field] = row[field]
            reasoning(row, sent, policy.echo)
            out.append(sent)
        else:
            out.append(row)
    return out


def sent_tokens(row: Any, policy: SendPolicy = SendPolicy()) -> int:
    """Estimated tokens of one row as Hermes sends it (sent_rows)."""
    return estimate_tokens(sent_rows([row], policy)[0])


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
        # Native replay fields can change the provider input without changing the visible text.
        "native": {name: attr(row, name) for name in (
            "reasoning_content", "reasoning_details", "codex_message_items", "codex_reasoning_items",
            "codex_checkpoint_items", "codex_reasoning_trimmed", "phase", "anthropic_content_blocks",
            "_anthropic_content_blocks", "call_id", "response_item_id")},
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("ascii")).hexdigest()
