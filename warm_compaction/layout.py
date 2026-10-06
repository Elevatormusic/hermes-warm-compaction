"""Builds the new history: the summary header, the summary, and the verbatim tail."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from .handoff import LEGACY_PREFIX
from .rows import attr, estimate_tokens, plain_text

HEADER_TEXT = "The summary of the earlier turns follows."
COPY_HEADING = "## Copied user messages"
COPY_EACH = 4_000
NO_COPIES = "(none)"
STEER_KIND = "steer"
SYNTHETIC_FLAGS = ("_dropped_toolcall_nudge",)
SYNTHETIC_PREFIXES = (
    "[System: Your previous response was truncated",
    "[System: The previous response was cut off",
    "[System: Your previous tool call",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)


def is_summary(row: Any, prefixes: Iterable[str]) -> bool:
    """Return True for a summary record: the Hermes summary flag or a known summary prefix."""
    if attr(row, "_compressed_summary"):
        return True
    text = plain_text(attr(row, "content")).lstrip()
    return any(prefix and text.startswith(prefix) for prefix in prefixes)


def is_real_user(row: Any, prefixes: Iterable[str]) -> bool:
    """Return True for a user row that a person wrote. Scaffolding rows and summaries are not real."""
    if not isinstance(row, dict) or row.get("role") != "user":
        return False
    for key, value in row.items():
        if value and isinstance(key, str) and key.startswith("_") and key.endswith("_synthetic"):
            return False
    if any(row.get(flag) for flag in SYNTHETIC_FLAGS):
        return False
    if row.get("display_kind") and row.get("display_kind") != STEER_KIND:
        return False
    text = plain_text(row.get("content")).strip()
    if not text or text.startswith(SYNTHETIC_PREFIXES):
        return False
    return not is_summary(row, prefixes)


def units(messages: list) -> list[tuple[int, int]]:
    """Return (start, end) index pairs. An assistant row with tool calls and its tool rows are one unit."""
    spans = []
    index = 0
    while index < len(messages):
        end = index + 1
        if attr(messages[index], "role") == "assistant" and attr(messages[index], "tool_calls"):
            while end < len(messages) and attr(messages[end], "role") == "tool":
                end += 1
        spans.append((index, end))
        index = end
    return spans


def tail_start(messages: list, tail_tokens: int, prefixes: Iterable[str]) -> tuple[int, dict[str, Any] | None]:
    """Return the first tail index and, in the middle of a task, a copy of the latest real user row."""
    prefixes = tuple(prefixes)
    spans = units(messages)
    if not spans:
        return 0, None
    start = spans[-1][0]
    size = sum(estimate_tokens(row) for row in messages[start:])
    for span_start, span_end in reversed(spans[:-1]):
        cost = sum(estimate_tokens(row) for row in messages[span_start:span_end])
        if size + cost > tail_tokens:
            break
        size += cost
        start = span_start
    for index in range(start, len(messages)):
        if is_real_user(messages[index], prefixes):
            return index, None
    latest = next((row for row in reversed(messages[:start]) if is_real_user(row, prefixes)), None)
    if latest is None:
        return start, None
    return start, {"role": "user", "content": copy.deepcopy(latest["content"])}


def copied_user_messages(messages: list, total_chars: int, prefixes: Iterable[str],
                         max_tokens: int | None = None) -> list[str]:
    """Return the newest real user messages that fit in total_chars and in max_tokens (estimated), each cut
    to COPY_EACH characters. The token limit keeps dense text, such as CJK text, inside the window."""
    prefixes = tuple(prefixes)
    chosen: list[str] = []
    used = tokens = 0
    for row in reversed(messages):
        if not is_real_user(row, prefixes):
            continue
        text = plain_text(row.get("content")).strip()[:COPY_EACH]
        cost = estimate_tokens(text)
        if used + len(text) > total_chars or (max_tokens is not None and tokens + cost > max_tokens):
            break
        chosen.append(text)
        used += len(text)
        tokens += cost
    chosen.reverse()
    return chosen


def quote(text: str) -> str:
    """Return the text as a Markdown block quote."""
    return "\n".join(">" + (" " + line if line else "") for line in text.splitlines()) or ">"


def summary_body(summary_text: str, copies: list[str], end_marker: str) -> str:
    """Return the text of the summary row."""
    copied = "\n\n".join(quote(text) for text in copies) if copies else NO_COPIES
    return f"{LEGACY_PREFIX}\n{summary_text.strip()}\n\n{COPY_HEADING}\n\n{copied}\n\n{end_marker}"


def _tail_row(row: Any, marker: str) -> Any:
    clean = copy.deepcopy(row)
    if isinstance(clean, dict):
        clean.pop(marker, None)
    return clean


def build(messages: list, summary_text: str, *, start: int, prepend: dict[str, Any] | None, copy_chars: int,
          header_prefix: str, prefixes: Iterable[str], end_marker: str,
          marker: str = "_db_persisted", copy_tokens: int | None = None) -> list | None:
    """Return the new history, or None when no row comes before the tail."""
    if start <= 0:
        return None
    prefixes = tuple(prefixes)
    header = f"{header_prefix}\n\n{HEADER_TEXT}"
    copies = copied_user_messages(messages[:start], copy_chars, prefixes, copy_tokens)
    body = summary_body(summary_text, copies, end_marker)
    rest = ([copy.deepcopy(prepend)] if prepend else []) + [_tail_row(row, marker) for row in messages[start:]]
    if attr(rest[0], "role") != "user":
        return [{"role": "user", "content": f"{header}\n\n{body}", "_compressed_summary": True}, *rest]
    return [
        {"role": "user", "content": header, "_compressed_summary": True},
        {"role": "assistant", "content": body, "_compressed_summary": True},
        *rest,
    ]
