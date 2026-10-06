"""Builds the new history: the summary header, the summary, and the verbatim tail."""

from __future__ import annotations

import copy
import functools
from typing import Any, Iterable

from .handoff import END_MARKER, LEGACY_PREFIX
from .rows import api_content, attr, estimate_tokens, hermes_value, plain_text, visible_text

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
# Hermes recovery nudges and continuation markers. The Hermes compressor matches them by exact text
# (ContextCompressor._is_synthetic_compression_user_turn). The Hermes value applies when Hermes has it; the
# text here is the value of Hermes 45871e10.
_COMPRESSOR = "agent.context_compressor"
_LOOP = "agent.conversation_loop"
HOST_NUDGES = (
    (_COMPRESSOR, "COMPRESSION_CONTINUATION_USER_CONTENT",
     "Continue from the compressed conversation context above. This marker exists because no human user turn "
     "was available."),
    (_COMPRESSOR, "_LEGACY_COMPRESSION_CONTINUATION_USER_CONTENT",
     "Continue from the compressed conversation context above. This marker exists because the compacted "
     "transcript contained no preserved user turn."),
    (_COMPRESSOR, "MAX_ITERATIONS_SUMMARY_REQUEST",
     "You've reached the maximum number of tool-calling iterations allowed. Please provide a final response "
     "summarizing what you've found and accomplished so far, without calling any more tools."),
    (_LOOP, "_CODEX_INCOMPLETE_NUDGE",
     "[System: Your previous response contained only internal reasoning and never produced a visible answer or "
     "tool call. Do not keep thinking. Produce your final answer as plain text now (or make the tool call you "
     "were planning).]"),
    (_LOOP, "_CODEX_ACK_CONTINUATION_NUDGE",
     "[System: Continue now. Execute the required tool calls and only send your final answer after completing "
     "the task.]"),
    (_LOOP, "_DEGENERATE_FINAL_NUDGE",
     "[System: Your previous message ended the turn with a fragment that is not a usable answer. If the task is "
     "unfinished, continue it and then give the complete answer. If that fragment WAS your complete answer, send "
     "it again exactly as before.]"),
    (_LOOP, "_DROPPED_TOOLCALL_NUDGE_CONTENT",
     "Your previous turn indicated a tool call but none was included. Do not narrate a plan or restate intent — "
     "issue the actual tool call now to continue the task."),
    (_LOOP, "_EMPTY_TOOL_RESPONSE_NUDGE",
     "You just executed tool calls but returned an empty response. Please process the tool results above and "
     "continue with the task."),
    (_LOOP, "_LENGTH_CONTINUATION_NETWORK_STUB",
     "[System: The previous response was cut off by a network error mid-stream — a transport interruption, NOT a "
     "change in your capabilities. Your tools are still fully available; call them as normal and ignore any "
     "earlier claim that you lack tool access. Continue the task from where you left off. Do not restart or "
     "repeat prior text.]"),
    (_LOOP, "_LEGACY_LENGTH_CONTINUATION_NETWORK_STUB",
     "[System: The previous response was cut off by a network error mid-stream. Continue exactly where you left "
     "off. Do not restart or repeat prior text. Finish the answer directly.]"),
    (_LOOP, "_LENGTH_CONTINUATION_OUTPUT_LIMIT",
     "[System: Your previous response was truncated by the output length limit. Continue exactly where you left "
     "off. Do not restart or repeat prior text. Finish the answer directly.]"),
)


@functools.lru_cache(maxsize=1)
def nudge_texts() -> frozenset[str]:
    """Return the exact texts of the Hermes recovery nudges. Read one time: the values do not change."""
    return frozenset(hermes_value(module, name, default).strip() for module, name, default in HOST_NUDGES)


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
    text = visible_text(row.get("content")).strip()
    if not text or text.startswith(SYNTHETIC_PREFIXES) or text in nudge_texts():
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
    prepend = {"role": "user", "content": copy.deepcopy(latest["content"])}
    for key in ("name", "api_content"):
        if latest.get(key):
            prepend[key] = copy.deepcopy(latest[key])
    return start, prepend


def copied_user_messages(messages: list, total_chars: int, prefixes: Iterable[str],
                         max_tokens: int | None = None) -> list[str]:
    """Return the newest real user messages that fit in total_chars and in max_tokens (estimated), each cut
    to COPY_EACH characters. Each copy is the text that Hermes sent (api_content when the row has it). The token
    limit counts the quoted form and keeps dense text, such as CJK text, inside the window."""
    prefixes = tuple(prefixes)
    chosen: list[str] = []
    used = tokens = 0
    for row in reversed(messages):
        if not is_real_user(row, prefixes):
            continue
        text = visible_text(api_content(row)).strip()[:COPY_EACH]
        cost = estimate_tokens(quote(text))
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


def escape_markers(text: str, end_marker: str) -> str:
    """Break each end marker in the text, so that only the real end marker ends the summary row. A reader
    splits the row at the first end marker."""
    for marker in (end_marker, END_MARKER):
        if marker:
            text = text.replace(marker, marker[0] + " " + marker[1:])
    return text


def summary_body(summary_text: str, copies: list[str], end_marker: str) -> str:
    """Return the text of the summary row."""
    copied = "\n\n".join(quote(escape_markers(text, end_marker)) for text in copies) if copies else NO_COPIES
    summary = escape_markers(summary_text.strip(), end_marker)
    return f"{LEGACY_PREFIX}\n{summary}\n\n{COPY_HEADING}\n\n{copied}\n\n{end_marker}"


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
    earlier = messages[:start]
    if prepend:
        # The prepended row is the latest real user row before the tail. Do not copy it a second time, and
        # count it in the copy budget.
        last = max((index for index, row in enumerate(earlier) if is_real_user(row, prefixes)), default=len(earlier))
        earlier = earlier[:last]
        if copy_tokens is not None:
            copy_tokens = max(copy_tokens - estimate_tokens(prepend), 0)
    copies = copied_user_messages(earlier, copy_chars, prefixes, copy_tokens)
    body = summary_body(summary_text, copies, end_marker)
    rest = ([copy.deepcopy(prepend)] if prepend else []) + [_tail_row(row, marker) for row in messages[start:]]
    if attr(rest[0], "role") != "user":
        return [{"role": "user", "content": f"{header}\n\n{body}", "_compressed_summary": True}, *rest]
    return [
        {"role": "user", "content": header, "_compressed_summary": True},
        {"role": "assistant", "content": body, "_compressed_summary": True},
        *rest,
    ]
