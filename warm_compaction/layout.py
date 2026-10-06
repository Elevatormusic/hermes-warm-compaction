"""Builds the new history: the summary header, the summary, and the verbatim tail."""

from __future__ import annotations

import copy
import functools
from typing import Any, Iterable

from .handoff import END_MARKER, LEGACY_PREFIX
from .rows import (api_content, attr, compact_json, cut_bounds, cut_middle, estimate_tokens, hermes_value, plain_text,
                   SendPolicy, sent_tokens, visible_text)

HEADER_TEXT = "The summary of the earlier turns follows."
COPY_HEADING = "## Copied user messages"
COPY_EACH = 4_000
MIN_COPY_CHARS = 200
# A payload that even the minimum cut does not fit: it is dropped, and the fallback summary gets all of it.
DROPPED = "[cut]"
CUT_NOTE = "[The middle of a newest row; its start and end stay after the summary.]\n"
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
    """Return True for a summary record: the Hermes summary flag, or (the Hermes session store drops the flag) the
    whole carrier: a known summary prefix with the end marker, or the plugin header row. A user message that only
    starts with a prefix is a real request."""
    if attr(row, "_compressed_summary"):
        return True
    text = plain_text(attr(row, "content")).strip()
    for prefix in prefixes:
        if prefix and text.startswith(prefix):
            rest = text[len(prefix):].strip()
            if END_MARKER in rest or rest == HEADER_TEXT:
                return True
    return False


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


def tail_start(messages: list, tail_tokens: int, prefixes: Iterable[str],
               policy: SendPolicy = SendPolicy()) -> tuple[int, dict[str, Any] | None]:
    """Return the first tail index and, in the middle of a task, a copy of the latest real user row."""
    prefixes = tuple(prefixes)
    spans = units(messages)
    if not spans:
        return 0, None
    start = spans[-1][0]
    # As Hermes sends the rows: the stored display text of a row with api_content is not in the request.
    size = sum(sent_tokens(row, policy) for row in messages[start:])
    for span_start, span_end in reversed(spans[:-1]):
        cost = sum(sent_tokens(row, policy) for row in messages[span_start:span_end])
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
    to COPY_EACH characters (its start and its end). The oldest copy that does not fit is cut to the remaining
    budget. Each copy is the text that Hermes sent (api_content when the row has it). The token limit counts
    the quoted form and keeps dense text, such as CJK text, inside the window."""
    prefixes = tuple(prefixes)
    chosen: list[str] = []
    used = tokens = 0
    for row in reversed(messages):
        if not is_real_user(row, prefixes):
            continue
        text = cut_middle(visible_text(api_content(row)).strip(), COPY_EACH)
        cost = estimate_tokens(quote(text))
        if used + len(text) > total_chars or (max_tokens is not None and tokens + cost > max_tokens):
            part = _fit_copy(text, total_chars - used, None if max_tokens is None else max_tokens - tokens)
            if part is not None:
                chosen.append(part)
            break
        chosen.append(text)
        used += len(text)
        tokens += cost
    chosen.reverse()
    return chosen


def _fit_copy(text: str, chars: int, tokens: int | None) -> str | None:
    """Return the start and the end of the text in at most chars characters and, in the quoted form, at most
    tokens estimated tokens. Return None when less than MIN_COPY_CHARS characters fit."""
    limit = min(len(text), chars)
    while limit >= MIN_COPY_CHARS:
        part = cut_middle(text, limit)
        cost = estimate_tokens(quote(part))
        if tokens is None or cost <= tokens:
            return part
        limit = min(limit - 1, limit * tokens // cost)
    return None


def fit_user_row(row: dict[str, Any], tokens: int, removed: list | None = None) -> dict[str, Any]:
    """Return the user row cut to its start and end, at most tokens estimated tokens (the text that Hermes
    sent; an attachment becomes a mark). At least MIN_COPY_CHARS characters stay: the tail starts with it. With a
    removed list, a cut adds one user row to it: the removed middle after CUT_NOTE."""
    text = visible_text(api_content(row)).strip()
    limit = len(text)
    while True:
        out = {"role": "user", "content": cut_middle(text, limit)}
        if attr(row, "name"):
            out["name"] = attr(row, "name")
        cost = estimate_tokens(out)
        if cost <= tokens or limit <= MIN_COPY_CHARS:
            if removed is not None and limit < len(text):
                first, second = cut_bounds(text, limit)
                removed.append({"role": "user", "content": CUT_NOTE + text[first:second]})
            return out
        limit = max(MIN_COPY_CHARS, min(limit - 1, limit * max(tokens, 0) // cost))


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


# The sent text fields of an assistant row that a model can make large: its reasoning.
REASONING_KEYS = ("reasoning_content", "reasoning")


def _tail_parts(row: Any) -> list[tuple[tuple, Any]]:
    """Return the sent payloads of a tail row that a cut can make smaller, as (key, value): the content (text, or
    each part of a list), the arguments of each tool call, and the reasoning text of an assistant row."""
    if not isinstance(row, dict):
        return []
    role = row.get("role")
    content = api_content(row) if role in ("user", "assistant") else row.get("content")
    parts: list[tuple[tuple, Any]] = []
    if isinstance(content, str):
        parts.append((("content",), content))
    elif isinstance(content, list):
        for index, item in enumerate(content):
            # The text shapes of visible_text: a raw string, and a text dictionary with or without its type.
            if isinstance(item, str):
                parts.append((("content", index), item))
            elif isinstance(item, dict) and item.get("type", "text") == "text" and isinstance(item.get("text"), str):
                parts.append((("content", index), item["text"]))
            elif isinstance(item, dict):
                parts.append((("media", index), item))
    if role == "assistant":
        for index, call in enumerate(row.get("tool_calls") or ()):
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and function.get("arguments") is not None:
                arguments = function["arguments"]
                parts.append((("arguments", index), arguments if isinstance(arguments, str) else compact_json(arguments)))
        parts.extend(((key,), row[key]) for key in REASONING_KEYS if isinstance(row.get(key), str))
    return parts


def _set_part(row: dict[str, Any], key: tuple, value: Any) -> dict[str, Any]:
    """Return a copy of the row with one payload replaced. A changed content drops the api_content sidecar: the
    new content is the sent text."""
    new = dict(row)
    if key[0] in ("content", "media"):
        content = api_content(row) if row.get("role") in ("user", "assistant") else row.get("content")
        new.pop("api_content", None)
        if len(key) == 1:
            new["content"] = value
        else:
            items = list(content)
            item = items[key[1]]
            if key[0] == "media" or isinstance(item, str):
                items[key[1]] = value
            else:
                items[key[1]] = {**item, "text": value}
            new["content"] = items
    elif key[0] == "arguments":
        calls = [dict(call) for call in row["tool_calls"]]
        # A cut JSON text is not JSON: the start and the end go into a JSON object, so that the call stays valid.
        calls[key[1]] = {**calls[key[1]], "function": {**calls[key[1]]["function"],
                                                       "arguments": compact_json({"truncated_arguments": value})}}
        new["tool_calls"] = calls
    else:
        new[key[0]] = value
    return new


def bound_tail(rows: list, tokens: int, removed: list | None = None, policy: SendPolicy = SendPolicy()) -> list:
    """Return the tail rows in about tokens estimated tokens. The tail keeps whole units, so the newest unit can
    be larger than the tail budget (a large user message, assistant reply, tool call, or tool result). Then the
    largest sent payloads (_tail_parts) are cut to their start and end, until the rows fit or no payload has more
    than MIN_COPY_CHARS characters. A tool call keeps its id and name, and its arguments stay a JSON object. A
    media part (an image, for example) is replaced by a short text. Other rows and fields stay as they are; signed
    reasoning_details stay, because a cut breaks the signature. When the minimum cuts are not enough (a small cap,
    or many small payloads), the payloads are dropped (DROPPED), largest first. Only the row structure (roles, tool
    call ids and names) can then stay above the cap.

    With a removed list, one row for each cut payload is added to it: the removed middle after CUT_NOTE, with the
    role (and the tool call id) of the row. The fallback summary can then keep what the tail cuts."""
    rows = list(rows)
    originals: dict[tuple, Any] = {}
    current: dict[tuple, Any] = {}
    limits: dict[tuple, int] = {}
    # First the cuts to start and end (at least MIN_COPY_CHARS), then, when they are not enough, the drops.
    floor = MIN_COPY_CHARS
    while sum(sent_tokens(row, policy) for row in rows) > tokens:
        cuttable = []
        for index, row in enumerate(rows):
            for key, value in _tail_parts(row):
                value = current.get((index, key), value)
                if (isinstance(value, str) and len(value) > max(floor, len(DROPPED))) or (
                        key[0] == "media" and (index, key) not in current
                        and estimate_tokens(value) > floor // 4):
                    cuttable.append((estimate_tokens(value), index, key, value))
        if not cuttable:
            if floor:
                floor = 0
                continue
            break
        cost, index, key, value = max(cuttable, key=lambda item: (item[0], -item[1]))
        originals.setdefault((index, key), value)
        if key[0] == "media":
            new_value: Any = {"type": "text", "text": f"[{value.get('type') or 'media'} removed]"}
        elif not floor:
            limits[(index, key)] = 0
            new_value = DROPPED
        else:
            target = cost - (sum(sent_tokens(row, policy) for row in rows) - tokens)
            limit = max(MIN_COPY_CHARS, min(len(value) - 1, len(value) * max(target, 0) // max(cost, 1)))
            # Cut the original text again: the kept start and end are then parts of the original text.
            limits[(index, key)] = limit
            new_value = cut_middle(originals[(index, key)], limit)
        current[(index, key)] = new_value
        rows[index] = _set_part(rows[index], key, new_value)
    if removed is not None:
        for index, key in sorted(originals, key=lambda item: (item[0], repr(item[1]))):
            original = originals[(index, key)]
            if key[0] == "media":
                text = f"(a {original.get('type') or 'media'} part was removed)"
            else:
                first, second = cut_bounds(original, limits[(index, key)])
                label = {"arguments": "tool call arguments: ", "reasoning_content": "reasoning: ",
                         "reasoning": "reasoning: "}.get(key[0], "")
                text = label + original[first:second]
            part = {"role": attr(rows[index], "role"), "content": CUT_NOTE + text}
            if part["role"] == "tool":
                part["tool_call_id"] = attr(rows[index], "tool_call_id")
            removed.append(part)
    return rows

def build(messages: list, summary_text: str, *, start: int, prepend: dict[str, Any] | None, copy_chars: int,
          header_prefix: str, prefixes: Iterable[str], end_marker: str,
          marker: str = "_db_persisted", copy_tokens: int | None = None, tail_tokens: int | None = None,
          policy: SendPolicy = SendPolicy()) -> list | None:
    """Return the new history, or None when no row comes before the tail. With tail_tokens, a tail above it
    is cut to it (bound_tail)."""
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
    tail = [_tail_row(row, marker) for row in messages[start:]]
    if tail_tokens is not None:
        tail = bound_tail(tail, tail_tokens, policy=policy)
    rest = ([copy.deepcopy(prepend)] if prepend else []) + tail
    if attr(rest[0], "role") != "user":
        return [{"role": "user", "content": f"{header}\n\n{body}", "_compressed_summary": True}, *rest]
    return [
        {"role": "user", "content": header, "_compressed_summary": True},
        {"role": "assistant", "content": body, "_compressed_summary": True},
        *rest,
    ]
