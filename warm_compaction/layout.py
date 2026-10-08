"""Builds the new history: the summary header, the summary, and the verbatim tail."""

from __future__ import annotations

import copy
import functools
import heapq
import re
from typing import Any
from collections.abc import Iterable

from .handoff import END_MARKER, LEGACY_PREFIX
from .rows import (api_content, attr, compact_json, cut_bounds, cut_middle, estimate_tokens, hermes_value, plain_text,
                   SendPolicy, attachment_mark, has_thought_signature, sent_reasoning_key, sent_tokens,
                   visible_text)

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
# Hermes 45871e10 rows from templates with variable parts: the whole text must match the template (Hermes itself
# matches only the start; a user message that only starts like one is a real request). The fixed nudges are in
# HOST_NUDGES.
_DROPPED_TOOLS_TAIL = (
    " was too large and the stream timed out before it could be delivered. Do NOT retry the same tool call with "
    "the same large content. Instead, break the content into multiple smaller tool calls (e.g. use multiple patch "
    "calls or write smaller files). Each tool call's arguments must be under ~8K tokens to avoid stream timeouts. "
    "The cut was a transport interruption, not a capability change \u2014 your tools remain fully available.]")
SYNTHETIC_TEMPLATES = (
    # agent.conversation_loop._get_continuation_prompt, with the dropped tool names.
    re.compile(re.escape("[System: Your previous tool call ") + r"\([^\n]*\)" + re.escape(_DROPPED_TOOLS_TAIL)),
    # tools.process_registry_notifications.format_process_notification (and the gateway copy): a completion or a
    # watch match, an optional attribution line, the command, and the output.
    re.compile(r'\[IMPORTANT: Background process \S+ (?:matched watch pattern "[^\n]*"|[^\n]*\(exit code [^\n]*\))'
               r"\.\n(?:[^\n]*\n)?Command: [^\n]*\n(?:Matched output|Output):\n.*\]", re.S),
    # tools.todo_tool.TodoStore.format_for_injection: the header and one line for each task.
    re.compile(re.escape("[Your active task list was preserved across context compression]")
               + r"(?:\n *- \[[^\]\n]*\] [^\n]*)+"),
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


# Hermes 45871e10 (agent.context_compressor._INFLIGHT_TASK_REPLAY_HEADER): after the end marker of a carrier,
# Hermes can restate the active request that was not finished.
INFLIGHT_REPLAY_HEADER = ("[STILL IN PROGRESS \u2014 this is the active request, restated after the compaction "
                          "boundary because it was not finished yet. Continue it; do not start over.]")


def _ends_carrier(text: str) -> bool:
    """True when the text ends as a summary carrier: its last end-marker line (a whole line that starts with
    END_MARKER and ends with ---, as each Hermes version writes it) has nothing after it, or only the Hermes
    restatement of the active request. The marker inside a sentence, or a request after it, is user text."""
    lines = text.split("\n")
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        if line.startswith(END_MARKER) and line.endswith("---"):
            after = "\n".join(lines[index + 1:]).strip()
            replay = hermes_value("agent.context_compressor", "_INFLIGHT_TASK_REPLAY_HEADER", INFLIGHT_REPLAY_HEADER)
            return not after or after.startswith(replay)
    return False


def is_summary(row: Any, prefixes: Iterable[str]) -> bool:
    """Return True for a summary record: the Hermes summary flag, or (the Hermes session store drops the flag) the
    whole carrier: a known summary prefix and an end marker that ends it (_ends_carrier), or the plugin header row.
    A user message that only starts with a prefix, or has the marker in its text, is a real request."""
    if attr(row, "_compressed_summary"):
        return True
    text = plain_text(attr(row, "content")).strip()
    for prefix in prefixes:
        if prefix and text.startswith(prefix):
            rest = text[len(prefix):].strip()
            if rest == HEADER_TEXT or _ends_carrier(rest):
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
    if not text or text in nudge_texts() or any(template.fullmatch(text) for template in SYNTHETIC_TEMPLATES):
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


# Between two copied messages in the summary row.
COPY_SEPARATOR = "\n\n"


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
        # With the separator in front of it (summary_body joins the quotes with a blank line).
        cost = estimate_tokens(COPY_SEPARATOR + quote(text))
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
        cost = estimate_tokens(COPY_SEPARATOR + quote(part))
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
    copied = COPY_SEPARATOR.join(quote(escape_markers(text, end_marker)) for text in copies) if copies else NO_COPIES
    summary = escape_markers(summary_text.strip(), end_marker)
    return f"{LEGACY_PREFIX}\n{summary}\n\n{COPY_HEADING}\n\n{copied}\n\n{end_marker}"


def _tail_row(row: Any, marker: str) -> Any:
    clean = copy.deepcopy(row)
    if isinstance(clean, dict):
        clean.pop(marker, None)
    return clean


def _tail_parts(row: Any, policy: SendPolicy = SendPolicy()) -> list[tuple[tuple, Any]]:
    """Return sent payloads that can be cut without breaking native replay."""
    if policy.native_mode in ("codex_responses", "anthropic_messages") and any(
            attr(row, key) for key in ("codex_message_items", "codex_reasoning_items", "reasoning_details",
                                      "anthropic_content_blocks")):
        # The canonical text and tool calls must still agree with their signed or encrypted replay blocks.
        return []
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
            # Gemini needs a function call with a thought signature back as it was: on a route that sends the
            # signature, its arguments are not a payload.
            if policy.signatures and has_thought_signature(call.get("extra_content") if isinstance(call, dict)
                                                           else None):
                continue
            if isinstance(function, dict) and function.get("arguments") is not None:
                arguments = function["arguments"]
                text = arguments if isinstance(arguments, str) else compact_json(arguments)
                parts.append((("arguments", index), text))
        key = sent_reasoning_key(row, policy) if policy.cut_reasoning else None
        if key is not None:
            parts.append(((key,), row[key]))
    return parts


def _set_parts(row: dict[str, Any], changes: dict[tuple, Any]) -> dict[str, Any]:
    """Return a copy of the row with payloads replaced (key: value, with the keys of _tail_parts), in one copy. A
    changed content drops the api_content sidecar: the new content is the sent text."""
    new = dict(row)
    content_keys = [key for key in changes if key[0] in ("content", "media")]
    if content_keys:
        content = api_content(row) if row.get("role") in ("user", "assistant") else row.get("content")
        new.pop("api_content", None)
        if ("content",) in changes:
            new["content"] = changes[("content",)]
        else:
            items = list(content)
            for key in content_keys:
                item = items[key[1]]
                if key[0] == "media" or isinstance(item, str):
                    items[key[1]] = changes[key]
                else:
                    items[key[1]] = {**item, "text": changes[key]}
            new["content"] = items
    argument_keys = [key for key in changes if key[0] == "arguments"]
    if argument_keys:
        calls = [dict(call) for call in row["tool_calls"]]
        for key in argument_keys:
            # A cut JSON text is not JSON: the start and the end go into a JSON object, so that the call stays
            # valid.
            calls[key[1]] = {**calls[key[1]], "function": {
                **calls[key[1]]["function"], "arguments": compact_json({"truncated_arguments": changes[key]})}}
        new["tool_calls"] = calls
    for key, value in changes.items():
        if key[0] not in ("content", "media", "arguments"):
            new[key[0]] = value
    return new


def _cuttable(key: tuple, value: Any, floor: int, marked: bool) -> bool:
    """True when a cut can make the payload smaller: a text above the floor, or a media part not yet marked."""
    if key[0] == "media":
        return not marked and estimate_tokens(value) > floor // 4
    return isinstance(value, str) and len(value) > max(floor, len(DROPPED))


def bound_tail(rows: list, tokens: int, removed: list | None = None, policy: SendPolicy = SendPolicy()) -> list:
    """Return the tail rows in about tokens estimated tokens. The tail keeps whole units, so the newest unit can
    be larger than the tail budget (a large user message, assistant reply, tool call, or tool result). Then the
    largest sent payloads (_tail_parts) are cut to their start and end, until the rows fit or no payload has more
    than MIN_COPY_CHARS characters. A tool call keeps its id and name, and its arguments stay a JSON object. A
    media part (an image, for example) is replaced by its attachment mark. Other rows and fields stay as they are;
    signed reasoning_details stay, because a cut breaks the signature. When the minimum cuts are not enough (a small
    cap, or many small payloads), the payloads are dropped (DROPPED), largest first. Only the row structure (roles,
    tool call ids and names) can then stay above the cap.

    With a removed list, one row for each cut payload is added to it: the removed middle after CUT_NOTE, with the
    role (and the tool call id) of the row. The fallback summary can then keep what the tail cuts."""
    base = list(rows)
    rows = list(rows)
    parts = {(index, key): value for index, row in enumerate(base) for key, value in _tail_parts(row, policy)}
    originals: dict[tuple, Any] = {}
    current: dict[tuple, Any] = {}
    limits: dict[tuple, int] = {}
    # First the cuts to start and end (at least MIN_COPY_CHARS), then, when they are not enough, the drops. Each
    # round measures the rows one time, cuts the largest payloads (a heap) until the estimated excess is gone, and
    # copies each changed row one time: a row with many parts does not make the cut quadratic.
    floor = MIN_COPY_CHARS
    while True:
        excess = sum(sent_tokens(row, policy) for row in rows) - tokens
        if excess <= 0:
            break
        heap = []
        for order, (place, value) in enumerate(parts.items()):
            value = current.get(place, value)
            if _cuttable(place[1], value, floor, place in current):
                heap.append((-estimate_tokens(value), place[0], order, place[1]))
        if not heap:
            if floor:
                floor = 0
                continue
            break
        heapq.heapify(heap)
        changed: set[int] = set()
        while heap and excess > 0:
            negative, index, _order, key = heapq.heappop(heap)
            cost = -negative
            value = current.get((index, key), parts[(index, key)])
            originals.setdefault((index, key), value)
            if key[0] == "media":
                # The mark keeps a web URL or a file name (short), not a data URL.
                new_value: Any = {"type": "text", "text": attachment_mark(value) + " (removed)"}
            elif not floor:
                limits[(index, key)] = 0
                new_value = DROPPED
            else:
                target = cost - excess
                limit = max(MIN_COPY_CHARS, min(len(value) - 1, len(value) * max(target, 0) // max(cost, 1)))
                # Cut the original text again: the kept start and end are then parts of the original text.
                limits[(index, key)] = limit
                new_value = cut_middle(originals[(index, key)], limit)
            current[(index, key)] = new_value
            changed.add(index)
            excess -= cost - estimate_tokens(new_value)
        by_row: dict[int, dict[tuple, Any]] = {}
        for (index, key), value in current.items():
            if index in changed:
                by_row.setdefault(index, {})[key] = value
        for index, changes in by_row.items():
            rows[index] = _set_parts(base[index], changes)
    if removed is not None:
        for index, key in sorted(originals, key=lambda item: (item[0], repr(item[1]))):
            original = originals[(index, key)]
            if key[0] == "media":
                text = f"(removed from the tail: {attachment_mark(original)})"
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
            # As Hermes sends it: api_content in place of the stored display text.
            copy_tokens = max(copy_tokens - sent_tokens(prepend, policy), 0)
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
