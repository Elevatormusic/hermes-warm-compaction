"""Keep the newest visible answer of a running turn inside the tail.

Hermes will not put a folded answer back at the end of a compressed candidate when a row before or after the
slot is an assistant row (the follower scan in agent/conversation_compression.py, which logs
"compression removed the visible reply"). The last answer of a turn that is still running would then vanish
from the display window while the work goes on. The plugin moves the fold point back to that answer instead,
so the answer stays in the tail as a real row. This mirrors the built-in compressor, which keeps its last
assistant row inside the tail.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from . import layout
from .rows import estimate_tokens, plain_text

logger = logging.getLogger("warm_compaction")


def last_visible_reply(messages: list, prefixes, before: int) -> int | None:
    """Return the index of the newest plain assistant answer before index ``before``, or None.

    A row with tool calls, a summary row, and a row without visible text are not answers: none of them is
    the text the user reads.
    """
    for index in range(before - 1, -1, -1):
        row = messages[index]
        if row.get("role") != "assistant" or row.get("tool_calls") or layout.is_summary(row, prefixes):
            continue
        if (plain_text(row.get("content")) or "").strip():
            return index
    return None


def apply(messages: list, start: int, prepend: dict[str, Any] | None, prefixes,
          tail_cap: int) -> tuple[int, dict[str, Any] | None]:
    """Move the fold point back to the newest visible answer when the fold would remove it.

    Give up and return the given point and copy unchanged when there is no such answer before the fold, when
    the row after the answer is not an assistant row (Hermes puts the answer back at the tail itself in that
    case), when the answer alone does not fit the tail cap (the same give-up rule as the built-in
    compressor), or when the answer is the first row. Otherwise the new tail starts at the answer, and the
    request copy follows the layout rule: a copy of the newest real user row before it.
    """
    reply = last_visible_reply(messages, prefixes, start)
    if reply is None or reply <= 0:
        return start, prepend
    follower = messages[reply + 1] if reply + 1 < len(messages) else None
    if follower is None or follower.get("role") != "assistant":
        # The turn does not run on past the answer: Hermes can put the answer back without an assistant row
        # on either side of the slot, so the old point stands.
        return start, prepend
    if estimate_tokens(messages[reply].get("content")) >= tail_cap:
        logger.info("Warm compaction tail guard yields: the answer alone does not fit the tail")
        return start, prepend
    latest = None
    for index in range(reply - 1, -1, -1):
        if layout.is_real_user(messages[index], prefixes):
            latest = messages[index]
            break
    new_prepend: dict[str, Any] | None = None
    if latest is not None:
        new_prepend = {"role": "user", "content": copy.deepcopy(latest["content"])}
        for key in ("name", "api_content"):
            if latest.get(key):
                new_prepend[key] = copy.deepcopy(latest[key])
    logger.info("Warm compaction tail guard: the fold stops before row %d instead of %d", reply, start)
    return reply, new_prepend
