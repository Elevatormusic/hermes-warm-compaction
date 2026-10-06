"""Synthetic rows and captures for the warm_compaction unit tests."""

from __future__ import annotations

import copy

from warm_compaction.rows import row_digest

ROUTE = ("fake-model", "http://127.0.0.1:9/v1", "chat_completions")
SYSTEM = {"role": "system", "content": "You are a test agent."}
HEADINGS_TEXT = (
    "## Goal\nFinish the test task.\n\n"
    "## User instructions\n- \"Use short names.\"\n\n"
    "## Current state\n- [OPEN] Write the report.\n\n"
    "## Key facts\n- The file is report.txt.\n\n"
    "## Next step\nWrite report.txt."
)


def user(text, **extra):
    """Return a user row."""
    return {"role": "user", "content": text, **extra}


def assistant(text, calls=(), **extra):
    """Return an assistant row. Each call is (id, name, arguments)."""
    row = {"role": "assistant", "content": text, **extra}
    if calls:
        row["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
            for call_id, name, arguments in calls
        ]
    return row


def tool(call_id, text, name=None):
    """Return a tool result row."""
    row = {"role": "tool", "tool_call_id": call_id, "content": text}
    if name:
        row["name"] = name
    return row


def wire(rows):
    """Return the request form of stored rows: the same fields without private keys."""
    return [{key: copy.deepcopy(value) for key, value in row.items() if not key.startswith("_")} for row in rows]


def capture_for(rows, reply, *, route=ROUTE, body_extra=None, session="s1", system=SYSTEM):
    """Return a capture in the form that CaptureStore.latest() gives."""
    body = {"model": route[0], "messages": [copy.deepcopy(system), *wire(rows)], **(body_extra or {})}
    return {
        "session_id": session,
        "route": tuple(route),
        "digests": [row_digest(row) for row in rows],
        "body": body,
        "reply": {
            "role": "assistant",
            "content": reply.get("content") or "",
            "tool_calls": [[call["id"], call["function"]["name"]] for call in reply.get("tool_calls", [])],
        },
        "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop",
        "captured_at": 0.0,
    }
