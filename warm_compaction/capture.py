"""The capture store. It joins the request hooks and the execution middleware by api_request_id."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from .rows import attr, plain_text, row_digest, tool_calls_of

MAX_OPEN = 8
MAX_SESSIONS = 16
CLIENT_OPTIONS = ("extra_body", "extra_headers", "extra_query", "timeout")
CAPTURE_FINISH = ("stop", "tool_calls")


class UnsupportedRequest(ValueError):
    """The request body cannot become a warm request. The message is the reason code."""


def final_body(request: Any) -> dict[str, Any]:
    """Return the JSON body that the OpenAI client sends: a copy with extra_body merged in."""
    if not isinstance(request, dict):
        raise UnsupportedRequest("request_not_mapping")
    if request.get("extra_headers") or request.get("extra_query"):
        raise UnsupportedRequest("request_options_unsupported")
    extra = request.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise UnsupportedRequest("request_options_unsupported")
    body = {key: value for key, value in request.items() if key not in CLIENT_OPTIONS}
    body.update(extra)
    try:
        return json.loads(json.dumps(body, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise UnsupportedRequest("request_not_json") from error


def unique_call_ids(calls: list[list[str]]) -> list[list[str]]:
    """Give a repeated tool-call id the name that Hermes gives it after this hook runs: the second c1 becomes
    c1_d2, the third c1_d3 (agent.message_sanitization.uniquify_tool_call_ids of Hermes 45871e10)."""
    seen: set[str] = set()
    out = []
    for call_id, name in calls:
        if call_id and call_id in seen:
            call_id = next(f"{call_id}_d{n}" for n in range(2, len(seen) + 3) if f"{call_id}_d{n}" not in seen)
        if call_id:
            seen.add(call_id)
        out.append([call_id, name])
    return out


class CaptureStore:
    """Keeps the latest usable main-model request of each session, in memory only."""

    def __init__(self, max_open: int = MAX_OPEN, max_sessions: int = MAX_SESSIONS,
                 clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.Lock()
        self._open: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_open = max_open
        self._max_sessions = max_sessions
        self._clock = clock

    def on_pre_api_request(self, api_request_id: Any = None, session_id: Any = None,
                           conversation_history: Any = None, model: Any = None, base_url: Any = None,
                           api_mode: Any = None, **_: Any) -> None:
        """Hook pre_api_request: keep the route and one digest for each stored history row."""
        if not api_request_id or not session_id or not isinstance(conversation_history, list):
            return None
        try:
            digests = [row_digest(row) for row in conversation_history]
        except Exception:
            return None
        entry = {"session_id": str(session_id), "route": (model, base_url, api_mode), "digests": digests, "body": None}
        with self._lock:
            self._open[str(api_request_id)] = entry
            self._open.move_to_end(str(api_request_id))
            while len(self._open) > self._max_open:
                self._open.popitem(last=False)
        return None

    def on_llm_execution(self, request: Any = None, next_call: Any = None, api_request_id: Any = None,
                         **_: Any) -> Any:
        """Middleware llm_execution: keep the final request body, then run the request unchanged."""
        try:
            self._keep_body(api_request_id, request)
        except Exception:
            pass
        return next_call()

    def _keep_body(self, api_request_id: Any, request: Any) -> None:
        with self._lock:
            entry = self._open.get(str(api_request_id or ""))
        if entry is None or entry["route"][2] != "chat_completions":
            return
        try:
            body = final_body(request)
        except UnsupportedRequest:
            body = None
        with self._lock:
            entry["body"] = body

    def on_post_api_request(self, api_request_id: Any = None, session_id: Any = None, finish_reason: Any = None,
                            assistant_message: Any = None, usage: Any = None, **_: Any) -> None:
        """Hook post_api_request: make the request the capture of its session when the reply is usable.

        The capture also keeps the prompt token count that the server measured for the request, when known.
        """
        with self._lock:
            entry = self._open.pop(str(api_request_id or ""), None)
        if entry is None or str(session_id or "") != entry["session_id"]:
            return None
        if finish_reason not in CAPTURE_FINISH or assistant_message is None:
            return None
        try:
            reply = {
                "role": "assistant",
                "content": plain_text(attr(assistant_message, "content")),
                "tool_calls": unique_call_ids(
                    [[call_id, name] for call_id, name, _arguments in tool_calls_of(assistant_message)]),
            }
        except Exception:
            return None
        measured = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(measured) is not int or measured <= 0:
            measured = None
        capture = {**entry, "reply": reply, "finish_reason": finish_reason, "prompt_tokens": measured,
                   "captured_at": self._clock()}
        with self._lock:
            self._sessions[entry["session_id"]] = capture
            self._sessions.move_to_end(entry["session_id"])
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        return None

    def latest(self, session_id: Any) -> dict[str, Any] | None:
        """Return a copy of the latest capture of a session, or None."""
        key = str(session_id or "")
        with self._lock:
            capture = self._sessions.get(key)
            if capture is None:
                return None
            self._sessions.move_to_end(key)
            return copy.deepcopy(capture)

    def forget(self, session_id: Any = None, **_: Any) -> None:
        """Hooks on_session_finalize and on_session_reset: remove the data of a session."""
        key = str(session_id or "")
        with self._lock:
            self._sessions.pop(key, None)
            for request_id in [rid for rid, entry in self._open.items() if entry["session_id"] == key]:
                del self._open[request_id]
        return None
