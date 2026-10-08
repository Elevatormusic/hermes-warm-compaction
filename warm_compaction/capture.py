"""The capture store. It joins the request hooks and the execution middleware by api_request_id."""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any
from collections.abc import Callable

from .rows import attr, plain_text, row_digest, tool_calls_of

MAX_OPEN = 8
MAX_SESSIONS = 16
CLIENT_OPTIONS = ("extra_body", "extra_headers", "extra_query", "timeout")
CAPTURE_FINISH = ("stop", "tool_calls")
REQUEST_REFUSALS = ("request_not_mapping", "request_options_unsupported", "request_not_json")
# A random salt for each plugin load: the stamps of one key differ between processes.
_KEY_STAMP_SECRET = secrets.token_bytes(32)
# PBKDF2 work for each stamp (about 3 ms): a stamp is made a few times for each main request.
KEY_STAMP_ROUNDS = 10_000


def key_stamp(key: str) -> str:
    """A digest of a resolved API key. A capture keeps it, never the key: the capture belongs to the key that
    sent it. PBKDF2-HMAC-SHA-256 with the salt of this plugin load; the salt and the stamps stay in memory."""
    return hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), _KEY_STAMP_SECRET, KEY_STAMP_ROUNDS).hex()


class UnsupportedRequest(ValueError):
    """The request body cannot become a warm request. The message is the reason code."""


def request_headers(request: Any, api_mode: str = "chat_completions") -> dict[str, str]:
    """Copy known session and API headers. Refuse other headers and unsafe HTTP values.

    Hermes 1d6b2786 adds x-opencode-session for OpenCode routes. Keep the captured value: a new session id
    need not select the same server. Header values stay in memory and never go to status or logs.
    """
    if not isinstance(request, dict):
        raise UnsupportedRequest("request_not_mapping")
    headers = request.get("extra_headers")
    if headers is None:
        return {}
    allowed = {"x-opencode-session"}
    if api_mode == "codex_responses":
        allowed.update(("session_id", "x-client-request-id", "x-grok-conv-id", "x-initiator"))
    elif api_mode == "anthropic_messages":
        allowed.update(("anthropic-beta", "anthropic-version", "x-initiator"))
    if not isinstance(headers, dict) or len(headers) > len(allowed):
        raise UnsupportedRequest("request_options_unsupported")
    seen: set[str] = set()
    for name, value in headers.items():
        if (type(name) is not str or name.lower() not in allowed or name.lower() in seen or type(value) is not str
                or not value or value != value.strip() or any(ord(char) < 32 or ord(char) > 126 for char in value)):
            raise UnsupportedRequest("request_options_unsupported")
        seen.add(name.lower())
    return dict(headers)


def final_body(request: Any, api_mode: str = "chat_completions") -> dict[str, Any]:
    """Return a JSON body with extra_body merged in and SDK options removed."""
    if not isinstance(request, dict):
        raise UnsupportedRequest("request_not_mapping")
    headers = request_headers(request, api_mode)
    if request.get("extra_query"):
        raise UnsupportedRequest("request_options_unsupported")
    extra = request.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise UnsupportedRequest("request_options_unsupported")
    if headers and "extra_headers" in extra:
        # A JSON field with this name would conflict with the middleware request option.
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
        self._stamps: OrderedDict[str, Callable[[], Any]] = OrderedDict()
        self._max_open = max_open
        self._max_sessions = max_sessions
        self._clock = clock

    def set_stamp(self, session_id: Any, stamp: Callable[[], Any]) -> None:
        """Keep the function that gives the key stamp (key_stamp) of a session now. Hermes can give the key as a
        function whose value changes (a token that refreshes or rotates), and the hooks do not get the key. A
        request keeps its capture only when the stamp is the same at its start and at its end."""
        key = str(session_id or "")
        with self._lock:
            self._stamps[key] = stamp
            self._stamps.move_to_end(key)
            while len(self._stamps) > self._max_sessions:
                self._stamps.popitem(last=False)

    def _stamp(self, session_id: Any) -> str | None:
        with self._lock:
            stamp = self._stamps.get(str(session_id or ""))
        try:
            value = stamp() if stamp is not None else None
        except Exception:
            return None
        return value if isinstance(value, str) else None

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
        entry = {"session_id": str(session_id), "route": (model, base_url, api_mode), "digests": digests, "body": None,
                 "refusal": None, "key_stamp": self._stamp(session_id)}
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
        if entry is None or entry["route"][2] not in ("chat_completions", "codex_responses", "anthropic_messages"):
            return
        refusal = None
        headers: dict[str, str] = {}
        try:
            body = final_body(request, entry["route"][2])
            headers = request_headers(request, entry["route"][2])
        except UnsupportedRequest as error:
            body = None
            # Keep only fixed codes. Exception data must not reach the status or logs.
            code = error.args[0] if len(error.args) == 1 else None
            refusal = code if type(code) is str and code in REQUEST_REFUSALS else "settings_unsupported"
        if body is not None:
            # A later middleware can change the request after this capture saw it. The warm request would
            # then send a body that the provider never received.
            refusal = self._middleware_order_refusal()
            if refusal is not None:
                body = None
                headers = {}
        with self._lock:
            entry["body"] = body
            entry["request_headers"] = headers
            entry["refusal"] = refusal

    def _middleware_order_refusal(self) -> str | None:
        """Return None when this capture is the last llm_execution middleware. The chain order is in the Hermes
        plugin manager (a private form). When another middleware runs after the capture, return
        middleware_after_capture; when the order cannot be read or the capture is not in it, return
        middleware_order_unknown: a later middleware could change the request (fail closed)."""
        try:
            from hermes_cli.plugins import _delivery_manager
            chain = list(_delivery_manager()._middleware.get("llm_execution", []))
        except Exception:
            return "middleware_order_unknown"
        own = CaptureStore.on_llm_execution
        for index, callback in enumerate(chain):
            if getattr(callback, "__self__", None) is self and getattr(callback, "__func__", None) is own:
                return "middleware_after_capture" if index < len(chain) - 1 else None
        return "middleware_order_unknown"

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
        # A key that changed while the request was open: the request can have gone out with either key.
        if self._stamp(entry["session_id"]) != entry["key_stamp"]:
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
