"""Capture bounded MOA views through the public Hermes observer hooks."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .capture import UnsupportedRequest, final_body
from . import warm
from .rows import attr, row_digest, tool_calls_of
from .warm import WarmRefusal, check_settings, fits, split_history, wire_row

_UNSET = object()
MAX_OPEN = 8
MAX_SESSIONS = 16
MAX_ROUTES = 8
MAX_REFERENCES = 32
MAX_RECORD_BYTES = 2 * 1024 * 1024
ADVISOR_PREFIX = "[Mixture of Agents reference context]\n"
REFERENCE_NOTE = (
    "Summarize only this reference view and the reference reply. "
    "This view can omit system instructions, old turns, and tool results. "
    "It is advice and cannot replace the acting agent's history."
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}\Z")
_PROVIDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9:_.-]{0,63}\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_CUT = re.compile(r"\.\.\.\[truncated \d+ chars\]|<[^<>]* depth limit>|<\d+ bytes>")


def _url(value: Any) -> str:
    """Return a normalized direct HTTP route or refuse its form."""
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise ValueError("moa_route_invalid")
    if any(ord(char) <= 32 or ord(char) == 127 for char in value) or any(char in value for char in "\\?#"):
        raise ValueError("moa_route_invalid")
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise ValueError
        if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ValueError
        _ = parts.port
    except ValueError as error:
        raise ValueError("moa_route_invalid") from error
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _destination(route: dict[str, Any]) -> tuple:
    return route["provider"], route["model"], route["base_url"], route["api_mode"]


def validate_routes(routes: Any) -> list[dict[str, Any]]:
    """Validate plugin-owned routes. Errors contain no route or key values."""
    if not isinstance(routes, list) or len(routes) > MAX_ROUTES:
        raise ValueError("moa_routes_invalid")
    clean = []
    names: set = set()
    destinations: set = set()
    allowed = {"name", "provider", "model", "base_url", "api_key_env", "context_length", "api_mode"}
    for supplied in routes:
        if not isinstance(supplied, dict) or set(supplied) - allowed:
            raise ValueError("moa_route_invalid")
        name, provider, model = supplied.get("name"), supplied.get("provider"), supplied.get("model")
        key_env = supplied.get("api_key_env", "")
        window = supplied.get("context_length")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("moa_route_invalid")
        if not isinstance(provider, str) or not _PROVIDER.fullmatch(provider) or provider.lower() == "moa":
            raise ValueError("moa_route_invalid")
        if not isinstance(model, str) or not model.strip() or model != model.strip() or len(model) > 256:
            raise ValueError("moa_route_invalid")
        if any(ord(char) < 32 or ord(char) == 127 for char in model):
            raise ValueError("moa_route_invalid")
        if not isinstance(key_env, str) or (key_env and not _ENV.fullmatch(key_env)):
            raise ValueError("moa_route_invalid")
        mode = supplied.get("api_mode", "chat_completions")
        if (type(window) is not int or window <= 0 or mode not in ("chat_completions", "codex_responses")
                or (mode == "codex_responses" and (provider.lower() != "openai-codex" or "api_key_env" in supplied))):
            raise ValueError("moa_route_invalid")
        if mode == "codex_responses":
            model = model.lower().rsplit("/", 1)[-1]
        route = dict(name=name, provider=provider.lower(), model=model, base_url=_url(supplied.get("base_url")),
                     context_length=window, api_mode=mode)
        if mode == "chat_completions":
            route["api_key_env"] = key_env
        destination = _destination(route)
        if name in names or destination in destinations:
            raise ValueError("moa_routes_duplicate")
        names.add(name)
        destinations.add(destination)
        clean.append(route)
    return clean


def _json_copy(value: Any) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
        raise WarmRefusal("moa_capture_too_large")
    return json.loads(encoded)


def _has_loss(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in ("_truncated", "_truncated_items") or _has_loss(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_loss(item) for item in value)
    return isinstance(value, str) and ("<redacted>" in value or bool(_CUT.search(value)))


def _body(request: Any, request_messages: Any, model: str, request_tools: Any = _UNSET) -> dict[str, Any]:
    """Restore raw messages only. All other replay fields must be complete."""
    if not isinstance(request, dict) or request.get("_truncated") or not isinstance(request.get("body"), dict):
        raise WarmRefusal("moa_payload_incomplete")
    source = request["body"]
    if not isinstance(source.get("messages"), list) or not isinstance(request_messages, list):
        raise WarmRefusal("moa_payload_incomplete")
    restored_fields = {"messages"}
    if request_tools is not _UNSET:
        if not isinstance(request_tools, list) or "tools" not in source:
            raise WarmRefusal("moa_payload_incomplete")
        restored_fields.add("tools")
    remaining = {key: value for key, value in source.items() if key not in restored_fields}
    if _has_loss(remaining):
        raise WarmRefusal("moa_payload_incomplete")
    if source.get("extra_headers"):
        raise WarmRefusal("moa_settings_unsupported")
    extra = source.get("extra_body")
    if isinstance(extra, dict) and any(key in extra for key in ("messages", "model")):
        raise WarmRefusal("moa_settings_unsupported")
    raw_messages = _json_copy(request_messages)
    if not raw_messages or any(not isinstance(row, dict) for row in raw_messages):
        raise WarmRefusal("moa_payload_incomplete")
    restored = _json_copy(remaining)
    restored["messages"] = raw_messages
    if request_tools is not _UNSET:
        restored["tools"] = _json_copy(request_tools)
    try:
        body = final_body(restored)
        check_settings(body)
    except (UnsupportedRequest, WarmRefusal) as error:
        raise WarmRefusal("moa_settings_unsupported") from error
    if body.get("model") != model:
        raise WarmRefusal("moa_route_changed")
    _json_copy(body)
    for key in ("max_tokens", "max_completion_tokens"):
        if key in body and (type(body[key]) is not int or body[key] <= 0):
            raise WarmRefusal("moa_settings_unsupported")
    return body


def _tokens(usage: Any) -> int | None:
    value = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return value if type(value) is int and value > 0 else None


def _attempt_key(api_request_id: Any, retry_count: Any) -> tuple:
    if not isinstance(api_request_id, str) or not api_request_id or type(retry_count) is not int or retry_count < 0:
        raise WarmRefusal("moa_missing_identity")
    return api_request_id, retry_count


class MoaStore:
    """Keep a bounded set of observed MOA attempts in memory only."""

    def __init__(self, routes: list[dict], include_references: bool = False) -> None:
        self.routes = validate_routes(routes)
        self.include_references = bool(include_references)
        self._routes = {_destination(route): route for route in self.routes}
        self._lock = threading.RLock()
        self._active: ContextVar = ContextVar("warm_compaction_moa_main", default=None)
        self._mains: OrderedDict = OrderedDict()
        self._pending: OrderedDict = OrderedDict()
        self._sessions: OrderedDict = OrderedDict()
        self._references: OrderedDict = OrderedDict()
        self._versions: OrderedDict = OrderedDict()
        self._version = 0
        self._capture_versions: dict[str, int] = {}

    def _forget_locked(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._capture_versions.pop(session_id, None)
        for mapping in (self._mains, self._pending, self._references):
            for key in [key for key, item in mapping.items() if item["session_id"] == session_id
                        or item.get("_owner_session") == session_id]:
                del mapping[key]

    def _bound_versions_locked(self) -> None:
        while len(self._versions) > MAX_SESSIONS:
            oldest, _ = self._versions.popitem(last=False)
            self._forget_locked(oldest)

    def _token_locked(self, session_id: str) -> int | None:
        if not session_id:
            return None
        if session_id not in self._versions:
            self._version += 1
            self._versions[session_id] = self._version
            self._bound_versions_locked()
        token = self._versions.get(session_id)
        if session_id in self._versions:
            self._versions.move_to_end(session_id)
        return token

    def _live_locked(self, session_id: str, token: int | None) -> bool:
        return token is not None and self._versions.get(session_id) == token

    def open_session(self, session_id: Any, session_check: Any = None, session_generation: Any = None) -> None:
        """Reopen only from the engine's current session-start call."""
        key = str(session_id or "")
        if not key:
            return
        with self._lock:
            if session_check is not None and not session_check():
                return
            if session_generation is not None and self._capture_versions.get(key) != session_generation:
                self._forget_locked(key)
                self._version += 1
                self._versions[key] = self._version
            if key in self._versions and self._versions[key] is None:
                self._version += 1
                self._versions[key] = self._version
            self._token_locked(key)
            self._bound_versions_locked()
            if session_generation is not None:
                self._capture_versions[key] = session_generation

    def _session(self, session_id: str, turn_id: str) -> dict | None:
        if not session_id or not turn_id:
            return None
        state = self._sessions.get(session_id)
        if state is None:
            state = {"turn_id": turn_id, "retired": []}
            self._sessions[session_id] = state
        elif state["turn_id"] != turn_id:
            if turn_id in state["retired"]:
                return None
            state["retired"] = [*state["retired"], state["turn_id"]][-MAX_SESSIONS:]
            state["turn_id"] = turn_id
            for key in [key for key in self._references if key[0] == session_id]:
                del self._references[key]
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > MAX_SESSIONS:
            oldest = next(iter(self._sessions))
            self._forget_locked(oldest)
            self._versions.pop(oldest, None)
        return state

    def pre_main(self, api_request_id: Any = None, session_id: Any = None, turn_id: Any = None,
                 provider: Any = None, conversation_history: Any = None, model: Any = None,
                 base_url: Any = None, api_mode: Any = None, session_check: Any = None,
                 session_generation: Any = None, **_: Any) -> None:
        """Freeze references that completed before this main request starts."""
        if provider != "moa" or not api_request_id:
            return
        try:
            session, turn = str(session_id or ""), str(turn_id or "")
            with self._lock:
                if session_check is not None and not session_check():
                    return
                previous_generation = self._capture_versions.get(session)
                if session_generation is not None and previous_generation not in (None, session_generation):
                    self._forget_locked(session)
                    self._version += 1
                    self._versions[session] = self._version
                token = self._token_locked(session)
                self._bound_versions_locked()
                if token is not None and session_generation is not None:
                    self._capture_versions[session] = session_generation
            if token is None:
                return
            digests = ([row_digest(row) for row in conversation_history]
                       if isinstance(conversation_history, list) else None)
            virtual_route = ((model, base_url, api_mode)
                             if all(value is not None for value in (model, base_url, api_mode)) else None)
            with self._lock:
                if not self._live_locked(session, token) or (session_check is not None and not session_check()):
                    return
                refs = []
                state = self._session(session, turn)
                if state is not None and digests is not None and self.include_references:
                    for key, record in self._references.items():
                        if key[:2] == (session, turn) and record.get("completed"):
                            boundary = record.get("boundary_digests")
                            if boundary is None:
                                record["boundary_digests"] = list(digests)
                                record["virtual_route"] = virtual_route
                            elif digests[:len(boundary)] != boundary:
                                continue
                            elif virtual_route is not None and record.get("virtual_route") not in (None, virtual_route):
                                continue
                            elif virtual_route is not None:
                                record["virtual_route"] = virtual_route
                            refs.append(copy.deepcopy(record))
                entry = {"session_id": session, "turn_id": turn, "references": refs, "aggregator": None,
                         "ambiguous": False, "_token": token,
                         "code": None if state is not None and digests is not None else "moa_missing_identity"}
                if api_request_id in self._mains:
                    entry["code"] = "moa_ambiguous_capture"
                self._mains[str(api_request_id)] = entry
                self._mains.move_to_end(str(api_request_id))
                while len(self._mains) > MAX_OPEN:
                    self._mains.popitem(last=False)
        except Exception:
            return

    def run_main(self, api_request_id: Any, next_call: Any) -> Any:
        """Set the main identity, call once, then restore the previous identity."""
        with self._lock:
            active = str(api_request_id) if str(api_request_id) in self._mains else None
        token = self._active.set(active)
        try:
            return next_call()
        finally:
            self._active.reset(token)

    def _observed_route(self, provider: Any, model: Any, base_url: Any, api_mode: Any) -> dict:
        if (api_mode not in ("chat_completions", "codex_responses")
                or (api_mode == "codex_responses" and not any(
                    route["api_mode"] == "codex_responses" for route in self.routes))):
            raise WarmRefusal("moa_api_mode_unsupported")
        try:
            destination = (str(provider or "").lower(), str(model or ""), _url(base_url), api_mode)
        except ValueError as error:
            raise WarmRefusal("moa_route_invalid") from error
        route = self._routes.get(destination)
        if route is None:
            raise WarmRefusal("moa_route_unconfigured")
        return copy.deepcopy(route)

    def on_pre_auxiliary_call(self, aux_task: Any = None, api_request_id: Any = None, retry_count: Any = 0,
                              session_id: Any = None, turn_id: Any = None, provider: Any = None,
                              model: Any = None, base_url: Any = None, api_mode: Any = None,
                              request: Any = None, request_messages: Any = None, streaming: Any = False,
                              native_request: Any = None, route_context: Any = None, extra_headers: Any = None,
                              request_tools: Any = _UNSET,
                              **_: Any) -> None:
        """Keep each physical attempt. A newer refusal replaces older success."""
        if (aux_task not in ("moa_aggregator", "moa_reference")
                or (aux_task == "moa_reference" and not self.include_references)):
            return
        # The coarse auxiliary preview does not own native attempts.
        if (api_mode == "codex_responses" and native_request is None
                and getattr(self, "native_available", None) is not False and any(
                route["api_mode"] == "codex_responses" for route in self.routes)):
            return
        session, turn = str(session_id or ""), str(turn_id or "")
        with self._lock:
            active_main = self._mains.get(self._active.get()) if aux_task == "moa_aggregator" else None
            owner_session = active_main["session_id"] if active_main is not None else session
            token = self._token_locked(owner_session)
        if token is None:
            return
        record = {"kind": str(aux_task), "session_id": session, "turn_id": turn, "api_request_id": api_request_id,
                  "retry_count": retry_count, "streaming": bool(streaming), "code": "moa_incomplete_capture",
                  "body": None, "route_config": None, "reply": None, "prompt_tokens": None,
                  "boundary_digests": None, "completed": False, "main_id": self._active.get(),
                  "_owner_session": owner_session, "_token": token}
        try:
            attempt = _attempt_key(api_request_id, retry_count)
            if not session or not turn:
                raise WarmRefusal("moa_missing_identity")
            route = self._observed_route(provider, model, base_url, api_mode)
            record["route_config"] = route
            record["name"] = route["name"]
            if api_mode == "codex_responses":
                if native_request is None and getattr(self, "native_available", None) is False:
                    raise WarmRefusal("moa_native_observer_unavailable")
                from .moa_native import capture_request
                record["body"], record["route_context"] = capture_request(
                    native_request, route_context, extra_headers, route, session, str(aux_task), attempt)
            else:
                record["body"] = _body(request, request_messages, str(model), request_tools)
            record["code"] = None
        except WarmRefusal as error:
            record["code"] = error.code
            attempt = (str(api_request_id or ""), retry_count) if type(retry_count) is int else ("", -1)
        except Exception:
            record["code"] = "moa_capture_error"
            attempt = (str(api_request_id or ""), retry_count) if type(retry_count) is int else ("", -1)
        try:
            with self._lock:
                if not self._live_locked(owner_session, token):
                    return
                if aux_task == "moa_aggregator":
                    main = self._mains.get(record["main_id"])
                    if main is None:
                        return
                    previous = main["aggregator"]
                    if (session, turn) != (main["session_id"], main["turn_id"]):
                        record["code"] = "moa_missing_identity"
                    if previous is not None:
                        if not previous.get("completed"):
                            main["ambiguous"] = True
                        if previous["api_request_id"] != api_request_id and previous.get("code") is None:
                            main["ambiguous"] = True
                        if previous["api_request_id"] == api_request_id and (
                            type(retry_count) is not int or type(previous["retry_count"]) is not int
                            or retry_count <= previous["retry_count"]
                        ):
                            main["ambiguous"] = True
                    main["aggregator"] = record
                else:
                    state = self._session(session, turn)
                    route = record["route_config"]
                    if state is None or route is None:
                        return
                    destination = (session, turn, _destination(route))
                    previous = self._references.get(destination)
                    if previous is not None:
                        if not previous.get("completed"):
                            previous["code"] = record["code"] = "moa_ambiguous_capture"
                        if previous["api_request_id"] != api_request_id and previous.get("boundary_digests") is None:
                            previous["code"] = record["code"] = "moa_ambiguous_capture"
                        if previous["api_request_id"] == api_request_id and (
                            type(retry_count) is not int or type(previous["retry_count"]) is not int
                            or retry_count <= previous["retry_count"]
                        ):
                            record["code"] = "moa_ambiguous_capture"
                    record["reference_key"] = destination
                    self._references[destination] = record
                    self._references.move_to_end(destination)
                    while len(self._references) > MAX_REFERENCES:
                        self._references.popitem(last=False)
                if attempt in self._pending:
                    self._pending[attempt]["code"] = record["code"] = "moa_ambiguous_capture"
                self._pending[attempt] = record
                self._pending.move_to_end(attempt)
                while len(self._pending) > MAX_OPEN:
                    _, removed = self._pending.popitem(last=False)
                    removed["code"] = "moa_capture_evicted"
        except Exception:
            return

    def on_post_auxiliary_call(self, aux_task: Any = None, api_request_id: Any = None, retry_count: Any = 0,
                               session_id: Any = None, turn_id: Any = None, provider: Any = None,
                               model: Any = None, base_url: Any = None, api_mode: Any = None,
                               streaming: Any = False, finish_reason: Any = None, response: Any = None,
                               usage: Any = None, error: Any = None, error_type: Any = None,
                               native_response: Any = None, route_context: Any = None, status: Any = None,
                               **_: Any) -> None:
        """Close an attempt. Do not promote missing, late, or failed replies."""
        if (api_mode == "codex_responses" and native_response is None and route_context is None
                and any(route["api_mode"] == "codex_responses" for route in self.routes)):
            return
        try:
            attempt = _attempt_key(api_request_id, retry_count)
            with self._lock:
                record = self._pending.pop(attempt, None)
                if record is None:
                    return
                if not self._live_locked(record["_owner_session"], record["_token"]):
                    return
                record["completed"] = True
                if record["code"] is not None:
                    return
                if ((str(session_id or ""), str(turn_id or ""), aux_task)
                        != (record["session_id"], record["turn_id"], record["kind"])):
                    raise WarmRefusal("moa_missing_identity")
                route = self._observed_route(provider, model, base_url, api_mode)
                if _destination(route) != _destination(record["route_config"]):
                    raise WarmRefusal("moa_route_changed")
                if error is not None or error_type is not None:
                    raise WarmRefusal("moa_provider_error")
                if bool(streaming) != record["streaming"]:
                    raise WarmRefusal("moa_incomplete_capture")
                if api_mode == "codex_responses":
                    from .moa_native import capture_response
                    record["native_response"], record["reply"] = capture_response(
                        native_response, route_context, record, status)
                    record["prompt_tokens"] = _tokens({"prompt_tokens": (usage or {}).get("input_tokens")})
                    return
                record["prompt_tokens"] = _tokens(usage)
                if aux_task == "moa_aggregator":
                    if not streaming and finish_reason not in ("stop", "tool_calls"):
                        raise WarmRefusal("moa_reply_incomplete")
                    return
                if streaming or finish_reason != "stop" or not isinstance(response, dict) or _has_loss(response):
                    raise WarmRefusal("moa_reply_incomplete")
                message = response.get("assistant_message")
                if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
                    raise WarmRefusal("moa_reply_incomplete")
                content = message.get("content")
                if (not isinstance(content, str) or not content.strip()
                        or message.get("tool_calls") or message.get("refusal")):
                    raise WarmRefusal("moa_reply_incomplete")
                record["reply"] = _json_copy({"role": "assistant", "content": content})
        except WarmRefusal as refusal:
            with self._lock:
                if "record" in locals() and record is not None:
                    record["code"] = refusal.code
        except Exception:
            with self._lock:
                if "record" in locals() and record is not None:
                    record["code"] = "moa_capture_error"

    def on_pre_auxiliary_native_request(self, **event: Any) -> None:
        """Use the generic native observer for configured MOA attempts only."""
        self.on_pre_auxiliary_call(**event)

    def on_post_auxiliary_native_request(self, **event: Any) -> None:
        """Close the corresponding native attempt without a provider call."""
        self.on_post_auxiliary_call(**event)

    def finish_main(self, api_request_id: Any, main_capture: dict) -> dict:
        """Attach only the final physical attempt and the frozen reference views."""
        with self._lock:
            main = self._mains.pop(str(api_request_id), None)
            if main is None or not self._live_locked(main["session_id"], main["_token"]):
                return {"code": "moa_missing_identity", "aggregator": None, "references": []}
            record = main["aggregator"]
            code = main["code"]
            if str(main_capture.get("session_id") or "") != main["session_id"]:
                code = "moa_missing_identity"
            if main["ambiguous"]:
                code = "moa_ambiguous_capture"
            if code is None and record is None:
                code = "moa_no_aggregator_capture"
            if code is None and not record.get("completed"):
                code = "moa_incomplete_capture"
            if code is None:
                code = record["code"]
            if record is not None:
                record = copy.deepcopy(record)
                if code is None and record["route_config"]["api_mode"] == "codex_responses":
                    from .moa_native import check_main_reply
                    try:
                        check_main_reply(record, main_capture.get("reply"))
                    except WarmRefusal as error:
                        code = error.code
                record["code"] = code
                record["reply"] = copy.deepcopy(main_capture.get("reply"))
                record["boundary_digests"] = list(main_capture.get("digests") or [])
                if record["prompt_tokens"] is None:
                    record["prompt_tokens"] = _tokens({"prompt_tokens": main_capture.get("prompt_tokens")})
            return {"code": code, "aggregator": record, "references": copy.deepcopy(main["references"])}

    def forget(self, session_id: Any = None, session_check: Any = None, **_: Any) -> None:
        """Remove a session and its pending attempts."""
        with self._lock:
            if session_check is not None and not session_check():
                return
            key = str(session_id or "")
            self._forget_locked(key)
            if key:
                self._versions[key] = None
                self._versions.move_to_end(key)
                self._bound_versions_locked()


def _record(record: Any) -> tuple[dict, dict]:
    if not isinstance(record, dict):
        raise WarmRefusal("moa_no_capture")
    if record.get("code"):
        raise WarmRefusal(record["code"])
    body, route = record.get("body"), record.get("route_config")
    if not record.get("completed") or not isinstance(body, dict) or not isinstance(route, dict):
        raise WarmRefusal("moa_incomplete_capture")
    try:
        validated = validate_routes([route])[0]
        if validated["api_mode"] == "codex_responses":
            from .moa_native import checked_body
            checked_body(body, validated["base_url"])
        else:
            check_settings(body)
    except (ValueError, WarmRefusal) as error:
        raise WarmRefusal("moa_settings_unsupported") from error
    if body.get("model") != validated["model"]:
        raise WarmRefusal("moa_route_changed")
    return copy.deepcopy(body), validated


def _shape(row: Any) -> tuple:
    return attr(row, "role"), attr(row, "tool_call_id") or None, tuple(
        (call_id, name) for call_id, name, _ in tool_calls_of(row)
    )


def _source_matches(sent: list, history: list) -> bool:
    offset = len(sent) - len(history)
    return offset >= 0 and all(attr(row, "role") in ("system", "developer") for row in sent[:offset]) and all(
        _shape(wire) == _shape(row) for wire, row in zip(sent[offset:], history)
    )


def _advisor_tail(sent: list) -> bool:
    return bool(sent) and attr(sent[-1], "role") == "user" and isinstance(attr(sent[-1], "content"), str) and (
        attr(sent[-1], "content").startswith(ADVISOR_PREFIX)
    )


def _same_rows(left: list, right: list) -> bool:
    return len(left) == len(right) and all(row_digest(a) == row_digest(b) for a, b in zip(left, right))


def _check_main_source(main_capture: dict, sent: list, history: list, route: dict) -> None:
    """Require the observed outer main rows, with at most one advisor block."""
    outer_body = main_capture.get("body")
    outer = outer_body.get("messages") if isinstance(outer_body, dict) else None
    virtual_route = main_capture.get("route") or ()
    if not isinstance(outer, list) or not virtual_route or outer_body.get("model") != virtual_route[0]:
        raise WarmRefusal("moa_source_transform_unsupported")
    canonical = outer[:-1] if _advisor_tail(outer) else outer
    try:
        warm.check_source({**outer_body, "model": route["model"], "messages": canonical}, history,
                          route["base_url"], warm.native_details_type(route["provider"]))
    except WarmRefusal as error:
        raise WarmRefusal("moa_source_transform_unsupported") from error
    if _same_rows(sent, outer):
        return
    if not _advisor_tail(outer) and _advisor_tail(sent) and _same_rows(sent[:-1], outer):
        return
    raise WarmRefusal("moa_source_transform_unsupported")


def _handoff_controls(body: dict, route: dict) -> None:
    """Use the public Chat handoff limits for each physical route."""
    body.pop("stop", None)
    body.pop("web_search_options", None)
    for field in warm.LIMIT_KEYS:
        value = body.get(field)
        if type(value) is int and value > 0:
            body[field] = min(max(value, warm.HANDOFF_MIN_TOKENS), warm.HANDOFF_MAX_TOKENS)
    if not any(type(body.get(field)) is int and body[field] > 0 for field in warm.LIMIT_KEYS):
        body[warm.limit_field(route["model"], route["base_url"])] = warm.HANDOFF_MAX_TOKENS


def build_aggregator(main_capture: dict, messages: list, current_route: tuple, context_length: int,
                     instruction: str) -> tuple[dict, dict]:
    """Append the accepted main reply and tool rows to the observed acting view."""
    if tuple(main_capture.get("route") or ()) != tuple(current_route):
        raise WarmRefusal("moa_route_changed")
    if len(current_route) != 3 or current_route[2] != "chat_completions":
        raise WarmRefusal("moa_api_mode_unsupported")
    moa = main_capture.get("moa") or {}
    if moa.get("code"):
        raise WarmRefusal(moa["code"])
    record = moa.get("aggregator")
    body, route = _record(record)
    if route["api_mode"] == "codex_responses":
        from .moa_native import build_aggregator as build_native
        return build_native(main_capture, record, messages, current_route, context_length, instruction)
    if record.get("kind") != "moa_aggregator" or record.get("session_id") != main_capture.get("session_id"):
        raise WarmRefusal("moa_missing_identity")
    if record.get("boundary_digests") != main_capture.get("digests"):
        raise WarmRefusal("moa_history_changed")
    try:
        new_rows, trailing = split_history(main_capture, messages)
    except WarmRefusal as error:
        raise WarmRefusal("moa_history_changed") from error
    history = messages[:len(main_capture["digests"])]
    sent = body["messages"]
    _check_main_source(main_capture, sent, history, route)
    echo = warm.needs_reasoning_echo(body, new_rows)
    added = [wire_row(row, echo, route["model"], route["base_url"], warm.native_details_type(route["provider"]))
             for row in (*new_rows, *trailing)]
    body["messages"] = [*sent, *warm._join_user_rows(added, instruction)]
    body["stream"] = False
    body.pop("stream_options", None)
    _handoff_controls(body, route)
    if type(context_length) is not int or context_length <= 0:
        raise WarmRefusal("moa_capacity_unknown")
    window = min(context_length, route["context_length"])
    if not fits(body, window, record.get("prompt_tokens"), len(sent)):
        raise WarmRefusal("moa_capacity")
    return body, route


def build_reference(record: dict, instruction: str) -> tuple[dict, dict]:
    """Summarize only an observed reference view and its accepted reply."""
    body, route = _record(record)
    if route["api_mode"] == "codex_responses":
        from .moa_native import build_reference as build_native
        return build_native(record, instruction + "\n\n" + REFERENCE_NOTE)
    reply = record.get("reply")
    if record.get("kind") != "moa_reference":
        raise WarmRefusal("moa_missing_identity")
    if (record.get("boundary_digests") is None or not isinstance(reply, dict)
            or not isinstance(reply.get("content"), str)):
        raise WarmRefusal("moa_reference_unbound")
    if reply.get("role") != "assistant" or not reply["content"].strip() or reply.get("tool_calls") or _has_loss(reply):
        raise WarmRefusal("moa_reply_incomplete")
    sent = body["messages"]
    body["messages"] = [*sent, copy.deepcopy(reply), {"role": "user", "content": instruction + "\n\n" + REFERENCE_NOTE}]
    body["stream"] = False
    body.pop("stream_options", None)
    _handoff_controls(body, route)
    if not fits(body, route["context_length"], record.get("prompt_tokens"), len(sent)):
        raise WarmRefusal("moa_capacity")
    return body, route


def resolve_key(route_config: dict) -> str:
    """Read only the explicit environment key at send time."""
    key_env = route_config.get("api_key_env", "")
    if not isinstance(key_env, str) or (key_env and not _ENV.fullmatch(key_env)):
        raise WarmRefusal("moa_route_invalid")
    if not key_env:
        return ""
    value = os.environ.get(key_env)
    if not value or not value.strip():
        raise WarmRefusal("moa_key_missing")
    if "\r" in value or "\n" in value:
        raise WarmRefusal("moa_key_invalid")
    return value
