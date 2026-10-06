"""Builds and sends the warm request: the captured request prefix, the new rows, and one instruction."""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .rows import attr, compact_json, estimate_tokens, plain_text, reply_text, row_digest, tool_calls_of

DEFAULT_RESERVE = 4096
TIMEOUT_S = 120.0
SAFETY = 1.1
SYSTEM_ROLES = ("system", "developer")
UNSUPPORTED_SETTINGS = ("response_format", "functions", "function_call", "modalities", "audio")

Post = Callable[[str, bytes, dict, float], tuple]


class WarmRefusal(Exception):
    """The warm request cannot run. The first argument is the refusal code."""

    @property
    def code(self) -> str:
        return str(self.args[0]) if self.args else "unknown"


def _matches_reply(row: Any, reply: dict[str, Any]) -> bool:
    if attr(row, "role") != "assistant":
        return False
    calls = [[call_id, name] for call_id, name, _arguments in tool_calls_of(row)]
    return calls == reply["tool_calls"] and reply_text(attr(row, "content")) == reply_text(reply["content"])


def split_history(capture: dict[str, Any], messages: list) -> tuple[list, list]:
    """Return (new rows, trailing user rows) after the captured rows. Refuse a changed history."""
    digests = capture["digests"]
    count = len(digests)
    if len(messages) <= count or [row_digest(row) for row in messages[:count]] != digests:
        raise WarmRefusal("history_changed")
    rest = messages[count:]
    if not _matches_reply(rest[0], capture["reply"]):
        raise WarmRefusal("history_changed")
    expected = {call_id for call_id, _name in capture["reply"]["tool_calls"]}
    answered: set = set()
    index = 1
    while index < len(rest) and attr(rest[index], "role") == "tool":
        call_id = attr(rest[index], "tool_call_id")
        if call_id not in expected or call_id in answered:
            raise WarmRefusal("history_changed")
        answered.add(call_id)
        index += 1
    if answered != expected:
        raise WarmRefusal("history_changed")
    trailing = rest[index:]
    if any(attr(row, "role") != "user" for row in trailing):
        raise WarmRefusal("history_changed")
    return rest[:index], trailing


def check_settings(body: dict[str, Any]) -> None:
    """Refuse request settings that the warm request cannot keep."""
    if not isinstance(body.get("messages"), list):
        raise WarmRefusal("settings_unsupported")
    if body.get("n") not in (None, 1):
        raise WarmRefusal("settings_unsupported")
    if body.get("tool_choice") not in (None, "auto", "none"):
        raise WarmRefusal("settings_unsupported")
    if any(body.get(key) is not None for key in UNSUPPORTED_SETTINGS):
        raise WarmRefusal("settings_unsupported")


def _shape(row: Any) -> tuple:
    calls = tuple((call_id, name) for call_id, name, _arguments in tool_calls_of(row))
    return attr(row, "role"), attr(row, "tool_call_id") or None, calls


def _words(content: Any) -> str:
    return " ".join(plain_text(content).split())


def _arguments(value: Any) -> Any:
    """Return the JSON value of tool-call arguments, so that a change of spacing or key order is not a change."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return " ".join(value.split())
    return value


def _same_row(wire: Any, row: Any) -> bool:
    """True when the sent row carries the stored row: the same shape, the same name or no name (the Hermes
    transport removes the name from some rows), the stored text inside the sent text (Hermes can add
    request-time context to a row), and the same tool-call arguments."""
    if _shape(wire) != _shape(row) or attr(wire, "name") not in (None, attr(row, "name")):
        return False
    if _words(attr(row, "content")) not in _words(attr(wire, "content")):
        return False
    return [_arguments(arguments) for _id, _name, arguments in tool_calls_of(wire)] == [
        _arguments(arguments) for _id, _name, arguments in tool_calls_of(row)]


def check_source(body: dict[str, Any], history_rows: list) -> None:
    """Refuse a request whose messages are not system rows followed by the stored rows. A row that a hook or
    a middleware rewrote would make the handoff summarize text that is not in the history it replaces."""
    sent = body["messages"]
    offset = len(sent) - len(history_rows)
    if offset < 0 or any(attr(row, "role") not in SYSTEM_ROLES for row in sent[:offset]):
        raise WarmRefusal("source_transform_unsupported")
    if not all(_same_row(wire, row) for wire, row in zip(sent[offset:], history_rows)):
        raise WarmRefusal("source_transform_unsupported")


def wire_row(row: Any) -> dict[str, Any]:
    """Return the wire form of a stored row. Keep only the fields that the API reads."""
    wire: dict[str, Any] = {"role": attr(row, "role"), "content": copy.deepcopy(attr(row, "content"))}
    for key in ("tool_call_id", "name"):
        value = attr(row, key)
        if value is not None:
            wire[key] = value
    calls = tool_calls_of(row)
    if calls:
        wire["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {
                "name": name,
                "arguments": arguments if isinstance(arguments, str) else compact_json(
                    arguments if arguments is not None else {}),
            }}
            for call_id, name, arguments in calls
        ]
    return wire


def fits(body: dict[str, Any], context_length: int, measured_tokens: int | None = None,
         measured_rows: int = 0) -> bool:
    """Return True when the request size plus the reply reserve fits in the context window.

    When the server measured the prompt of the first measured_rows messages, use that count and estimate only
    the rows after them. Otherwise estimate all messages and tools.
    """
    if context_length <= 0:
        return True
    reserve = body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_RESERVE
    messages = body.get("messages") or []
    if type(measured_tokens) is int and measured_tokens > 0 and 0 < measured_rows <= len(messages):
        size = measured_tokens + estimate_tokens({"messages": messages[measured_rows:]}) * SAFETY
    else:
        size = estimate_tokens({"messages": messages, "tools": body.get("tools")}) * SAFETY
    return size + int(reserve) <= context_length


def build_request(capture: dict[str, Any], messages: list, route: tuple, context_length: int,
                  instruction: str) -> dict[str, Any]:
    """Return the warm request body. Raise WarmRefusal when a condition is not true."""
    if route[2] != "chat_completions":
        raise WarmRefusal("api_mode_unsupported")
    if tuple(capture["route"]) != tuple(route):
        raise WarmRefusal("route_changed")
    body = capture.get("body")
    if body is None:
        raise WarmRefusal("settings_unsupported")
    check_settings(body)
    new_rows, trailing = split_history(capture, messages)
    check_source(body, messages[: len(capture["digests"])])
    request = dict(body)
    # Send the trailing user rows too: the tail can keep only the newest of them, and compaction removes the others.
    request["messages"] = [*body["messages"], *(wire_row(row) for row in (*new_rows, *trailing)),
                           {"role": "user", "content": instruction}]
    request["stream"] = False
    request.pop("stream_options", None)
    if not fits(request, context_length, capture.get("prompt_tokens"), len(body["messages"])):
        raise WarmRefusal("capacity")
    return request


def urllib_post(url: str, data: bytes, headers: dict[str, str], timeout_s: float) -> tuple[int, bytes]:
    """POST data with the standard library. Return (status, body). Raise TimeoutError on a time-out."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies_environment()))
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        try:
            return error.code, error.read()
        finally:
            error.close()
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError(str(error.reason)) from error
        raise


def _api_key_text(api_key: Any) -> str:
    if callable(api_key):
        try:
            api_key = api_key()
        except Exception:
            return ""
    return api_key if isinstance(api_key, str) else ""


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def send(body: dict[str, Any], base_url: str, api_key: Any, timeout_s: float = TIMEOUT_S,
         post: Post | None = None) -> dict[str, Any]:
    """Send the warm request one time. Return the reply fields and the usage, or raise WarmRefusal."""
    if not base_url:
        raise WarmRefusal("provider_error")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = _api_key_text(api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode("utf-8")
    started = time.monotonic()
    try:
        status, raw = (post or urllib_post)(str(base_url).rstrip("/") + "/chat/completions", data, headers, timeout_s)
    except TimeoutError as error:
        raise WarmRefusal("timeout") from error
    except Exception as error:
        raise WarmRefusal("provider_error") from error
    elapsed = time.monotonic() - started
    if not 200 <= int(status) < 300:
        raise WarmRefusal("provider_error")
    try:
        payload = json.loads(raw.decode("utf-8"))
        choice = payload["choices"][0]
        message = choice["message"]
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "tool_calls": bool(message.get("tool_calls")),
            "refusal": bool(message.get("refusal")),
            "prompt_tokens": _optional_int(usage.get("prompt_tokens")),
            "completion_tokens": _optional_int(usage.get("completion_tokens")),
            "cached_tokens": _optional_int(details.get("cached_tokens")),
            "elapsed_s": round(elapsed, 3),
        }
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise WarmRefusal("incomplete_response") from error
