"""Builds and sends the warm request: the captured request prefix, the new rows, and one instruction."""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .rows import (
    api_content, attr, compact_json, estimate_tokens, hermes_value, plain_text, reply_text, row_digest,
    tool_calls_of,
)

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
    # Hermes can keep the sent reply text in api_content and leave content empty or normalized.
    text = reply_text(reply["content"])
    return calls == reply["tool_calls"] and text in (reply_text(attr(row, "content")), reply_text(api_content(row)))


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


def _is_text(part: Any) -> bool:
    return isinstance(part, str) or (
        isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str))


def _parts(content: Any) -> tuple[list[str], list]:
    """Return the text runs between the non-text parts and the non-text parts, in order. A message with n image,
    audio, or file parts has n + 1 text runs. White space stays: it can change code, tables, or commands."""
    if not isinstance(content, list):
        return [plain_text(content)], []
    runs: list[str] = []
    media: list = []
    current: list[str] = []
    for part in content:
        if _is_text(part):
            current.append(part if isinstance(part, str) else part["text"])
        else:
            runs.append("\n".join(current))
            media.append(part)
            current = []
    runs.append("\n".join(current))
    return runs, media


def _arguments(value: Any) -> Any:
    """Return the JSON value of tool-call arguments, so that a change of spacing or key order is not a change."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return " ".join(value.split())
    return value


def _same_row(wire: Any, row: Any) -> bool:
    """True when the sent row carries the stored row: the same shape; the same name (the Hermes transport
    removes the name from tool rows only); the same image, audio, and file parts in the same order; each stored
    text run inside the sent text run at the same place (Hermes can add request-time context to a row); and the
    same tool-call arguments. The stored text is the api_content sidecar when the row has one."""
    if _shape(wire) != _shape(row):
        return False
    name = attr(wire, "name")
    if name != attr(row, "name") and not (name is None and attr(row, "role") == "tool"):
        return False
    sent_runs, sent_media = _parts(attr(wire, "content"))
    stored_runs, stored_media = _parts(api_content(row))
    if sent_media != stored_media or not all(
            stored.strip() in sent for stored, sent in zip(stored_runs, sent_runs)):
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


def _reasoning_policy(source: dict, wire: dict, needs_pad: bool) -> None:
    """The reasoning_content rule of Hermes 45871e10 (agent.message_sanitization.apply_reasoning_content_policy):
    a thinking-mode route (DeepSeek, Kimi, MiMo) needs the field on every assistant row; other routes reject it."""
    if source.get("role") != "assistant":
        return
    if not needs_pad:
        wire.pop("reasoning_content", None)
        return
    existing, reasoning = source.get("reasoning_content"), source.get("reasoning")
    if isinstance(existing, str):
        wire["reasoning_content"] = existing or " "
    elif isinstance(reasoning, str) and reasoning and not source.get("tool_calls"):
        wire["reasoning_content"] = reasoning
    else:
        wire["reasoning_content"] = " "


def needs_reasoning_echo(body: dict[str, Any], rows: list) -> bool:
    """True when the route needs reasoning_content on assistant rows: Hermes sent it on a captured assistant row,
    or a new assistant row has it from the provider."""
    return any(isinstance(row, dict) and row.get("role") == "assistant" and "reasoning_content" in row
               for row in body.get("messages") or ()) or any(
        attr(row, "role") == "assistant" and isinstance(attr(row, "reasoning_content"), str) for row in rows)


def _model_consumes_thought_signature(model: Any) -> bool:
    """Hermes 45871e10 (agent.transports.chat_completions): Gemini and Gemma models need the tool-call
    extra_content replayed; other providers reject it."""
    name = str(model or "").lower()
    return "gemini" in name or "gemma" in name


def _signature(extra: Any) -> Any:
    """Return the tool-call extra_content when it has a usable thought signature, else None."""
    if not isinstance(extra, dict):
        return None
    candidate = extra.get("thought_signature")
    google = extra.get("google")
    if candidate is None and isinstance(google, dict):
        candidate = google.get("thought_signature")
    return copy.deepcopy(extra) if isinstance(candidate, str) and candidate.strip() else None


def wire_row(row: Any, reasoning_echo: bool = False, model: Any = None) -> dict[str, Any]:
    """Return the wire form of a stored row, as Hermes sends it: only the fields that the API reads, the
    api_content sidecar of a user or assistant row in place of its content, reasoning_content on an assistant
    row when the route needs it, and the thought signature (extra_content) of a tool call for a Gemini model."""
    wire: dict[str, Any] = {"role": attr(row, "role"), "content": copy.deepcopy(api_content(row))}
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
        consumes = hermes_value("agent.transports.chat_completions", "_model_consumes_thought_signature",
                                _model_consumes_thought_signature)(model)
        stored = attr(row, "tool_calls") or []
        for call, source in zip(wire["tool_calls"], stored):
            extra = _signature(attr(source, "extra_content")) if consumes else None
            if extra is not None:
                call["extra_content"] = extra
    if isinstance(row, dict):
        policy = hermes_value("agent.message_sanitization", "apply_reasoning_content_policy", _reasoning_policy)
        policy(row, wire, reasoning_echo)
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
    echo = needs_reasoning_echo(body, new_rows)
    added = [wire_row(row, echo, body.get("model")) for row in (*new_rows, *trailing)]
    request["messages"] = [*body["messages"], *added, {"role": "user", "content": instruction}]
    request["stream"] = False
    request.pop("stream_options", None)
    # A stop sequence of the main request could cut the handoff after the five headings.
    request.pop("stop", None)
    if not fits(request, context_length, capture.get("prompt_tokens"), len(body["messages"])):
        raise WarmRefusal("capacity")
    return request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow a redirect. A followed redirect would send the Authorization header to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def urllib_post(url: str, data: bytes, headers: dict[str, str], timeout_s: float) -> tuple[int, bytes]:
    """POST data with the standard library. Return (status, body). Raise TimeoutError on a time-out. A redirect
    is not followed: its status comes back, and the caller treats it as a provider error."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies_environment()),
                                         _NoRedirect())
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
