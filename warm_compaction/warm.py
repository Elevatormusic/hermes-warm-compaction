"""Builds and sends the warm request: the captured request prefix, the new rows, and one instruction."""

from __future__ import annotations

import collections
import copy
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from collections.abc import Callable

from . import protocol
from .rows import (
    api_content, attr, compact_json, estimate_tokens, has_thought_signature, hermes_value, plain_text,
    reasoning_policy as _reasoning_policy, replay_details, reply_text, row_digest, tool_calls_of,
)

DEFAULT_RESERVE = 4096
# The reply limit of the handoff. The instruction asks for at most 600 words; the upper limit leaves room for a
# thinking model that counts its reasoning in the same limit.
HANDOFF_MIN_TOKENS = 2048
HANDOFF_MAX_TOKENS = 8192
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
    if len(messages) < count:
        raise WarmRefusal("history_changed:capture_ahead")
    if [row_digest(row) for row in messages[:count]] != digests:
        raise WarmRefusal("history_changed:digest")
    if len(messages) == count:
        raise WarmRefusal("history_changed:reply_missing")
    rest = messages[count:]
    if not _matches_reply(rest[0], capture["reply"]):
        raise WarmRefusal("history_changed:reply")
    # A multiset: one tool row for each tool call, also when an id repeats.
    expected = collections.Counter(call_id for call_id, _name in capture["reply"]["tool_calls"])
    answered: collections.Counter = collections.Counter()
    index = 1
    while index < len(rest) and attr(rest[index], "role") == "tool":
        call_id = attr(rest[index], "tool_call_id")
        if answered[call_id] >= expected[call_id]:
            raise WarmRefusal("history_changed:tool_extra")
        answered[call_id] += 1
        index += 1
    if answered != expected:
        raise WarmRefusal("history_changed:tool_count")
    trailing = rest[index:]
    if any(attr(row, "role") != "user" for row in trailing):
        raise WarmRefusal("history_changed:trailing")
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


def _arguments(value: Any) -> tuple[str, str]:
    """Return a canonical form of tool-call arguments: a change of spacing or key order is not a change, but a
    change of a JSON type is (true and 1 are different, which Python == does not see). Text that is not JSON
    stays as it is."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return "text", value
    return "json", json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _same_row(wire: Any, row: Any) -> bool:
    """True when the sent row carries the stored row: the same shape; the same name (the Hermes transport
    removes the name from tool rows only); the same image, audio, and file parts in the same order; each stored
    text run equal to the sent text run at the same place, or a complete string with all trailing whitespace
    removed; and the same tool-call arguments. The stored text is the api_content sidecar when the row has
    one. Leading whitespace and whitespace in content lists must stay unchanged."""
    if _shape(wire) != _shape(row):
        return False
    name = attr(wire, "name")
    if name != attr(row, "name") and not (name is None and attr(row, "role") == "tool"):
        return False
    sent_content, stored_content = attr(wire, "content"), api_content(row)
    sent_runs, sent_media = _parts(sent_content)
    stored_runs, stored_media = _parts(stored_content)
    same_text = all(stored == sent for stored, sent in zip(stored_runs, sent_runs))
    # Hermes e36a8180 strips complete strings during request assembly. Allow only the trailing removal;
    # a change of leading whitespace can change code indentation. Do not normalize content lists.
    trailing_removed = (isinstance(sent_content, str) and isinstance(stored_content, str)
                        and sent_content == stored_content.rstrip())
    if sent_media != stored_media or not (same_text or trailing_removed):
        return False
    return [_arguments(arguments) for _id, _name, arguments in tool_calls_of(wire)] == [
        _arguments(arguments) for _id, _name, arguments in tool_calls_of(row)]


# Fields that Hermes adds to a sent row on some routes, apart from wire_row: the prompt caching marker.
TRANSPORT_FIELDS = frozenset({"cache_control"})


def _no_extra_fields(wire: Any, expected: dict[str, Any]) -> bool:
    """True when the sent row and its tool calls have no field that Hermes does not send for the stored row (a
    provider control that a middleware added, for example): the model reads it."""
    if not isinstance(wire, dict):
        return True
    if set(wire) - set(expected) - TRANSPORT_FIELDS:
        return False
    for call, want in zip(wire.get("tool_calls") or [], expected.get("tool_calls") or []):
        if isinstance(call, dict) and (set(call) - set(want) or (
                isinstance(call.get("function"), dict) and set(call["function"]) - set(want.get("function") or {}))):
            return False
        # The same keys with another type value: the provider reads another kind of call.
        if isinstance(call, dict) and "type" in call and call["type"] != want.get("type"):
            return False
    return True


def _same_replay_fields(wire: Any, expected: dict[str, Any]) -> bool:
    """True when the sent row has the reasoning fields and thought signatures that Hermes replays for the stored
    row on this route. The provider reads them, so a change makes a prefix that the history does not have."""
    if any(attr(wire, key) != expected.get(key) for key in ("reasoning_content", "reasoning_details")):
        return False
    return [attr(call, "extra_content") for call in attr(wire, "tool_calls") or []] == [
        call.get("extra_content") for call in expected.get("tool_calls") or []]


def check_source(body: dict[str, Any], history_rows: list, base_url: Any = None,
                 native_type: str | None = None) -> None:
    """Refuse a request whose messages are not system rows followed by the stored rows. A row that a hook or
    a middleware rewrote would make the handoff summarize text that is not in the history it replaces."""
    sent = body["messages"]
    offset = len(sent) - len(history_rows)
    if offset < 0 or any(attr(row, "role") not in SYSTEM_ROLES for row in sent[:offset]):
        raise WarmRefusal("source_transform_unsupported")
    if not all(_same_row(wire, row) for wire, row in zip(sent[offset:], history_rows)):
        raise WarmRefusal("source_transform_unsupported")
    # The replayed fields follow the captured request only: the new rows do not change what Hermes sent.
    echo = needs_reasoning_echo(body, [])
    expected = [wire_row(row, echo, body.get("model"), base_url, native_type) for row in history_rows]
    if not all(_same_replay_fields(wire, want) and _no_extra_fields(wire, want)
               for wire, want in zip(sent[offset:], expected)):
        raise WarmRefusal("source_transform_unsupported")


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
    """Return a copy of the tool-call extra_content when it has a usable thought signature, else None."""
    return copy.deepcopy(extra) if has_thought_signature(extra) else None


REPLAY_DETAILS_HOSTS = ("openrouter.ai", "nousresearch.com")
LIMIT_KEYS = ("max_tokens", "max_completion_tokens")


def _forces_completion_field(model: Any) -> bool:
    """Hermes 45871e10 (utils.model_forces_max_completion_tokens): OpenAI families that reject max_tokens."""
    name = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return name.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"))


def limit_field(model: Any, base_url: Any) -> str:
    """The reply limit field that Hermes 45871e10 sends on the route (AIAgent._max_tokens_param):
    max_completion_tokens for OpenAI, Azure OpenAI, GitHub Copilot, and the newer OpenAI model families, else
    max_tokens."""
    host = (urllib.parse.urlparse(str(base_url or "")).hostname or "").lower()
    forces = hermes_value("utils", "model_forces_max_completion_tokens", _forces_completion_field)
    if (host == "api.openai.com" or host == "openai.azure.com" or host.endswith(".openai.azure.com")
            or host.endswith(".githubcopilot.com") or forces(model)):
        return "max_completion_tokens"
    return "max_tokens"


def _route_replays_reasoning_details(base_url: Any) -> bool:
    """Hermes 45871e10 (agent.transports.chat_completions): only OpenRouter and the Nous Portal read replayed
    reasoning_details; strict routes reject the field."""
    host = (urllib.parse.urlparse(str(base_url or "")).hostname or "").lower()
    return any(host == name or host.endswith("." + name) for name in REPLAY_DETAILS_HOSTS)


def native_details_type(provider: Any) -> str | None:
    """The native reasoning carrier type of the provider profile (native_reasoning_details_type of Hermes
    45871e10), or None. Hermes replays that carrier on every route of the provider."""
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider) if provider else None
        value = getattr(profile, "native_reasoning_details_type", None)
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _replay_details(details: Any, native_type: str | None = None) -> list | None:
    """A copy of rows.replay_details: reasoning_details without private native-assistant carriers, except the
    carrier of the provider profile, or None."""
    kept = replay_details(details, native_type)
    return copy.deepcopy(kept) if kept is not None else None


def wire_row(row: Any, reasoning_echo: bool = False, model: Any = None, base_url: Any = None,
             native_type: str | None = None) -> dict[str, Any]:
    """Return the wire form of a stored row, as Hermes sends it: only the fields that the API reads, the
    api_content sidecar of a user or assistant row in place of its content, reasoning_content on an assistant
    row when the route needs it, the thought signature (extra_content) of a tool call for a Gemini model, and
    reasoning_details of an assistant row on a route that replays it or for a provider profile with a native
    carrier type (native_type: that carrier only of the native carriers)."""
    wire: dict[str, Any] = {"role": attr(row, "role"), "content": copy.deepcopy(api_content(row))}
    for key in ("tool_call_id", "name"):
        value = attr(row, key)
        # The Hermes transport removes the name from tool rows; strict providers reject it there.
        if value is not None and not (key == "name" and attr(row, "role") == "tool"):
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
    if attr(row, "role") == "assistant" and (native_type or hermes_value(
            "agent.transports.chat_completions", "_route_replays_reasoning_details",
            _route_replays_reasoning_details)(base_url)):
        details = _replay_details(attr(row, "reasoning_details"), native_type)
        if details is not None:
            wire["reasoning_details"] = details
    if isinstance(row, dict):
        policy = hermes_value("agent.message_sanitization", "apply_reasoning_content_policy", _reasoning_policy)
        policy(row, wire, reasoning_echo)
    return wire


def _join_content(first: Any, second: Any) -> Any:
    """The rule of Hermes 45871e10 (agent.agent_runtime_helpers._merge_user_content): two texts join with a blank
    line; lists append as separate parts. None for another shape."""
    if isinstance(first, str) and isinstance(second, str):
        return first + ("\n\n" if first and second else "") + second
    first_parts = first if isinstance(first, list) else [{"type": "text", "text": first}] if first else []
    second_parts = second if isinstance(second, list) else [{"type": "text", "text": second}] if second else []
    if not isinstance(first, (str, list)) or not isinstance(second, (str, list)):
        return None
    return [*first_parts, *second_parts]


def ends_with_instruction(row: Any, instruction: str) -> bool:
    """True when the row is a user row whose last block is the host instruction (it can join the last user row
    of the history; see _join_user_rows)."""
    if attr(row, "role") != "user":
        return False
    content = attr(row, "content")
    if isinstance(content, str):
        return content == instruction or content.endswith("\n\n" + instruction)
    return (isinstance(content, list) and bool(content) and isinstance(content[-1], dict)
            and content[-1].get("type") == "text" and content[-1].get("text") == instruction)


def _join_user_rows(rows: list, instruction: str) -> list:
    """Join adjacent user rows of the same author (no name, or the same name), then join the host instruction
    to the last user row (it keeps its name: the ordinary request also ended with that row). The ordinary
    request has no adjacent user rows (Hermes joins them), and strict chat templates refuse them. The
    instruction is the last block; it says that it comes from the host."""
    out: list = []
    for row in rows:
        last = out[-1] if out else None
        if (last is not None and last.get("role") == row.get("role") == "user" and last.get("name") == row.get("name")
                and set(last) <= {"role", "content", "name"} and set(row) <= {"role", "content", "name"}):
            joined = _join_content(last.get("content"), row.get("content"))
            if joined is not None:
                out[-1] = {**last, "content": joined}
                continue
        out.append(row)
    last = out[-1] if out else None
    if last is not None and last.get("role") == "user" and set(last) <= {"role", "content", "name"}:
        joined = _join_content(last.get("content"), instruction)
        if joined is not None:
            out[-1] = {**last, "content": joined}
            return out
    return [*out, {"role": "user", "content": instruction}]


def reply_reserve(body: Any) -> int:
    """The reply reserve of a request: the larger positive max_tokens or max_completion_tokens (a server can
    honor either), else DEFAULT_RESERVE."""
    values = [body.get(key) for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
              ] if isinstance(body, dict) else []
    values = [value for value in values if type(value) is int and value > 0]
    return max(values) if values else DEFAULT_RESERVE


def fits(body: dict[str, Any], context_length: int, measured_tokens: int | None = None,
         measured_rows: int = 0, api_mode: str = "chat_completions") -> bool:
    """Return True when the request size plus the reply reserve fits in the context window.

    When the server measured the prompt of the first measured_rows messages, use that count and estimate only
    the rows after them. Otherwise estimate all messages and tools.
    """
    if context_length <= 0:
        return True
    reserve = reply_reserve(body)
    field = protocol.history_key(api_mode)
    messages = body.get(field) or []
    if type(measured_tokens) is int and measured_tokens > 0 and 0 < measured_rows <= len(messages):
        size = measured_tokens + estimate_tokens({field: messages[measured_rows:]}) * SAFETY
    else:
        size = estimate_tokens({key: value for key, value in body.items()
                                if key == field or isinstance(value, (str, list, dict))}) * SAFETY
    if api_mode == "codex_responses" and not any(
            type(body.get(key)) is int and body[key] > 0 for key in ("max_output_tokens", "max_tokens")):
        reserve = max(DEFAULT_RESERVE, min(context_length // 4, 65_536))
    return size + int(reserve) <= context_length


def build_request(capture: dict[str, Any], messages: list, route: tuple, context_length: int,
                  instruction: str, native_type: str | None = None) -> dict[str, Any]:
    """Return the warm request body. Raise WarmRefusal when a condition is not true. native_type: the native
    reasoning carrier type of the provider profile (native_details_type)."""
    if route[2] == "codex_responses":
        from .responses import build_request as build_native
        return build_native(capture, messages, route, context_length, instruction, native_type)
    if route[2] == "anthropic_messages":
        from .anthropic import build_request as build_native
        return build_native(capture, messages, route, context_length, instruction, native_type)
    if route[2] != "chat_completions":
        raise WarmRefusal("api_mode_unsupported")
    if tuple(capture["route"]) != tuple(route):
        raise WarmRefusal("route_changed")
    body = capture.get("body")
    if body is None:
        raise WarmRefusal(capture.get("refusal") or "settings_unsupported")
    check_settings(body)
    new_rows, trailing = split_history(capture, messages)
    check_source(body, messages[: len(capture["digests"])], route[1], native_type)
    request = dict(body)
    # Send the trailing user rows too: the tail can keep only the newest of them, and compaction removes the others.
    echo = needs_reasoning_echo(body, new_rows)
    added = [wire_row(row, echo, body.get("model"), route[1], native_type) for row in (*new_rows, *trailing)]
    request["messages"] = [*body["messages"], *_join_user_rows(added, instruction)]
    request["stream"] = False
    request.pop("stream_options", None)
    # A stop sequence of the main request could cut the handoff after the five headings.
    request.pop("stop", None)
    # A web search costs a search and can bring text that is not in the conversation into the handoff.
    request.pop("web_search_options", None)
    # The reply limit of the main request is for another task: a small one cuts the handoff, a large one reserves
    # space that the handoff does not need. Keep the field that the route uses.
    for key in LIMIT_KEYS:
        value = body.get(key)
        if type(value) is int and value > 0:
            request[key] = min(max(value, HANDOFF_MIN_TOKENS), HANDOFF_MAX_TOKENS)
    # Without a limit the server default applies: a small one cuts the handoff, a large one can take more space
    # than the capacity check reserves. The handoff gets HANDOFF_MAX_TOKENS in the field of the route.
    if not any(type(body.get(key)) is int and body.get(key) > 0 for key in LIMIT_KEYS):
        request[limit_field(body.get("model"), route[1])] = HANDOFF_MAX_TOKENS
    if not fits(request, context_length, capture.get("prompt_tokens"), len(body["messages"])):
        raise WarmRefusal("capacity")
    return request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow a redirect. A followed redirect would send the Authorization header to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


# A reply is at most a few tens of kilobytes (the handoff gate allows 24 KB of text). A larger body is not read
# whole: a wrong endpoint or a gateway must not fill the memory of the Hermes process.
MAX_RESPONSE_BYTES = 1 << 20


def urllib_post(url: str, data: bytes, headers: dict[str, str], timeout_s: float,
                context: ssl.SSLContext | None = None) -> tuple[int, bytes]:
    """POST data with the standard library. Return (status, body). Raise TimeoutError on a time-out. A redirect
    is not followed: its status comes back, and the caller treats it as a provider error. context is the TLS
    context of the route (route_tls). The body is read to at most MAX_RESPONSE_BYTES + 1 bytes."""
    handlers: list = [urllib.request.ProxyHandler(urllib.request.getproxies_environment()), _NoRedirect()]
    if context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return response.status, response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        # The error body is not used.
        error.close()
        return error.code, b""
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError(str(error.reason)) from error
        raise


def api_key_text(api_key: Any) -> str:
    """The key as text: the value of a key function now, or "" when it cannot be read."""
    if callable(api_key):
        try:
            api_key = api_key()
        except Exception:
            return ""
    return api_key if isinstance(api_key, str) else ""


_FINISH_ALIASES = {"max_tokens": "length", "end": "stop", "function_call": "tool_calls"}


def _normalize_finish_reason(raw: Any) -> Any:
    """The rule of Hermes 45871e10 (agent.message_sanitization.normalize_finish_reason): the lowercase OpenAI
    value, with the aliases of other providers. Hermes applies it in its transport; this request does not use it."""
    if not isinstance(raw, str) or not raw:
        return raw
    lowered = raw.lower()
    return _FINISH_ALIASES.get(lowered, lowered)


def route_headers(api_key: Any, base_url: Any, provider: Any,
                  api_mode: str = "chat_completions") -> dict[str, str]:
    """Return the default headers that the Hermes OpenAI client sends on this route, in the order of Hermes
    45871e10 (agent.agent_init): the host headers or the provider profile headers, then model.default_headers,
    then providers.<name>.extra_headers. Provider headers go with every request (attribution, a User-Agent for a
    WAF, gateway credentials). Raise WarmRefusal("headers_unknown") when a source cannot be read: a request
    without them can be refused or go to another cache. The engine adds supported per-request session headers
    from the capture after this step. The values can be credentials: never log them."""
    url = str(base_url or "")
    if api_mode == "anthropic_messages":
        try:
            from hermes_cli.config_providers import get_custom_provider_extra_headers
            headers = get_custom_provider_extra_headers(url) or {}
        except Exception as error:
            raise WarmRefusal("headers_unknown") from error
        return native_headers(api_key_text(api_key), url, provider, api_mode, dict(headers))
    try:
        from agent.agent_init import _host_default_headers_factory
        from agent.auxiliary_client import _apply_user_default_headers
        from hermes_cli.config_providers import get_custom_provider_extra_headers
        factory = _host_default_headers_factory(url)
        if factory is not None:
            headers = dict(factory(api_key_text(api_key), url) or {})
        else:
            from providers import get_provider_profile
            profile = get_provider_profile(provider) if provider else None
            headers = dict(getattr(profile, "default_headers", None) or {})
        headers = dict(_apply_user_default_headers(headers) or {})
        headers.update(get_custom_provider_extra_headers(url) or {})
    except Exception as error:
        raise WarmRefusal("headers_unknown") from error
    return {str(key): str(value) for key, value in headers.items()}


def route_tls(base_url: Any) -> ssl.SSLContext | None:
    """Return the TLS context of an https route as the Hermes client has it (ssl_ca_cert of a custom provider,
    through agent.ssl_verify.resolve_httpx_verify), or None for the default context. Raise
    WarmRefusal("tls_unknown") when the setting cannot be read: with another trust setting, the request fails
    or goes to a server that the main requests do not trust. Raise WarmRefusal("tls_unverified") for
    ssl_verify: false: the plugin does not send without certificate checks."""
    url = str(base_url or "")
    if not url.lower().startswith("https:"):
        return None
    try:
        from agent.ssl_verify import resolve_httpx_verify
        from hermes_cli.config_providers import get_custom_provider_tls_settings
        tls = get_custom_provider_tls_settings(url) or {}
        verify = resolve_httpx_verify(ca_bundle=tls.get("ssl_ca_cert"), ssl_verify=tls.get("ssl_verify"),
                                      base_url=url)
    except Exception as error:
        raise WarmRefusal("tls_unknown") from error
    if isinstance(verify, ssl.SSLContext):
        return verify
    if verify is False:
        # ssl_verify: false. The plugin does not send without certificate checks; the fallback runs.
        raise WarmRefusal("tls_unverified")
    if verify is True:
        return None
    raise WarmRefusal("tls_unknown")


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def native_headers(key: str, base_url: str, provider: str, api_mode: str, headers: dict) -> dict:
    """Resolve native authentication before the middleware sees a warm request."""
    if api_mode == "anthropic_messages":
        from .anthropic import route_headers as messages_headers
        return messages_headers(key, base_url, headers, provider)
    if api_mode == "codex_responses" and urllib.parse.urlsplit(base_url).hostname == "chatgpt.com":
        try:
            from agent.codex_headers import codex_cloudflare_headers
            required = codex_cloudflare_headers(key, base_url=base_url)
            names = {name.lower() for name in required}
            headers = {name: value for name, value in headers.items() if name.lower() not in names}
            return {**headers, **required}
        except Exception as error:
            raise WarmRefusal("headers_unknown") from error
    return headers


def send(body: dict[str, Any], base_url: str, api_key: Any, timeout_s: float = TIMEOUT_S,
         post: Post | None = None, extra_headers: dict[str, str] | None = None,
         ssl_context: ssl.SSLContext | None = None, api_mode: str = "chat_completions",
         provider: str = "") -> dict[str, Any]:
    """Send the warm request one time. Return the reply fields and the usage, or raise WarmRefusal.
    extra_headers are the client default headers of the route (route_headers); they win, as in the SDK."""
    if not base_url:
        raise WarmRefusal("provider_error")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = api_key_text(api_key)
    if key and api_mode != "anthropic_messages":
        headers["Authorization"] = f"Bearer {key}"
    headers.update(extra_headers or {})
    headers = native_headers(key, base_url, provider, api_mode, headers)
    if api_mode == "codex_responses" and body.get("stream"):
        headers["Accept"] = "text/event-stream"
    data = json.dumps(body).encode("utf-8")
    started = time.monotonic()
    try:
        # The query of the route URL stays after the path: Hermes sends it as the client's default_query (the
        # api-version of an Azure route, for example).
        parts = urllib.parse.urlsplit(str(base_url))
        path = parts.path.rstrip("/")
        endpoint = "/chat/completions"
        if api_mode == "codex_responses":
            endpoint = "/responses"
        elif api_mode == "anthropic_messages":
            endpoint = "/messages" if path.endswith("/v1") else "/v1/messages"
        elif api_mode != "chat_completions":
            raise WarmRefusal("api_mode_unsupported")
        url = urllib.parse.urlunsplit(parts._replace(path=path + endpoint))
        if ssl_context is None:
            status, raw = (post or urllib_post)(url, data, headers, timeout_s)
        else:
            status, raw = (post or urllib_post)(url, data, headers, timeout_s, context=ssl_context)
    except TimeoutError as error:
        raise WarmRefusal("timeout") from error
    except Exception as error:
        raise WarmRefusal("provider_error") from error
    elapsed = time.monotonic() - started
    if not 200 <= int(status) < 300:
        raise WarmRefusal("provider_error")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise WarmRefusal("response_too_large")
    try:
        if api_mode != "chat_completions":
            from . import anthropic, responses
            parser = responses.parse_reply if api_mode == "codex_responses" else anthropic.parse_reply
            result = parser(protocol.envelope(raw, api_mode))
            return {**result, "elapsed_s": round(elapsed, 3)}
        payload = json.loads(raw.decode("utf-8"))
        choice = payload["choices"][0]
        message = choice["message"]
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return {
            "content": message.get("content"),
            "finish_reason": hermes_value("agent.message_sanitization", "normalize_finish_reason",
                                          _normalize_finish_reason)(choice.get("finish_reason")),
            "tool_calls": bool(message.get("tool_calls")),
            "refusal": bool(message.get("refusal")),
            "prompt_tokens": _optional_int(usage.get("prompt_tokens")),
            "completion_tokens": _optional_int(usage.get("completion_tokens")),
            "cached_tokens": _optional_int(details.get("cached_tokens")),
            "elapsed_s": round(elapsed, 3),
        }
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise WarmRefusal("incomplete_response") from error
