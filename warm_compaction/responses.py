"""A bounded Responses adapter for plain text and function-tool history."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any
from urllib.parse import urlsplit

from .rows import api_content
from .warm import HANDOFF_MAX_TOKENS, HANDOFF_MIN_TOKENS, WarmRefusal, fits, split_history

_PHASES = {"commentary", "final_answer"}
_OPAQUE_FIELDS = ("reasoning_details",)
_STATE_FIELDS = ("previous_response_id", "conversation", "context_management", "background", "truncation")
_SETTINGS = {"model", "instructions", "input", "store", "tools", "tool_choice", "parallel_tool_calls",
             "prompt_cache_key", "prompt_cache_retention", "reasoning", "include", "service_tier", "text",
             "max_output_tokens", "temperature"}


def _safe_text(text: str, base_url: Any) -> None:
    if is_codex_backend(base_url):
        visible = "".join(char for char in text if unicodedata.category(char) != "Cf")
        if re.search(r"<\|(start|end|channel|message|constrain|return|call)\|>", visible):
            raise WarmRefusal("source_transform_unsupported")


def _safe_tree(value: Any, base_url: Any) -> None:
    if isinstance(value, str):
        _safe_text(value, base_url)
    elif isinstance(value, dict):
        for key, item in value.items():
            _safe_tree(key, base_url)
            _safe_tree(item, base_url)
    elif isinstance(value, list):
        for item in value:
            _safe_tree(item, base_url)


def is_codex_backend(base_url: Any) -> bool:
    """Return True for the consumer Codex route used by Hermes."""
    parsed = urlsplit(str(base_url or ""))
    return parsed.hostname == "chatgpt.com" and parsed.path.startswith("/backend-api/codex")


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        and isinstance(part.get("text"), str) and set(part) <= {"type", "text"} for part in content
    ):
        return "".join(part["text"] for part in content)
    raise WarmRefusal("source_transform_unsupported")


def _call_id(raw: Any) -> str:
    """Read the stored call ID. Refuse IDs that need a generated replacement."""
    if not isinstance(raw, str) or not raw.strip():
        raise WarmRefusal("source_transform_unsupported")
    value = raw.strip().split("|", 1)[0]
    if value.startswith("fc_"):
        value = "call_" + value[3:]
    if not value or len(value) > 64:
        raise WarmRefusal("source_transform_unsupported")
    return value


def _arguments(value: Any) -> str:
    """Use the same compact JSON form for a valid function argument object."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as error:
            raise WarmRefusal("source_transform_unsupported") from error
    else:
        parsed = value
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise WarmRefusal("source_transform_unsupported") from error


def _message(row: dict, content: str, typed: bool) -> dict:
    role = row["role"]
    result = {"type": "message", "role": role, "content": content}
    if typed:
        result["content"] = [{"type": "output_text" if role == "assistant" else "input_text", "text": content}]
    if role == "assistant" and row.get("phase") in _PHASES:
        result["phase"] = row["phase"]
    return result


def _issuer(base_url: Any) -> str:
    """Use the named Hermes issuer classes. Custom routes keep their URL stamp."""
    parsed = urlsplit(str(base_url or ""))
    host = parsed.hostname or ""
    if host == "api.x.ai" or host.endswith(".x.ai"):
        return "xai_responses"
    if host == "githubcopilot.com" or host.endswith(".githubcopilot.com"):
        return "github_responses"
    if is_codex_backend(base_url):
        return "codex_backend"
    if parsed.query or parsed.fragment or parsed.username or parsed.port not in (None, 80, 443):
        return "other:" + str(base_url or "").strip()
    return "other:" + parsed._replace(scheme=parsed.scheme.lower(), netloc=host,
                                      path=parsed.path.rstrip("/")).geturl()


def _replay_reasoning(row: dict, base_url: Any, model: Any, seen: set, require_provenance: bool = False) -> list:
    """Replay verified reasoning bytes. Native compaction checkpoints remain unsupported."""
    raw_items = row.get("codex_reasoning_items") or []
    if not isinstance(raw_items, list):
        raise WarmRefusal("source_transform_unsupported")
    items = []
    for raw in raw_items:
        if not isinstance(raw, dict) or raw.get("type") != "reasoning":
            raise WarmRefusal("source_transform_unsupported")
        encrypted = raw.get("encrypted_content")
        if not isinstance(encrypted, str) or not encrypted:
            raise WarmRefusal("source_transform_unsupported")
        stamp = raw.get("_issuer_kind")
        if require_provenance and (not isinstance(stamp, str) or not isinstance(raw.get("_issuer_model"), str)):
            raise WarmRefusal("source_transform_unsupported")
        if isinstance(stamp, str) and stamp.startswith("other:"):
            stamp = _issuer(stamp[6:])
        if stamp is not None and stamp != _issuer(base_url):
            raise WarmRefusal("source_transform_unsupported")
        if raw.get("_issuer_model") is not None and raw["_issuer_model"] != model:
            raise WarmRefusal("source_transform_unsupported")
        item_id = raw.get("id")
        if item_id:
            if not isinstance(item_id, str) or item_id in seen:
                raise WarmRefusal("source_transform_unsupported")
            seen.add(item_id)
        summary = raw.get("summary") or []
        if not isinstance(summary, list) or any(
            not isinstance(part, dict) or part.get("type") != "summary_text"
            or not isinstance(part.get("text"), str) or set(part) - {"type", "text"} for part in summary
        ):
            raise WarmRefusal("source_transform_unsupported")
        for part in summary:
            _safe_text(part["text"], base_url)
        items.append({"type": "reasoning", "encrypted_content": encrypted, "summary": copy.deepcopy(summary)})
    return items


def _replay_messages(row: dict, content: str, base_url: Any) -> list:
    replay = row.get("codex_message_items")
    if not replay:
        return []
    if not isinstance(replay, list):
        raise WarmRefusal("source_transform_unsupported")
    items, final_text = [], []
    linked = bool(row.get("codex_reasoning_items") or row.get("codex_reasoning_trimmed"))
    host = urlsplit(str(base_url or "")).hostname or ""
    copilot = host == "githubcopilot.com" or host.endswith(".githubcopilot.com")
    for raw in replay:
        if not isinstance(raw, dict) or raw.get("type") != "message" or raw.get("role") != "assistant":
            raise WarmRefusal("source_transform_unsupported")
        parts = raw.get("content")
        if not isinstance(parts, list) or not parts or any(
            not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str)
            for part in parts
        ):
            raise WarmRefusal("source_transform_unsupported")
        text = "".join(part["text"] for part in parts)
        item = {"type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": part["text"]} for part in parts]}
        if raw.get("status") not in (None, "completed") or raw.get("phase") not in (None, *_PHASES):
            raise WarmRefusal("source_transform_unsupported")
        if raw.get("phase"):
            item["phase"] = raw["phase"]
        if raw.get("phase") != "commentary":
            final_text.append(text.strip())
        item_id = raw.get("id")
        if isinstance(item_id, str) and item_id.strip() and len(item_id.strip()) <= 64 and not copilot and not linked:
            if not is_codex_backend(base_url) or item_id.strip().startswith("msg"):
                item["id"] = item_id.strip()
        items.append(item)
    if "\n".join(final_text).strip() != content:
        raise WarmRefusal("source_transform_unsupported")
    return items


def wire_rows(rows: list, base_url: Any = None, *, used_ids: set | None = None, model: Any = None,
              used_reasoning_ids: set | None = None, require_provenance: bool = False) -> list:
    """Render text, verified native replay, and matched function-tool rows."""
    typed = is_codex_backend(base_url)
    used = set(used_ids or ())
    pending: set[str] = set()
    seen_reasoning: set[str] = set(used_reasoning_ids or ())
    output = []
    for row in rows:
        if not isinstance(row, dict) or any(row.get(key) for key in _OPAQUE_FIELDS):
            raise WarmRefusal("source_transform_unsupported")
        role = row.get("role")
        if role not in {"user", "assistant", "tool"}:
            raise WarmRefusal("source_transform_unsupported")
        content = _text(api_content(row))
        _safe_text(content, base_url)
        # Hermes removes outer whitespace from scalar user and assistant content.
        if role in {"user", "assistant"} and isinstance(api_content(row), str):
            content = content.strip()
        if role == "tool":
            call_id = _call_id(row.get("tool_call_id"))
            if call_id not in pending:
                raise WarmRefusal("source_transform_unsupported")
            pending.remove(call_id)
            output.append({"type": "function_call_output", "call_id": call_id, "output": content})
            continue
        calls = row.get("tool_calls") or []
        if role == "user" and (calls or row.get("codex_message_items")):
            raise WarmRefusal("source_transform_unsupported")
        reasoning = _replay_reasoning(row, base_url, model, seen_reasoning, require_provenance
                                      ) if role == "assistant" else []
        output.extend(reasoning)
        replay = _replay_messages(row, content, base_url) if role == "assistant" else []
        if replay:
            output.extend(replay)
        elif content or role == "user":
            output.append(_message(row, content, typed or isinstance(api_content(row), list)))
        elif reasoning and not calls:
            output.append(_message(row, " ", typed))
        if not isinstance(calls, list):
            raise WarmRefusal("source_transform_unsupported")
        for call in calls:
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                raise WarmRefusal("source_transform_unsupported")
            function = call.get("function")
            if not isinstance(function, dict):
                raise WarmRefusal("source_transform_unsupported")
            name = function.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise WarmRefusal("source_transform_unsupported")
            call_id = _call_id(call.get("call_id") or call.get("id"))
            if call_id in used:
                raise WarmRefusal("source_transform_unsupported")
            used.add(call_id)
            pending.add(call_id)
            arguments = _arguments(function.get("arguments", "{}"))
            _safe_text(arguments, base_url)
            output.append({"type": "function_call", "call_id": call_id, "name": name,
                           "arguments": arguments})
    if pending:
        raise WarmRefusal("source_transform_unsupported")
    return output


def check_source(body: dict[str, Any], history: list, base_url: Any = None, native_type: Any = None) -> None:
    """Require that the captured input is the supported form of the stored history."""
    check_settings(body, base_url)
    want = wire_rows(history, base_url, model=body.get("model"))
    got = body.get("input")
    # The Responses API treats an input item that has a role and no "type" as a message.
    # Hermes omits that default, so normalise it before the comparison.
    if isinstance(got, list):
        got = [({**item, "type": "message"} if isinstance(item, dict) and "type" not in item and "role" in item
                else item) for item in got]
    if got != want:
        raise WarmRefusal("source_transform_unsupported")


def check_settings(body: dict[str, Any], base_url: Any = None) -> None:
    """Refuse stateful and forced-output request settings."""
    if (set(body) - _SETTINGS or not isinstance(body.get("input"), list)
            or any(value is None for value in body.values())):
        raise WarmRefusal("settings_unsupported")
    # The main terminal runs preflight again after middleware. Accept only unchanged scalar forms.
    for field in ("model", "instructions"):
        value = body.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise WarmRefusal("source_transform_unsupported")
    _safe_text(body["instructions"], base_url)
    for field, kind in (("reasoning", dict), ("include", list)):
        if field in body and not isinstance(body[field], kind):
            raise WarmRefusal("source_transform_unsupported")
    if "service_tier" in body and (not isinstance(body["service_tier"], str) or not body["service_tier"]
                                   or body["service_tier"] != body["service_tier"].strip()):
        raise WarmRefusal("source_transform_unsupported")
    if "max_output_tokens" in body and (type(body["max_output_tokens"]) is not int or body["max_output_tokens"] <= 0):
        raise WarmRefusal("source_transform_unsupported")
    if "temperature" in body and type(body["temperature"]) is not float:
        raise WarmRefusal("source_transform_unsupported")
    if "prompt_cache_key" in body and (not isinstance(body["prompt_cache_key"], str) or not body["prompt_cache_key"]
                                       or body["prompt_cache_key"] != body["prompt_cache_key"].strip()
                                       or len(body["prompt_cache_key"]) > 64):
        raise WarmRefusal("source_transform_unsupported")
    if body.get("store") is not False or body.get("tool_choice") not in (None, "auto", "none"):
        raise WarmRefusal("settings_unsupported")
    if any(body.get(key) is not None for key in _STATE_FIELDS):
        raise WarmRefusal("settings_unsupported")
    text = body.get("text")
    if text is not None and (not isinstance(text, dict) or not text or set(text) - {"verbosity"}):
        raise WarmRefusal("settings_unsupported")
    tools = body.get("tools", [])
    if not isinstance(tools, list) or any(not isinstance(tool, dict) or tool.get("type") != "function"
                                         for tool in tools):
        raise WarmRefusal("settings_unsupported")
    for tool in tools:
        if (set(tool) != {"type", "name", "description", "strict", "parameters"}
                or not isinstance(tool.get("name"), str) or not tool["name"]
                or tool["name"] != tool["name"].strip() or not isinstance(tool.get("description"), str)
                or type(tool.get("strict")) is not bool or not isinstance(tool.get("parameters"), dict)):
            raise WarmRefusal("source_transform_unsupported")
    _safe_tree(tools, base_url)
    if is_codex_backend(base_url) and body.get("prompt_cache_retention") is not None:
        # Hermes drops this value after execution middleware. It is not a wire capture.
        raise WarmRefusal("source_transform_unsupported")


def reply_reserve(body: dict, context_length: int = 0) -> int:
    """Return the output reserve. An unknown consumer limit uses a conservative reserve."""
    value = body.get("max_output_tokens")
    if type(value) is int and value > 0:
        return value
    return max(4096, min(context_length // 4, 65_536))


def build_request(capture: dict[str, Any], messages: list, route: tuple, context_length: int,
                  instruction: str, native_type: Any = None) -> dict:
    """Keep the captured input prefix and add supported new rows and the handoff request."""
    if route[2] != "codex_responses":
        raise WarmRefusal("api_mode_unsupported")
    if tuple(capture.get("route") or ()) != tuple(route):
        raise WarmRefusal("route_changed")
    body = capture.get("body")
    if not isinstance(body, dict):
        raise WarmRefusal(capture.get("refusal") or "no_capture")
    check_settings(body, route[1])
    new_rows, trailing = split_history(capture, messages)
    count = len(capture["digests"])
    check_source(body, messages[:count], route[1])
    used_ids = {item["call_id"] for item in body["input"] if item.get("type") == "function_call"}
    reasoning_ids = {item.get("id") for row in messages[:count] for item in (row.get("codex_reasoning_items") or [])
                     if item.get("id")}
    added = wire_rows([*new_rows, *trailing], route[1], used_ids=used_ids, model=body.get("model"),
                      used_reasoning_ids=reasoning_ids, require_provenance=True)
    added.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": instruction}]})
    request = copy.deepcopy(body)
    request["input"] = [*request["input"], *added]
    request["stream"] = True
    _safe_text(instruction, route[1])
    if not is_codex_backend(route[1]):
        limit = body.get("max_output_tokens")
        request["max_output_tokens"] = (
            min(max(limit, HANDOFF_MIN_TOKENS), HANDOFF_MAX_TOKENS)
            if type(limit) is int and limit > 0 else HANDOFF_MAX_TOKENS
        )
    if not fits(request, context_length, capture.get("prompt_tokens"), len(body["input"]),
                api_mode="codex_responses"):
        raise WarmRefusal("capacity")
    return request


def _counter(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def parse_reply(payload: dict) -> dict:
    """Read a terminal Responses payload. The caller assembles streaming events first."""
    if not isinstance(payload, dict) or not isinstance(payload.get("output"), list):
        raise WarmRefusal("incomplete_response")
    status = payload.get("status")
    content = []
    tool_calls = refusal = False
    for item in payload["output"]:
        if not isinstance(item, dict):
            raise WarmRefusal("incomplete_response")
        kind = item.get("type")
        if kind in {"reasoning"}:
            continue
        if kind != "message":
            tool_calls = True
            continue
        if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
            raise WarmRefusal("incomplete_response")
        if item.get("status") not in (None, "completed") or item.get("phase") not in (None, *_PHASES):
            raise WarmRefusal("incomplete_response")
        for part in item["content"]:
            if not isinstance(part, dict):
                raise WarmRefusal("incomplete_response")
            if part.get("type") == "refusal":
                refusal = True
            elif part.get("type") == "output_text" and isinstance(part.get("text"), str):
                if item.get("phase") != "commentary":
                    content.append(part["text"])
            else:
                raise WarmRefusal("incomplete_response")
    usage = payload.get("usage", {})
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise WarmRefusal("incomplete_response")
    details = usage.get("input_tokens_details", {})
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise WarmRefusal("incomplete_response")
    return {"content": "".join(content), "finish_reason": "stop" if status == "completed" else "length",
            "tool_calls": tool_calls, "refusal": refusal or bool(payload.get("error")),
            "prompt_tokens": _counter(usage.get("input_tokens")),
            "completion_tokens": _counter(usage.get("output_tokens")),
            "cached_tokens": _counter(details.get("cached_tokens"))}
