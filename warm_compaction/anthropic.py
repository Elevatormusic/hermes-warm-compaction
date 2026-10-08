"""Build a bounded warm request for the Anthropic Messages API."""

from __future__ import annotations

import copy
import json
from typing import Any
from urllib.parse import urlsplit

from .rows import api_content, attr, tool_calls_of
from .warm import HANDOFF_MAX_TOKENS, HANDOFF_MIN_TOKENS, WarmRefusal, fits, split_history

COMMON_BETAS = "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
OPAQUE_ROWS = ("reasoning_details", "reasoning_content", "anthropic_content_blocks", "_anthropic_content_blocks")
SETTINGS = {"model", "messages", "system", "tools", "tool_choice", "max_tokens", "stream", "stop_sequences",
            "thinking", "output_config", "temperature", "top_p", "top_k", "metadata", "service_tier",
            "cache_control"}


def _refuse() -> None:
    raise WarmRefusal("source_transform_unsupported")


def _text(content: Any, *, empty: str = "") -> list[dict]:
    if content is None:
        content = ""
    if isinstance(content, str):
        return [{"type": "text", "text": content or empty}] if content or empty else []
    if isinstance(content, list) and all(isinstance(p, dict) and p.get("type") == "text"
                                         and isinstance(p.get("text"), str)
                                         and set(p) <= {"type", "text", "cache_control"} for p in content):
        return [{"type": "text", "text": p["text"]} for p in content]
    _refuse()


def _id(value: Any) -> str:
    if not isinstance(value, str) or not value or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in value):
        _refuse()
    return value


def _thinking(blocks: Any) -> list[dict]:
    if blocks is None:
        return []
    if not isinstance(blocks, list):
        _refuse()
    result = []
    for block in blocks:
        if not isinstance(block, dict):
            _refuse()
        fields = {"thinking": {"type", "thinking", "signature"}, "redacted_thinking": {"type", "data"}}
        kind = block.get("type")
        if kind not in fields or set(block) != fields[kind] or any(type(v) is not str for v in block.values()):
            _refuse()
        if not block.get("signature" if kind == "thinking" else "data"):
            _refuse()
        result.append(copy.deepcopy(block))
    return result


def _assistant_blocks(row: Any, text: list[dict], calls: list[dict], base_url: Any) -> list[dict]:
    if attr(row, "_anthropic_content_blocks"):
        _refuse()
    details = _thinking(attr(row, "reasoning_details"))
    mirror = attr(row, "reasoning_content")
    if mirror and (not details or mirror != "\n\n".join(b["thinking"] for b in details if b["type"] == "thinking")):
        _refuse()
    ordered = attr(row, "anthropic_content_blocks")
    if ordered:
        if not isinstance(ordered, list):
            _refuse()
        clean = []
        for block in ordered:
            if not isinstance(block, dict):
                _refuse()
            kind = block.get("type")
            if kind in ("thinking", "redacted_thinking"):
                clean.extend(_thinking([block]))
            elif kind == "text":
                clean.extend(_text([block]))
            elif kind == "tool_use" and set(block) == {"type", "id", "name", "input"}:
                clean.append(copy.deepcopy(block))
            else:
                _refuse()
        if ([b for b in clean if b["type"] == "tool_use"] != calls
                or "\n".join(b["text"] for b in clean if b["type"] == "text") != "\n".join(b["text"] for b in text)
                or [b for b in clean if b["type"] in ("thinking", "redacted_thinking")] != details):
            _refuse()
    else:
        clean = [*details, *text, *calls]
    host = (urlsplit(str(base_url or "https://api.anthropic.com")).hostname or "").lower()
    if details and host != "api.anthropic.com":
        # Generic gateways can strip or reject native signatures. That renderer is not reproduced here.
        _refuse()
    return clean


def _rows(rows: list, *, new: bool = False, base_url: Any = None) -> list[dict]:
    result: list[dict] = []
    pending: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        role = attr(row, "role")
        if role in ("system", "developer"):
            if new:
                _refuse()
            continue
        if role != "assistant" and any(attr(row, key) for key in OPAQUE_ROWS):
            _refuse()
        if role == "tool":
            call_id = _id(attr(row, "tool_call_id"))
            if call_id not in pending or not isinstance(attr(row, "content"), str):
                _refuse()
            pending.remove(call_id)
            content = [{"type": "tool_result", "tool_use_id": call_id,
                        "content": attr(row, "content") or "(no output)"}]
            role = "user"
        elif role in ("user", "assistant"):
            if pending:
                _refuse()
            content = _text(api_content(row), empty="(empty message)" if role == "user" else "")
            calls = []
            for call_id, name, arguments in tool_calls_of(row):
                if role != "assistant" or not isinstance(name, str) or not name:
                    _refuse()
                call_id = _id(call_id)
                if call_id in seen:
                    _refuse()
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except (TypeError, ValueError):
                    _refuse()
                if not isinstance(args, dict):
                    _refuse()
                calls.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
                pending.add(call_id)
                seen.add(call_id)
            if role == "assistant":
                content = _assistant_blocks(row, content, calls, base_url)
            content = content or [{"type": "text", "text": "(empty message)"}]
        else:
            _refuse()
        if result and result[-1]["role"] == role:
            if role == "assistant" and any(b["type"] in ("thinking", "redacted_thinking") for b in content):
                _refuse()
            result[-1]["content"].extend(content)
        else:
            result.append({"role": role, "content": content})
    if pending:
        _refuse()
    return result


def _native_rows(rows: Any) -> list[dict]:
    if not isinstance(rows, list):
        _refuse()
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role", "content"} or row["role"] not in ("user", "assistant"):
            _refuse()
        blocks = _text(row["content"]) if isinstance(row["content"], str) else row["content"]
        if not isinstance(blocks, list):
            _refuse()
        clean = []
        for block in blocks:
            if not isinstance(block, dict):
                _refuse()
            part = {k: v for k, v in block.items() if k != "cache_control"}
            kind = part.get("type")
            if kind in ("thinking", "redacted_thinking") and row["role"] == "assistant":
                clean.extend(_thinking([part]))
                continue
            fields = {"text": {"type", "text"}, "tool_use": {"type", "id", "name", "input"},
                      "tool_result": {"type", "tool_use_id", "content"}}
            if kind not in fields or set(part) != fields[kind]:
                _refuse()
            clean.append(part)
        normalized.append({"role": row["role"], "content": clean})
    return normalized


def check_source(body: dict[str, Any], history: list, base_url: Any = None, native_type: str | None = None) -> None:
    """Check that the text and tool data in the captured prefix describe the stored history."""
    del native_type
    expected = _rows(history, base_url=base_url)
    actual = _native_rows(body.get("messages"))
    if actual != expected:
        _refuse()


def build_request(capture: dict[str, Any], messages: list, route: tuple, context_length: int,
                  instruction: str, native_type: str | None = None) -> dict[str, Any]:
    """Keep the native prefix and append a text and tool handoff turn."""
    del native_type
    if route[2] != "anthropic_messages":
        raise WarmRefusal("api_mode_unsupported")
    if tuple(capture["route"]) != tuple(route):
        raise WarmRefusal("route_changed")
    body = capture.get("body")
    if body is None:
        raise WarmRefusal(capture.get("refusal") or "settings_unsupported")
    if set(body) - SETTINGS or not isinstance(body.get("messages"), list):
        raise WarmRefusal("settings_unsupported")
    if isinstance(body.get("output_config"), dict) and set(body["output_config"]) - {"effort"}:
        raise WarmRefusal("settings_unsupported")
    choice = body.get("tool_choice")
    if choice is not None and choice not in ({"type": "auto"}, {"type": "none"}):
        raise WarmRefusal("settings_unsupported")
    if any(not isinstance(t, dict) or "input_schema" not in t or t.get("type") for t in body.get("tools", [])):
        raise WarmRefusal("settings_unsupported")
    new_rows, trailing = split_history(capture, messages)
    check_source(body, messages[:len(capture["digests"])], route[1])
    added = _rows([*new_rows, *trailing], new=True, base_url=route[1])
    block = {"type": "text", "text": instruction}
    if added and added[-1]["role"] == "user":
        added[-1]["content"].append(block)
    else:
        added.append({"role": "user", "content": [block]})
    if body["messages"] and added and body["messages"][-1]["role"] == added[0]["role"]:
        # Merging into the old turn would change its cache boundary or a thinking signature.
        _refuse()
    request = copy.deepcopy(body)
    request["messages"].extend(added)
    request["stream"] = False
    request.pop("stop_sequences", None)
    limit = body.get("max_tokens")
    limit = limit if type(limit) is int and limit > 0 else HANDOFF_MAX_TOKENS
    request["max_tokens"] = min(max(limit, HANDOFF_MIN_TOKENS), HANDOFF_MAX_TOKENS)
    thinking = request.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")
        if type(budget) is not int or budget <= 0:
            raise WarmRefusal("settings_unsupported")
        request["max_tokens"] = max(request["max_tokens"], budget + HANDOFF_MIN_TOKENS)
    if not fits(request, context_length, capture.get("prompt_tokens"), len(body["messages"]),
                api_mode="anthropic_messages"):
        raise WarmRefusal("capacity")
    return request


def route_headers(key: str, base_url: Any, existing_headers: dict, provider: str = "") -> dict[str, str]:
    """Use API-key auth for native Anthropic and ordinary Messages gateways. Refuse other auth routes."""
    url = urlsplit(str(base_url or "https://api.anthropic.com"))
    host = (url.hostname or "").lower()
    oauth = key and not key.startswith("sk-ant-api") and key.startswith(("sk-ant-", "eyJ", "cc-"))
    if oauth or provider in {"bedrock", "nous"}:
        raise WarmRefusal("auth_unsupported")
    if any(part in host for part in ("minimax", "moonshot", "kimi", "azure", "anthropic.aws")):
        raise WarmRefusal("auth_unsupported")
    if any(host == name or host.endswith("." + name) for name in (
            "api.commandcode.ai", "palantirfoundry.com", "inference-api.nousresearch.com")):
        raise WarmRefusal("auth_unsupported")
    for name, value in existing_headers.items():
        if name.lower() == "authorization":
            raise WarmRefusal("auth_unsupported")
        if name.lower() == "x-api-key" and value != key:
            raise WarmRefusal("auth_unsupported")
    headers = {k: v for k, v in existing_headers.items() if k.lower() not in ("authorization", "x-api-key")}
    if not any(k.lower() == "anthropic-version" for k in headers):
        headers["anthropic-version"] = "2023-06-01"
    if not any(k.lower() == "anthropic-beta" for k in headers):
        headers["anthropic-beta"] = COMMON_BETAS
    if key:
        headers["x-api-key"] = key
    return headers


def parse_reply(payload: dict) -> dict[str, Any]:
    """Read the completed native message and its optional token counters."""
    try:
        if payload.get("type") != "message" or payload.get("role") != "assistant":
            raise ValueError
        blocks = payload["content"]
        if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
            raise ValueError
        text = []
        for block in blocks:
            if block.get("type") == "text":
                if not isinstance(block.get("text"), str):
                    raise ValueError
                text.append(block["text"])
        reason = payload.get("stop_reason")
        finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                  "model_context_window_exceeded": "length", "tool_use": "tool_calls",
                  "refusal": "content_filter"}.get(reason)
        usage = payload.get("usage")
        usage = {} if usage is None else usage
        if not isinstance(usage, dict):
            raise ValueError
        counters = [usage.get(k) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
        def count(value):
            return value if type(value) is int and value >= 0 else None
        prompt = sum(counters) if all(count(v) is not None for v in counters) else None
        return {"content": "\n".join(text), "finish_reason": finish,
                "tool_calls": any(b.get("type") not in ("text", "thinking", "redacted_thinking") for b in blocks),
                "refusal": reason == "refusal", "prompt_tokens": prompt,
                "completion_tokens": count(usage.get("output_tokens")),
                "cached_tokens": count(usage.get("cache_read_input_tokens"))}
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        raise WarmRefusal("incomplete_response") from error
