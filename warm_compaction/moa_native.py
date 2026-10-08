"""Validate native MOA captures and append handoffs with the Responses adapter."""

from __future__ import annotations

import copy
import re
from typing import Any

from . import responses
from .rows import row_digest
from .warm import WarmRefusal, split_history

_SCHEMA = "hermes.native-route.v1"
_CONTEXT_FIELDS = {
    "schema", "provider", "model", "base_url", "api_mode", "profile_key", "session_id", "cache_scope",
    "aux_task", "api_request_id", "retry_count", "credential_fingerprint", "headers_fingerprint", "signature",
    "extra_headers", "settings_fingerprint", "input_count", "input_fingerprint",
}
_STAMPS = ("profile_key", "credential_fingerprint", "headers_fingerprint", "settings_fingerprint",
           "input_fingerprint", "signature")


def checked_body(body: Any, base_url: str) -> dict:
    """Keep stream outside the source adapter. The host sends it unchanged."""
    if not isinstance(body, dict) or body.get("stream") is not True:
        raise WarmRefusal("moa_settings_unsupported")
    source = {key: value for key, value in body.items() if key != "stream"}
    responses.check_settings(source, base_url)
    return source


def capture_request(body: Any, context: Any, headers: Any, route: dict, session: str,
                    task: str, attempt: tuple) -> tuple[dict, dict]:
    """Check the public context shape. Hermes checks its signature at dispatch."""
    from .moa import _json_copy, _url
    body, context = _json_copy(body), _json_copy(context)
    if not isinstance(context, dict) or set(context) != _CONTEXT_FIELDS or context.get("schema") != _SCHEMA:
        raise WarmRefusal("moa_native_context_invalid")
    for field in _STAMPS:
        if not isinstance(context.get(field), str) or not re.fullmatch("[0-9a-f]{64}", context[field]):
            raise WarmRefusal("moa_native_context_invalid")
    if (context.get("provider"), context.get("model"), context.get("api_mode")) != (
            route["provider"], route["model"], route["api_mode"]):
        raise WarmRefusal("moa_route_changed")
    try:
        if _url(context.get("base_url")) != route["base_url"]:
            raise WarmRefusal("moa_route_changed")
    except ValueError as error:
        raise WarmRefusal("moa_native_context_invalid") from error
    if (context.get("session_id"), context.get("aux_task"), context.get("api_request_id"),
            context.get("retry_count")) != (session, task, *attempt):
        raise WarmRefusal("moa_missing_identity")
    if (not isinstance(context.get("cache_scope"), str) or not context["cache_scope"]
            or not isinstance(context.get("extra_headers"), dict) or headers != context["extra_headers"]):
        raise WarmRefusal("moa_native_context_invalid")
    source = checked_body(body, route["base_url"])
    if source["model"] != context["model"] or type(context["input_count"]) is not int:
        raise WarmRefusal("moa_route_changed")
    if context["input_count"] != len(source["input"]):
        raise WarmRefusal("moa_native_context_invalid")
    return body, context


def native_row(payload: dict, route: dict) -> dict:
    """Use only the complete raw reply to make native replay fields."""
    from .moa import _json_copy
    if (not isinstance(payload, dict) or payload.get("status") != "completed"
            or payload.get("error") or payload.get("incomplete_details")):
        raise WarmRefusal("moa_reply_incomplete")
    parsed = responses.parse_reply(payload)
    if parsed["refusal"]:
        raise WarmRefusal("moa_reply_incomplete")
    reasoning, messages, calls, final_text = [], [], [], []
    for item in payload["output"]:
        kind = item.get("type")
        if kind == "reasoning":
            reasoning.append({**copy.deepcopy(item), "_issuer_kind": responses._issuer(route["base_url"]),
                              "_issuer_model": route["model"]})
        elif kind == "message":
            messages.append(copy.deepcopy(item))
            if item.get("phase") != "commentary":
                final_text.append("".join(part["text"] for part in item["content"]).strip())
        elif kind == "function_call":
            if (item.get("status") not in (None, "completed") or not isinstance(item.get("name"), str)
                    or not isinstance(item.get("call_id"), str) or not isinstance(item.get("arguments"), str)):
                raise WarmRefusal("moa_reply_incomplete")
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
        else:
            raise WarmRefusal("moa_reply_incomplete")
    row = {"role": "assistant", "content": "\n".join(final_text)}
    if calls:
        row["tool_calls"] = calls
    if reasoning:
        row["codex_reasoning_items"] = reasoning
    if messages:
        row["codex_message_items"] = messages
    # The existing adapter checks reasoning provenance, message phases, and text.
    if not calls:
        responses.wire_rows([row], route["base_url"], model=route["model"], require_provenance=True)
    return _json_copy(row)


def capture_response(payload: Any, context: Any, record: dict, status: Any) -> tuple[dict, dict]:
    """Keep one terminal reply on the exact request identity."""
    from .moa import _json_copy
    if context != record.get("route_context") or status != "completed":
        raise WarmRefusal("moa_reply_incomplete")
    payload = _json_copy(payload)
    return payload, native_row(payload, record["route_config"])


def check_main_reply(record: dict, reply: Any) -> None:
    """Bind the raw native reply to the accepted main text and tool identities."""
    row = native_row(record["native_response"], record["route_config"])
    calls = [[responses._call_id(call["id"]), call["function"]["name"]] for call in row.get("tool_calls", [])]
    if not isinstance(reply, dict) or row["content"].strip() != (reply.get("content") or "").strip():
        raise WarmRefusal("moa_reply_mismatch")
    expected = [[responses._call_id(call_id), name] for call_id, name in reply.get("tool_calls", [])]
    if calls != expected:
        raise WarmRefusal("moa_reply_mismatch")


def _enrich_reply(row: dict, record: dict) -> dict:
    """Add observed native fields to a private copy. Keep real history intact."""
    native = native_row(record["native_response"], record["route_config"])
    # Check tool arguments too. A same-name call can still request another action.
    if row.get("tool_calls"):
        actual = [(responses._call_id(call["id"]), call["function"]["name"],
                   responses._arguments(call["function"].get("arguments", "{}"))) for call in row["tool_calls"]]
        expected = [(responses._call_id(call["id"]), call["function"]["name"],
                     responses._arguments(call["function"]["arguments"]))
                    for call in native.get("tool_calls", [])]
        if actual != expected:
            raise WarmRefusal("moa_reply_mismatch")
    enriched = copy.deepcopy(row)
    for field in ("codex_reasoning_items", "codex_message_items"):
        if enriched.get(field) and enriched[field] != native.get(field):
            raise WarmRefusal("moa_reply_mismatch")
        if field in native:
            enriched[field] = native[field]
    return enriched


def build_aggregator(main: dict, record: dict, messages: list, current_route: tuple,
                     window: int, instruction: str) -> tuple[dict, dict]:
    """Keep the exact acting prefix. Append the accepted native reply and tools."""
    from .moa import ADVISOR_PREFIX, _record
    body, route = _record(record)
    if type(window) is not int or window <= 0:
        raise WarmRefusal("moa_capacity_unknown")
    if (record.get("kind") != "moa_aggregator" or record.get("session_id") != main.get("session_id")
            or record.get("boundary_digests") != main.get("digests")
            or tuple(main.get("route") or ()) != current_route):
        raise WarmRefusal("moa_missing_identity")
    check_main_reply(record, main.get("reply"))
    new, trailing = split_history(main, messages)
    history = copy.deepcopy(messages[:len(main["digests"])])
    source = checked_body(body, route["base_url"])
    expected = responses.wire_rows(history, route["base_url"], model=route["model"])
    advisor = []
    if source["input"] != expected:
        final = source["input"][-1] if source["input"] else {}
        content = responses._text(final.get("content"))
        if (source["input"][:-1] != expected or final.get("role") != "user"
                or not content.startswith(ADVISOR_PREFIX)):
            raise WarmRefusal("moa_source_transform_unsupported")
        advisor = [{"role": "user", "content": content}]
    synthetic_history = [*history, *advisor]
    copied = [_enrich_reply(new[0], record), *copy.deepcopy(new[1:]), *copy.deepcopy(trailing)]
    physical = (route["model"], route["base_url"], "codex_responses")
    capture = {"route": physical, "digests": [row_digest(row) for row in synthetic_history], "body": source,
               "reply": copy.deepcopy(main["reply"]), "prompt_tokens": record.get("prompt_tokens")}
    built = responses.build_request(capture, [*synthetic_history, *copied], physical,
                                    min(window, route["context_length"]), instruction)
    # The host capability requires all original controls and the exact prefix.
    built = {**body, "input": built["input"]}
    return built, route


def build_reference(record: dict, instruction: str) -> tuple[dict, dict]:
    """Use only the reference's own native request and terminal reply."""
    from .moa import _record
    body, route = _record(record)
    if record.get("kind") != "moa_reference" or record.get("boundary_digests") is None:
        raise WarmRefusal("moa_reference_unbound")
    reply = native_row(record["native_response"], route)
    if reply.get("tool_calls") or not reply["content"].strip():
        raise WarmRefusal("moa_reply_incomplete")
    added = responses.wire_rows([reply], route["base_url"], model=route["model"], require_provenance=True)
    added.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": instruction}]})
    result = {**body, "input": [*body["input"], *added]}
    from .warm import fits
    if not fits(result, route["context_length"], record.get("prompt_tokens"), len(body["input"]), "codex_responses"):
        raise WarmRefusal("moa_capacity")
    return result, route
