"""Request field names and bounded native reply envelopes."""

from __future__ import annotations

import json
from typing import Any

SUPPORTED = ("chat_completions", "codex_responses", "anthropic_messages")
MAX_DONE_ITEMS = 256


def history_key(api_mode: str) -> str:
    """Return the field with the conversation input."""
    return "input" if api_mode == "codex_responses" else "messages"


def ends_with_instruction(row: Any, instruction: str) -> bool:
    """True when the last user input block is the host instruction."""
    if not isinstance(row, dict) or row.get("role") != "user":
        return False
    content = row.get("content")
    if isinstance(content, str):
        return content == instruction or content.endswith("\n\n" + instruction)
    return (isinstance(content, list) and bool(content) and isinstance(content[-1], dict)
            and content[-1].get("type") in ("text", "input_text") and content[-1].get("text") == instruction)


def check_source(body: dict, history: list, route: tuple, native_type: str | None = None) -> None:
    """Check that the captured input represents the stored history."""
    from . import anthropic, responses, warm
    if route[2] == "codex_responses":
        responses.check_source(body, history, route[1], native_type)
    elif route[2] == "anthropic_messages":
        anthropic.check_source(body, history, route[1], native_type)
    else:
        warm.check_source(body, history, route[1], native_type)


def envelope(raw: bytes, api_mode: str) -> dict:
    """Read JSON or a complete SSE reply. Never use partial text as a summary."""
    from .warm import WarmRefusal
    try:
        text = raw.decode("utf-8")
        if text.lstrip().startswith("{"):
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError
            return payload
        events = []
        for frame in text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
            if data and data != "[DONE]":
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError
                events.append(event)
        if api_mode == "codex_responses":
            return _responses_stream(events)
        if api_mode == "anthropic_messages":
            return _anthropic_stream(events)
    except (UnicodeError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise WarmRefusal("incomplete_response") from error
    raise WarmRefusal("incomplete_response")


def _completed_item(item: Any) -> bool:
    """Validate a complete done item. Return True for final assistant text."""
    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
        raise ValueError
    kind = item.get("type")
    if not isinstance(kind, str) or not kind or item.get("status") not in (None, "completed"):
        raise ValueError
    if kind == "reasoning":
        summary = item.get("summary")
        if not isinstance(summary, list) or any(not isinstance(part, dict)
                                               or part.get("type") != "summary_text"
                                               or not isinstance(part.get("text"), str) for part in summary):
            raise ValueError
        if item.get("encrypted_content") is not None and not isinstance(item["encrypted_content"], str):
            raise ValueError
    if kind != "message":
        # The reply parser keeps tools and refusals out of an accepted handoff.
        return False
    if (item.get("status") != "completed" or item.get("role") != "assistant"
            or not isinstance(item.get("content"), list) or not item["content"]):
        raise ValueError
    text = False
    for part in item["content"]:
        if not isinstance(part, dict):
            raise ValueError
        if part.get("type") == "output_text" and isinstance(part.get("text"), str):
            text = text or bool(part["text"].strip())
        elif part.get("type") != "refusal" or not isinstance(part.get("refusal"), str):
            raise ValueError
    return text and item.get("phase") != "commentary"


def _responses_stream(events: list[dict]) -> dict:
    """Use completed done items when the completed terminal has no output list."""
    payload = None
    done: dict[int, dict] = {}
    announced: dict[int, tuple[str, str]] = {}
    referenced: dict[int, str | None] = {}
    ids: set[str] = set()
    text_done = False
    progress_kinds = {"response.output_text.delta", "response.output_text.done", "response.content_part.added",
                      "response.content_part.done", "response.function_call_arguments.delta",
                      "response.function_call_arguments.done"}
    for event in events:
        kind = event.get("type")
        if payload is not None or kind in ("response.incomplete", "response.failed", "error", "response.error"):
            raise ValueError
        if kind in ("response.output_item.added", "response.output_item.done"):
            index, item = event.get("output_index"), event.get("item")
            if type(index) is not int or not 0 <= index < MAX_DONE_ITEMS or not isinstance(item, dict):
                raise ValueError
            identity = (item.get("id"), item.get("type"))
            if any(not isinstance(value, str) or not value for value in identity):
                raise ValueError
            if kind == "response.output_item.added":
                if index in announced or index in done or any(other[0] == identity[0] for other in announced.values()):
                    raise ValueError
                announced[index] = identity
            else:
                if index in done or identity[0] in ids or (index in announced and announced[index] != identity):
                    raise ValueError
                if any(other_index != index and other[0] == identity[0] for other_index, other in announced.items()):
                    raise ValueError
                text_done = _completed_item(item) or text_done
                done[index] = item
                ids.add(identity[0])
        elif kind == "response.completed":
            terminal = event.get("response")
            if (not isinstance(terminal, dict) or terminal.get("status") != "completed"
                    or terminal.get("error") or terminal.get("incomplete_details")):
                raise ValueError
            payload = dict(terminal)
            output = payload.get("output")
            if output is not None and not isinstance(output, list):
                raise ValueError
            if output:
                # A terminal output and a done item cannot disagree about one output position.
                if any(index >= len(output) or output[index] != item for index, item in done.items()):
                    raise ValueError
                if any(index >= len(output) or not isinstance(output[index], dict)
                       or (output[index].get("id"), output[index].get("type")) != identity
                       for index, identity in announced.items()):
                    raise ValueError
            else:
                if (not text_done or set(done) != set(range(len(done))) or set(announced) - set(done)):
                    raise ValueError
                payload["output"] = [done[index] for index in range(len(done))]
            if any(index >= len(payload["output"]) or not isinstance(payload["output"][index], dict)
                   or (item_id is not None and payload["output"][index].get("id") != item_id)
                   for index, item_id in referenced.items()):
                raise ValueError
        elif kind in progress_kinds:
            # These events can report progress. Their text never supplies a missing done item.
            index = event.get("output_index")
            if type(index) is not int or not 0 <= index < MAX_DONE_ITEMS or index in done:
                raise ValueError
            item_id = event.get("item_id")
            if item_id is not None and (not isinstance(item_id, str) or not item_id):
                raise ValueError
            if index in announced and item_id is not None and item_id != announced[index][0]:
                raise ValueError
            if index in referenced and item_id is not None and referenced[index] not in (None, item_id):
                raise ValueError
            referenced[index] = item_id or referenced.get(index)
    if payload is None:
        raise ValueError
    return payload


def _anthropic_stream(events: list[dict]) -> dict:
    """Assemble only a complete native Messages stream."""
    payload: dict = {}
    blocks: dict[int, dict] = {}
    closed: set[int] = set()
    stopped = False
    for event in events:
        kind = event.get("type")
        if stopped or kind == "error":
            raise ValueError
        if kind == "message_start":
            if payload:
                raise ValueError
            payload = dict(event["message"])
            if not isinstance(payload.get("usage", {}), dict):
                raise ValueError
            payload["usage"] = dict(payload.get("usage", {}))
        elif kind == "content_block_start":
            index = event["index"]
            if not payload or type(index) is not int or index != len(blocks):
                raise ValueError
            blocks[index] = dict(event["content_block"])
        elif kind == "content_block_delta":
            index = event["index"]
            block, delta = blocks[index], event["delta"]
            if type(index) is not int or index in closed or not isinstance(delta, dict):
                raise ValueError
            compatible = {"text": {"text_delta"}, "thinking": {"thinking_delta", "signature_delta"},
                          "tool_use": {"input_json_delta"}}
            if delta.get("type") not in compatible.get(block.get("type"), set()):
                raise ValueError
            if delta.get("type") == "text_delta":
                if block.get("type") != "text":
                    raise ValueError
                block["text"] = block.get("text", "") + delta["text"]
        elif kind == "content_block_stop":
            index = event["index"]
            if type(index) is not int or index not in blocks or index in closed:
                raise ValueError
            closed.add(index)
        elif kind == "message_delta":
            if not payload or not isinstance(event.get("delta"), dict) or not isinstance(event.get("usage", {}), dict):
                raise ValueError
            payload.update(event["delta"])
            payload["usage"].update(event.get("usage", {}))
        elif kind == "message_stop":
            if not payload or set(blocks) != closed or not payload.get("stop_reason"):
                raise ValueError
            stopped = True
    if not stopped:
        raise ValueError
    payload["content"] = [blocks[index] for index in sorted(blocks)]
    return payload
