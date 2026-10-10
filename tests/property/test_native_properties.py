"""Bounded native API properties. All data is synthetic; no provider request runs."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from warm_compaction import anthropic, handoff, protocol, responses, warm
from warm_compaction.capture import CaptureStore
from warm_compaction.rows import row_digest

RESPONSES = ("fake-model", "http://127.0.0.1:9/v1", "codex_responses")
CODEX = ("fake-model", "https://chatgpt.com/backend-api/codex", "codex_responses")
MESSAGES = ("fake-model", "http://127.0.0.1:9", "anthropic_messages")
ANTHROPIC = ("fake-model", "https://api.anthropic.com", "anthropic_messages")
TEXT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789éλ\n\t", min_size=1, max_size=40)
SEALED = st.binary(min_size=1, max_size=40).map(bytes.hex)
PROPERTIES = settings(max_examples=60, deadline=None, derandomize=True, database=None)
INSTRUCTION = "Write a synthetic handoff."


def user(text):
    return {"role": "user", "content": "Synthetic " + text}


def assistant(text, calls=(), **extra):
    row = {"role": "assistant", "content": "Synthetic " + text, **extra}
    if calls:
        row["tool_calls"] = [{"id": call_id, "type": "function", "function": {
            "name": "read", "arguments": json.dumps(arguments)}} for call_id, arguments in calls]
    return row


def tool(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": "Synthetic " + text}


@st.composite
def histories(draw):
    """Make complete turns with distinct tool IDs and a final user request."""
    rows = []
    turns = draw(st.lists(st.tuples(TEXT, TEXT, st.booleans()), min_size=1, max_size=5))
    for index, (question, answer, use_tool) in enumerate(turns):
        rows.append(user(question))
        calls = [(f"call_{index}", {"path": answer, "index": index})] if use_tool else []
        rows.append(assistant(answer, calls))
        if use_tool:
            rows.append(tool(f"call_{index}", answer))
    rows.append(user(draw(TEXT)))
    return rows


def capture(rows, reply, route):
    """Make a native capture with fake route and session values."""
    body = {"model": route[0]}
    if route[2] == "codex_responses":
        body.update(instructions="Synthetic system rule.", store=False,
                    input=responses.wire_rows(rows, route[1], model=route[0]))
    else:
        body.update(system=[{"type": "text", "text": "Synthetic system rule.",
                             "cache_control": {"type": "ephemeral"}}],
                    messages=anthropic._rows(rows), max_tokens=4096, thinking={"type": "disabled"})
    return {"session_id": "synthetic-session", "route": route, "body": body,
            "digests": [row_digest(row) for row in rows], "reply": {
                "content": reply["content"], "tool_calls": [[call["id"], call["function"]["name"]]
                                                            for call in reply.get("tool_calls", [])]}}


def build(saved, rows, route):
    adapter = responses if route[2] == "codex_responses" else anthropic
    return adapter.build_request(saved, rows, route, 200_000, INSTRUCTION)


def summary(text):
    return "\n\n".join(heading + "\nSynthetic " + text for heading in handoff.HEADINGS)


def sse(events):
    return "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events).encode()


def response_events(text):
    item = {"id": "msg_synthetic", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}
    return [{"type": "response.output_item.added", "output_index": 0,
             "item": {"id": item["id"], "type": "message"}},
            {"type": "response.output_text.delta", "output_index": 0, "item_id": item["id"], "delta": text},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": {"status": "completed", "output": []}}]


def message_events(text, chunks):
    points = sorted({0, len(text), *(point % (len(text) + 1) for point in chunks)})
    return [{"type": "message_start", "message": {"type": "message", "role": "assistant", "usage": {}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            *[{"type": "content_block_delta", "index": 0, "delta": {
                "type": "text_delta", "text": text[start:end]}} for start, end in zip(points, points[1:])],
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
            {"type": "message_stop"}]


def accepted(raw, mode):
    """Run the same envelope, native reply, and handoff gates as the warm path."""
    try:
        payload = protocol.envelope(raw, mode)
        reply = (responses if mode == "codex_responses" else anthropic).parse_reply(payload)
    except warm.WarmRefusal:
        return None
    return handoff.gate(reply)[0]


class NativeHistoryProperties(unittest.TestCase):
    @PROPERTIES
    @given(histories(), TEXT, st.sampled_from((RESPONSES, CODEX, MESSAGES)), st.lists(TEXT, max_size=4))
    def test_extension_keeps_captured_prefix_and_does_not_change_inputs(self, rows, answer, route, new_users):
        reply = assistant(answer)
        saved = capture(rows, reply, route)
        current = [*rows, reply, *(user(text) for text in new_users)]
        before = copy.deepcopy((saved, current))
        body = build(saved, current, route)
        field = protocol.history_key(route[2])
        prefix = before[0]["body"][field]
        self.assertEqual(json.dumps(body[field][:len(prefix)]), json.dumps(prefix))
        self.assertEqual((saved, current), before)
        # Editing the result must not edit the in-memory capture.
        body[field][0].clear()
        self.assertEqual((saved, current), before)

    @PROPERTIES
    @given(histories(), TEXT, st.sampled_from((RESPONSES, MESSAGES)), st.integers(min_value=0, max_value=50))
    def test_changed_stored_history_cannot_use_the_capture(self, rows, answer, route, position):
        reply = assistant(answer)
        saved = capture(rows, reply, route)
        changed = copy.deepcopy(rows)
        changed[position % len(changed)]["content"] += " changed"
        with self.assertRaises(warm.WarmRefusal):
            build(saved, [*changed, reply], route)

    @PROPERTIES
    @given(histories(), TEXT, st.sampled_from((RESPONSES, MESSAGES)), st.integers(min_value=1, max_value=4), st.data())
    def test_new_tool_pairs_match_even_with_result_order_changes(self, rows, answer, route, count, data):
        calls = [(f"new_{index}", {"path": answer, "index": index}) for index in range(count)]
        reply = assistant(answer, calls)
        results = [tool(call_id, answer) for call_id, _ in calls]
        order = data.draw(st.permutations(results))
        saved = capture(rows, reply, route)
        body = build(saved, [*rows, reply, *order], route)
        if route[2] == "codex_responses":
            tail = body["input"][len(saved["body"]["input"]):]
            sent_calls = [item["call_id"] for item in tail if item["type"] == "function_call"]
            sent_results = [item["call_id"] for item in tail if item["type"] == "function_call_output"]
        else:
            blocks = [block for row in body["messages"][len(saved["body"]["messages"]):]
                      for block in row["content"]]
            sent_calls = [block["id"] for block in blocks if block["type"] == "tool_use"]
            sent_results = [block["tool_use_id"] for block in blocks if block["type"] == "tool_result"]
        self.assertCountEqual(sent_calls, sent_results)
        self.assertEqual(len(sent_calls), count)
        for broken in (results[:-1], [*results, results[0]], [tool("unknown", answer), *results[1:]]):
            with self.assertRaises(warm.WarmRefusal):
                build(saved, [*rows, reply, *broken], route)

    @PROPERTIES
    @given(TEXT, SEALED, SEALED)
    def test_signed_messages_payload_stays_intact_or_refuses(self, text, signature, redacted):
        rows, reply = [user(text)], assistant(text)
        blocks = [{"type": "thinking", "thinking": text, "signature": signature},
                  {"type": "redacted_thinking", "data": redacted}]
        reply.update(reasoning_details=blocks, reasoning_content=text)
        saved = capture(rows, reply, ANTHROPIC)
        before = copy.deepcopy(reply)
        body = build(saved, [*rows, reply], ANTHROPIC)
        self.assertEqual(body["messages"][-2]["content"][:2], blocks)
        self.assertEqual(reply, before)
        invalid = copy.deepcopy(reply)
        invalid["reasoning_details"][0].pop("signature")
        with self.assertRaises(warm.WarmRefusal):
            build(saved, [*rows, invalid], ANTHROPIC)
        custom = capture(rows, reply, MESSAGES)
        with self.assertRaises(warm.WarmRefusal):
            build(custom, [*rows, reply], MESSAGES)

    @PROPERTIES
    @given(TEXT, SEALED, st.sampled_from(("_issuer_model", "_issuer_kind")))
    def test_encrypted_responses_payload_stays_intact_or_refuses(self, text, sealed, stamp):
        rows = [user(text)]
        native = {"type": "reasoning", "encrypted_content": sealed,
                  "_issuer_kind": "codex_backend", "_issuer_model": CODEX[0]}
        reply = assistant(text, codex_reasoning_items=[native])
        saved = capture(rows, reply, CODEX)
        before = copy.deepcopy(reply)
        body = build(saved, [*rows, reply], CODEX)
        replayed = [item for item in body["input"] if item["type"] == "reasoning"]
        self.assertEqual([item["encrypted_content"] for item in replayed], [sealed])
        self.assertEqual(reply, before)
        invalid = copy.deepcopy(reply)
        invalid["codex_reasoning_items"][0][stamp] = "foreign"
        with self.assertRaises(warm.WarmRefusal):
            build(saved, [*rows, invalid], CODEX)


class NativeStreamProperties(unittest.TestCase):
    @PROPERTIES
    @given(TEXT, st.lists(st.integers(min_value=0, max_value=1000), max_size=12))
    def test_messages_chunk_boundaries_do_not_change_the_handoff(self, text, chunks):
        handoff_text = summary(text)
        self.assertEqual(accepted(sse(message_events(handoff_text, chunks)), "anthropic_messages"),
                         handoff_text.strip())

    @PROPERTIES
    @given(TEXT, st.sampled_from(("terminal", "block_stop", "wrong_delta", "length", "error", "duplicate_stop")))
    def test_unsafe_messages_event_sequences_cannot_yield_a_handoff(self, text, fault):
        events = message_events(summary(text), [])
        if fault == "terminal":
            events.pop()
        elif fault == "block_stop":
            events = [event for event in events if event["type"] != "content_block_stop"]
        elif fault == "wrong_delta":
            events[2]["delta"]["type"] = "thinking_delta"
        elif fault == "length":
            events[-2]["delta"]["stop_reason"] = "max_tokens"
        elif fault == "error":
            events.insert(3, {"type": "error", "error": {"type": "synthetic_error"}})
        else:
            events.append({"type": "message_stop"})
        self.assertIsNone(accepted(sse(events), "anthropic_messages"))

    @PROPERTIES
    @given(TEXT, st.sampled_from(("terminal", "done", "identity", "duplicate", "length", "tool", "refusal")))
    def test_unsafe_responses_event_sequences_cannot_yield_a_handoff(self, text, fault):
        events = response_events(summary(text))
        self.assertEqual(accepted(sse(events), "codex_responses"), summary(text).strip())
        if fault == "terminal":
            events.pop()
        elif fault == "done":
            events.pop(2)
        elif fault == "identity":
            events[2]["item"]["id"] = "msg_foreign"
        elif fault == "duplicate":
            events.insert(3, copy.deepcopy(events[2]))
        elif fault == "length":
            events[-1]["response"]["status"] = "incomplete"
        elif fault == "tool":
            events.insert(3, {"type": "response.output_item.done", "output_index": 1,
                              "item": {"id": "fc_synthetic", "type": "function_call", "name": "read"}})
        else:
            events[2]["item"]["content"].append({"type": "refusal", "refusal": "Synthetic refusal."})
        self.assertIsNone(accepted(sse(events), "codex_responses"))

    @PROPERTIES
    @given(TEXT, st.sampled_from(("codex_responses", "anthropic_messages")), st.integers(min_value=0, max_value=5000))
    def test_truncated_native_stream_cannot_yield_a_handoff(self, text, mode, position):
        events = response_events(summary(text)) if mode == "codex_responses" else message_events(summary(text), [])
        raw = sse(events)
        # The cut removes at least the final JSON brace, not only the optional SSE separator.
        cut = position % (len(raw) - 2)
        self.assertIsNone(accepted(raw[:cut], mode))

    @PROPERTIES
    @given(TEXT, st.sampled_from(("codex_responses", "anthropic_messages")), st.integers(min_value=0, max_value=5000))
    def test_malformed_native_bytes_cannot_yield_a_handoff(self, text, mode, position):
        events = response_events(summary(text)) if mode == "codex_responses" else message_events(summary(text), [])
        raw = sse(events)
        cut = position % len(raw)
        self.assertIsNone(accepted(raw[:cut] + b"\xff" + raw[cut:], mode))


class CaptureLifecycleProperties(unittest.TestCase):
    def setUp(self):
        # The separate property command has no tests/ path. Load only the existing synthetic Hermes stand-in.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        self.addCleanup(sys.path.pop, 0)
        import wc_hermes_stub
        wc_hermes_stub.install(self)

    @PROPERTIES
    @given(histories(), TEXT, st.sampled_from((RESPONSES, MESSAGES)))
    def test_summary_failure_keeps_native_history_capture_and_counters(self, rows, answer, route):
        from warm_compaction.engine import CompactionAborted, WarmCompactionEngine
        saved = capture(rows, assistant(answer), route)
        store = CaptureStore()
        engine = WarmCompactionEngine(store=store, settings={"warm": False, "tail_tokens": 1})
        engine.update_model(route[0], 200_000, base_url=route[1], api_mode=route[2],
                            provider="synthetic-provider", api_key="synthetic-key")
        engine.on_session_start("synthetic-session")
        saved["key_stamp"] = engine._key_stamp()
        store._sessions["synthetic-session"] = copy.deepcopy(saved)
        current = copy.deepcopy([*rows, assistant(answer)])
        before = copy.deepcopy(current)
        captured = copy.deepcopy(store.latest("synthetic-session"))
        with self.assertRaises(CompactionAborted):
            engine.compress(current)
        self.assertEqual(current, before)
        self.assertEqual(store.latest("synthetic-session"), captured)
        self.assertEqual(engine.compression_count, 0)
        self.assertIsNone(engine._pending_warm_result)
        self.assertEqual(engine._warm_failures, 0)
        self.assertEqual((engine.warm_last["path"], engine.warm_last["fallback_reason"]), ("aborted", "unavailable"))

    @PROPERTIES
    @given(histories(), TEXT, st.sampled_from((RESPONSES, MESSAGES)),
           st.lists(st.sampled_from(("same", "route", "provider", "credential", "session", "history")),
                    min_size=1, max_size=12))
    def test_capture_cannot_cross_identity_or_history_changes(self, rows, answer, route, actions):
        from warm_compaction.engine import WarmCompactionEngine
        reply = assistant(answer)
        saved = capture(rows, reply, route)
        store = CaptureStore()
        engine = WarmCompactionEngine(store=store)
        engine.update_model(*route[:1], 200_000, base_url=route[1], api_mode=route[2],
                            provider="synthetic-provider", api_key="synthetic-key")
        engine.on_session_start("synthetic-session")
        saved["key_stamp"] = engine._key_stamp()
        store._sessions["synthetic-session"] = copy.deepcopy(saved)
        current = copy.deepcopy([*rows, reply])
        valid = True
        for step, action in enumerate(actions):
            if action == "same":
                engine.update_model(engine._wc_route[0], 200_000, base_url=engine._wc_route[1],
                                    api_mode=engine._wc_route[2], provider=engine._wc_provider,
                                    api_key=engine._wc_api_key)
            elif action in ("route", "provider", "credential"):
                provider = f"changed-provider-{step}" if action == "provider" else engine._wc_provider
                engine.update_model("changed-model" if action == "route" else engine._wc_route[0],
                                    200_000, base_url=engine._wc_route[1], api_mode=engine._wc_route[2],
                                    provider=provider,
                                    api_key=f"changed-key-{step}" if action == "credential" else engine._wc_api_key)
                valid = False
            elif action == "session":
                engine.on_session_start(f"changed-session-{step}")
                valid = False
            else:
                current[0]["content"] += " changed"
                valid = False
            record = {}
            with patch.object(engine, "_execute", return_value={"content": summary("state"), "finish_reason": "stop",
                              "tool_calls": False, "refusal": False, "prompt_tokens": None,
                              "cached_tokens": None}) as send:
                candidate = engine._warm_summary(current, store.latest(engine._wc_session_id), None, "", (),
                                                 record, engine._attempt())
            self.assertEqual(candidate is not None, valid)
            self.assertEqual(send.called, valid)

    @PROPERTIES
    @given(TEXT, st.lists(st.sampled_from(("same", "rotate", "forget", "wrong_session")), min_size=1, max_size=10))
    def test_open_capture_is_not_published_after_key_or_session_changes(self, text, actions):
        import wc_hermes_stub
        store = CaptureStore()
        credential = ["synthetic-stamp"]
        store.set_stamp("synthetic-session", lambda: credential[0])
        rows = [user(text)]
        store.on_pre_api_request(api_request_id="synthetic-request", session_id="synthetic-session",
                                 conversation_history=rows, model=RESPONSES[0], base_url=RESPONSES[1],
                                 api_mode=RESPONSES[2])
        wc_hermes_stub.CAPTURE_CHAIN[:] = [store.on_llm_execution]
        store.on_llm_execution(api_request_id="synthetic-request", request=capture(rows, assistant(text),
                               RESPONSES)["body"], next_call=lambda: None)
        valid = True
        session = "synthetic-session"
        for step, action in enumerate(actions):
            if action == "rotate":
                credential[0] = f"changed-stamp-{step}"
                valid = False
            elif action == "forget":
                store.forget("synthetic-session")
                valid = False
            elif action == "wrong_session":
                session = "different-session"
                valid = False
        store.on_post_api_request(api_request_id="synthetic-request", session_id=session, finish_reason="stop",
                                  assistant_message=SimpleNamespace(content="Synthetic reply", tool_calls=[]))
        self.assertEqual(store.latest("synthetic-session") is not None, valid)


if __name__ == "__main__":
    unittest.main()
