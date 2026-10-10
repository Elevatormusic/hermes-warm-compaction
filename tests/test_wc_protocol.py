"""Tests for native reply completion, request headers, and history stamps."""

import copy
import json
import unittest
from unittest.mock import patch

from warm_compaction.capture import UnsupportedRequest, final_body, request_headers
from warm_compaction.protocol import MAX_DONE_ITEMS, envelope
from warm_compaction.rows import SendPolicy, row_digest, sent_tokens
from warm_compaction.warm import WarmRefusal, fits, send


def sse(events):
    """Encode synthetic events."""
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()


class EnvelopeTest(unittest.TestCase):
    def test_responses_requires_one_completed_terminal(self):
        response = {"status": "completed", "output": [{"type": "message", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": "synthetic"}]}]}
        event = {"type": "response.completed", "response": response}
        self.assertEqual(envelope(sse([event]), "codex_responses"), response)
        for events in ([{"type": "response.output_text.delta", "delta": "partial"}], [event, event],
                       [event, {"type": "response.failed"}], [{"type": "response.incomplete"}]):
            with self.subTest(events=events), self.assertRaises(WarmRefusal):
                envelope(sse(events), "codex_responses")

    def done_message(self, index=0, item_id="msg_synthetic", **changes):
        return {"type": "response.output_item.done", "output_index": index, "item": {
            "id": item_id, "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "synthetic summary", "annotations": []}], **changes}}

    def completed(self, **changes):
        return {"type": "response.completed", "response": {"status": "completed", "output": None, **changes}}

    def test_completed_responses_fill_null_empty_and_absent_output_from_done_items(self):
        from warm_compaction.responses import parse_reply
        done = self.done_message()
        for output in (None, [], "absent"):
            terminal = self.completed(usage={"input_tokens": 12, "output_tokens": 3})
            if output == "absent":
                terminal["response"].pop("output")
            else:
                terminal["response"]["output"] = output
            before = copy.deepcopy((done, terminal))
            with self.subTest(output=output):
                payload = envelope(sse([done, terminal]), "codex_responses")
                self.assertEqual(payload["output"], [done["item"]])
                self.assertEqual(parse_reply(payload)["content"], "synthetic summary")
                self.assertEqual(payload["usage"], {"input_tokens": 12, "output_tokens": 3})
                self.assertEqual((done, terminal), before)

    def test_done_items_use_output_order_and_match_added_items(self):
        reasoning = {"type": "response.output_item.done", "output_index": 0, "item": {
            "id": "rs_synthetic", "type": "reasoning", "summary": [], "encrypted_content": "fake"}}
        message = self.done_message(1)
        added = {"type": "response.output_item.added", "output_index": 1, "item": {
            "id": message["item"]["id"], "type": "message", "role": "assistant", "status": "in_progress",
            "content": []}}
        payload = envelope(sse([added, message, reasoning, self.completed()]), "codex_responses")
        self.assertEqual(payload["output"], [reasoning["item"], message["item"]])

    def test_missing_done_items_and_delta_only_text_are_refused(self):
        delta = {"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_synthetic",
                 "content_index": 0, "delta": "synthetic partial summary"}
        for events in ([self.completed()], [delta, self.completed()], [self.done_message()],
                       [self.done_message(), {"type": "response.incomplete", "response": {"status": "incomplete"}}],
                       [self.done_message(), {"type": "response.failed"}],
                       [self.done_message(), {"type": "error"}, self.completed()],
                       [self.done_message(), {"type": "response.error"}, self.completed()]):
            with self.subTest(events=events), self.assertRaises(WarmRefusal) as caught:
                envelope(sse(events), "codex_responses")
            self.assertEqual(caught.exception.code, "incomplete_response")

    def test_malformed_incomplete_and_nontext_done_items_are_refused(self):
        cases = [self.done_message(status="in_progress"), self.done_message(role="user"),
                 self.done_message(id=""), self.done_message(content=None), self.done_message(content=[]),
                 self.done_message(content=[{"type": "output_text", "text": None}]),
                 self.done_message(content=[{"type": "output_text", "text": " "}]),
                 self.done_message(content=[{"type": "image", "data": "fake"}]),
                 self.done_message(phase="commentary"),
                 {"type": "response.output_item.done", "output_index": 0, "item": []},
                 {"type": "response.output_item.done", "output_index": 0, "item": {
                     "id": "rs_synthetic", "type": "reasoning", "summary": []}},
                 {"type": "response.output_item.done", "output_index": 0, "item": {
                     "id": "rs_synthetic", "type": "reasoning", "summary": "wrong"}}]
        for done in cases:
            with self.subTest(done=done), self.assertRaises(WarmRefusal):
                envelope(sse([done, self.completed()]), "codex_responses")

    def test_done_indexes_ids_and_terminal_output_cannot_conflict(self):
        done = self.done_message()
        added = {"type": "response.output_item.added", "output_index": 0, "item": {
            "id": "msg_synthetic", "type": "message"}}
        wrong_added = copy.deepcopy(added)
        wrong_added["item"]["id"] = "msg_other"
        wrong_terminal = self.completed(output=[{**done["item"], "content": [{
            "type": "output_text", "text": "changed"}]}])
        cases = [[done, done, self.completed()], [done, self.done_message(item_id="other"), self.completed()],
                 [done, self.done_message(1), self.completed()], [wrong_added, done, self.completed()],
                 [added, added, done, self.completed()], [done, wrong_terminal]]
        for index in (-1, 1, False, "0", MAX_DONE_ITEMS):
            cases.append([self.done_message(index), self.completed()])
        for events in cases:
            with self.subTest(events=events), self.assertRaises(WarmRefusal):
                envelope(sse(events), "codex_responses")

    def test_completed_done_items_reject_invalid_terminal_and_trailing_events(self):
        done = self.done_message()
        terminals = [self.completed(status="incomplete"), self.completed(output="invalid"),
                     self.completed(error={"code": "synthetic_error"}),
                     self.completed(incomplete_details={"reason": "synthetic_limit"})]
        for terminal in terminals:
            with self.subTest(terminal=terminal), self.assertRaises(WarmRefusal):
                envelope(sse([done, terminal]), "codex_responses")
        for trailing in ({"type": "response.output_text.delta", "delta": "trailing"},
                         self.done_message(), self.completed(), {"type": "response.in_progress"}):
            with self.subTest(trailing=trailing), self.assertRaises(WarmRefusal):
                envelope(sse([done, self.completed(), trailing]), "codex_responses")

    def test_progress_cannot_change_an_announced_id_or_reopen_a_done_item(self):
        done = self.done_message()
        added = {"type": "response.output_item.added", "output_index": 0, "item": {
            "id": "msg_synthetic", "type": "message"}}
        delta = {"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_other", "delta": "text"}
        for events in ([added, delta, done, self.completed()], [done, delta, self.completed()]):
            with self.subTest(events=events), self.assertRaises(WarmRefusal):
                envelope(sse(events), "codex_responses")
        for index, item_id in ((1, "msg_missing"), (0, "msg_other"), (False, "msg_synthetic")):
            delta = {"type": "response.output_text.delta", "output_index": index, "item_id": item_id, "delta": "text"}
            with self.subTest(index=index, item_id=item_id), self.assertRaises(WarmRefusal):
                envelope(sse([delta, done, self.completed()]), "codex_responses")

    def anthropic_events(self):
        return [{"type": "message_start", "message": {
            "type": "message", "role": "assistant", "usage": {"input_tokens": 12}, "content": []}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "synthetic"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"}]

    def test_anthropic_requires_stop_and_closed_blocks(self):
        events = self.anthropic_events()
        payload = envelope(sse(events), "anthropic_messages")
        self.assertEqual(payload["content"], [{"type": "text", "text": "synthetic"}])
        self.assertEqual(payload["usage"], {"input_tokens": 12, "output_tokens": 3})
        for bad in (events[:-1], [*events[:3], *events[4:]], [*events, {"type": "error"}]):
            with self.subTest(events=bad), self.assertRaises(WarmRefusal):
                envelope(sse(bad), "anthropic_messages")

    def test_json_and_invalid_utf8(self):
        self.assertEqual(envelope(b'{"status":"completed"}', "codex_responses"), {"status": "completed"})
        for raw in (b'\xff', b'[]', b'data: {broken}\n\n'):
            with self.subTest(raw=raw), self.assertRaises(WarmRefusal):
                envelope(raw, "codex_responses")

    def test_native_stream_refuses_wrong_delta_and_block_index(self):
        for index in (-1, 2, False):
            events = self.anthropic_events()
            events[1]["index"] = index
            with self.subTest(index=index), self.assertRaises(WarmRefusal):
                envelope(sse(events), "anthropic_messages")
        events = self.anthropic_events()
        events[2]["delta"] = {"type": "thinking_delta", "thinking": "wrong block"}
        with self.assertRaises(WarmRefusal):
            envelope(sse(events), "anthropic_messages")


class NativeCaptureTest(unittest.TestCase):
    def test_headers_stay_out_of_json(self):
        headers = {"session_id": "synthetic-session", "x-client-request-id": "synthetic-cache"}
        request = {"model": "fake", "input": [], "extra_headers": headers}
        self.assertEqual(request_headers(request, "codex_responses"), headers)
        self.assertNotIn("extra_headers", final_body(request, "codex_responses"))
        for bad in ({"Authorization": "fake"}, {"session_id": "ok", "Session_ID": "other"},
                    {"session_id": "bad\r\nheader"}):
            with self.subTest(headers=bad), self.assertRaises(UnsupportedRequest):
                request_headers({"extra_headers": bad}, "codex_responses")

    def test_native_sidecar_changes_history_stamp(self):
        row = {"role": "assistant", "content": "synthetic"}
        for field in ("phase", "codex_message_items", "codex_reasoning_items", "reasoning_details",
                      "anthropic_content_blocks"):
            changed = copy.deepcopy(row)
            changed[field] = "synthetic-new-value"
            with self.subTest(field=field):
                self.assertNotEqual(row_digest(row), row_digest(changed))


class NativeSendTest(unittest.TestCase):
    def test_responses_dispatch_keeps_query_and_reads_terminal(self):
        calls = []
        payload = {"status": "completed", "output": [{"type": "message", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": "synthetic summary"}]}]}
        def post(url, data, headers, timeout_s):
            calls.append((url, json.loads(data), headers))
            return 200, sse([{"type": "response.completed", "response": payload}])
        result = send({"input": [], "stream": True}, "http://127.0.0.1:9/v1?api-version=fake", "fake",
                      post=post, api_mode="codex_responses")
        self.assertEqual(calls[0][0], "http://127.0.0.1:9/v1/responses?api-version=fake")
        self.assertEqual(calls[0][2]["Accept"], "text/event-stream")
        self.assertEqual(result["content"], "synthetic summary")
        self.assertIsNone(result["cached_tokens"])

    def test_messages_dispatch_uses_native_auth_and_endpoint(self):
        calls = []
        payload = {"type": "message", "role": "assistant", "stop_reason": "end_turn",
                   "content": [{"type": "text", "text": "synthetic summary"}]}
        def post(url, data, headers, timeout_s):
            calls.append((url, headers))
            return 200, json.dumps(payload).encode()
        for path in ("/v1", ""):
            result = send({"messages": []}, "http://127.0.0.1:9" + path, "fake", post=post,
                          api_mode="anthropic_messages")
            self.assertEqual(calls[-1][0], "http://127.0.0.1:9/v1/messages")
            self.assertEqual(calls[-1][1]["x-api-key"], "fake")
            self.assertNotIn("Authorization", calls[-1][1])
            self.assertEqual(result["finish_reason"], "stop")

    def test_capacity_counts_native_instructions(self):
        self.assertFalse(fits({"input": [], "instructions": "x" * 40000, "max_output_tokens": 2048},
                              10000, api_mode="codex_responses"))
        self.assertFalse(fits({"messages": [], "system": "x" * 40000, "max_tokens": 2048},
                              10000, api_mode="anthropic_messages"))


class NativeTailTest(unittest.TestCase):
    def native_history(self, mode, replay_chars):
        """Make a fresh engine and synthetic replay history with no captured request."""
        from test_wc_engine import EngineTest, old_turns
        fixture = EngineTest("test_threshold_comes_from_the_setting")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        engine = fixture.make(warm=False)
        engine.update_model(model="fake", context_length=16384, base_url="http://127.0.0.1:9/v1",
                            api_key="fake", provider="custom", api_mode=mode)
        replay = ({"codex_reasoning_items": [{"type": "reasoning", "encrypted_content": "x" * replay_chars}]}
                  if mode == "codex_responses" else {"anthropic_content_blocks": [
                      {"type": "redacted_thinking", "data": "x" * replay_chars},
                      {"type": "text", "text": "synthetic"}]})
        rows = [*old_turns(3), {"role": "user", "content": "synthetic last request"},
                {"role": "assistant", "content": "synthetic", **replay}]
        return fixture, engine, rows

    def test_native_replay_above_threshold_without_capture_keeps_history(self):
        for mode in ("codex_responses", "anthropic_messages"):
            with self.subTest(mode=mode):
                fixture, engine, rows = self.native_history(mode, 36000)
                self.assertIsNone(engine._store.latest(engine._wc_session_id))
                self.assertGreater(sent_tokens(rows[-1], engine._policy(rows)), engine.threshold_tokens)
                self.assertLess(sent_tokens(rows[-1], engine._policy(rows)) + 4096, engine.context_length)
                fixture.assert_aborted(engine, rows, "capacity")
                self.assertEqual(engine.compression_count, 0)
                self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("aborted", "disabled"))

    def test_native_replay_above_unknown_overhead_budget_keeps_history(self):
        for mode in ("codex_responses", "anthropic_messages"):
            with self.subTest(mode=mode):
                fixture, engine, rows = self.native_history(mode, 20000)
                replay_tokens = sent_tokens(rows[-1], engine._policy(rows))
                self.assertGreater(replay_tokens, engine.threshold_tokens // 2)
                self.assertLess(replay_tokens, engine.threshold_tokens)
                fixture.assert_aborted(engine, rows, "capacity")
                self.assertEqual(engine.compression_count, 0)
                self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("aborted", "disabled"))

    def test_native_replay_with_known_overhead_must_fit_below_threshold(self):
        from warm_compaction import anthropic, responses
        for mode in ("codex_responses", "anthropic_messages"):
            with self.subTest(mode=mode):
                fixture, engine, rows = self.native_history(mode, 30000)
                from warm_compaction.engine import request_overhead
                system = "synthetic system " + "s" * 4000
                body = ({"model": "fake", "instructions": system, "store": False,
                         "input": responses.wire_rows(rows[:-1]), "max_output_tokens": 4096}
                        if mode == "codex_responses" else {"model": "fake", "system": system,
                            "messages": anthropic._rows(rows[:-1]), "max_tokens": 4096})
                capture = {"route": engine._wc_route, "digests": [row_digest(row) for row in rows[:-1]], "body": body,
                           "reply": {"content": rows[-1]["content"], "tool_calls": []}}
                with patch.object(fixture.store, "latest", return_value=capture):
                    self.assertIs(engine._budget_capture(capture, rows), capture)
                    total = sent_tokens(rows[-1], engine._policy(rows)) + request_overhead(capture, rows)
                    self.assertGreater(total, engine.threshold_tokens)
                    self.assertLess(total + 4096, engine.context_length)
                    fixture.assert_aborted(engine, rows, "capacity")
                self.assertEqual(engine.compression_count, 0)
                self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("aborted", "disabled"))

    def test_small_native_replay_without_capture_can_compact(self):
        from warm_compaction.rows import estimate_tokens, sent_rows
        for mode in ("codex_responses", "anthropic_messages"):
            with self.subTest(mode=mode):
                _fixture, engine, rows = self.native_history(mode, 5000)
                before = copy.deepcopy(rows)
                result = engine.compress(rows)
                self.assertIsNot(result, rows)
                self.assertEqual(rows, before)
                self.assertEqual(result[-1], rows[-1])
                self.assertLess(estimate_tokens(sent_rows(result, engine._policy(rows))), engine.threshold_tokens // 2)
                self.assertEqual(engine.compression_count, 1)
                self.assertEqual(engine.warm_last["path"], "fallback")

    def test_transformed_native_settings_cannot_supply_a_budget_capture(self):
        from test_wc_engine import EngineTest
        from warm_compaction.responses import wire_rows
        fixture = EngineTest("test_threshold_comes_from_the_setting")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        engine = fixture.engine
        engine.update_model(model="fake", context_length=16384, base_url="http://127.0.0.1:9/v1",
                            api_key="fake", provider="custom", api_mode="codex_responses")
        rows = [{"role": "user", "content": "synthetic"}]
        capture = {"route": engine._wc_route, "digests": [row_digest(rows[0])], "body": {
            "model": "fake", "instructions": "synthetic system", "store": False, "input": wire_rows(rows)},
            "reply": {"content": "synthetic reply", "tool_calls": []}}
        rows.append({"role": "assistant", "content": "synthetic reply"})
        self.assertIs(engine._budget_capture(capture, rows), capture)
        for field, value in (("instructions", ""), ("instructions", " synthetic system "), ("store", None)):
            bad = copy.deepcopy(capture)
            if field == "store":
                bad["body"].pop(field)
            else:
                bad["body"][field] = value
            with self.subTest(field=field, value=value):
                self.assertIsNone(engine._budget_capture(bad, rows))

    def test_native_replay_counts_and_cannot_be_cut(self):
        from warm_compaction.layout import bound_tail
        for mode, field in (("codex_responses", "codex_reasoning_items"),
                            ("anthropic_messages", "anthropic_content_blocks")):
            row = {"role": "assistant", "content": "synthetic", field: [{"data": "x" * 50000}]}
            policy = SendPolicy(echo=False, cut_reasoning=False, native_mode=mode)
            with self.subTest(mode=mode):
                self.assertGreater(sent_tokens(row, policy), 10000)
                self.assertEqual(bound_tail([row], 100, policy=policy), [row])

    def test_an_indivisible_native_tail_over_capacity_keeps_history(self):
        from test_wc_engine import EngineTest, old_turns
        fixture = EngineTest("test_threshold_comes_from_the_setting")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        engine = fixture.engine
        engine.update_model(model="fake", context_length=16384, base_url="http://127.0.0.1:9/v1",
                            api_key="fake", provider="custom", api_mode="codex_responses")
        rows = [*old_turns(3), {"role": "user", "content": "synthetic last request"},
                {"role": "assistant", "content": "synthetic", "codex_reasoning_items": [{"data": "x" * 100000}]}]
        fixture.assert_aborted(engine, rows, "capacity")
        self.assertEqual(engine.compression_count, 0)
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("aborted", "no_capture"))


if __name__ == "__main__":
    unittest.main()
