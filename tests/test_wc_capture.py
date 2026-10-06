"""Tests for the capture store."""

import unittest
from types import SimpleNamespace

from warm_compaction.capture import CaptureStore, UnsupportedRequest, final_body
from warm_compaction.rows import row_digest
from wc_fixtures import user


def request(**extra):
    return {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], **extra}


def reply_object(content="done", calls=()):
    tool_calls = [SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments="{}")) for call_id, name in calls]
    return SimpleNamespace(content=content, tool_calls=tool_calls or None)


class FinalBodyTest(unittest.TestCase):
    def test_merges_extra_body_and_drops_client_options(self):
        body = final_body(request(extra_body={"top_k": 20}, timeout=30))
        self.assertEqual(body["top_k"], 20)
        self.assertNotIn("extra_body", body)
        self.assertNotIn("timeout", body)

    def test_refuses_headers_query_and_values_that_are_not_json(self):
        cases = (
            (request(extra_headers={"x": "1"}), "request_options_unsupported"),
            (request(extra_query={"q": "1"}), "request_options_unsupported"),
            (request(bad=object()), "request_not_json"),
            (["not", "a", "mapping"], "request_not_mapping"),
        )
        for value, code in cases:
            with self.subTest(code=code), self.assertRaises(UnsupportedRequest) as caught:
                final_body(value)
            self.assertEqual(str(caught.exception), code)


class CaptureStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = CaptureStore(clock=lambda: 5.0)
        self.history = [user("u")]

    def run_request(self, request_id="r1", session="s1", api_mode="chat_completions", finish="stop",
                    body=None, message=None, usage=None):
        self.store.on_pre_api_request(api_request_id=request_id, session_id=session, conversation_history=self.history,
                                      model="m", base_url="http://x/v1", api_mode=api_mode, request={"sanitized": 1})
        calls = []
        result = self.store.on_llm_execution(request=body if body is not None else request(),
                                             next_call=lambda: calls.append(1) or "response", api_request_id=request_id,
                                             original_request=None, telemetry_schema_version=1)
        self.store.on_post_api_request(api_request_id=request_id, session_id=session, finish_reason=finish,
                                       assistant_message=message or reply_object(), usage=usage)
        return result, calls

    def test_keeps_the_measured_prompt_count(self):
        self.run_request(usage={"prompt_tokens": 107_554, "input_tokens": 34, "cache_read_tokens": 107_520})
        self.assertEqual(self.store.latest("s1")["prompt_tokens"], 107_554)
        for usage in (None, {}, {"prompt_tokens": 0}, {"prompt_tokens": True}, {"prompt_tokens": "9"}):
            with self.subTest(usage=usage):
                self.run_request(request_id="r2", usage=usage)
                self.assertIsNone(self.store.latest("s1")["prompt_tokens"])

    def test_joins_the_three_calls(self):
        result, calls = self.run_request()
        self.assertEqual((result, calls), ("response", [1]))
        capture = self.store.latest("s1")
        self.assertEqual(capture["digests"], [row_digest(self.history[0])])
        self.assertEqual(capture["route"], ("m", "http://x/v1", "chat_completions"))
        self.assertEqual(capture["body"]["messages"][1]["content"], "u")
        self.assertEqual(capture["reply"], {"role": "assistant", "content": "done", "tool_calls": []})
        self.assertEqual((capture["finish_reason"], capture["captured_at"]), ("stop", 5.0))

    def test_a_repeated_call_id_gets_the_hermes_rename(self):
        # Hermes renames a repeated id in one reply (c1, c1 -> c1, c1_d2) after this hook runs.
        self.run_request(finish="tool_calls", message=reply_object("", [("c1", "read"), ("c1", "ls"), ("c1", "cat")]))
        self.assertEqual(self.store.latest("s1")["reply"]["tool_calls"],
                         [["c1", "read"], ["c1_d2", "ls"], ["c1_d3", "cat"]])

    def test_no_body_when_an_execution_middleware_runs_after_the_capture(self):
        # A later middleware can change the request after the capture saw it. The capture cannot see that change.
        import wc_hermes_stub
        wc_hermes_stub.install(self)

        def other(request=None, next_call=None, **context):
            return next_call()
        for chain, kept in (([self.store.on_llm_execution, other], False), ([other, self.store.on_llm_execution], True)):
            wc_hermes_stub.PLUGINS._delivery_manager = lambda chain=chain: SimpleNamespace(
                _middleware={"llm_execution": chain})
            self.addCleanup(lambda: wc_hermes_stub.PLUGINS.__dict__.pop("_delivery_manager", None))
            with self.subTest(kept=kept):
                self.run_request()
                capture = self.store.latest("s1")
                self.assertEqual(capture["body"] is not None, kept)
                self.assertEqual(capture.get("refusal"), None if kept else "middleware_after_capture")

    def test_keeps_tool_call_ids_and_names(self):
        self.run_request(finish="tool_calls", message=reply_object("", [("c1", "read")]))
        self.assertEqual(self.store.latest("s1")["reply"]["tool_calls"], [["c1", "read"]])

    def test_other_api_mode_keeps_no_body(self):
        self.run_request(api_mode="codex_responses")
        self.assertIsNone(self.store.latest("s1")["body"])

    def test_unusable_finish_reason_keeps_no_capture(self):
        self.run_request(finish="length")
        self.assertIsNone(self.store.latest("s1"))

    def test_session_mismatch_keeps_no_capture(self):
        self.store.on_pre_api_request(api_request_id="r1", session_id="s1", conversation_history=[], model="m",
                                      base_url="b", api_mode="chat_completions")
        self.store.on_post_api_request(api_request_id="r1", session_id="s2", finish_reason="stop",
                                       assistant_message=reply_object())
        self.assertIsNone(self.store.latest("s1"))
        self.assertIsNone(self.store.latest("s2"))

    def test_capture_error_never_stops_the_request(self):
        class Broken(dict):
            def items(self):
                raise RuntimeError("broken")

        result, calls = self.run_request(body=Broken(model="m"))
        self.assertEqual((result, calls), ("response", [1]))
        self.assertIsNone(self.store.latest("s1")["body"])

    def test_downstream_errors_pass_through(self):
        self.store.on_pre_api_request(api_request_id="r1", session_id="s1", conversation_history=[], model="m",
                                      base_url="b", api_mode="chat_completions")

        def fail():
            raise ConnectionError("down")

        with self.assertRaises(ConnectionError):
            self.store.on_llm_execution(request=request(), next_call=fail, api_request_id="r1")

    def test_limits_open_requests_and_sessions(self):
        store = CaptureStore(max_open=2, max_sessions=2)
        for index in range(3):
            store.on_pre_api_request(api_request_id=f"r{index}", session_id="s", conversation_history=[],
                                     model="m", base_url="b", api_mode="chat_completions")
        store.on_post_api_request(api_request_id="r0", session_id="s", finish_reason="stop",
                                  assistant_message=reply_object())
        self.assertIsNone(store.latest("s"))
        for index in range(3):
            store.on_pre_api_request(api_request_id=f"q{index}", session_id=f"s{index}", conversation_history=[],
                                     model="m", base_url="b", api_mode="chat_completions")
            store.on_post_api_request(api_request_id=f"q{index}", session_id=f"s{index}", finish_reason="stop",
                                      assistant_message=reply_object())
        self.assertIsNone(store.latest("s0"))
        self.assertIsNotNone(store.latest("s2"))

    def test_forget_removes_a_session(self):
        self.run_request()
        self.store.forget(session_id="s1", reason="end")
        self.assertIsNone(self.store.latest("s1"))

    def test_latest_returns_a_copy(self):
        self.run_request()
        self.store.latest("s1")["body"]["messages"].clear()
        self.assertEqual(len(self.store.latest("s1")["body"]["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
