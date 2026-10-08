"""Tests for the capture store."""

import copy
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import wc_hermes_stub
from warm_compaction.capture import CaptureStore, UnsupportedRequest, final_body, key_stamp
from warm_compaction.rows import row_digest
from wc_fixtures import user


def request(**extra):
    return {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], **extra}


def reply_object(content="done", calls=()):
    tool_calls = [SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments="{}"))
                  for call_id, name in calls]
    return SimpleNamespace(content=content, tool_calls=tool_calls or None)


class KeyStampTest(unittest.TestCase):
    def test_stamps_are_stable_and_distinct(self):
        keys = ("synthetic-key-a", "synthetic-key-b", "", "synthetic-\u03ba\u03bb\u03b5\u03b9\u03b4\u03af")
        stamps = []
        for key in keys:
            with self.subTest(key=key):
                stamp = key_stamp(key)
                self.assertEqual(stamp, key_stamp(key))
                self.assertRegex(stamp, r"^[0-9a-f]{64}$")
                stamps.append(stamp)
        self.assertEqual(len(set(stamps)), len(keys))

    def test_fresh_interpreters_have_different_stamps(self):
        source_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from warm_compaction.capture import key_stamp; "
            "print(key_stamp('synthetic-key-a'))"
        )
        command = [sys.executable, "-I", "-B", "-c", code, str(source_root)]
        stamps = [subprocess.check_output(command, text=True, timeout=10).strip() for _ in range(2)]
        self.assertNotEqual(stamps[0], stamps[1])


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
            (request(extra_headers={"x-opencode-session": "id"}, extra_body={"extra_headers": {}}),
             "request_options_unsupported"),
            (request(bad=object()), "request_not_json"),
            (["not", "a", "mapping"], "request_not_mapping"),
        )
        for value, code in cases:
            with self.subTest(code=code), self.assertRaises(UnsupportedRequest) as caught:
                final_body(value)
            self.assertEqual(str(caught.exception), code)


class CaptureStoreTest(unittest.TestCase):
    def setUp(self):
        wc_hermes_stub.install(self)
        self.store = CaptureStore(clock=lambda: 5.0)
        wc_hermes_stub.CAPTURE_CHAIN.append(self.store.on_llm_execution)
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

    def test_session_header_is_separate_and_copied(self):
        body = request(extra_headers={"X-OpenCode-Session": "synthetic-session"})
        before = copy.deepcopy(body)
        self.run_request(body=body)
        capture = self.store.latest("s1")
        self.assertIsNone(capture["refusal"])
        self.assertEqual(capture["request_headers"], before["extra_headers"])
        self.assertNotIn("extra_headers", capture["body"])
        self.assertEqual(body, before)
        body["extra_headers"]["X-OpenCode-Session"] = "changed"
        capture["request_headers"].clear()
        self.assertEqual(self.store.latest("s1")["request_headers"], before["extra_headers"])

    def test_invalid_session_headers_are_not_kept(self):
        for headers in (
            {"x-opencode-session": ""}, {"x-opencode-session": " "},
            {"x-opencode-session": "id\r\nAuthorization: private"},
            {"x-opencode-session": "id\x00"}, {"x-opencode-session": "id\x7f"},
            {"x-opencode-session": "id\t"}, {"x-opencode-session": "id\u00e9"},
            {"x-opencode-session": 7}, {"x-opencode-session": None},
            {"x-opencode-session": "id", "Authorization": "private"},
            {"x-opencode-session": "id", "X-OpenCode-Session": "other"},
            {"x-other-session": "private"}, ["private"], "private",
        ):
            with self.subTest(headers=headers):
                self.run_request(body=request(extra_headers=headers))
                capture = self.store.latest("s1")
                self.assertEqual(capture["refusal"], "request_options_unsupported")
                self.assertIsNone(capture["body"])
                self.assertEqual(capture.get("request_headers", {}), {})
                self.assertNotIn("private", repr(capture))

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

    def test_keeps_fixed_request_refusals_and_runs_the_request_unchanged(self):
        cases = (
            (["synthetic-private-request"], "request_not_mapping"),
            (request(extra_headers={"Authorization": "synthetic-private-key"}), "request_options_unsupported"),
            (request(extra_query={"token": "synthetic-private-key"}), "request_options_unsupported"),
            (request(extra_body=["synthetic-private-request"]), "request_options_unsupported"),
            (request(value=SimpleNamespace(secret="synthetic-private-value")), "request_not_json"),
            (request(value=float("nan")), "request_not_json"),
            (request(value=float("inf")), "request_not_json"),
        )
        for body, code in cases:
            with self.subTest(code=code):
                before = body.copy()
                result, calls = self.run_request(body=body)
                capture = self.store.latest("s1")
                self.assertEqual((result, calls), ("response", [1]))
                self.assertEqual(body, before)
                self.assertEqual((capture["body"], capture["refusal"]), (None, code))
                self.assertNotIn("synthetic-private", repr(capture))

    def test_unknown_exception_data_does_not_become_a_refusal(self):
        for data in ("synthetic-private-exception", SimpleNamespace(secret="synthetic-private-value")):
            with self.subTest(data_type=type(data).__name__), patch(
                    "warm_compaction.capture.final_body", side_effect=UnsupportedRequest(data)):
                result, calls = self.run_request()
                capture = self.store.latest("s1")
                self.assertEqual((result, calls), ("response", [1]))
                self.assertEqual((capture["body"], capture["refusal"]), (None, "settings_unsupported"))
                self.assertNotIn("synthetic-private", repr(capture))

    def test_a_valid_capture_replaces_an_earlier_request_refusal(self):
        self.run_request(body=request(extra_headers={"x": "synthetic-private-value"}))
        self.assertEqual(self.store.latest("s1")["refusal"], "request_options_unsupported")
        self.run_request(request_id="r2")
        capture = self.store.latest("s1")
        self.assertEqual(capture["body"], request())
        self.assertIsNone(capture["refusal"])

    def test_no_body_when_an_execution_middleware_runs_after_the_capture(self):
        # A later middleware can change the request after the capture saw it. The capture cannot see that change.
        def other(request=None, next_call=None, **context):
            return next_call()
        own = self.store.on_llm_execution
        for chain, kept in (([own, other], False), ([other, own], True)):
            wc_hermes_stub.CAPTURE_CHAIN[:] = chain
            with self.subTest(kept=kept):
                self.run_request()
                capture = self.store.latest("s1")
                self.assertEqual(capture["body"] is not None, kept)
                self.assertEqual(capture.get("refusal"), None if kept else "middleware_after_capture")

    def test_no_body_when_the_middleware_order_cannot_be_read(self):
        # A later Hermes can keep the chain in another form. Then a later middleware can change the request
        # without this capture seeing it: fail closed.
        def broken():
            raise AttributeError("_middleware")
        cases = (("no manager", lambda: wc_hermes_stub.PLUGINS.__dict__.pop("_delivery_manager")),
                 ("manager fails", lambda: setattr(wc_hermes_stub.PLUGINS, "_delivery_manager", broken)),
                 ("capture not in the chain", wc_hermes_stub.CAPTURE_CHAIN.clear))
        manager = wc_hermes_stub.PLUGINS._delivery_manager
        for name, change in cases:
            wc_hermes_stub.CAPTURE_CHAIN[:] = [self.store.on_llm_execution]
            wc_hermes_stub.PLUGINS._delivery_manager = manager
            change()
            with self.subTest(name):
                self.run_request()
                capture = self.store.latest("s1")
                self.assertEqual((capture["body"], capture["refusal"]), (None, "middleware_order_unknown"))
        wc_hermes_stub.PLUGINS._delivery_manager = manager

    def test_keeps_tool_call_ids_and_names(self):
        self.run_request(finish="tool_calls", message=reply_object("", [("c1", "read")]))
        self.assertEqual(self.store.latest("s1")["reply"]["tool_calls"], [["c1", "read"]])

    def test_other_api_mode_keeps_no_body(self):
        self.run_request(api_mode="bedrock_converse")
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
    def test_forget_during_digest_work_cannot_restore_an_open_request(self):
        def digest(row):
            self.store.forget("s1")
            return row_digest(row)
        with patch("warm_compaction.capture.row_digest", side_effect=digest):
            self.run_request()
        self.assertIsNone(self.store.latest("s1"))
        self.assertEqual(len(self.store._open), 0)

    def test_forget_and_reopen_during_digest_work_refuses_the_old_token(self):
        def digest(row):
            self.store.forget("s1")
            self.store.open_session("s1")
            return row_digest(row)
        with patch("warm_compaction.capture.row_digest", side_effect=digest):
            self.run_request()
        self.assertIsNone(self.store.latest("s1"))
        self.run_request(request_id="r2")
        self.assertIsNotNone(self.store.latest("s1"))

    def test_forget_during_moa_finish_cannot_publish_the_old_capture(self):
        store = self.store
        class Moa:
            def pre_main(self, **kwargs):
                pass
            def run_main(self, request_id, next_call):
                return next_call()
            def finish_main(self, request_id, capture):
                result = {"code": None, "aggregator": {"synthetic": True}, "references": []}
                store.forget("s1")
                return result
            def forget(self, **kwargs):
                pass
        self.store._moa = Moa()
        self.run_request()
        self.assertIsNone(self.store.latest("s1"))

    def test_forget_and_reopen_during_post_refuses_the_old_capture(self):
        store = self.store
        class Moa:
            first = True
            def pre_main(self, **kwargs):
                pass
            def run_main(self, request_id, next_call):
                return next_call()
            def finish_main(self, request_id, capture):
                if self.first:
                    self.first = False
                    store.forget("s1")
                    store.open_session("s1")
                return {"code": None, "aggregator": None, "references": []}
            def forget(self, **kwargs):
                pass
            def open_session(self, *args, **kwargs):
                pass
        self.store._moa = Moa()
        self.run_request()
        self.assertIsNone(self.store.latest("s1"))
        self.run_request(request_id="r2")
        self.assertIsNotNone(self.store.latest("s1"))

    def test_callbacks_cannot_reopen_a_forgotten_session(self):
        self.run_request()
        self.store.forget("s1")
        result, calls = self.run_request(request_id="r2")
        self.assertEqual((result, calls), ("response", [1]))
        self.assertIsNone(self.store.latest("s1"))
        self.store.open_session("s1")
        self.run_request(request_id="r3")
        self.assertIsNotNone(self.store.latest("s1"))

    def test_token_eviction_invalidates_digest_work_in_progress(self):
        self.store = CaptureStore(max_sessions=1)
        def digest(row):
            self.store.open_session("other")
            return row_digest(row)
        with patch("warm_compaction.capture.row_digest", side_effect=digest):
            self.run_request()
        self.assertIsNone(self.store.latest("s1"))
        self.assertEqual(len(self.store._open), 0)
        self.assertEqual(len(self.store._versions), 1)

    def test_moa_gets_virtual_route_and_an_old_epoch_guard(self):
        from warm_compaction.moa import MoaStore
        tracker = MoaStore([])
        self.store._moa = tracker
        original = tracker.pre_main
        received = []
        def delayed(**kwargs):
            received.append(kwargs)
            self.store.forget("s1")
            self.store.open_session("s1")
            original(**kwargs)
        tracker.pre_main = delayed
        self.store.on_pre_api_request(api_request_id="r1", session_id="s1", turn_id="t1", provider="moa",
                                      conversation_history=self.history, model="preset", base_url="moa://local",
                                      api_mode="chat_completions")
        self.assertEqual((received[0]["model"], received[0]["base_url"], received[0]["api_mode"]),
                         ("preset", "moa://local", "chat_completions"))
        self.assertFalse(received[0]["session_check"]())
        self.assertEqual(len(tracker._mains), 0)
        self.assertEqual(len(self.store._open), 0)

    def test_lifecycle_tokens_do_not_leave_the_capture_store(self):
        self.run_request()
        self.assertNotIn("_token", self.store.latest("s1"))
        for index in range(20):
            self.store.forget(f"closed-{index}")
        self.assertLessEqual(len(self.store._versions), self.store._max_sessions)

    def test_delayed_moa_forget_cannot_close_an_explicit_reopen(self):
        from warm_compaction.moa import MoaStore
        tracker = MoaStore([])
        self.store._moa = tracker
        self.store.open_session("s1")
        original = tracker.forget
        def delayed(**kwargs):
            self.store.open_session("s1")
            original(**kwargs)
        tracker.forget = delayed
        self.store.forget("s1")
        self.assertIsNotNone(tracker._versions["s1"])
        self.run_request(request_id="r2")
        self.assertIsNotNone(self.store.latest("s1"))

    def test_cross_store_reopen_invalidates_an_old_aux_copy(self):
        from warm_compaction.moa import MoaStore, _body
        from test_wc_moa import REF_ROUTE, ROUTE, event
        tracker = MoaStore([ROUTE, REF_ROUTE], include_references=True)
        self.store._moa = tracker
        self.store.open_session("s1")
        original = tracker.forget
        def delayed(**kwargs):
            self.store.open_session("s1")
            original(**kwargs)
        tracker.forget = delayed
        def body(*args):
            result = _body(*args)
            self.store.forget("s1")
            return result
        with patch("warm_compaction.moa._body", side_effect=body):
            tracker.on_pre_auxiliary_call(**event(REF_ROUTE, task="moa_reference", request_id="r1"))
        self.assertEqual(len(tracker._references), 0)
        self.assertEqual(len(tracker._pending), 0)

    def test_explicit_reopens_keep_both_token_maps_bounded(self):
        from warm_compaction.moa import MAX_SESSIONS, MoaStore
        tracker = MoaStore([])
        self.store._moa = tracker
        for index in range(MAX_SESSIONS + 4):
            self.store.open_session(f"s{index}")
        self.assertLessEqual(len(self.store._versions), self.store._max_sessions)
        self.assertLessEqual(len(tracker._versions), MAX_SESSIONS)
        self.assertLessEqual(len(tracker._capture_versions), MAX_SESSIONS)


if __name__ == "__main__":
    unittest.main()
