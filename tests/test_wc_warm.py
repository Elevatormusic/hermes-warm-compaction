"""Tests for the warm request."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from warm_compaction.rows import estimate_tokens
from warm_compaction.warm import (
    DEFAULT_RESERVE, SAFETY, WarmRefusal, build_request, check_settings, fits, send, split_history, urllib_post,
    wire_row,
)
from wc_fixtures import ROUTE, assistant, capture_for, tool, user

INSTRUCTION = "Write the handoff."


def history():
    return [user("u1"), assistant("a1"), user("u2")]


class SplitHistoryTest(unittest.TestCase):
    def test_reply_then_tool_rows_then_user_rows(self):
        rows = history()
        reply = assistant("", [("c1", "read", "{}"), ("c2", "ls", "{}")])
        capture = capture_for(rows, reply)
        messages = [*rows, reply, tool("c2", "r2"), tool("c1", "r1"), user("u3")]
        new_rows, trailing = split_history(capture, messages)
        self.assertEqual([row["role"] for row in new_rows], ["assistant", "tool", "tool"])
        self.assertEqual(trailing, [user("u3")])

    def test_reply_match_ignores_think_and_outer_space(self):
        rows = history()
        capture = capture_for(rows, assistant("<think>t</think>\nanswer"))
        new_rows, trailing = split_history(capture, [*rows, assistant("answer ")])
        self.assertEqual((len(new_rows), trailing), (1, []))

    def test_changed_history_is_refused(self):
        rows = history()
        reply = assistant("", [("c1", "read", "{}")])
        capture = capture_for(rows, reply)
        cases = {
            "changed row": [user("u1"), assistant("other"), user("u2"), reply, tool("c1", "r")],
            "no reply": rows,
            "other reply": [*rows, assistant("", [("c9", "read", "{}")]), tool("c9", "r")],
            "missing tool row": [*rows, reply],
            "unknown tool row": [*rows, reply, tool("c1", "r"), tool("c7", "r")],
            "assistant after tools": [*rows, reply, tool("c1", "r"), assistant("next")],
        }
        for label, messages in cases.items():
            with self.subTest(label), self.assertRaises(WarmRefusal) as caught:
                split_history(capture, messages)
            self.assertEqual(caught.exception.code, "history_changed")


class WireRowTest(unittest.TestCase):
    def test_keeps_api_fields_only(self):
        row = assistant(None, [("c1", "read", {"path": "a b"})], reasoning="r", _db_persisted=True)
        self.assertEqual(wire_row(row), {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a b\"}"}}]})
        self.assertEqual(wire_row(tool("c1", "out", name="read")),
                         {"role": "tool", "content": "out", "tool_call_id": "c1", "name": "read"})


class SettingsTest(unittest.TestCase):
    def test_refuses_settings_that_change_the_reply_form(self):
        for extra in ({"n": 2}, {"tool_choice": "required"}, {"tool_choice": {"type": "function"}},
                      {"response_format": {"type": "json_object"}}, {"functions": []}, {"modalities": ["audio"]}):
            with self.subTest(extra=extra), self.assertRaises(WarmRefusal) as caught:
                check_settings({"messages": [], **extra})
            self.assertEqual(caught.exception.code, "settings_unsupported")

    def test_accepts_auto_and_none_tool_choice(self):
        check_settings({"messages": [], "tool_choice": "auto", "n": 1})
        check_settings({"messages": [], "tool_choice": "none"})

    def test_fits_uses_the_reply_reserve(self):
        body = {"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 100}
        self.assertTrue(fits(body, 300))
        self.assertFalse(fits(body, 200))

    def test_fits_uses_the_measured_count_for_the_captured_rows(self):
        # The byte estimate of the two captured rows is above 2,000 tokens. The server measured 900.
        captured = [{"role": "user", "content": "x" * 4_000}, {"role": "assistant", "content": "y" * 4_000}]
        body = {"messages": [*captured, {"role": "user", "content": "z" * 40}], "max_tokens": 100}
        self.assertFalse(fits(body, 1_500))
        self.assertTrue(fits(body, 1_500, measured_tokens=900, measured_rows=2))
        self.assertFalse(fits(body, 1_000, measured_tokens=900, measured_rows=2))
        # A count that does not match the rows is not used.
        self.assertFalse(fits(body, 1_500, measured_tokens=900, measured_rows=4))
        self.assertFalse(fits(body, 1_500, measured_tokens=None, measured_rows=2))


class BuildRequestTest(unittest.TestCase):
    def setUp(self):
        self.rows = history()
        self.reply = assistant("", [("c1", "read", "{}")])
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]

    def build(self, capture, route=ROUTE, context_length=100_000):
        return build_request(capture, self.messages, route, context_length, INSTRUCTION)

    def test_appends_new_rows_and_the_instruction_to_the_captured_messages(self):
        extra = {"stream": True, "stream_options": {"include_usage": True}, "temperature": 0.2}
        capture = capture_for(self.rows, self.reply, body_extra=extra)
        body = self.build(capture)
        sent = capture["body"]["messages"]
        self.assertEqual(json.dumps(body["messages"][: len(sent)]), json.dumps(sent))
        self.assertEqual(body["messages"][len(sent):], [
            wire_row(self.reply), wire_row(tool("c1", "r1")), wire_row(user("u3")),
            {"role": "user", "content": INSTRUCTION}])
        self.assertEqual((body["stream"], body["temperature"]), (False, 0.2))
        self.assertNotIn("stream_options", body)

    def test_sends_every_trailing_user_row(self):
        # The tail can keep only the newest of several user rows. The handoff must see the older ones too.
        self.messages = [*self.messages, user("u4 " + "x" * 5_000)]
        body = self.build(capture_for(self.rows, self.reply))
        self.assertEqual(body["messages"][-3:-1], [wire_row(user("u3")), wire_row(user("u4 " + "x" * 5_000))])

    def test_refusal_codes(self):
        good = capture_for(self.rows, self.reply)
        cases = [
            ("api_mode_unsupported", good, (ROUTE[0], ROUTE[1], "codex_responses")),
            ("route_changed", good, ("other-model", ROUTE[1], ROUTE[2])),
            ("settings_unsupported", dict(good, body=None), ROUTE),
            ("settings_unsupported", capture_for(self.rows, self.reply, body_extra={"n": 3}), ROUTE),
        ]
        for code, capture, route in cases:
            with self.subTest(code), self.assertRaises(WarmRefusal) as caught:
                self.build(capture, route)
            self.assertEqual(caught.exception.code, code)

    def test_refuses_a_changed_source_shape(self):
        capture = capture_for(self.rows, self.reply)
        capture["body"]["messages"].insert(1, {"role": "user", "content": "injected"})
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def _tool_round(self):
        rows = [user("u1"), assistant("", [("c0", "read", "{\"path\":\"a\"}")]), tool("c0", "r0"), user("u2")]
        self.rows, self.messages = rows, [*rows, self.reply, tool("c1", "r1"), user("u3")]
        return capture_for(self.rows, self.reply)

    def test_refuses_a_rewritten_row_with_the_same_shape(self):
        capture = self._tool_round()
        capture["body"]["messages"][-1]["content"] = "A different request."
        rewritten = self._tool_round()
        rewritten["body"]["messages"][-3]["tool_calls"][0]["function"]["arguments"] = "{\"path\":\"other\"}"
        for capture in (capture, rewritten):
            with self.subTest(), self.assertRaises(WarmRefusal) as caught:
                self.build(capture)
            self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_refuses_a_renamed_row_and_accepts_a_removed_name(self):
        self.rows = [user("u1", name="alice"), assistant("a1"), user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        renamed = capture_for(self.rows, self.reply)
        renamed["body"]["messages"][1]["name"] = "bob"
        with self.assertRaises(WarmRefusal) as caught:
            self.build(renamed)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")
        # The Hermes transport removes the name from some rows. That is not a rewrite.
        stripped = capture_for(self.rows, self.reply)
        stripped["body"]["messages"][1].pop("name")
        self.build(stripped)

    def test_refuses_changed_media_parts(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        other = {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}
        for stored in ([{"type": "text", "text": "see this"}, image], [image]):
            self.rows = [user(stored), assistant("a1"), user("u2")]
            self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
            self.build(capture_for(self.rows, self.reply))
            for sent in ([part for part in stored if part is not image], [other if part is image else part
                                                                          for part in stored]):
                capture = capture_for(self.rows, self.reply)
                capture["body"]["messages"][1]["content"] = sent
                with self.subTest(stored=stored, sent=sent), self.assertRaises(WarmRefusal) as caught:
                    self.build(capture)
                self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_accepts_request_time_context_and_reformatted_arguments(self):
        capture = self._tool_round()
        sent = capture["body"]["messages"]
        sent[-1]["content"] += "\n\n[recalled context: the user wants short answers]"
        sent[-3]["tool_calls"][0]["function"]["arguments"] = "{ \"path\": \"a\" }"
        self.assertEqual(self.build(capture)["messages"][: len(sent)], sent)

    def test_refuses_when_the_window_is_too_small(self):
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture_for(self.rows, self.reply), context_length=4_096)
        self.assertEqual(caught.exception.code, "capacity")

    def test_capacity_uses_the_measured_prompt_count_of_the_capture(self):
        capture = capture_for(self.rows, self.reply)
        body = self.build(capture, context_length=0)
        sent = len(capture["body"]["messages"])
        whole = estimate_tokens({"messages": body["messages"], "tools": None}) * SAFETY + DEFAULT_RESERVE
        window = int(whole) - 1
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture, context_length=window)
        self.assertEqual(caught.exception.code, "capacity")
        capture["prompt_tokens"] = 1
        self.assertLess(1 + estimate_tokens({"messages": body["messages"][sent:]}) * SAFETY + DEFAULT_RESERVE, window)
        self.assertEqual(self.build(capture, context_length=window)["messages"], body["messages"])


def fake_post(status, payload, calls):
    def post(url, data, headers, timeout_s):
        calls.append((url, json.loads(data), headers, timeout_s))
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return post


class SendTest(unittest.TestCase):
    def test_reads_the_reply_and_the_usage(self):
        calls = []
        payload = {"choices": [{"message": {"content": "text"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                             "prompt_tokens_details": {"cached_tokens": 96}}}
        reply = send({"messages": []}, "http://h/v1/", "k", post=fake_post(200, payload, calls))
        self.assertEqual(calls[0][0], "http://h/v1/chat/completions")
        self.assertEqual(calls[0][2]["Authorization"], "Bearer k")
        self.assertEqual((reply["content"], reply["finish_reason"], reply["tool_calls"], reply["refusal"]),
                         ("text", "stop", False, False))
        self.assertEqual((reply["prompt_tokens"], reply["completion_tokens"], reply["cached_tokens"]), (100, 10, 96))

    def test_missing_cached_tokens_is_unknown_and_no_key_sends_no_header(self):
        calls = []
        payload = {"choices": [{"message": {"content": "t"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5}}
        reply = send({"messages": []}, "http://h/v1", "", post=fake_post(200, payload, calls))
        self.assertIsNone(reply["cached_tokens"])
        self.assertNotIn("Authorization", calls[0][2])

    def test_callable_key(self):
        calls = []
        payload = {"choices": [{"message": {"content": "t"}, "finish_reason": "stop"}]}
        send({"messages": []}, "http://h/v1", lambda: "token", post=fake_post(200, payload, calls))
        self.assertEqual(calls[0][2]["Authorization"], "Bearer token")

    def test_error_codes(self):
        def slow(*_args):
            raise TimeoutError("slow")

        def down(*_args):
            raise ConnectionRefusedError("down")

        cases = [
            ("timeout", slow),
            ("provider_error", down),
            ("provider_error", fake_post(500, {"error": "x"}, [])),
            ("incomplete_response", fake_post(200, b"not json", [])),
            ("incomplete_response", fake_post(200, {"choices": []}, [])),
        ]
        for code, post in cases:
            with self.subTest(code), self.assertRaises(WarmRefusal) as caught:
                send({"messages": []}, "http://h/v1", "k", post=post)
            self.assertEqual(caught.exception.code, code)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        status = 500 if self.path.endswith("/fail") else 200
        data = json.dumps({"size": len(body), "auth": self.headers.get("Authorization")}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


class UrllibPostTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def test_posts_and_returns_status_and_body(self):
        status, raw = urllib_post(self.base + "/v1/chat/completions", b"{}", {"Authorization": "Bearer k"}, 5.0)
        self.assertEqual((status, json.loads(raw)), (200, {"size": 2, "auth": "Bearer k"}))

    def test_http_error_returns_the_status(self):
        status, _raw = urllib_post(self.base + "/fail", b"{}", {}, 5.0)
        self.assertEqual(status, 500)


if __name__ == "__main__":
    unittest.main()
