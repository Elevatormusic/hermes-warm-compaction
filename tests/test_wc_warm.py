"""Tests for the warm request."""

import copy
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from warm_compaction.rows import estimate_tokens
from warm_compaction.warm import (
    HANDOFF_MAX_TOKENS, HANDOFF_MIN_TOKENS, SAFETY, WarmRefusal, build_request, check_settings, ends_with_instruction,
    fits, send, split_history, urllib_post, wire_row,
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


class RepeatedCallIdTest(unittest.TestCase):
    def test_tool_results_are_counted_by_occurrence(self):
        rows = history()
        reply = assistant("", [("c1", "read", "{}"), ("c1", "ls", "{}")])
        capture = capture_for(rows, reply)
        new_rows, _trailing = split_history(capture, [*rows, reply, tool("c1", "r1"), tool("c1", "r2")])
        self.assertEqual(len(new_rows), 3)
        with self.assertRaises(WarmRefusal):
            split_history(capture, [*rows, reply, tool("c1", "r1"), tool("c1", "r2"), tool("c1", "r3")])
        with self.assertRaises(WarmRefusal):
            split_history(capture, [*rows, reply, tool("c1", "r1")])


class ApiContentHistoryTest(unittest.TestCase):
    def test_the_reply_matches_by_its_api_content(self):
        rows = history()
        capture = capture_for(rows, assistant("The answer is 4."))
        new_rows, _trailing = split_history(capture, [*rows, assistant("", api_content="The answer is 4.")])
        self.assertEqual(len(new_rows), 1)

    def test_the_digest_covers_the_api_content(self):
        from warm_compaction.rows import row_digest
        self.assertNotEqual(row_digest(user("hi", api_content="[a]\n\nhi")),
                            row_digest(user("hi", api_content="[b]\n\nhi")))
        self.assertNotEqual(row_digest(user("hi", api_content="[a]\n\nhi")), row_digest(user("hi")))
        self.assertEqual(row_digest(dict(tool("c1", "r"), api_content="x")), row_digest(tool("c1", "r")))


class WireRowTest(unittest.TestCase):
    def test_keeps_api_fields_only(self):
        row = assistant(None, [("c1", "read", {"path": "a b"})], reasoning="r", _db_persisted=True)
        self.assertEqual(wire_row(row), {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a b\"}"}}]})
        # The Hermes transport removes the name from tool rows; strict providers reject it there.
        self.assertEqual(wire_row(tool("c1", "out", name="read")),
                         {"role": "tool", "content": "out", "tool_call_id": "c1"})
        self.assertEqual(wire_row(user("hi", name="alice")), {"role": "user", "content": "hi", "name": "alice"})

    def test_sends_the_api_content_of_user_and_assistant_rows(self):
        self.assertEqual(wire_row(user("hi", api_content="ctx\n\nhi")), {"role": "user", "content": "ctx\n\nhi"})
        self.assertEqual(wire_row(assistant("", api_content="answer")), {"role": "assistant", "content": "answer"})
        self.assertEqual(wire_row(assistant("shown", api_content="")), {"role": "assistant", "content": "shown"})
        row = dict(tool("c1", "out"), api_content="other")
        self.assertEqual(wire_row(row), {"role": "tool", "content": "out", "tool_call_id": "c1"})


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

    def test_fits_reserves_the_larger_of_the_two_reply_limits(self):
        body = {"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 100,
                "max_completion_tokens": 5_000}
        self.assertFalse(fits(body, 1_000))
        self.assertTrue(fits(body, 6_000))

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
        # The instruction joins the trailing user row: no two adjacent user rows.
        self.assertEqual(body["messages"][len(sent):], [
            wire_row(self.reply), wire_row(tool("c1", "r1")), {"role": "user", "content": "u3\n\n" + INSTRUCTION}])
        self.assertTrue(ends_with_instruction(body["messages"][-1], INSTRUCTION))
        self.assertFalse(ends_with_instruction({"role": "user", "content": "u3"}, INSTRUCTION))
        self.assertEqual((body["stream"], body["temperature"]), (False, 0.2))
        self.assertNotIn("stream_options", body)

    def test_appended_assistant_rows_keep_reasoning_content_on_an_echo_route(self):
        # A thinking-mode route (DeepSeek, Kimi) needs reasoning_content on every assistant row. The captured
        # rows show that Hermes sends it on this route.
        self.rows = [user("u1"), assistant("a1", reasoning_content=" "), user("u2")]
        self.messages = [*self.rows, assistant("", [("c1", "read", "{}")], reasoning="look"), tool("c1", "r1")]
        sent = self.build(capture_for(self.rows, self.reply))["messages"]
        self.assertEqual(sent[-3]["reasoning_content"], " ")
        self.messages[-2] = assistant("", [("c1", "read", "{}")], reasoning_content="look")
        self.assertEqual(self.build(capture_for(self.rows, self.reply))["messages"][-3]["reasoning_content"], "look")
        # A route without it keeps the field out (strict providers reject it).
        self.rows = history()
        self.messages = [*self.rows, assistant("", [("c1", "read", "{}")], reasoning="look"), tool("c1", "r1")]
        self.assertNotIn("reasoning_content", self.build(capture_for(self.rows, self.reply))["messages"][-3])

    def test_the_native_carrier_of_the_provider_profile_is_replayed(self):
        # Hermes 45871e10 replays the <provider>.native_assistant carrier that the provider profile declares
        # (native_reasoning_details_type), also on a route that does not replay other reasoning_details.
        from warm_compaction.warm import WarmRefusal, check_source, wire_row
        from wc_fixtures import SYSTEM, wire
        carrier = [{"type": "acme.native_assistant", "data": "n"}]
        row = assistant("a1", reasoning_details=carrier)
        self.assertNotIn("reasoning_details", wire_row(row, base_url=ROUTE[1]))
        self.assertNotIn("reasoning_details", wire_row(row, base_url=ROUTE[1], native_type="other.native_assistant"))
        self.assertEqual(wire_row(row, base_url=ROUTE[1], native_type="acme.native_assistant")["reasoning_details"],
                         carrier)
        rows = [user("u1"), row, user("u2")]
        body = {"model": ROUTE[0], "messages": [SYSTEM, *wire(rows)]}
        check_source(body, rows, ROUTE[1], native_type="acme.native_assistant")
        with self.assertRaises(WarmRefusal):
            check_source(body, rows, ROUTE[1])

    def test_appended_tool_calls_keep_the_gemini_thought_signature(self):
        signed = {"google": {"thought_signature": "sig-1"}}
        reply = assistant("", [("c1", "read", "{}")])
        reply["tool_calls"][0]["extra_content"] = signed
        reply["tool_calls"][0]["call_id"] = "c1"
        self.messages = [*self.rows, reply, tool("c1", "r1")]
        gemini = capture_for(self.rows, self.reply, route=("gemini-3-pro", ROUTE[1], ROUTE[2]))
        sent = build_request(gemini, self.messages, ("gemini-3-pro", ROUTE[1], ROUTE[2]), 100_000,
                             INSTRUCTION)["messages"][-3]
        self.assertEqual(sent["tool_calls"][0], {"id": "c1", "type": "function", "extra_content": signed,
                                                 "function": {"name": "read", "arguments": "{}"}})
        # Other models reject the field; an empty signature is not sent.
        other = self.build(capture_for(self.rows, self.reply))["messages"][-3]
        self.assertNotIn("extra_content", other["tool_calls"][0])
        reply["tool_calls"][0]["extra_content"] = {"google": {"thought_signature": " "}}
        sent = build_request(gemini, self.messages, ("gemini-3-pro", ROUTE[1], ROUTE[2]), 100_000,
                             INSTRUCTION)["messages"][-3]
        self.assertNotIn("extra_content", sent["tool_calls"][0])

    def test_the_stop_setting_is_not_sent(self):
        # A stop sequence of the main request could cut the handoff after the five headings.
        body = self.build(capture_for(self.rows, self.reply, body_extra={"stop": ["\n## Next"], "temperature": 0.2}))
        self.assertNotIn("stop", body)
        self.assertEqual(body["temperature"], 0.2)

    def test_web_search_is_not_sent(self):
        # A web search in the handoff request costs a search and can bring text that is not in the conversation.
        body = self.build(capture_for(self.rows, self.reply, body_extra={"web_search_options": {}}))
        self.assertNotIn("web_search_options", body)

    def test_the_handoff_has_its_own_reply_limit(self):
        # The reply limit of the main request is for another task: a small one cuts the handoff, a large one
        # reserves space that the handoff does not need.
        cases = (({"max_tokens": 128}, {"max_tokens": HANDOFF_MIN_TOKENS}),
                 ({"max_tokens": 3_000}, {"max_tokens": 3_000}),
                 ({"max_tokens": 60_000}, {"max_tokens": HANDOFF_MAX_TOKENS}),
                 ({"max_completion_tokens": 500}, {"max_completion_tokens": HANDOFF_MIN_TOKENS}),
                 # Without a limit the server default applies: it can cut the handoff, or reserve more than the
                 # capacity check does. The handoff gets its own limit in the field that Hermes uses for the route.
                 ({}, {"max_tokens": HANDOFF_MAX_TOKENS}))
        for extra, expected in cases:
            with self.subTest(extra=extra):
                body = self.build(capture_for(self.rows, self.reply, body_extra=extra))
                self.assertEqual({key: body[key] for key in ("max_tokens", "max_completion_tokens") if key in body},
                                 expected)

    def test_a_route_that_needs_max_completion_tokens_gets_it(self):
        for route in (("gpt-5.1", ROUTE[1], ROUTE[2]), ("m", "https://api.openai.com/v1", ROUTE[2]),
                      ("m", "https://x.openai.azure.com/v1", ROUTE[2])):
            with self.subTest(route=route):
                capture = capture_for(self.rows, self.reply, route=route)
                body = self.build(capture, route=route)
                self.assertEqual(body.get("max_completion_tokens"), HANDOFF_MAX_TOKENS)
                self.assertNotIn("max_tokens", body)

    def test_a_large_main_reply_limit_does_not_refuse_a_handoff_that_fits(self):
        capture = capture_for(self.rows, self.reply, body_extra={"max_tokens": 60_000})
        self.build(capture, context_length=20_000)

    def test_sends_every_trailing_user_row(self):
        # The tail can keep only the newest of several user rows. The handoff must see the older ones too.
        # Strict chat templates refuse two adjacent user rows: they are joined, as Hermes joins them, and the
        # host instruction is the last block of the last user row.
        self.messages = [*self.messages, user("u4 " + "x" * 5_000)]
        body = self.build(capture_for(self.rows, self.reply))
        self.assertEqual(body["messages"][-1], {"role": "user",
                                                "content": "u3\n\nu4 " + "x" * 5_000 + "\n\n" + INSTRUCTION})
        self.assertEqual(body["messages"][-2]["role"], "tool")
        roles = [row["role"] for row in body["messages"]]
        self.assertFalse(any(a == b == "user" for a, b in zip(roles, roles[1:])))

    def test_the_instruction_joins_a_named_user_row_and_other_authors_stay_apart(self):
        # Two history rows of different authors stay apart. The instruction joins the last user row and keeps
        # its name: the ordinary request also ended with that row.
        self.messages = [*self.messages[:-1], user("u3", name="alice"), user("u4", name="bob")]
        body = self.build(capture_for(self.rows, self.reply))
        self.assertEqual(body["messages"][-2:], [wire_row(user("u3", name="alice")),
                                                 {"role": "user", "name": "bob", "content": "u4\n\n" + INSTRUCTION}])

    def test_refusal_codes(self):
        good = capture_for(self.rows, self.reply)
        cases = [
            ("api_mode_unsupported", good, (ROUTE[0], ROUTE[1], "bedrock_converse")),
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
        for changed in (capture, rewritten):
            with self.subTest(), self.assertRaises(WarmRefusal) as caught:
                self.build(changed)
            self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_refuses_a_changed_tool_call_type(self):
        # The same keys with another type value: the provider reads another kind of call than the stored one.
        self.rows = [user("u1"), assistant("", [("c0", "read", "{}")]), tool("c0", "r0"), user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        capture = capture_for(self.rows, self.reply)
        capture["body"]["messages"][2]["tool_calls"][0]["type"] = "custom"
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")
        self.build(capture_for(self.rows, self.reply))

    def test_refuses_a_sent_row_with_an_extra_field(self):
        # A field that the stored row does not give (a provider control, for example) changes what the model reads.
        for index, field in ((1, {"recipient": "x"}), (-1, {"prefix": True})):
            capture = capture_for(self.rows, self.reply)
            capture["body"]["messages"][index].update(field)
            with self.subTest(field=field), self.assertRaises(WarmRefusal) as caught:
                self.build(capture)
            self.assertEqual(caught.exception.code, "source_transform_unsupported")
        # Hermes prompt caching marks rows with cache_control on some routes.
        capture = capture_for(self.rows, self.reply)
        capture["body"]["messages"][1]["cache_control"] = {"type": "ephemeral"}
        self.build(capture)

    def test_refuses_a_renamed_or_unnamed_user_row_and_accepts_an_unnamed_tool_row(self):
        self.rows = [user("u1", name="alice"), assistant("", [("c0", "read", "{}")]), tool("c0", "r0", name="read"),
                     user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        for change in (lambda sent: sent[1].update(name="bob"), lambda sent: sent[1].pop("name")):
            capture = capture_for(self.rows, self.reply)
            change(capture["body"]["messages"])
            with self.subTest(), self.assertRaises(WarmRefusal) as caught:
                self.build(capture)
            self.assertEqual(caught.exception.code, "source_transform_unsupported")
        # The Hermes transport removes the name from tool rows only. That is not a rewrite.
        stripped = capture_for(self.rows, self.reply)
        stripped["body"]["messages"][3].pop("name")
        self.build(stripped)

    def test_refuses_moved_media_parts(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        self.rows = [user([{"type": "text", "text": "first"}, image, {"type": "text", "text": "caption"}]),
                     assistant("a1"), user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        capture = capture_for(self.rows, self.reply)
        capture["body"]["messages"][1]["content"] = [image, {"type": "text", "text": "first"},
                                                     {"type": "text", "text": "caption"}]
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")
        # Text added inside a text run is a rewrite too: Hermes sends the stored text (api_content).
        capture = capture_for(self.rows, self.reply)
        capture["body"]["messages"][1]["content"][0]["text"] = "[context]\n\nfirst"
        with self.assertRaises(WarmRefusal):
            self.build(capture)

    def test_the_stored_api_content_is_the_sent_text(self):
        self.rows = [user("hi", api_content="[recalled: short answers]\n\nhi"),
                     assistant("", api_content="The answer is 4."), user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        capture = capture_for(self.rows, self.reply)
        for row in capture["body"]["messages"][1:3]:
            row["content"] = row.pop("api_content")
        self.build(capture)
        capture["body"]["messages"][2]["content"] = "The answer is 5."
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")

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

    def test_refuses_a_changed_json_type_in_arguments(self):
        for old, new in (('{"path":"a","force":true}', '{"path":"a","force":1}'),
                         ('{"path":"a","n":0}', '{"path":"a","n":false}')):
            rows = [user("u1"), assistant("", [("c0", "read", old)]), tool("c0", "r0"), user("u2")]
            self.rows, self.messages = rows, [*rows, self.reply, tool("c1", "r1"), user("u3")]
            capture = capture_for(self.rows, self.reply)
            capture["body"]["messages"][2]["tool_calls"][0]["function"]["arguments"] = new
            with self.subTest(new=new), self.assertRaises(WarmRefusal) as caught:
                self.build(capture)
            self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_appended_rows_keep_reasoning_details_on_a_replaying_route(self):
        details = [{"type": "reasoning.encrypted", "data": "abc"}, {"type": "x.native_assistant", "data": "n"}]
        reply = assistant("", [("c1", "read", "{}")], reasoning_details=details)
        self.messages = [*self.rows, reply, tool("c1", "r1")]
        for base_url, expected in (("https://openrouter.ai/api/v1", [details[0]]),
                                   ("https://api.example.com/v1", None)):
            route = (ROUTE[0], base_url, ROUTE[2])
            sent = build_request(capture_for(self.rows, self.reply, route=route), self.messages, route, 100_000,
                                 INSTRUCTION)["messages"][-3]
            with self.subTest(base_url=base_url):
                self.assertEqual(sent.get("reasoning_details"), expected)

    def test_refuses_changed_reasoning_fields_in_the_captured_rows(self):
        # The provider reads reasoning_content, reasoning_details, and the thought signature of a captured
        # assistant row. A middleware that changed them made a prefix that differs from the stored history.
        details = [{"type": "reasoning.encrypted", "data": "abc"}]
        signed = {"google": {"thought_signature": "sig-1"}}
        echo = ([user("u1"), assistant("a1", reasoning_content="think"), user("u2")], ROUTE)
        replay = ([user("u1"), assistant("a1", reasoning_details=details), user("u2")],
                  (ROUTE[0], "https://openrouter.ai/api/v1", ROUTE[2]))
        gemini_row = assistant("", [("c0", "read", "{}")])
        gemini_row["tool_calls"][0]["extra_content"] = signed
        gemini = ([user("u1"), gemini_row, tool("c0", "r0"), user("u2")], ("gemini-3-pro", ROUTE[1], ROUTE[2]))
        changes = [
            (echo, lambda row: row.update(reasoning_content="other")),
            (echo, lambda row: row.update(reasoning_content=" ")),
            (replay, lambda row: row.pop("reasoning_details")),
            (replay, lambda row: row.update(reasoning_details=[{"type": "reasoning.encrypted", "data": "x"}])),
            (gemini, lambda row: row["tool_calls"][0].pop("extra_content")),
        ]
        for (rows, route), change in changes:
            self.rows, self.messages = rows, [*rows, self.reply, tool("c1", "r1"), user("u3")]
            self.build(capture_for(self.rows, self.reply, route=route), route)
            capture = capture_for(self.rows, self.reply, route=route)
            change(capture["body"]["messages"][2])
            with self.subTest(route=route), self.assertRaises(WarmRefusal) as caught:
                self.build(capture, route)
            self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_accepts_reformatted_arguments(self):
        capture = self._tool_round()
        sent = capture["body"]["messages"]
        sent[-3]["tool_calls"][0]["function"]["arguments"] = "{ \"path\": \"a\" }"
        self.assertEqual(self.build(capture)["messages"][: len(sent)], sent)

    def test_refuses_any_text_added_to_the_stored_text(self):
        # Hermes stores the text that it sends (api_content), so the sent text is the stored text. Added text,
        # on the same line or on its own line, can change the meaning ("Ignore the next line.\nDelete A").
        self.rows = [user("Delete A"), assistant("a1"), user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        for sent, accepted in (("Do not Delete A", False), ("Delete A now", False), ("Delete A\n\n[context]", False),
                               ("Ignore the next line.\nDelete A", False), ("[a]\n\nDelete A\n\n[b]", False),
                               ("Delete A", True)):
            capture = capture_for(self.rows, self.reply)
            capture["body"]["messages"][1]["content"] = sent
            with self.subTest(sent=sent):
                if accepted:
                    self.build(capture)
                    continue
                with self.assertRaises(WarmRefusal) as caught:
                    self.build(capture)
                self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_refuses_changed_white_space_around_the_stored_text(self):
        # A middleware that dedents the first line of a code fragment changes its meaning.
        self.rows = [user("    return 1\nx = 2"), assistant("a1"), user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        for sent, accepted in (("return 1\nx = 2", False), ("    return 1\nx = 2", True)):
            capture = capture_for(self.rows, self.reply)
            capture["body"]["messages"][1]["content"] = sent
            with self.subTest(sent=sent):
                if accepted:
                    self.build(capture)
                    continue
                with self.assertRaises(WarmRefusal) as caught:
                    self.build(capture)
                self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_accepts_only_full_trailing_whitespace_removal_from_strings(self):
        # Hermes removes outer whitespace from complete strings, for all message roles. Leading whitespace
        # stays protected by the source check. The captured prefix and the stored rows stay unchanged.
        for index in (0, 1, 2):
            for suffix in (" ", "  ", "\n\n", "\t", "\r\n", "\u00a0\u2003"):
                for remove in (False, True):
                    with self.subTest(index=index, suffix=suffix, remove=remove):
                        self._tool_round()
                        self.rows[index]["content"] = "synthetic text" + suffix
                        capture = capture_for(self.rows, self.reply)
                        sent = capture["body"]["messages"]
                        if remove:
                            sent[index + 1]["content"] = "synthetic text"
                        before_capture, before_messages = copy.deepcopy(capture), copy.deepcopy(self.messages)
                        body = self.build(capture)
                        self.assertEqual(body["messages"][:len(sent)], before_capture["body"]["messages"])
                        self.assertEqual(capture, before_capture)
                        self.assertEqual(self.messages, before_messages)

    def test_trailing_whitespace_removal_uses_the_effective_api_content(self):
        cases = (
            user("display text", api_content="sent text \n"),
            assistant("display text", api_content="sent text \n"),
            user([{"type": "text", "text": "display text"}], api_content="sent text \n"),
            user("sent text \n", api_content=""),
            assistant("sent text \n", api_content=""),
            dict(tool("c0", "sent text \n"), api_content="ignored sidecar"),
        )
        for row in cases:
            with self.subTest(row=row):
                self.rows = [row]
                self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
                capture = capture_for(self.rows, self.reply)
                capture["body"]["messages"][1] = wire_row(row)
                capture["body"]["messages"][1]["content"] = "sent text"
                before_capture, before_messages = copy.deepcopy(capture), copy.deepcopy(self.messages)
                body = self.build(capture)
                self.assertEqual(body["messages"][:2], before_capture["body"]["messages"])
                self.assertEqual(capture, before_capture)
                self.assertEqual(self.messages, before_messages)
                capture["body"]["messages"][1]["content"] = "display text"
                with self.assertRaises(WarmRefusal) as caught:
                    self.build(capture)
                self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_refuses_other_string_whitespace_changes(self):
        cases = (
            ("text", "text "),
            ("text ", "text  "),
            ("text \n", "text "),
            ("text \n", "text\n"),
            ("text ", "text\t"),
            ("    text \n", "text"),
            ("text \n", " text"),
            ("two  words \n", "two words"),
            ("text\u200b", "text"),
        )
        for stored, sent in cases:
            with self.subTest(stored=stored, sent=sent):
                self.rows = [user(stored)]
                self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
                capture = capture_for(self.rows, self.reply)
                capture["body"]["messages"][1]["content"] = sent
                with self.assertRaises(WarmRefusal) as caught:
                    self.build(capture)
                self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_refuses_whitespace_removal_from_text_parts(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        text = {"type": "text", "text": "text \n"}
        trimmed = {"type": "text", "text": "text"}
        cases = (
            ([text], [trimmed]),
            (["text \n"], ["text"]),
            ([text], "text"),
            ("text \n", [trimmed]),
            ([text, image], [trimmed, image]),
            ([image, text], [image, trimmed]),
            ([text, image, text], [text, image, trimmed]),
        )
        for stored, sent in cases:
            with self.subTest(stored=stored, sent=sent):
                self.rows = [user(stored)]
                self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
                capture = capture_for(self.rows, self.reply)
                capture["body"]["messages"][1]["content"] = sent
                with self.assertRaises(WarmRefusal) as caught:
                    self.build(capture)
                self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_trailing_whitespace_removal_does_not_bypass_other_source_checks(self):
        changes = (
            lambda row: row.update(role="user"),
            lambda row: row.update(name="other"),
            lambda row: row.update(reasoning_content="other"),
            lambda row: row.update(recipient="other"),
            lambda row: row["tool_calls"][0]["function"].update(arguments='{"path":"other"}'),
        )
        for change in changes:
            with self.subTest(change=change):
                self._tool_round()
                self.rows[1]["content"] = "call text \n"
                capture = capture_for(self.rows, self.reply)
                sent = capture["body"]["messages"][2]
                sent["content"] = "call text"
                change(sent)
                with self.assertRaises(WarmRefusal) as caught:
                    self.build(capture)
                self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_refuses_changed_whitespace_inside_the_text(self):
        self.rows = [user("def f():\n    return 1"), assistant("a1"), user("u2")]
        self.messages = [*self.rows, self.reply, tool("c1", "r1"), user("u3")]
        capture = capture_for(self.rows, self.reply)
        capture["body"]["messages"][1]["content"] = "def f():\n  return 1"
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_a_capture_without_a_body_gives_its_refusal(self):
        for code in ("middleware_after_capture", "middleware_order_unknown", "request_not_mapping",
                     "request_options_unsupported", "request_not_json"):
            capture = dict(capture_for(self.rows, self.reply), body=None, refusal=code)
            with self.subTest(code=code), self.assertRaises(WarmRefusal) as caught:
                self.build(capture)
            self.assertEqual(caught.exception.code, code)

    def test_refuses_when_the_window_is_too_small(self):
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture_for(self.rows, self.reply), context_length=4_096)
        self.assertEqual(caught.exception.code, "capacity")

    def test_capacity_uses_the_measured_prompt_count_of_the_capture(self):
        capture = capture_for(self.rows, self.reply)
        body = self.build(capture, context_length=0)
        sent = len(capture["body"]["messages"])
        # The capture has no reply limit: the handoff gets HANDOFF_MAX_TOKENS, and the check reserves it.
        whole = estimate_tokens({"messages": body["messages"], "tools": None}) * SAFETY + HANDOFF_MAX_TOKENS
        window = int(whole) - 1
        with self.assertRaises(WarmRefusal) as caught:
            self.build(capture, context_length=window)
        self.assertEqual(caught.exception.code, "capacity")
        capture["prompt_tokens"] = 1
        added = estimate_tokens({"messages": body["messages"][sent:]})
        self.assertLess(1 + added * SAFETY + HANDOFF_MAX_TOKENS, window)
        self.assertEqual(self.build(capture, context_length=window)["messages"], body["messages"])


def fake_post(status, payload, calls):
    def post(url, data, headers, timeout_s):
        calls.append((url, json.loads(data), headers, timeout_s))
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return post


class RouteTlsTest(unittest.TestCase):
    def setUp(self):
        import wc_hermes_stub
        wc_hermes_stub.install(self)
        self.stub = wc_hermes_stub

    def test_tls_of_the_route(self):
        import ssl
        from warm_compaction.warm import route_tls
        self.assertIsNone(route_tls("http://127.0.0.1:9/v1"))
        self.stub.TLS_VERIFY.append(True)
        self.assertIsNone(route_tls("https://h/v1"))
        # The plugin does not send without certificate checks (ssl_verify: false): the warm path stops.
        self.stub.TLS_VERIFY[:] = [False]
        with self.assertRaises(WarmRefusal) as caught:
            route_tls("https://h/v1")
        self.assertEqual(caught.exception.code, "tls_unverified")
        own = ssl.create_default_context()
        self.stub.TLS_VERIFY[:] = [own]
        self.assertIs(route_tls("https://h/v1"), own)

    def test_an_unreadable_tls_setting_stops_the_warm_request(self):
        from warm_compaction.warm import route_tls
        self.stub.TLS_VERIFY[:] = [RuntimeError("changed")]
        with self.assertRaises(WarmRefusal) as caught:
            route_tls("https://h/v1")
        self.assertEqual(caught.exception.code, "tls_unknown")

    def test_the_context_goes_to_the_post(self):
        import ssl
        context = ssl.create_default_context()
        seen = []

        def post(url, data, headers, timeout_s, context=None):
            seen.append(context)
            return 200, json.dumps({"choices": [{"message": {"content": "t"}, "finish_reason": "stop"}]}).encode()
        send({"messages": []}, "https://h/v1", "k", post=post, ssl_context=context)
        send({"messages": []}, "http://h/v1", "k", post=fake_post(200, {"choices": [
            {"message": {"content": "t"}, "finish_reason": "stop"}]}, []))
        self.assertEqual(seen, [context])


class SendTest(unittest.TestCase):
    def test_the_finish_reason_is_normalized(self):
        # Some OpenAI-compatible servers send STOP or MAX_TOKENS. The gate needs the contract value.
        for raw, expected in (("STOP", "stop"), ("MAX_TOKENS", "length"), ("end", "stop"), ("stop", "stop")):
            payload = {"choices": [{"message": {"content": "text"}, "finish_reason": raw}], "usage": {}}
            with self.subTest(raw=raw):
                reply = send({"messages": []}, "http://h/v1", "k", post=fake_post(200, payload, []))
                self.assertEqual(reply["finish_reason"], expected)

    def test_extra_headers_are_sent(self):
        calls = []
        payload = {"choices": [{"message": {"content": "text"}, "finish_reason": "stop"}], "usage": {}}
        send({"messages": []}, "http://h/v1", "k", post=fake_post(200, payload, calls),
             extra_headers={"X-Title": "Hermes Agent"})
        self.assertEqual((calls[0][2]["X-Title"], calls[0][2]["Authorization"]), ("Hermes Agent", "Bearer k"))

    def test_the_warm_request_sends_a_user_agent(self):
        # Python urllib adds "Python-urllib/x.y" when the request sets no User-Agent.
        # Cloudflare-fronted routes refuse that signature with HTTP 403 (error 1010).
        calls = []
        payload = {"choices": [{"message": {"content": "text"}, "finish_reason": "stop"}], "usage": {}}
        send({"messages": []}, "http://h/v1", "k", post=fake_post(200, payload, calls))
        agent = calls[0][2]["User-Agent"]
        self.assertTrue(agent, "the warm request must set a User-Agent")
        self.assertNotIn("urllib", agent.lower(), "the library User-Agent is refused by a bot rule")

    def test_reads_the_reply_and_the_usage(self):
        calls = []
        payload = {"choices": [{"message": {"content": "text"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                             "prompt_tokens_details": {"cached_tokens": 96}}}
        reply = send({"messages": []}, "http://h/v1/", "k", post=fake_post(200, payload, calls))
        self.assertEqual(calls[0][0], "http://h/v1/chat/completions")
        # Hermes sends the query of the route URL (Azure api-version) as the client's default_query.
        send({"messages": []}, "https://x.openai.azure.com/openai/deployments/d?api-version=2024-10-21", "k",
             post=fake_post(200, payload, calls))
        self.assertEqual(calls[1][0],
                         "https://x.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21")
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
    def do_GET(self):
        # Only a followed redirect comes here.
        self.server.redirect_hits.append((self.path, self.headers.get("Authorization")))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.endswith("/moved"):
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status = 500 if self.path.endswith("/fail") else 200
        data = json.dumps({"size": len(body), "auth": self.headers.get("Authorization")}).encode("utf-8")
        if self.path.endswith("/large"):
            data = b" " * (3 << 20)
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
        self.server.redirect_hits = self.redirect_hits = []

    def test_posts_and_returns_status_and_body(self):
        status, raw = urllib_post(self.base + "/v1/chat/completions", b"{}", {"Authorization": "Bearer k"}, 5.0)
        self.assertEqual((status, json.loads(raw)), (200, {"size": 2, "auth": "Bearer k"}))

    def test_http_error_returns_the_status(self):
        status, _raw = urllib_post(self.base + "/fail", b"{}", {}, 5.0)
        self.assertEqual(status, 500)

    def test_a_large_response_is_not_read_whole(self):
        # A wrong endpoint or a gateway can send a large body: the read stops after the limit.
        from warm_compaction.warm import MAX_RESPONSE_BYTES
        status, raw = urllib_post(self.base + "/large", b"{}", {}, 5.0)
        self.assertEqual(status, 200)
        self.assertEqual(len(raw), MAX_RESPONSE_BYTES + 1)
        with self.assertRaises(WarmRefusal) as caught:
            send({"model": "m", "messages": []}, self.base + "/v1", "k", post=lambda *args, **kwargs: (200, raw))
        self.assertEqual(caught.exception.code, "response_too_large")

    def test_a_redirect_is_not_followed(self):
        # A followed redirect would send the Authorization header to the redirect target.
        status, _raw = urllib_post(self.base + "/moved", b"{}", {"Authorization": "Bearer k"}, 5.0)
        self.assertEqual(status, 302)
        self.assertEqual(self.redirect_hits, [])


if __name__ == "__main__":
    unittest.main()
