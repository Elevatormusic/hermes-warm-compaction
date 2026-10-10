"""Synthetic checks for the bounded Anthropic Messages adapter."""

import copy
import unittest

from warm_compaction import anthropic
from warm_compaction.rows import row_digest
from warm_compaction.warm import WarmRefusal
from wc_fixtures import assistant, tool, user

ROUTE = ("claude-sonnet-4-6", "https://api.anthropic.com", "anthropic_messages")
INSTRUCTION = "Write the handoff."


def capture(rows, reply):
    """Make an invented capture. No provider or real credentials are used."""
    return {"route": ROUTE, "digests": [row_digest(r) for r in rows],
            "reply": {"content": reply["content"], "tool_calls": [
                [c["id"], c["function"]["name"]] for c in reply.get("tool_calls", [])]},
            "body": {"model": ROUTE[0], "system": [{"type": "text", "text": "Synthetic rule.",
                                                      "cache_control": {"type": "ephemeral"}}],
                     "messages": anthropic._rows(rows), "max_tokens": 4096,
                     "thinking": {"type": "disabled"}}}


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.rows = [user("first"), assistant("answer"), user("next")]
        self.reply = assistant("result")
        self.capture = capture(self.rows, self.reply)
        self.messages = [*self.rows, self.reply, user("latest")]

    def build(self, cap=None, messages=None, context=100000):
        return anthropic.build_request(cap or self.capture, messages or self.messages, ROUTE, context, INSTRUCTION)

    def refusal(self, code, callback):
        with self.assertRaises(WarmRefusal) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_keeps_the_native_prefix_and_settings_without_mutation(self):
        self.capture["body"].update(tool_choice={"type": "auto"}, tools=[{
            "name": "read", "description": "Synthetic tool", "input_schema": {"type": "object"}}],
            stop_sequences=["cut"], output_config={"effort": "medium"})
        self.capture["body"]["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        before = copy.deepcopy((self.capture, self.messages))
        body = self.build()
        self.assertEqual(body["messages"][:3], before[0]["body"]["messages"])
        for name in ("system", "tools", "tool_choice", "thinking", "output_config"):
            self.assertEqual(body[name], before[0]["body"][name])
        self.assertEqual(body["messages"][-1], {"role": "user", "content": [
            {"type": "text", "text": "latest"}, {"type": "text", "text": INSTRUCTION}]})
        self.assertFalse(body["stream"])
        self.assertNotIn("stop_sequences", body)
        self.assertEqual((self.capture, self.messages), before)

    def test_complete_new_tool_pairs_keep_arguments_and_results(self):
        reply = assistant("call", [("call1", "read", '{"path":"synthetic"}')])
        cap = capture(self.rows, reply)
        body = self.build(cap, [*self.rows, reply, tool("call1", "synthetic output"), user("latest")])
        self.assertEqual(body["messages"][-2]["content"][-1], {
            "type": "tool_use", "id": "call1", "name": "read", "input": {"path": "synthetic"}})
        self.assertEqual(body["messages"][-1]["content"][0], {
            "type": "tool_result", "tool_use_id": "call1", "content": "synthetic output"})

    def test_source_tool_rows_are_matched_after_native_folding(self):
        rows = [user("task"), assistant("call", [("c1", "read", "{}")]), tool("c1", "output"), user("next")]
        cap = capture(rows, self.reply)
        body = self.build(cap, [*rows, self.reply])
        self.assertEqual(body["messages"][:len(cap["body"]["messages"])], cap["body"]["messages"])
        cap["body"]["messages"][1]["content"][1]["input"] = {"changed": True}
        self.refusal("source_transform_unsupported", lambda: self.build(cap, [*rows, self.reply]))

    def test_captured_thinking_stays_intact(self):
        self.rows[1]["reasoning_details"] = [{"type": "thinking", "thinking": "synthetic", "signature": "fake"}]
        self.capture = capture(self.rows, self.reply)
        block = {"type": "thinking", "thinking": "synthetic", "signature": "fake"}
        body = self.build()
        self.assertEqual(body["messages"][1]["content"][0], block)

    def test_new_reasoning_and_media_are_refused(self):
        for key in anthropic.OPAQUE_ROWS:
            with self.subTest(key=key):
                reply = dict(self.reply, **{key: "synthetic opaque field"})
                self.refusal("source_transform_unsupported", lambda reply=reply: self.build(
                    messages=[*self.rows, reply]))
        reply = dict(self.reply, api_content=[{"type": "image", "source": {"type": "base64", "data": "fake"}}])
        # The plain content itself must be unsupported too: api_content is a string-only sidecar in Hermes.
        reply["content"] = reply["api_content"]
        cap = capture(self.rows, self.reply)
        cap["reply"]["content"] = ""
        self.refusal("source_transform_unsupported", lambda: self.build(cap, [*self.rows, reply]))

    def test_changed_capture_text_and_native_media_are_refused(self):
        self.capture["body"]["messages"][0]["content"][0]["text"] = "changed"
        self.refusal("source_transform_unsupported", self.build)
        self.capture["body"]["messages"][0]["content"] = [{"type": "image", "source": {}}]
        self.refusal("source_transform_unsupported", self.build)

    def test_forced_tools_server_tools_and_stateful_options_are_refused(self):
        for change in ({"tool_choice": {"type": "any"}}, {"tool_choice": {"type": "tool", "name": "read"}},
                       {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
                       {"context_management": {"edits": []}}, {"container": "synthetic"},
                       {"output_config": {"format": {"type": "json_schema"}}}):
            with self.subTest(change=change):
                cap = copy.deepcopy(self.capture)
                cap["body"].update(change)
                self.refusal("settings_unsupported", lambda cap=cap: self.build(cap))

    def test_missing_tool_result_and_changed_route_are_refused(self):
        reply = assistant("call", [("c1", "read", "{}")])
        cap = capture(self.rows, reply)
        self.refusal("history_changed:tool_count", lambda: self.build(cap, [*self.rows, reply]))
        cap["route"] = ("changed", *ROUTE[1:])
        self.refusal("route_changed", lambda: self.build(cap))

    def test_signed_ordered_source_rows_are_refused(self):
        self.rows[1]["anthropic_content_blocks"] = [{"type": "text", "text": "other"}]
        self.capture["digests"] = [row_digest(r) for r in self.rows]
        self.refusal("source_transform_unsupported", self.build)

    def test_size_estimate_includes_system_and_keeps_manual_thinking_budget(self):
        cap = copy.deepcopy(self.capture)
        cap["body"]["system"] = "synthetic " * 10000
        self.refusal("capacity", lambda: self.build(cap, context=9000))
        cap["prompt_tokens"] = 10
        self.build(cap, context=9000)
        cap["body"]["thinking"] = {"type": "enabled", "budget_tokens": 8000}
        body = self.build(cap)
        self.assertEqual(body["thinking"], cap["body"]["thinking"])
        self.assertEqual(body["max_tokens"], 8000 + anthropic.HANDOFF_MIN_TOKENS)
        self.refusal("capacity", lambda: self.build(cap, context=9000))

    def test_signed_and_redacted_reply_fields_follow_the_hermes_normalizer(self):
        # Hermes bc2e4d37 agent/transports/anthropic.py:65-105 keeps these signed blocks in reasoning_details.
        blocks = [{"type": "thinking", "thinking": "synthetic thought", "signature": "fake-signed-value"},
                  {"type": "redacted_thinking", "data": "fake-redacted-data"}]
        reply = dict(self.reply, reasoning_content="synthetic thought", reasoning_details=blocks)
        body = self.build(messages=[*self.rows, reply, user("latest")])
        self.assertEqual(body["messages"][-2]["content"], [*blocks, {"type": "text", "text": "result"}])
        self.assertEqual(reply["reasoning_details"], blocks)

    def test_ordered_thinking_tool_carrier_keeps_exact_block_order(self):
        # The normalizer adds anthropic_content_blocks for signed thinking interleaved with tool use.
        first = {"type": "thinking", "thinking": "before", "signature": "fake-first"}
        second = {"type": "thinking", "thinking": "after", "signature": "fake-second"}
        call = {"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "synthetic"}}
        ordered = [first, {"type": "text", "text": "call"}, call, second]
        reply = assistant("call", [("c1", "read", '{"path":"synthetic"}')])
        reply.update(reasoning_details=[first, second], reasoning_content="before\n\nafter",
                     anthropic_content_blocks=ordered)
        cap = capture(self.rows, reply)
        messages = [*self.rows, reply, tool("c1", "output")]
        body = self.build(cap, messages)
        self.assertEqual(body["messages"][-2]["content"], ordered)
        changed = copy.deepcopy(reply)
        changed["anthropic_content_blocks"][2]["input"] = {"path": "changed"}
        self.refusal("source_transform_unsupported", lambda: self.build(
            cap, [*self.rows, changed, tool("c1", "output")]))

    def test_changed_or_unknown_thinking_fields_are_refused(self):
        blocks = [{"type": "thinking", "thinking": "synthetic", "signature": "fake"}]
        self.rows[1].update(reasoning_details=blocks, reasoning_content="synthetic")
        self.capture = capture(self.rows, self.reply)
        self.capture["body"]["messages"][1]["content"][0]["signature"] = "changed"
        self.refusal("source_transform_unsupported", self.build)
        for details in ([{"type": "thinking", "thinking": "unsigned"}],
                        [{"type": "redacted_thinking", "data": ""}],
                        [{"type": "custom.native_assistant", "payload": "opaque"}],
                        [{**blocks[0], "output_only": True}]):
            reply = dict(self.reply, reasoning_details=details)
            with self.subTest(details=details):
                self.refusal("source_transform_unsupported", lambda reply=reply: self.build(
                    messages=[*self.rows, reply]))


class HeaderTest(unittest.TestCase):
    def test_api_key_auth_keeps_custom_headers(self):
        headers = anthropic.route_headers("sk-ant-api-synthetic", ROUTE[1], {
            "X-Title": "Synthetic"})
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers["x-api-key"], "sk-ant-api-synthetic")
        self.assertEqual(headers["anthropic-beta"], anthropic.COMMON_BETAS)
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["X-Title"], "Synthetic")

    def test_known_headers_keep_their_case_and_value(self):
        defaults = {"Anthropic-Beta": "synthetic-beta", "Anthropic-Version": "synthetic-version"}
        headers = anthropic.route_headers("fake", "http://127.0.0.1:9", defaults)
        self.assertEqual(headers, {**defaults, "x-api-key": "fake"})
        self.assertEqual(defaults, {"Anthropic-Beta": "synthetic-beta", "Anthropic-Version": "synthetic-version"})

    def test_oauth_bearer_and_signed_cloud_routes_are_refused(self):
        for key, base, provider in (("sk-ant-oat-fake", ROUTE[1], ""), ("eyJfake", ROUTE[1], ""),
                                   ("fake", "https://api.minimax.io/anthropic", ""),
                                   ("fake", "https://synthetic.azure.com/anthropic", ""),
                                   ("fake", "https://api.commandcode.ai", ""),
                                   ("fake", "https://synthetic.palantirfoundry.com", ""),
                                   ("fake", "https://inference-api.nousresearch.com", ""),
                                   ("fake", "https://synthetic.example", "nous"),
                                   ("fake", ROUTE[1], "bedrock")):
            with self.subTest(base=base), self.assertRaises(WarmRefusal) as caught:
                anthropic.route_headers(key, base, {}, provider)
            self.assertEqual(caught.exception.code, "auth_unsupported")

    def test_a_route_header_with_another_credential_is_refused(self):
        for headers in ({"Authorization": "Bearer other"}, {"Authorization": "Bearer fake"},
                        {"X-Api-Key": "other"}):
            with self.subTest(headers=headers), self.assertRaises(WarmRefusal) as caught:
                anthropic.route_headers("fake", ROUTE[1], headers)
            self.assertEqual(caught.exception.code, "auth_unsupported")


class ParseTest(unittest.TestCase):
    def reply(self, **changes):
        return {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Synthetic handoff"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                                                      "cache_creation_input_tokens": 5, "output_tokens": 8}, **changes}

    def test_native_usage_counts_all_input_components(self):
        reply = anthropic.parse_reply(self.reply())
        self.assertEqual((reply["prompt_tokens"], reply["cached_tokens"], reply["completion_tokens"]), (105, 90, 8))
        self.assertEqual((reply["content"], reply["finish_reason"], reply["tool_calls"], reply["refusal"]),
                         ("Synthetic handoff", "stop", False, False))

    def test_stops_tools_refusals_and_unknown_counters(self):
        for reason, expected in (("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length"),
                                 ("refusal", "content_filter"), ("tool_use", "tool_calls"), (None, None)):
            with self.subTest(reason=reason):
                reply = anthropic.parse_reply(self.reply(stop_reason=reason, usage={}))
                self.assertEqual(reply["finish_reason"], expected)
                self.assertEqual(reply["refusal"], reason == "refusal")
                self.assertIsNone(reply["cached_tokens"])
                self.assertIsNone(reply["prompt_tokens"])
        reply = anthropic.parse_reply(self.reply(content=[{"type": "tool_use", "name": "read"}]))
        self.assertTrue(reply["tool_calls"])

    def test_thinking_is_not_summary_text_and_malformed_replies_are_refused(self):
        reply = anthropic.parse_reply(self.reply(content=[{"type": "thinking", "thinking": "synthetic"},
                                                         {"type": "text", "text": "summary"}]))
        self.assertEqual(reply["content"], "summary")
        for payload in ({}, self.reply(content="wrong"), self.reply(content=[{"type": "text", "text": None}]),
                        self.reply(usage="wrong"), self.reply(usage=[]), self.reply(usage=False)):
            with self.subTest(payload=payload), self.assertRaises(WarmRefusal) as caught:
                anthropic.parse_reply(payload)
            self.assertEqual(caught.exception.code, "incomplete_response")


if __name__ == "__main__":
    unittest.main()
