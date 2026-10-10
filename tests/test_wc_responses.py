"""Synthetic Responses request and reply checks. No provider request runs here."""

import copy
import unittest

from warm_compaction import responses
from warm_compaction.rows import row_digest
from warm_compaction.warm import WarmRefusal
from wc_fixtures import assistant, tool, user

ROUTE = ("fake-model", "http://127.0.0.1:9/v1", "codex_responses")
CODEX = ("fake-model", "https://chatgpt.com/backend-api/codex", "codex_responses")
INSTRUCTION = "Write a handoff. Use no tools."


def capture(rows, reply, route=ROUTE, **controls):
    """Return a synthetic capture with the supported wire form."""
    return {"route": route, "digests": [row_digest(row) for row in rows],
            "body": {"model": route[0], "instructions": "Synthetic system text.", "store": False,
                     "input": responses.wire_rows(rows, route[1], model=route[0]), **controls},
            "reply": {"content": reply.get("content") or "", "tool_calls": [
                [call["id"], call["function"]["name"]] for call in reply.get("tool_calls", [])]}}


def response(status="completed", **extra):
    """Return a synthetic terminal provider payload."""
    return {"status": status, "output": [{"type": "message", "role": "assistant", "status": "completed",
                                           "content": [{"type": "output_text", "text": "Synthetic summary."}]}],
            **extra}


class ResponsesSourceTest(unittest.TestCase):
    def test_plain_text_and_compact_tool_arguments_match(self):
        rows = [user("  question  "), assistant(None, [("call_a", "read", '{"z": 1, "a": "é"}')]),
                tool("call_a", " result ")]
        wire = responses.wire_rows(rows)
        self.assertEqual(wire[0], {"type": "message", "role": "user", "content": "question"})
        self.assertEqual(wire[1], {"type": "function_call", "call_id": "call_a", "name": "read",
                                   "arguments": '{"a":"\\u00e9","z":1}'})
        self.assertEqual(wire[2]["output"], " result ")
        responses.check_source({"model": ROUTE[0], "instructions": "Synthetic system text.",
                                "store": False, "input": wire}, rows)

    def test_codex_text_is_typed(self):
        wire = responses.wire_rows([user("u"), assistant("a", phase="final_answer")], CODEX[1])
        self.assertEqual(wire[0]["content"], [{"type": "input_text", "text": "u"}])
        self.assertEqual(wire[1]["content"], [{"type": "output_text", "text": "a"}])
        self.assertEqual(wire[1]["phase"], "final_answer")

    def test_an_input_item_without_a_type_is_a_message(self):
        # The Responses API reads an input item that has a role and no "type" as a message.
        # Hermes sends that form, so the source check must accept it.
        rows = [user("u"), assistant("a")]
        wire = responses.wire_rows(rows, ROUTE[1], model=ROUTE[0])
        untyped = [{key: value for key, value in item.items() if key != "type"} for item in wire]
        self.assertEqual(untyped[0], {"role": "user", "content": "u"})
        responses.check_source({"model": ROUTE[0], "instructions": "Synthetic system text.",
                                "store": False, "input": untyped}, rows)

    def test_native_text_replay_keeps_phase_status_and_id(self):
        row = assistant("a", codex_message_items=[{
            "type": "message", "role": "assistant", "id": "msg_fake", "status": "completed",
            "phase": "final_answer", "content": [{"type": "output_text", "text": "a", "annotations": []}]}])
        wire = responses.wire_rows([row], CODEX[1])
        self.assertEqual(wire[0]["id"], "msg_fake")
        self.assertEqual(wire[0]["status"], "completed")
        self.assertEqual(wire[0]["phase"], "final_answer")
        self.assertEqual(wire[0]["content"], [{"type": "output_text", "text": "a"}])
        foreign = copy.deepcopy(row)
        foreign["codex_message_items"][0]["id"] = "foreign_id"
        self.assertNotIn("id", responses.wire_rows([foreign], CODEX[1])[0])

    def test_composite_tool_ids_use_call_id(self):
        rows = [assistant(None, [("call_a|fc_a", "read", "{}")]), tool("call_a|fc_a", "done")]
        wire = responses.wire_rows(rows)
        self.assertEqual([item["call_id"] for item in wire], ["call_a", "call_a"])

    def test_copilot_host_check_requires_a_domain_boundary(self):
        row = assistant("a", codex_message_items=[{
            "type": "message", "role": "assistant", "id": "msg_synthetic", "status": "completed",
            "content": [{"type": "output_text", "text": "a"}]}])
        for host, expected in (("githubcopilot.com", False), ("api.githubcopilot.com", False),
                               ("evilgithubcopilot.com", True), ("githubcopilot.com.evil.test", True)):
            with self.subTest(host=host):
                wire = responses.wire_rows([row], "https://" + host + "/v1")
                self.assertEqual("id" in wire[0], expected)

    def test_source_change_is_refused(self):
        rows = [user("original")]
        body = {"model": ROUTE[0], "instructions": "Synthetic system text.", "store": False,
                "input": responses.wire_rows([user("changed")])}
        with self.assertRaises(WarmRefusal) as caught:
            responses.check_source(body, rows)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_source_check_refuses_settings_that_cannot_supply_budget_estimates(self):
        rows, reply = [user("q")], assistant("a")
        for field, value in (("instructions", ""), ("instructions", " Synthetic system text. "), ("store", None)):
            body = capture(rows, reply)["body"]
            if value is None:
                body.pop(field)
            else:
                body[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(WarmRefusal):
                responses.check_source(body, rows, ROUTE[1])

    def test_main_preparation_whitespace_is_required_and_prefix_is_kept(self):
        rows, reply = [user("question ")], assistant("answer")
        saved = capture(rows, reply)
        responses.check_source(saved["body"], rows)
        built = responses.build_request(saved, [*rows, reply], ROUTE, 100_000, INSTRUCTION)
        self.assertEqual(built["input"][0], saved["body"]["input"][0])
        saved["body"]["input"][0]["content"] = "question "
        with self.assertRaises(WarmRefusal):
            responses.check_source(saved["body"], rows)

    def test_unsafe_source_rows_are_refused(self):
        replay = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "old"}]}
        cases = [
            [user([{"type": "image_url", "image_url": {"url": "https://example.invalid/fake.png"}}])],
            [assistant("a", codex_reasoning_items=[{"type": "compaction", "encrypted_content": "fake"}])],
            [assistant("new", codex_message_items=[replay])],
            [assistant(None, [("x" * 65, "read", "{}")]), tool("x" * 65, "done")],
            [assistant(None, [("a", "read", "{}"), ("a", "read", "{}")]), tool("a", "done")],
            [tool("unknown", "done")],
            [assistant(None, [("a", "read", "{}")])],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(WarmRefusal):
                responses.wire_rows(rows)

    def test_native_reasoning_and_message_replay_match_the_final_wire_shape(self):
        row = assistant("final", codex_reasoning_items=[{
            "type": "reasoning", "id": "rs_fake", "encrypted_content": "fake-sealed-bytes",
            "summary": [{"type": "summary_text", "text": "Synthetic reasoning summary."}],
            "_issuer_kind": "codex_backend", "_issuer_model": CODEX[0]}], codex_message_items=[{
                "type": "message", "role": "assistant", "id": "msg_fake", "status": "completed",
                "phase": "final_answer", "content": [{"type": "output_text", "text": "final"}]}])
        wire = responses.wire_rows([row], CODEX[1], model=CODEX[0])
        self.assertEqual(wire[0], {"type": "reasoning", "encrypted_content": "fake-sealed-bytes",
                                   "summary": [{"type": "summary_text", "text": "Synthetic reasoning summary."}]})
        self.assertNotIn("id", wire[1])
        body = {"model": CODEX[0], "instructions": "Synthetic system text.", "store": False, "input": wire}
        responses.check_source(body, [row], CODEX[1])
        for changed_key, changed_value in (("encrypted_content", "changed-sealed-bytes"),
                                           ("_issuer_kind", "xai_responses"), ("_issuer_model", "foreign-model")):
            changed = copy.deepcopy(row)
            changed["codex_reasoning_items"][0][changed_key] = changed_value
            with self.subTest(changed_key=changed_key), self.assertRaises(WarmRefusal):
                responses.check_source(body, [changed], CODEX[1])
        changed = copy.deepcopy(row)
        changed["codex_message_items"][0]["phase"] = "commentary"
        with self.assertRaises(WarmRefusal):
            responses.check_source(body, [changed], CODEX[1])

    def test_reasoning_then_commentary_then_final_keeps_the_item_order(self):
        row = assistant("final", codex_reasoning_items=[{"type": "reasoning", "encrypted_content": "fake"}],
                        codex_message_items=[{
                            "type": "message", "role": "assistant", "status": "completed", "phase": phase,
                            "content": [{"type": "output_text", "text": text}]}
                            for phase, text in (("commentary", "working"), ("final_answer", "final"))])
        wire = responses.wire_rows([row], CODEX[1])
        self.assertEqual([item["type"] for item in wire], ["reasoning", "message", "message"])
        self.assertEqual([item["phase"] for item in wire[1:]], ["commentary", "final_answer"])

    def test_reasoning_tool_pair_and_new_reply_keep_native_bytes(self):
        rows = [user("q")]
        reply = assistant(None, [("call_fake", "read", "{}")], codex_reasoning_items=[{
            "type": "reasoning", "id": "rs_fake", "encrypted_content": "fake-sealed-bytes",
            "_issuer_kind": "codex_backend", "_issuer_model": CODEX[0]}])
        saved = capture(rows, reply, CODEX)
        built = responses.build_request(saved, [*rows, reply, tool("call_fake", "done")],
                                        CODEX, 100_000, INSTRUCTION)
        self.assertEqual([item["type"] for item in built["input"]],
                         ["message", "reasoning", "function_call", "function_call_output", "message"])
        self.assertEqual(built["input"][1]["encrypted_content"], "fake-sealed-bytes")

    def test_new_codex_text_that_the_host_would_change_is_refused(self):
        for text in ("<|start|>", "<\u200b|start|>"):
            with self.subTest(text=text), self.assertRaises(WarmRefusal):
                responses.wire_rows([user(text)], CODEX[1])

    def test_reasoning_ids_cannot_repeat_across_the_capture_boundary(self):
        native = {"type": "reasoning", "id": "rs_fake", "encrypted_content": "fake-sealed-bytes"}
        rows = [user("q"), assistant("old", codex_reasoning_items=[native]), user("q2")]
        reply = assistant("new", codex_reasoning_items=[copy.deepcopy(native)])
        saved = capture(rows, reply, CODEX)
        with self.assertRaises(WarmRefusal):
            responses.build_request(saved, [*rows, reply], CODEX, 100_000, INSTRUCTION)


class ResponsesBuildTest(unittest.TestCase):
    def test_capture_ahead_has_a_history_refusal(self):
        for route in (ROUTE, CODEX):
            with self.subTest(route=route):
                rows = [user("First synthetic question."), assistant("First synthetic answer."),
                        user("Next synthetic question.")]
                saved = capture(rows, assistant("Captured synthetic answer."), route)
                messages = rows[:-1]
                original_capture, original_messages = copy.deepcopy(saved), copy.deepcopy(messages)
                with self.assertRaises(WarmRefusal) as caught:
                    responses.build_request(saved, messages, route, 100_000, INSTRUCTION)
                self.assertEqual(caught.exception.args, ("history_changed:capture_ahead",))
                self.assertEqual(caught.exception.code, "history_changed:capture_ahead")
                self.assertEqual(saved, original_capture)
                self.assertEqual(messages, original_messages)

    def test_changed_digest_has_a_history_refusal(self):
        for route in (ROUTE, CODEX):
            with self.subTest(route=route):
                rows, reply = [user("Original synthetic question.")], assistant("Synthetic answer.")
                saved = capture(rows, reply, route)
                messages = [user("Changed synthetic question."), reply]
                original_capture, original_messages = copy.deepcopy(saved), copy.deepcopy(messages)
                with self.assertRaises(WarmRefusal) as caught:
                    responses.build_request(saved, messages, route, 100_000, INSTRUCTION)
                self.assertEqual(caught.exception.args, ("history_changed:digest",))
                self.assertEqual(caught.exception.code, "history_changed:digest")
                self.assertEqual(saved, original_capture)
                self.assertEqual(messages, original_messages)

    def test_request_preserves_prefix_and_non_generation_controls(self):
        rows, reply = [user("question")], assistant("answer")
        controls = {"prompt_cache_key": "fake-cache", "prompt_cache_retention": "24h", "reasoning": {"effort": "low"},
                    "include": ["reasoning.encrypted_content"], "tool_choice": "auto", "parallel_tool_calls": True,
                    "tools": [{"type": "function", "name": "read", "description": "Read a fake file.",
                               "parameters": {"type": "object"}, "strict": False}],
                    "text": {"verbosity": "low"}, "max_output_tokens": 100}
        saved = capture(rows, reply, **controls)
        original = copy.deepcopy(saved)
        built = responses.build_request(saved, [*rows, reply, user("follow up")], ROUTE, 100_000, INSTRUCTION)
        self.assertEqual(saved, original)
        self.assertEqual(built["input"][:len(saved["body"]["input"])], saved["body"]["input"])
        for key, value in controls.items():
            if key != "max_output_tokens":
                self.assertEqual(built[key], value)
        self.assertEqual(built["max_output_tokens"], 2048)
        self.assertTrue(built["stream"])
        self.assertEqual(built["input"][-1]["content"][0]["text"], INSTRUCTION)

    def test_codex_has_no_added_output_limit(self):
        rows, reply = [user("question")], assistant("answer")
        built = responses.build_request(capture(rows, reply, CODEX), [*rows, reply], CODEX, 100_000, INSTRUCTION)
        self.assertNotIn("max_output_tokens", built)
        self.assertTrue(built["stream"])
        self.assertEqual(responses.reply_reserve(built, 100_000), 25_000)

    def test_new_tool_pair_is_added_without_changing_prefix(self):
        rows = [user("question")]
        reply = assistant(None, [("call_a|fc_a", "read", "{}")])
        saved = capture(rows, reply)
        built = responses.build_request(saved, [*rows, reply, tool("call_a|fc_a", "result")],
                                        ROUTE, 100_000, INSTRUCTION)
        self.assertEqual([item["type"] for item in built["input"]],
                         ["message", "function_call", "function_call_output", "message"])

    def test_forced_output_and_stateful_controls_are_refused(self):
        rows, reply = [user("q")], assistant("a")
        cases = [{"tool_choice": "required"}, {"text": {"format": {"type": "json_schema"}}},
                 {"previous_response_id": "fake"}, {"conversation": "fake"},
                 {"context_management": [{"type": "compaction"}]}, {"store": True},
                 {"tools": [{"type": "web_search"}]}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(WarmRefusal):
                responses.build_request(capture(rows, reply, **options), [*rows, reply], ROUTE, 100_000, INSTRUCTION)

    def test_consumer_wire_transform_is_refused(self):
        rows, reply = [user("q")], assistant("a")
        with self.assertRaises(WarmRefusal) as caught:
            responses.build_request(capture(rows, reply, CODEX, prompt_cache_retention="24h"),
                                    [*rows, reply], CODEX, 100_000, INSTRUCTION)
        self.assertEqual(caught.exception.code, "source_transform_unsupported")

    def test_late_scalar_transforms_are_refused(self):
        rows, reply = [user("q")], assistant("a")
        cases = [{"instructions": " Synthetic system text. "}, {"model": " fake-model "},
                 {"prompt_cache_key": "x" * 65}, {"prompt_cache_key": " cache "},
                 {"reasoning": "low"}, {"include": "reasoning.encrypted_content"},
                 {"temperature": True}, {"service_tier": " auto "},
                 {"parallel_tool_calls": None}, {"prompt_cache_retention": None}, {"text": None},
                 {"tools": [{"type": "function", "name": " read ", "parameters": {},
                             "description": "synthetic", "strict": False}]}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(WarmRefusal):
                responses.build_request(capture(rows, reply, **options), [*rows, reply], ROUTE, 100_000, INSTRUCTION)

    def test_missing_store_false_cannot_enable_provider_default_storage(self):
        rows, reply = [user("q")], assistant("a")
        saved = capture(rows, reply)
        saved["body"].pop("store")
        with self.assertRaises(WarmRefusal):
            responses.build_request(saved, [*rows, reply], ROUTE, 100_000, INSTRUCTION)

    def test_new_opaque_reply_needs_issuer_and_model_stamps(self):
        rows = [user("q")]
        reply = assistant("a", codex_reasoning_items=[{"type": "reasoning", "encrypted_content": "fake"}])
        saved = capture(rows, reply, CODEX)
        with self.assertRaises(WarmRefusal):
            responses.build_request(saved, [*rows, reply], CODEX, 100_000, INSTRUCTION)

    def test_capacity_counts_the_consumer_unknown_output_reserve(self):
        rows, reply = [user("q")], assistant("a")
        saved = capture(rows, reply, CODEX)
        saved["prompt_tokens"] = 98_000
        with self.assertRaises(WarmRefusal) as caught:
            responses.build_request(saved, [*rows, reply], CODEX, 100_000, INSTRUCTION)
        self.assertEqual(caught.exception.code, "capacity")


class ResponsesReplyTest(unittest.TestCase):
    def test_completed_reply_reads_usage_and_cache_counter(self):
        parsed = responses.parse_reply(response(usage={"input_tokens": 200, "output_tokens": 30,
                                                       "input_tokens_details": {"cached_tokens": 150}}))
        self.assertEqual(parsed, {"content": "Synthetic summary.", "finish_reason": "stop", "tool_calls": False,
                                  "refusal": False, "prompt_tokens": 200, "completion_tokens": 30,
                                  "cached_tokens": 150})

    def test_missing_counters_are_unknown(self):
        parsed = responses.parse_reply(response(usage={"input_tokens": True, "output_tokens": -1}))
        self.assertIsNone(parsed["prompt_tokens"])
        self.assertIsNone(parsed["completion_tokens"])
        self.assertIsNone(parsed["cached_tokens"])

    def test_non_completed_status_fails_the_summary_gate(self):
        for status in ("incomplete", "failed", "cancelled", None, "in_progress"):
            with self.subTest(status=status):
                self.assertEqual(responses.parse_reply(response(status))["finish_reason"], "length")

    def test_tool_and_refusal_outputs_are_flagged(self):
        for item, flag in (({"type": "function_call", "name": "read"}, "tool_calls"),
                           ({"type": "web_search_call"}, "tool_calls"),
                           ({"type": "message", "role": "assistant", "content": [
                               {"type": "refusal", "refusal": "fake"}]}, "refusal")):
            with self.subTest(item=item):
                self.assertTrue(responses.parse_reply(response(output=[item]))[flag])

    def test_commentary_and_reasoning_are_not_final_summary_text(self):
        payload = response()
        payload["output"].insert(0, {"type": "reasoning", "summary": []})
        payload["output"].insert(1, {"type": "message", "role": "assistant", "phase": "commentary",
                                     "content": [{"type": "output_text", "text": "Working."}]})
        self.assertEqual(responses.parse_reply(payload)["content"], "Synthetic summary.")

    def test_bad_terminal_payload_is_refused(self):
        cases = [{}, response(output=None), response(usage=[]), response(output=[{
            "type": "message", "role": "assistant", "status": "incomplete", "content": []}]),
            response(output=[{"type": "message", "role": "assistant", "content": [{"type": "audio"}]}])]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(WarmRefusal):
                responses.parse_reply(payload)
