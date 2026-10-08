"""Check bounded MOA captures with synthetic hook payloads."""

import copy
import os
import threading
import unittest
from unittest.mock import patch

from warm_compaction.moa import (
    ADVISOR_PREFIX, MAX_OPEN, MAX_RECORD_BYTES, MAX_REFERENCES, MAX_ROUTES, MAX_SESSIONS,
    MoaStore, build_aggregator, build_reference, resolve_key, validate_routes,
)
from warm_compaction.warm import WarmRefusal
from wc_fixtures import SYSTEM, assistant, capture_for, tool, user

VIRTUAL = ("test-preset", "moa://local", "chat_completions")
ROUTE = {"name": "acting", "provider": "custom", "model": "acting-model",
         "base_url": "http://127.0.0.1:9/v1", "context_length": 131072, "api_key_env": ""}
REF_ROUTE = {**ROUTE, "name": "advisor", "model": "advisor-model"}


def event(route=ROUTE, *, task="moa_aggregator", request_id="a1", retry=0, session="s1", turn="t1",
          messages=None, streaming=False, extra=None):
    rows = copy.deepcopy(messages if messages is not None else [SYSTEM, user("Task")])
    body = {"model": route["model"], "messages": rows, "max_tokens": 256, "temperature": 0.3,
            "stream": streaming, **(extra or {})}
    return {"aux_task": task, "api_request_id": request_id, "retry_count": retry,
            "session_id": session, "turn_id": turn, "provider": route["provider"], "model": route["model"],
            "base_url": route["base_url"], "api_mode": "chat_completions", "streaming": streaming,
            "request": {"method": "POST", "body": body}, "request_messages": rows}


def post_event(pre, *, finish="stop", error=None, content="Reference advice", usage=None, calls=None):
    result = {key: value for key, value in pre.items() if key not in ("request", "request_messages")}
    result.update(finish_reason=finish, error=error, error_type=None, usage=usage,
                  response={"assistant_message": {"role": "assistant", "content": content, "tool_calls": calls}})
    return result


class RoutesTest(unittest.TestCase):
    def test_normalizes_routes_without_reading_keys(self):
        with patch.dict(os.environ, {"WC_TEST_MOA_KEY": "first"}):
            route = validate_routes([{**ROUTE, "base_url": "HTTP://LOCALHOST:9/v1///", "provider": "CUSTOM",
                                      "api_key_env": "WC_TEST_MOA_KEY"}])[0]
        self.assertEqual(route["base_url"], "http://localhost:9/v1")
        self.assertEqual(route["provider"], "custom")
        self.assertNotIn("api_key", route)
        with patch.dict(os.environ, {"WC_TEST_MOA_KEY": "second"}):
            self.assertEqual(resolve_key(route), "second")

    def test_invalid_route_forms_are_rejected(self):
        changes = (
            {"base_url": "moa://local"}, {"base_url": "file:///tmp/x"},
            {"base_url": "http://name:secret@localhost/v1"}, {"base_url": "http://x/v1?q=secret"},
            {"base_url": "http://x/v1#fragment"}, {"base_url": "http://x/v1?"}, {"base_url": "http://x/v1#"},
            {"base_url": "http://x:bad/v1"},
            {"base_url": "http://x\\other/v1"}, {"base_url": " http://x/v1"},
            {"context_length": 0}, {"context_length": True}, {"context_length": "131072"},
            {"api_mode": "responses"}, {"provider": "moa"}, {"name": "unsafe name"},
            {"name": "naïve"}, {"api_key_env": "BAD-VAR"}, {"model": " m "}, {"model": "m\nsecret"},
            {"api_key": "secret"},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError) as caught:
                validate_routes([{**ROUTE, **change}])
            self.assertTrue(str(caught.exception).startswith("moa_"))
            self.assertNotIn("secret", str(caught.exception))

    def test_duplicate_routes_names_and_route_limit(self):
        cases = ([ROUTE, {**ROUTE, "name": "other", "base_url": ROUTE["base_url"] + "/"}],
                 [ROUTE, {**REF_ROUTE, "name": "acting"}], None,
                 [{**ROUTE, "name": f"r{i}", "model": f"m{i}"} for i in range(MAX_ROUTES + 1)])
        for routes in cases:
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                MoaStore(routes)

    def test_no_key_missing_key_and_bad_key(self):
        self.assertEqual(resolve_key(ROUTE), "")
        route = {**ROUTE, "api_key_env": "WC_TEST_MOA_MISSING_KEY"}
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(WarmRefusal) as caught:
            resolve_key(route)
        self.assertEqual(caught.exception.code, "moa_key_missing")
        with patch.dict(os.environ, {"WC_TEST_MOA_MISSING_KEY": "value\nunsafe"}), self.assertRaises(WarmRefusal):
            resolve_key(route)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.store = MoaStore([ROUTE, REF_ROUTE], include_references=True)
        self.history = [user("Task")]
        self.reply = assistant("Done")

    def begin(self, request_id="m1", session="s1", turn="t1", history=None):
        self.store.pre_main(api_request_id=request_id, session_id=session, turn_id=turn, provider="moa",
                            conversation_history=self.history if history is None else history,
                            model=VIRTUAL[0], base_url=VIRTUAL[1], api_mode=VIRTUAL[2])

    def finish(self, request_id="m1", session="s1", history=None):
        cap = capture_for(self.history if history is None else history, self.reply, route=VIRTUAL, session=session)
        cap["moa"] = self.store.finish_main(request_id, cap)
        return cap

    def round(self, pre=None, post=None, request_id="m1"):
        pre = event() if pre is None else pre
        def send():
            self.store.on_pre_auxiliary_call(**pre)
            self.store.on_post_auxiliary_call(**(post_event(pre) if post is None else post))
            return "unchanged"
        return self.store.run_main(request_id, send)

    def reference(self, request_id="r1", content="Reference advice", **extra):
        pre = event(REF_ROUTE, task="moa_reference", request_id=request_id, **extra)
        self.store.on_pre_auxiliary_call(**pre)
        self.store.on_post_auxiliary_call(**post_event(pre, content=content))
        return pre


class CaptureTest(StoreCase):
    def test_streamed_aggregator_uses_the_accepted_main_reply(self):
        self.begin()
        pre = event(streaming=True)
        self.assertEqual(self.round(pre, post_event(pre, finish=None, content=None, usage=None)), "unchanged")
        cap = self.finish()
        record = cap["moa"]["aggregator"]
        self.assertIsNone(cap["moa"]["code"])
        self.assertEqual(record["reply"], cap["reply"])
        self.assertEqual(record["route_config"]["name"], "acting")

    def test_nonstream_finish_and_usage(self):
        self.begin()
        pre = event()
        self.round(pre, post_event(pre, usage={"prompt_tokens": 2000}))
        record = self.finish()["moa"]["aggregator"]
        self.assertIsNone(record["code"])
        self.assertEqual(record["prompt_tokens"], 2000)

    def test_raw_messages_restore_long_rows_and_deep_message_fields(self):
        self.begin()
        raw = [SYSTEM, user("x" * 100000, fields={"a": {"b": {"c": "complete"}}})]
        pre = event(messages=raw)
        pre["request"] = copy.deepcopy(pre["request"])
        pre["request"]["body"]["messages"] = [SYSTEM, user("x" * 8000 + "...[truncated 92000 chars]")]
        self.round(pre)
        raw[-1]["content"] = "changed"
        self.assertEqual(len(self.finish()["moa"]["aggregator"]["body"]["messages"][-1]["content"]), 100000)

    def test_raw_tools_restore_exact_descriptions_and_deep_schemas_only_when_present(self):
        raw_tools = [{"type": "function", "function": {
            "name": "read", "description": "Synthetic description " * 1000,
            "parameters": {"type": "object", "properties": {
                "items": {"type": "array", "items": {"type": "string"}}}}}}]
        preview_tools = [{"type": "function", "function": {
            "name": "read", "description": "Synthetic...[truncated 1 chars]",
            "parameters": {"type": "object", "properties": "<dict depth limit>"}}}]
        self.begin("missing")
        self.round(event(extra={"tools": preview_tools}), request_id="missing")
        self.assertEqual(self.finish("missing")["moa"]["code"], "moa_payload_incomplete")
        self.begin("exact")
        pre = event(extra={"tools": preview_tools})
        pre["request_tools"] = copy.deepcopy(raw_tools)
        self.round(pre, request_id="exact")
        pre["request_tools"][0]["function"]["description"] = "Mutated after observation"
        self.assertEqual(self.finish("exact")["moa"]["aggregator"]["body"]["tools"], raw_tools)

    def test_raw_tools_do_not_restore_missing_or_other_incomplete_fields(self):
        for extra, tools in (({}, []), ({"tools": []}, None),
                             ({"tools": [], "reasoning": "<dict depth limit>"}, [])):
            with self.subTest(extra=extra, tools=tools):
                self.begin()
                pre = event(extra=extra)
                pre["request_tools"] = tools
                self.round(pre)
                self.assertEqual(self.finish()["moa"]["code"], "moa_payload_incomplete")

    def test_incomplete_settings_do_not_use_raw_messages_to_hide_loss(self):
        losses = ({"tools": [{"description": "...[truncated 1 chars]"}]},
                  {"extra_body": {"schema": "<dict depth limit>"}}, {"extra_body": {"x": "<3 bytes>"}},
                  {"tools": [{"_truncated_items": 3}]}, {"extra_body": {"authorization": "<redacted>"}})
        for index, loss in enumerate(losses):
            with self.subTest(loss=loss):
                self.begin(f"m{index}")
                self.round(event(extra=loss), request_id=f"m{index}")
                self.assertEqual(self.finish(f"m{index}")["moa"]["code"], "moa_payload_incomplete")

    def test_whole_preview_unknown_route_modes_and_settings_refuse(self):
        cases = []
        preview = event()
        preview["request"] = {"_truncated": True, "preview": "partial"}
        cases.append((preview, "moa_payload_incomplete"))
        unknown = event()
        unknown["base_url"] = "http://127.0.0.1:10/v1"
        cases.append((unknown, "moa_route_unconfigured"))
        mode = event()
        mode["api_mode"] = "codex_responses"
        cases.append((mode, "moa_api_mode_unsupported"))
        for extra in ({"extra_headers": {"x": "1"}}, {"extra_query": {"x": "1"}},
                      {"extra_body": {"messages": []}}, {"extra_body": {"model": "other"}},
                      {"tool_choice": "required"}, {"response_format": {}}, {"max_tokens": "256"}):
            cases.append((event(extra=extra), "moa_settings_unsupported"))
        mismatch = event(extra={"model": "other"})
        cases.append((mismatch, "moa_route_changed"))
        for index, (pre, code) in enumerate(cases):
            with self.subTest(code=code):
                self.begin(f"m{index}")
                self.round(pre, request_id=f"m{index}")
                self.assertEqual(self.finish(f"m{index}")["moa"]["code"], code)

    def test_retry_replaces_success_with_failure(self):
        self.begin()
        def send():
            first = event(retry=0)
            self.store.on_pre_auxiliary_call(**first)
            self.store.on_post_auxiliary_call(**post_event(first))
            second = event(retry=1)
            self.store.on_pre_auxiliary_call(**second)
            self.store.on_post_auxiliary_call(**post_event(second, error="synthetic failure"))
        self.store.run_main("m1", send)
        result = self.finish()["moa"]
        self.assertEqual(result["code"], "moa_provider_error")
        self.assertEqual(result["aggregator"]["retry_count"], 1)

    def test_retry_uses_the_new_body_after_an_error(self):
        self.begin()
        def send():
            first = event()
            self.store.on_pre_auxiliary_call(**first)
            self.store.on_post_auxiliary_call(**post_event(first, error="synthetic failure"))
            second = event(retry=1, messages=[SYSTEM, user("Task"), user(ADVISOR_PREFIX + "Advice")])
            self.store.on_pre_auxiliary_call(**second)
            self.store.on_post_auxiliary_call(**post_event(second))
        self.store.run_main("m1", send)
        cap = self.finish()
        self.assertIsNone(cap["moa"]["code"])
        self.assertEqual(len(cap["moa"]["aggregator"]["body"]["messages"]), 3)

    def test_concurrent_or_old_order_aggregators_are_ambiguous(self):
        for index, concurrent in enumerate((True, False)):
            with self.subTest(concurrent=concurrent):
                main_id = f"m{index}"
                self.begin(main_id)
                def send(concurrent=concurrent):
                    first = event(retry=1)
                    self.store.on_pre_auxiliary_call(**first)
                    if not concurrent:
                        self.store.on_post_auxiliary_call(**post_event(first))
                    second = event(retry=0)
                    self.store.on_pre_auxiliary_call(**second)
                    self.store.on_post_auxiliary_call(**post_event(second))
                    self.store.on_post_auxiliary_call(**post_event(first))
                self.store.run_main(main_id, send)
                self.assertEqual(self.finish(main_id)["moa"]["code"], "moa_ambiguous_capture")

    def test_context_is_restored_on_exception_and_calls_once(self):
        self.begin()
        calls = []
        def fail():
            calls.append(1)
            raise ConnectionError("synthetic")
        with self.assertRaises(ConnectionError):
            self.store.run_main("m1", fail)
        self.assertEqual(calls, [1])
        self.store.on_pre_auxiliary_call(**event())
        self.store.on_post_auxiliary_call(**post_event(event()))
        self.assertEqual(self.finish()["moa"]["code"], "moa_no_aggregator_capture")

    def test_thread_contexts_keep_sessions_separate(self):
        barrier = threading.Barrier(2)
        errors = []
        def worker(index):
            try:
                main_id, session, turn = f"m{index}", f"s{index}", f"t{index}"
                self.begin(main_id, session, turn)
                def send():
                    pre = event(request_id=f"a{index}", session=session, turn=turn)
                    self.store.on_pre_auxiliary_call(**pre)
                    barrier.wait(timeout=5)
                    self.store.on_post_auxiliary_call(**post_event(pre))
                self.store.run_main(main_id, send)
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        for index in range(2):
            cap = self.finish(f"m{index}", f"s{index}")
            self.assertIsNone(cap["moa"]["code"])
            self.assertEqual(cap["moa"]["aggregator"]["api_request_id"], f"a{index}")

    def test_references_bind_before_main_and_keep_the_old_boundary(self):
        self.reference(content="Full advice")
        self.begin()
        self.round()
        first = self.finish()["moa"]["references"][0]
        second_history = [*self.history, self.reply, user("Next")]
        self.begin("m2", history=second_history)
        self.round(event(request_id="a2", messages=[SYSTEM, *second_history]), request_id="m2")
        second = self.finish("m2", history=second_history)["moa"]["references"][0]
        self.assertEqual(first["boundary_digests"], second["boundary_digests"])
        self.assertEqual(second["reply"], {"role": "assistant", "content": "Full advice"})

    def test_reference_completed_after_pre_main_is_not_attached(self):
        pre = event(REF_ROUTE, task="moa_reference", request_id="r1")
        self.store.on_pre_auxiliary_call(**pre)
        self.begin()
        self.store.on_post_auxiliary_call(**post_event(pre))
        self.round()
        self.assertEqual(self.finish()["moa"]["references"], [])
        self.begin("m2")
        self.round(event(request_id="a2"), request_id="m2")
        self.assertEqual(len(self.finish("m2")["moa"]["references"]), 1)

    def test_no_references_from_another_turn_or_changed_history(self):
        self.reference()
        self.begin()
        self.round()
        self.finish()
        self.begin("m2", history=[user("Changed")])
        self.round(event(request_id="a2"), request_id="m2")
        self.assertEqual(self.finish("m2", history=[user("Changed")])["moa"]["references"], [])
        self.begin("m3", turn="t2")
        self.round(event(request_id="a3", turn="t2"), request_id="m3")
        self.assertEqual(self.finish("m3")["moa"]["references"], [])

    def test_duplicate_reference_destination_does_not_invent_slots(self):
        first = event(REF_ROUTE, task="moa_reference", request_id="r1")
        second = event(REF_ROUTE, task="moa_reference", request_id="r2")
        self.store.on_pre_auxiliary_call(**first)
        self.store.on_pre_auxiliary_call(**second)
        self.store.on_post_auxiliary_call(**post_event(first))
        self.store.on_post_auxiliary_call(**post_event(second))
        self.begin()
        self.round()
        record = self.finish()["moa"]["references"][0]
        self.assertEqual(record["code"], "moa_ambiguous_capture")

    def test_sequential_duplicate_reference_calls_before_main_are_ambiguous(self):
        self.reference(request_id="r1")
        self.reference(request_id="r2")
        self.begin()
        self.round()
        self.assertEqual(self.finish()["moa"]["references"][0]["code"], "moa_ambiguous_capture")

    def test_reference_recovery_after_a_bound_view_is_a_fresh_boundary(self):
        self.reference(request_id="r1")
        self.begin()
        self.round()
        first = self.finish()["moa"]["references"][0]
        self.reference(request_id="r2")
        history = [*self.history, self.reply, user("Next")]
        self.begin("m2", history=history)
        self.round(event(request_id="a2", messages=[SYSTEM, *history]), request_id="m2")
        second = self.finish("m2", history=history)["moa"]["references"][0]
        self.assertIsNone(second["code"])
        self.assertGreater(len(second["boundary_digests"]), len(first["boundary_digests"]))

    def test_preset_switch_cannot_reuse_a_bound_reference_in_the_same_turn(self):
        self.reference()
        self.begin()
        self.round()
        self.finish()
        self.store.pre_main(api_request_id="m2", session_id="s1", turn_id="t1", provider="moa",
                            conversation_history=self.history, model="changed",
                            base_url=VIRTUAL[1], api_mode=VIRTUAL[2])
        self.round(event(request_id="a2"), request_id="m2")
        self.assertEqual(self.finish("m2")["moa"]["references"], [])

    def test_distinct_successful_aggregator_calls_are_ambiguous(self):
        self.begin()
        def send():
            for request_id in ("a1", "a2"):
                pre = event(request_id=request_id)
                self.store.on_pre_auxiliary_call(**pre)
                self.store.on_post_auxiliary_call(**post_event(pre))
        self.store.run_main("m1", send)
        self.assertEqual(self.finish()["moa"]["code"], "moa_ambiguous_capture")

    def test_invalid_retry_replaces_a_success_without_stopping_the_call(self):
        self.begin()
        def send():
            pre = event()
            self.store.on_pre_auxiliary_call(**pre)
            self.store.on_post_auxiliary_call(**post_event(pre))
            self.store.on_pre_auxiliary_call(**event(retry="bad"))
            return "unchanged"
        self.assertEqual(self.store.run_main("m1", send), "unchanged")
        self.assertEqual(self.finish()["moa"]["code"], "moa_ambiguous_capture")

    def test_reference_reply_must_be_complete_text_without_tools(self):
        cases = ({"content": "...[truncated 10 chars]"}, {"calls": [{"id": "c"}]},
                 {"finish": "length"}, {"content": ""})
        for index, change in enumerate(cases):
            with self.subTest(change=change):
                pre = event(REF_ROUTE, task="moa_reference", request_id=f"r{index}")
                self.store.on_pre_auxiliary_call(**pre)
                self.store.on_post_auxiliary_call(**post_event(pre, **change))
                self.begin(f"m{index}")
                self.round(event(request_id=f"a{index}"), request_id=f"m{index}")
                self.assertEqual(self.finish(f"m{index}")["moa"]["references"][0]["code"], "moa_reply_incomplete")

    def test_forget_discards_pending_and_late_replies(self):
        self.reference()
        self.begin()
        pre = event()
        self.store.run_main("m1", lambda: self.store.on_pre_auxiliary_call(**pre))
        self.store.forget("s1")
        self.store.on_post_auxiliary_call(**post_event(pre))
        self.assertEqual(self.finish()["moa"]["code"], "moa_missing_identity")
        self.assertEqual(len(self.store._references), 0)
        self.assertEqual(len(self.store._pending), 0)

    def test_late_reference_pre_after_forget_is_ignored_until_new_main(self):
        self.store.forget("s1")
        self.reference()
        self.assertEqual(len(self.store._references), 0)
        self.begin()
        self.assertEqual(len(self.store._mains), 0)
        self.store.open_session("s1")
        self.begin("m2")
        self.reference(request_id="r2")
        self.assertEqual(len(self.store._references), 1)

    def test_forget_and_reopen_during_pre_main_refuses_old_digest_work(self):
        from warm_compaction.rows import row_digest
        def digest(row):
            self.store.forget("s1")
            self.store.open_session("s1")
            return row_digest(row)
        with patch("warm_compaction.moa.row_digest", side_effect=digest):
            self.begin()
        self.assertEqual(len(self.store._mains), 0)
        self.begin("m2")
        self.round(event(request_id="a2"), request_id="m2")
        self.assertIsNone(self.finish("m2")["moa"]["code"])

    def test_forget_and_reopen_during_aux_copy_refuses_old_body(self):
        from warm_compaction.moa import _body
        def body(*args):
            result = _body(*args)
            self.store.forget("s1")
            self.store.open_session("s1")
            return result
        with patch("warm_compaction.moa._body", side_effect=body):
            self.reference()
        self.assertEqual(len(self.store._references), 0)
        self.assertEqual(len(self.store._pending), 0)

    def test_moa_token_eviction_invalidates_pre_main_work_in_progress(self):
        from warm_compaction.rows import row_digest
        def digest(row):
            for index in range(MAX_SESSIONS):
                self.store.open_session(f"other-{index}")
            return row_digest(row)
        with patch("warm_compaction.moa.row_digest", side_effect=digest):
            self.begin()
        self.assertEqual(len(self.store._mains), 0)
        self.assertLessEqual(len(self.store._versions), MAX_SESSIONS)

    def test_capture_limits_and_large_raw_body(self):
        self.begin()
        self.round(event(messages=[SYSTEM, user("x" * (MAX_RECORD_BYTES + 1))]))
        self.assertEqual(self.finish()["moa"]["code"], "moa_capture_too_large")
        for index in range(MAX_SESSIONS + 3):
            self.begin(f"m{index}", f"s{index}", f"t{index}")
        self.assertLessEqual(len(self.store._mains), MAX_OPEN)
        self.assertLessEqual(len(self.store._sessions), MAX_SESSIONS)
        self.assertLessEqual(len(self.store._references), MAX_REFERENCES)


class BuildersTest(StoreCase):
    def acting_capture(self, *, source=None, extra=None):
        self.begin()
        self.round(event(messages=source, extra=extra))
        return self.finish()

    def test_aggregator_keeps_prefix_settings_and_complete_tool_group(self):
        self.reply = assistant("", calls=[("call-1", "read", "{}")])
        source = [SYSTEM, *self.history, user(ADVISOR_PREFIX + "Advice")]
        cap = self.acting_capture(source=source,
                                  extra={"extra_body": {"top_k": 30}, "stream_options": {"include_usage": True}})
        rows = [*self.history, self.reply, tool("call-1", "Result"), user("Next request")]
        body, route = build_aggregator(cap, rows, VIRTUAL, 131072, "Handoff")
        self.assertEqual(body["messages"][:len(source)], source)
        self.assertEqual(body["messages"][-2]["tool_call_id"], "call-1")
        self.assertEqual(body["messages"][-1], user("Next request\n\nHandoff"))
        self.assertNotIn(user("Next request"), body["messages"])
        self.assertEqual(body["top_k"], 30)
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(route["name"], "acting")
        self.assertFalse(body["stream"])
        self.assertNotIn("stream_options", body)

    def test_request_text_change_with_the_same_role_refuses(self):
        cap = self.acting_capture()
        cap["body"]["messages"][-1]["content"] = "Altered task"
        cap["moa"]["aggregator"]["body"]["messages"][-1]["content"] = "Altered task"
        with self.assertRaises(WarmRefusal) as caught:
            build_aggregator(cap, [*self.history, self.reply], VIRTUAL, 131072, "Handoff")
        self.assertEqual(caught.exception.code, "moa_source_transform_unsupported")

    def test_handoff_controls_keep_public_limits(self):
        cap = self.acting_capture(extra={"max_tokens": 50, "stop": ["##"],
                                         "web_search_options": {"search_context_size": "low"}})
        body, _ = build_aggregator(cap, [*self.history, self.reply], VIRTUAL, 131072, "Handoff")
        self.assertEqual(body["max_tokens"], 2048)
        self.assertNotIn("stop", body)
        self.assertNotIn("web_search_options", body)
        cap["moa"]["aggregator"]["body"].pop("max_tokens")
        body, _ = build_aggregator(cap, [*self.history, self.reply], VIRTUAL, 131072, "Handoff")
        self.assertEqual(body["max_tokens"], 8192)

    def test_aggregator_refuses_merged_guidance_missing_tools_and_changed_history(self):
        cap = self.acting_capture(source=[SYSTEM, user("Task\n" + ADVISOR_PREFIX + "merged")])
        with self.assertRaises(WarmRefusal) as caught:
            build_aggregator(cap, [*self.history, self.reply], VIRTUAL, 131072, "Handoff")
        self.assertEqual(caught.exception.code, "moa_source_transform_unsupported")
        transformed = copy.deepcopy(cap)
        transformed["moa"]["aggregator"]["body"]["messages"] = [SYSTEM, assistant("wrong role")]
        with self.assertRaises(WarmRefusal) as caught:
            build_aggregator(transformed, [*self.history, self.reply], VIRTUAL, 131072, "Handoff")
        self.assertEqual(caught.exception.code, "moa_source_transform_unsupported")
        with self.assertRaises(WarmRefusal) as caught:
            build_aggregator(cap, [user("Changed"), self.reply], VIRTUAL, 131072, "Handoff")
        self.assertEqual(caught.exception.code, "moa_history_changed")
        self.reply = assistant("", calls=[("c1", "read", "{}")])
        cap = self.acting_capture()
        with self.assertRaises(WarmRefusal):
            build_aggregator(cap, [*self.history, self.reply], VIRTUAL, 131072, "Handoff")

    def test_virtual_route_change_and_capacity_refuse(self):
        cap = self.acting_capture()
        rows = [*self.history, self.reply]
        for route, window, code in ((('other', 'moa://local', 'chat_completions'), 131072, "moa_route_changed"),
                                    (VIRTUAL, 50, "moa_capacity"), (VIRTUAL, 0, "moa_capacity_unknown")):
            with self.subTest(code=code), self.assertRaises(WarmRefusal) as caught:
                build_aggregator(cap, rows, route, window, "Handoff")
            self.assertEqual(caught.exception.code, code)
        cap["moa"]["aggregator"]["prompt_tokens"] = 131072
        with self.assertRaises(WarmRefusal) as caught:
            build_aggregator(cap, rows, VIRTUAL, 131072, "Handoff")
        self.assertEqual(caught.exception.code, "moa_capacity")

    def test_reference_request_uses_its_own_view_and_reply(self):
        reference_view = [SYSTEM, user("Only the trimmed view")]
        self.reference(messages=reference_view, content="Own reference reply")
        self.begin()
        self.round()
        record = self.finish()["moa"]["references"][0]
        body, route = build_reference(record, "Handoff")
        self.assertEqual(body["messages"][:len(reference_view)], reference_view)
        self.assertEqual(body["messages"][-2], assistant("Own reference reply"))
        self.assertIn("reference view", body["messages"][-1]["content"])
        self.assertNotIn(assistant("Done"), body["messages"])
        self.assertEqual(route["name"], "advisor")

    def test_unbound_reference_and_reference_capacity_refuse(self):
        self.reference()
        self.begin()
        self.round()
        record = self.finish()["moa"]["references"][0]
        unbound = {**record, "boundary_digests": None}
        with self.assertRaises(WarmRefusal) as caught:
            build_reference(unbound, "Handoff")
        self.assertEqual(caught.exception.code, "moa_reference_unbound")
        record["route_config"]["context_length"] = 100
        with self.assertRaises(WarmRefusal) as caught:
            build_reference(record, "Handoff")
        self.assertEqual(caught.exception.code, "moa_capacity")


if __name__ == "__main__":
    unittest.main()
