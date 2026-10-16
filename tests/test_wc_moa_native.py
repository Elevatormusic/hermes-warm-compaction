"""Native MOA checks with synthetic routes, hooks, and terminal replies."""

import copy
import unittest
from types import SimpleNamespace

import test_wc_engine as engine_fixtures
from wc_fixtures import HEADINGS_TEXT, SYSTEM, assistant, tool, user, wire
from warm_compaction import responses
from warm_compaction.capture import CaptureStore
from warm_compaction.moa import ADVISOR_PREFIX, MoaStore, build_aggregator, build_reference, validate_routes
from warm_compaction.warm import WarmRefusal

VIRTUAL = ("preset", "moa://local", "chat_completions")
BASE = "https://chatgpt.com/backend-api/codex"


def route(name="acting"):
    return {"name": name, "provider": "openai-codex", "model": name, "base_url": BASE,
            "api_mode": "codex_responses", "context_length": 200000}


def terminal(text="Accepted answer", calls=None, reasoning=False):
    output = []
    if reasoning:
        output.append({"type": "reasoning", "id": "rs_synthetic", "encrypted_content": "synthetic-sealed",
                       "summary": [{"type": "summary_text", "text": "Synthetic thought"}]})
    output.append({"type": "message", "role": "assistant", "status": "completed", "phase": "final_answer",
                   "id": "msg_synthetic", "content": [{"type": "output_text", "text": text}]})
    output.extend(calls or [])
    return {"status": "completed", "output": output,
            "usage": {"input_tokens": 500, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 400}}}


def event(target, rows, task="moa_aggregator", request_id="a1", retry=0, instructions="Synthetic acting system."):
    body = {"model": target["model"], "instructions": instructions, "store": False, "stream": True,
            "input": responses.wire_rows(rows, target["base_url"], model=target["model"]),
            "reasoning": {"effort": "low"}, "include": ["reasoning.encrypted_content"]}
    context = {"schema": "hermes.native-route.v1", "provider": target["provider"], "model": target["model"],
               "base_url": target["base_url"], "api_mode": target["api_mode"], "session_id": "s1",
               "cache_scope": "scope-synthetic", "aux_task": task, "api_request_id": request_id,
               "retry_count": retry, "extra_headers": {"session_id": "s1"}, "input_count": len(body["input"])}
    for field in ("profile_key", "credential_fingerprint", "headers_fingerprint", "settings_fingerprint",
                  "input_fingerprint", "signature"):
        context[field] = "a" * 64
    return {"aux_task": task, "api_request_id": request_id, "retry_count": retry, "session_id": "s1",
            "turn_id": "t1", "task_id": "task-synthetic", "platform": "cli", "provider": target["provider"],
            "model": target["model"], "base_url": target["base_url"], "api_mode": target["api_mode"],
            "streaming": True, "native_request": body, "route_context": context,
            "extra_headers": {"session_id": "s1"}}


def post(pre, payload=None, **extra):
    return {**{key: value for key, value in pre.items() if key not in ("native_request", "extra_headers")},
            "native_response": terminal() if payload is None else payload, "status": "completed", "error_type": None,
            "usage": {"input_tokens": 500}, **extra}


class NativeStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = MoaStore([route(), route("reference")], include_references=True)
        self.rows = [user("Task")]
        self.reply = assistant("Accepted answer")

    def capture(self, pre=None, payload=None, coarse=False):
        from wc_fixtures import capture_for
        pre = event(route(), self.rows) if pre is None else pre
        self.store.pre_main(api_request_id="main-1", session_id="s1", turn_id="t1", provider="moa",
                            conversation_history=self.rows, model=VIRTUAL[0],
                                 base_url=VIRTUAL[1], api_mode=VIRTUAL[2])
        def run():
            if coarse:
                self.store.on_pre_auxiliary_call(**{key: value for key, value in pre.items()
                                                   if key not in ("native_request", "route_context", "extra_headers")})
            self.store.on_pre_auxiliary_native_request(**pre)
            self.store.on_post_auxiliary_native_request(**post(pre, payload))
        self.store.run_main("main-1", run)
        cap = capture_for(self.rows, self.reply, route=VIRTUAL)
        cap["moa"] = self.store.finish_main("main-1", cap)
        return cap

    def test_native_route_alias_is_normalized_without_key_access(self):
        normalized = validate_routes([{**route(), "model": "OpenAI/ACTING"}])[0]
        self.assertEqual(normalized["model"], "acting")
        self.assertNotIn("api_key_env", normalized)
        for change in ({"api_key_env": ""}, {"api_key_env": "SYNTHETIC"}, {"provider": "custom"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_routes([{**route(), **change}])

    def test_exact_native_capture_owns_the_attempt_after_coarse_preview(self):
        cap = self.capture(coarse=True)
        self.assertIsNone(cap["moa"]["code"])
        self.assertTrue(cap["moa"]["aggregator"]["body"]["stream"])

    def test_native_context_identity_and_header_changes_refuse(self):
        for field, value in (("signature", "invalid"), ("session_id", "other"), ("model", "other"),
                             ("input_count", 2), ("extra_headers", {}), ("schema", "other")):
            with self.subTest(field=field):
                self.setUp()
                pre = event(route(), self.rows)
                pre["route_context"][field] = value
                self.assertIsNotNone(self.capture(pre)["moa"]["code"])

    def test_partial_failed_missing_and_foreign_post_replies_refuse(self):
        for payload in ({"status": "incomplete", "output": []}, {"status": "failed", "output": []},
                        {"status": "completed", "output": []}, terminal("Different reply")):
            with self.subTest(payload=payload):
                self.setUp()
                self.assertIsNotNone(self.capture(payload=payload)["moa"]["code"])

    def test_native_aggregator_keeps_controls_prefix_and_native_reply_fields(self):
        cap = self.capture(payload=terminal(reasoning=True))
        original = copy.deepcopy(cap["moa"]["aggregator"]["body"])
        messages = [*self.rows, self.reply, user("New user request")]
        before = copy.deepcopy(messages)
        built, _ = build_aggregator(cap, messages, VIRTUAL, 200000, "Handoff")
        self.assertEqual(built["input"][:len(original["input"])], original["input"])
        self.assertEqual({key: value for key, value in built.items() if key != "input"},
                         {key: value for key, value in original.items() if key != "input"})
        self.assertEqual(built["input"][1]["encrypted_content"], "synthetic-sealed")
        self.assertEqual(built["input"][2]["phase"], "final_answer")
        self.assertIn("New user request", str(built["input"]))
        self.assertEqual(messages, before)

    def test_aggregator_advisor_block_stays_in_exact_native_prefix(self):
        pre = event(route(), [*self.rows, user(ADVISOR_PREFIX + "Advice")])
        cap = self.capture(pre)
        built, _ = build_aggregator(cap, [*self.rows, self.reply], VIRTUAL, 200000, "Handoff")
        self.assertEqual(built["input"][:2], pre["native_request"]["input"])

    def test_native_source_transform_refuses(self):
        cap = self.capture(event(route(), [user("Changed task")]))
        with self.assertRaises(WarmRefusal):
            build_aggregator(cap, [*self.rows, self.reply], VIRTUAL, 200000, "Handoff")

    def test_native_tool_reply_matches_ids_names_arguments_and_results(self):
        self.reply = assistant("Accepted answer", [("call_synthetic", "read", '{"path":"x"}')])
        calls = [{"type": "function_call", "call_id": "call_synthetic", "name": "read",
                  "arguments": '{"path":"x"}', "status": "completed"}]
        cap = self.capture(payload=terminal(calls=calls, reasoning=True))
        messages = [*self.rows, self.reply, tool("call_synthetic", "Synthetic result")]
        built, _ = build_aggregator(cap, messages, VIRTUAL, 200000, "Handoff")
        self.assertIn("function_call_output", str(built["input"]))
        changed = copy.deepcopy(messages)
        changed[1]["tool_calls"][0]["function"]["arguments"] = '{"path":"changed"}'
        with self.assertRaises(WarmRefusal) as caught:
            build_aggregator(cap, changed, VIRTUAL, 200000, "Handoff")
        self.assertEqual(caught.exception.code, "moa_reply_mismatch")

    def test_native_legacy_call_id_matches_normalized_history(self):
        self.reply = assistant("Accepted answer", [("call_synthetic", "read", '{"path":"x"}')])
        calls = [{"type": "function_call", "call_id": "fc_synthetic", "name": "read",
                  "arguments": '{"path":"x"}', "status": "completed"}]
        cap = self.capture(payload=terminal(calls=calls, reasoning=True))
        self.assertIsNone(cap["moa"]["code"])
        messages = [*self.rows, self.reply, tool("call_synthetic", "Synthetic result")]
        built, _ = build_aggregator(cap, messages, VIRTUAL, 200000, "Handoff")
        pairs = [row["call_id"] for row in built["input"]
                 if row.get("type") in ("function_call", "function_call_output")]
        self.assertEqual(pairs, ["call_synthetic", "call_synthetic"])

    def test_capacity_and_replay_sidecar_conflicts_refuse(self):
        cap = self.capture(payload=terminal(reasoning=True))
        for window in (0, 50):
            with self.subTest(window=window), self.assertRaises(WarmRefusal):
                build_aggregator(cap, [*self.rows, self.reply], VIRTUAL, window, "Handoff")
        changed = {**self.reply, "codex_message_items": [{"type": "message", "role": "assistant", "content": []}]}
        with self.assertRaises(WarmRefusal):
            build_aggregator(cap, [*self.rows, changed], VIRTUAL, 200000, "Handoff")

    def test_native_reference_has_only_its_own_view_and_reply(self):
        pre = event(route("reference"), [user("Reference view")], task="moa_reference", request_id="r1",
                    instructions="Synthetic reference system.")
        self.store.on_pre_auxiliary_native_request(**pre)
        self.store.on_post_auxiliary_native_request(**post(pre, terminal("Reference reply", reasoning=True)))
        cap = self.capture()
        reference = cap["moa"]["references"][0]
        built, _ = build_reference(reference, "Handoff")
        self.assertEqual(built["instructions"], "Synthetic reference system.")
        self.assertIn("Reference view", str(built["input"]))
        self.assertIn("Reference reply", str(built["input"]))
        self.assertNotIn("Accepted answer", str(built))
        self.assertNotIn("Synthetic acting system.", str(built))


class NativeEngineTest(unittest.TestCase):
    def setUp(self):
        self.helper = engine_fixtures.EngineTest()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.tracker = MoaStore([route(), route("reference")], include_references=True)
        self.helper.store = CaptureStore(moa=self.tracker)
        engine_fixtures.wc_hermes_stub.CAPTURE_CHAIN[:] = [self.helper.store.on_llm_execution]
        self.engine = self.helper.make(moa_routes=[route(), route("reference")], moa_references=True)
        self.engine.update_model(VIRTUAL[0], 200000, base_url=VIRTUAL[1], provider="moa", api_mode=VIRTUAL[2])
        self.rows = [*engine_fixtures.old_turns(), user("Current task")]
        self.reply = assistant("Accepted answer")
        self.messages = [*self.rows, self.reply]
        self.calls = []
        self.helper.llm.complete_native = self.complete_native

    def complete_native(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        target = kwargs["route_context"]
        return SimpleNamespace(native_response=terminal(HEADINGS_TEXT), usage=None,
                               audit={"provider": target["provider"], "model": target["model"],
                                      "task": "warm_compaction", "api_request_id": "aux-synthetic"})

    def seed(self, reference=False):
        if reference:
            pre = event(route("reference"), [user("Reference view")], task="moa_reference", request_id="r1",
                        instructions="Reference system.")
            self.tracker.on_pre_auxiliary_native_request(**pre)
            self.tracker.on_post_auxiliary_native_request(**post(pre, terminal("Reference reply")))
        store = self.helper.store
        store.on_pre_api_request(api_request_id="main-1", session_id="s1", turn_id="t1", provider="moa",
                                 conversation_history=self.rows, model=VIRTUAL[0],
                                 base_url=VIRTUAL[1], api_mode=VIRTUAL[2])
        pre = event(route(), self.rows)
        def request():
            self.tracker.on_pre_auxiliary_native_request(**pre)
            self.tracker.on_post_auxiliary_native_request(**post(pre))
        store.on_llm_execution(api_request_id="main-1", request={"model": VIRTUAL[0],
                               "messages": [SYSTEM, *wire(self.rows)]}, next_call=request)
        store.on_post_api_request(api_request_id="main-1", session_id="s1", finish_reason="stop",
                                  assistant_message=engine_fixtures.reply_object(self.reply))

    def test_native_api_writes_one_history_and_no_direct_http_request(self):
        self.seed()
        result = self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["path"], "warm")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["expected_session_id"], "s1")
        self.assertEqual(self.calls[0]["task"], "warm_compaction")
        self.assertIn(HEADINGS_TEXT, result[1]["content"])
        self.assertEqual(self.helper.post.calls, [])
        self.assertEqual(self.helper.llm.calls, [])

    def test_native_references_use_distinct_contexts_before_aggregator(self):
        self.seed(reference=True)
        self.engine.compress(self.messages)
        self.assertEqual([call["route_context"]["model"] for call in self.calls], ["reference", "acting"])
        reference, acting = [call["native_request"] for call in self.calls]
        self.assertNotIn("Current task", str(reference))
        self.assertIn("untrusted advisory data", str(acting))

    def test_missing_native_api_uses_explicit_fallback(self):
        self.seed()
        self.helper.llm.complete_native = None
        self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["reason"], "moa_native_api_unavailable")
        self.assertEqual(len(self.helper.llm.calls), 1)
        self.assertEqual(self.calls, [])

    def test_native_middleware_rewrite_block_and_retry_keep_public_guards(self):
        cases = ("rewrite", "block", "retry")
        for action in cases:
            with self.subTest(action=action):
                self.seed()
                self.calls.clear()
                def middleware(request, next_call, action=action, **context):
                    if action == "rewrite":
                        request["input"][0]["content"] = [{"type": "input_text", "text": "Changed task"}]
                        return next_call()
                    if action == "block":
                        return {"content": "mock"}
                    next_call()
                    return next_call()
                stub = engine_fixtures.wc_hermes_stub
                stub.EXECUTION_MIDDLEWARE[:] = [middleware]
                self.engine.compress(self.messages)
                self.assertEqual(len(self.calls), 1 if action == "retry" else 0)
                self.assertEqual(self.engine.warm_last["reason"],
                                 {"rewrite": "middleware_rewrite", "block": "middleware_changed_reply",
                                  "retry": "middleware_repeated"}[action])
                stub.EXECUTION_MIDDLEWARE.clear()

    def test_native_route_change_after_send_keeps_history(self):
        self.seed()
        complete = self.complete_native
        def change(**kwargs):
            result = complete(**kwargs)
            self.engine.update_model("other", 200000, base_url="moa://changed", provider="moa", api_mode=VIRTUAL[2])
            return result
        self.helper.llm.complete_native = change
        result = self.engine.compress(self.messages)
        self.assertIs(result, self.messages)
        self.assertEqual(self.helper.llm.calls, [])

    def test_native_history_policy_uses_only_the_current_acting_capture(self):
        self.seed(reference=True)
        self.assertEqual(self.engine._policy(self.messages).native_mode, "codex_responses")
        capture = self.helper.store._sessions["s1"]
        # A native reference does not make a Chat aggregator a native actor.
        capture["moa"]["aggregator"]["route_config"]["api_mode"] = "chat_completions"
        self.assertEqual(self.engine._policy(self.messages).native_mode, "")
        capture["moa"]["aggregator"]["route_config"]["api_mode"] = "codex_responses"
        capture["moa"]["code"] = "moa_reply_mismatch"
        self.assertEqual(self.engine._policy(self.messages).native_mode, "")

    def test_native_partial_reply_and_wrong_audit_use_fallback(self):
        for invalid in ("partial", "audit"):
            with self.subTest(invalid=invalid):
                self.seed()
                def complete(invalid=invalid, **kwargs):
                    result = self.complete_native(**kwargs)
                    if invalid == "partial":
                        result.native_response["status"] = "incomplete"
                    else:
                        result.audit["provider"] = "other"
                    return result
                self.helper.llm.complete_native = complete
                self.engine.compress(self.messages)
                self.assertEqual(self.engine.warm_last["path"], "fallback")


if __name__ == "__main__":
    unittest.main()
