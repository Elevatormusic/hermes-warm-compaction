"""Tests for the warm_compaction context engine with stand-in Hermes modules."""

import json
import unittest
from types import SimpleNamespace

import wc_hermes_stub
from warm_compaction.fallback import END_LINE
from wc_fixtures import HEADINGS_TEXT, ROUTE, SYSTEM, assistant, tool, user, wire


class FakeLlm:
    def __init__(self, text=HEADINGS_TEXT.replace("Finish the test task.", "Fallback summary.") + "\n" + END_LINE,
                 error=None):
        self.text, self.error, self.calls = text, error, []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text, usage=SimpleNamespace(input_tokens=50))


def fake_post(content=HEADINGS_TEXT, cached=900):
    def post(url, data, headers, timeout_s):
        post.calls.append({"url": url, "body": json.loads(data), "headers": headers})
        payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                             "prompt_tokens_details": {"cached_tokens": cached}}}
        return 200, json.dumps(payload).encode("utf-8")
    post.calls = []
    return post


def reply_object(row):
    calls = [SimpleNamespace(id=call["id"], function=SimpleNamespace(
        name=call["function"]["name"], arguments=call["function"]["arguments"])) for call in row.get("tool_calls", [])]
    return SimpleNamespace(content=row.get("content"), tool_calls=calls or None)


def old_turns(count=30):
    rows = []
    for index in range(count):
        rows += [user(f"ask {index} " + "x" * 2000), assistant(f"answer {index} " + "y" * 2000)]
    return rows


class EngineTest(unittest.TestCase):
    def setUp(self):
        wc_hermes_stub.install(self)
        from warm_compaction.capture import CaptureStore
        from warm_compaction.engine import WarmCompactionEngine
        self.engine_class = WarmCompactionEngine
        self.store = CaptureStore()
        wc_hermes_stub.CAPTURE_CHAIN.append(self.store.on_llm_execution)
        self.llm = FakeLlm()
        self.post = fake_post()
        self.engine = self.make()

    def make(self, **settings):
        engine = self.engine_class(store=self.store, llm=self.llm, post=self.post, settings=settings or None)
        engine.on_session_start("s1", platform="cli")
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        return engine

    def seed(self, rows, reply, session="s1"):
        """Run one main-model request through the capture store, in the Hermes order."""
        self.store.on_pre_api_request(api_request_id="r1", session_id=session, conversation_history=list(rows),
                                      model=ROUTE[0], base_url=ROUTE[1], api_mode=ROUTE[2])
        self.store.on_llm_execution(request={"model": ROUTE[0], "messages": [SYSTEM, *wire(rows)]},
                                    next_call=lambda: None, api_request_id="r1")
        self.store.on_post_api_request(api_request_id="r1", session_id=session,
                                       finish_reason="tool_calls" if reply.get("tool_calls") else "stop",
                                       assistant_message=reply_object(reply))

    def test_threshold_comes_from_the_setting(self):
        self.assertEqual(self.engine.threshold_tokens, 100_000)
        self.assertFalse(self.engine.should_compress(99_999))
        self.assertTrue(self.engine.should_compress(100_000))
        self.engine.update_from_response({"prompt_tokens": 120_000, "completion_tokens": 10})
        self.assertTrue(self.engine.should_compress())

    def test_model_threshold_override(self):
        engine = self.engine_class(store=self.store)
        engine.model_thresholds = {"fake": 0.25}
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_mode=ROUTE[2])
        self.assertEqual(engine.threshold_tokens, 50_000)

    def test_response_clears_the_wait_flag(self):
        self.engine.awaiting_real_usage_after_compression = True
        self.engine.update_from_response({"prompt_tokens": 7})
        self.assertFalse(self.engine.awaiting_real_usage_after_compression)
        self.assertEqual((self.engine.last_prompt_tokens, self.engine.last_real_prompt_tokens), (7, 7))
        self.engine.update_from_response({})
        self.assertEqual((self.engine.last_prompt_tokens, self.engine.last_real_prompt_tokens), (0, 7))

    def test_preflight_uses_the_estimate(self):
        self.assertFalse(self.engine.should_compress_preflight([user("x")]))
        self.assertTrue(self.engine.should_compress_preflight([user("x" * 400_004)]))

    def test_manual_warm_path(self):
        rows = old_turns()
        reply = assistant("answer final")
        self.seed(rows, reply)
        new = self.engine.compress([*rows, reply])
        status = self.engine.get_status()["warm_last"]
        self.assertEqual((status["path"], status["reason"], status["prompt_tokens"], status["cached_tokens"]),
                         ("warm", "accepted", 1000, 900))
        sent = self.post.calls[0]["body"]["messages"]
        self.assertEqual(sent[: len(rows) + 1], [SYSTEM, *wire(rows)])
        self.assertEqual(sent[len(rows) + 1], {"role": "assistant", "content": "answer final"})
        self.assertTrue(sent[-1]["content"].startswith("Stop the current task now."))
        self.assertEqual(self.post.calls[0]["headers"]["Authorization"], "Bearer k")
        self.assertEqual([row["role"] for row in new[:3]], ["user", "assistant", "user"])
        self.assertTrue(new[0]["content"].startswith(wc_hermes_stub.SUMMARY_PREFIX))
        self.assertIn(HEADINGS_TEXT, new[1]["content"])
        self.assertTrue(new[1]["content"].endswith(wc_hermes_stub.END_MARKER))
        self.assertEqual(new[-1], reply)
        self.assertEqual((self.llm.calls, self.engine.compression_count), ([], 1))

    def test_mid_task_warm_path_sends_the_tool_rows(self):
        rows = [*old_turns(), user("run the tool")]
        reply = assistant("", [("c1", "read", "{\"path\":\"a\"}")])
        self.seed(rows, reply)
        engine = self.make(tail_tokens=10)
        new = engine.compress([*rows, reply, tool("c1", "file text")])
        sent = self.post.calls[0]["body"]["messages"]
        self.assertEqual([row["role"] for row in sent[-3:]], ["assistant", "tool", "user"])
        self.assertEqual(engine.warm_last["path"], "warm")
        self.assertEqual([row["role"] for row in new], ["user", "assistant", "user", "assistant", "tool"])
        self.assertEqual(new[2], {"role": "user", "content": "run the tool"})

    def test_fallback_without_a_capture(self):
        new = self.engine.compress([*old_turns(), assistant("done")])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]), ("fallback", "no_capture"))
        self.assertEqual(self.llm.calls[0][1]["task"], "warm_compaction")
        self.assertIn("Fallback summary.", new[1]["content"])
        self.assertEqual(self.post.calls, [])

    def test_the_fallback_summarizes_only_the_removed_rows(self):
        from warm_compaction import layout
        rows = [*old_turns(), assistant("done")]
        start, _prepend = layout.tail_start(rows, self.engine._tail_tokens(), self.engine._prefixes())
        self.engine.compress(rows)
        sent = self.llm.calls[0][0][1]["content"]
        self.assertIn(rows[start - 1]["content"][:20], sent)
        self.assertNotIn("done", sent.split("\n"))
        self.assertNotIn(rows[-2]["content"][:20], sent)

    def test_gate_refusal_keeps_the_tokens_and_falls_back(self):
        self.post = fake_post(content="no headings")
        engine = self.make()
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        engine.compress([*rows, reply])
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"], engine.warm_last["prompt_tokens"]),
                         ("fallback", "gate:heading_missing", 1000))

    def test_warm_request_runs_through_the_execution_middleware(self):
        seen = []

        def audit(request=None, next_call=None, **context):
            seen.append((len(request["messages"]), context.get("purpose"), context.get("api_request_id")))
            return next_call()
        wc_hermes_stub.EXECUTION_MIDDLEWARE.append(audit)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual(self.engine.warm_last["path"], "warm")
        self.assertEqual(seen, [(len(rows) + 3, "warm_compaction", None)])

    def test_a_blocking_or_mocking_middleware_stops_the_warm_request(self):
        def block(request=None, next_call=None, **context):
            return None  # A policy middleware blocks by not calling next_call.

        def mock(request=None, next_call=None, **context):
            return SimpleNamespace(choices=[])
        for middleware, reason in ((block, "middleware_changed_reply"), (mock, "middleware_changed_reply")):
            with self.subTest(reason):
                wc_hermes_stub.EXECUTION_MIDDLEWARE[:] = [middleware]
                self.post.calls.clear()
                rows = old_turns()
                reply = assistant("final")
                self.seed(rows, reply)
                self.engine.compress([*rows, reply])
                self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                                 ("fallback", reason))
                self.assertEqual(self.post.calls, [])

    def test_a_replacement_reply_of_the_right_shape_stops_the_warm_request(self):
        def replace(request=None, next_call=None, **context):
            return {"content": HEADINGS_TEXT, "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 5,
                    "cached_tokens": 0, "tool_calls": False, "refusal": False, "elapsed_s": 0.0}

        def change(request=None, next_call=None, **context):
            reply = next_call()
            reply["content"] = HEADINGS_TEXT.replace("report.txt", "other.txt")
            return reply
        for middleware in (replace, change):
            with self.subTest(middleware.__name__):
                wc_hermes_stub.EXECUTION_MIDDLEWARE[:] = [middleware]
                rows = old_turns()
                reply = assistant("final")
                self.seed(rows, reply)
                self.engine.compress([*rows, reply])
                self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                                 ("fallback", "middleware_changed_reply"))

    def test_a_second_send_is_refused_before_it_sends(self):
        def twice(request=None, next_call=None, **context):
            first = next_call()
            try:
                next_call()
            except Exception:
                pass
            return first
        wc_hermes_stub.EXECUTION_MIDDLEWARE.append(twice)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual(len(self.post.calls), 1)
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("fallback", "middleware_repeated"))

    def test_capacity_is_checked_again_after_the_request_middleware(self):
        def expand(request=None, **context):
            request["messages"][-2]["content"] += " " + "context " * 200_000
            return {"request": request}
        wc_hermes_stub.REQUEST_MIDDLEWARE.append(expand)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]), ("fallback", "capacity"))
        self.assertEqual(self.post.calls, [])

    def test_the_warm_request_has_the_default_headers_of_the_hermes_client(self):
        # Provider headers (attribution, a WAF User-Agent, credentials) go with every request of the route.
        wc_hermes_stub.HOST_HEADERS.update({"X-Title": "Hermes Agent"})
        wc_hermes_stub.USER_HEADERS.update({"User-Agent": "allowed-agent"})
        wc_hermes_stub.CUSTOM_HEADERS.update({"X-Gateway-Key": "synthetic-header-secret"})
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual(self.engine.warm_last["path"], "warm")
        headers = self.post.calls[0]["headers"]
        self.assertEqual((headers["X-Title"], headers["User-Agent"], headers["X-Gateway-Key"], headers["Authorization"]),
                         ("Hermes Agent", "allowed-agent", "synthetic-header-secret", "Bearer k"))
        self.assertNotIn("synthetic-header-secret", json.dumps(self.engine.warm_last))
        # Without a host factory, the provider profile gives the headers.
        wc_hermes_stub.HOST_HEADERS.clear()
        wc_hermes_stub.PROFILE_HEADERS.update({"User-Agent": "profile-agent"})
        wc_hermes_stub.USER_HEADERS.clear()
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual(self.post.calls[1]["headers"]["User-Agent"], "profile-agent")

    def test_an_unreadable_header_source_stops_the_warm_request(self):
        def broken(base_url):
            raise AttributeError("changed")
        self.addCleanup(setattr, wc_hermes_stub.AGENT_INIT, "_host_default_headers_factory",
                        wc_hermes_stub.AGENT_INIT._host_default_headers_factory)
        wc_hermes_stub.AGENT_INIT._host_default_headers_factory = broken
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("fallback", "headers_unknown"))
        self.assertEqual(self.post.calls, [])

    def test_a_request_middleware_that_removes_the_instruction_stops_the_warm_request(self):
        def drop(request=None, **context):
            return {"request": {**request, "messages": request["messages"][:-1]}}
        wc_hermes_stub.REQUEST_MIDDLEWARE.append(drop)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("fallback", "middleware_rewrite"))
        self.assertEqual(self.post.calls, [])

    def test_a_request_middleware_that_changes_the_captured_part_stops_the_warm_request(self):
        # The captured body already went through the request middleware. A second pass would apply it twice.
        def prepend(request=None, **context):
            return {"request": {**request, "messages": [{"role": "system", "content": "policy"},
                                                        *request["messages"]]}}
        wc_hermes_stub.REQUEST_MIDDLEWARE.append(prepend)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("fallback", "middleware_rewrite"))
        self.assertEqual(self.post.calls, [])

    def test_a_rewriting_middleware_stops_the_warm_request(self):
        # The capture keeps the body before later middleware. A middleware that rewrites requests can have
        # rewritten the captured request too, so the warm request is not sent.
        def rewrite(request=None, next_call=None, **context):
            return next_call({**request, "messages": request["messages"][1:]})
        wc_hermes_stub.EXECUTION_MIDDLEWARE.append(rewrite)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("fallback", "middleware_rewrite"))
        self.assertEqual(self.post.calls, [])

    def test_an_in_place_rewrite_stops_the_warm_request(self):
        # The request and the comparison base must not be the same object.
        def rewrite(request=None, next_call=None, **context):
            request["messages"][-1]["content"] = "Continue the task."
            return next_call()
        wc_hermes_stub.EXECUTION_MIDDLEWARE.append(rewrite)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("fallback", "middleware_rewrite"))
        self.assertEqual(self.post.calls, [])

    def test_the_warm_request_goes_through_the_request_middleware(self):
        def redact(request=None, **context):
            seen.append(context.get("purpose"))
            text = json.dumps(request).replace("SECRET", "[redacted]")
            return {"request": json.loads(text)}
        seen = []
        wc_hermes_stub.REQUEST_MIDDLEWARE.append(redact)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply, user("the key is SECRET")])
        self.assertEqual(self.engine.warm_last["path"], "warm")
        self.assertEqual(seen, ["warm_compaction"])
        sent = json.dumps(self.post.calls[0]["body"])
        self.assertNotIn("SECRET", sent)
        self.assertIn("the key is [redacted]", sent)

    def test_copy_allowance_leaves_room_for_the_actual_summary(self):
        from warm_compaction.engine import CARRIER_TOKENS
        from warm_compaction.rows import estimate_tokens
        self.engine.update_model(model=ROUTE[0], context_length=20_000, base_url=ROUTE[1], api_key="k",
                                 provider="custom", api_mode=ROUTE[2])
        self.assertEqual((self.engine.threshold_tokens, self.engine._tail_tokens()), (10_000, 5_000))
        tail = [user("t " + "x" * 16_000)]
        for summary in ("short", "## Key facts\n" + "- /very/long/path/name_" * 600):
            with self.subTest(summary=len(summary)):
                allowance = self.engine._copy_tokens(tail, summary)
                self.assertLessEqual(estimate_tokens(tail) + estimate_tokens(summary) + CARRIER_TOKENS + allowance,
                                     self.engine.threshold_tokens)
        self.assertEqual(self.engine._copy_tokens(tail, "short"), 5_000)
        self.assertLess(self.engine._copy_tokens(tail, summary), 5_000)
        engine = self.make()  # A large window keeps the full allowance.
        self.assertEqual(engine._copy_tokens(tail, summary), engine._tail_tokens())
        # The system prompt and the tool schemas count too.
        self.assertEqual(self.engine._copy_tokens(tail, "short", 4_000), 10_000 - 4_000 - CARRIER_TOKENS
                         - estimate_tokens(tail) - estimate_tokens("short"))

    def test_copy_allowance_leaves_room_for_the_reply(self):
        # threshold 0.95 of a 20K window: the prompt after compaction must leave the reply reserve free.
        engine = self.make(threshold=0.95)
        engine.update_model(model=ROUTE[0], context_length=20_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        from warm_compaction.engine import CARRIER_TOKENS
        from warm_compaction.rows import estimate_tokens
        tail = [user("t " + "x" * 8_000)]
        used = estimate_tokens(tail) + estimate_tokens("short") + CARRIER_TOKENS
        self.assertEqual(engine._copy_tokens(tail, "short", 0, 4_096),
                         min(engine._tail_tokens(), 20_000 - 4_096 - used))

    def test_request_reserve_comes_from_the_capture(self):
        from warm_compaction.engine import request_reserve
        from warm_compaction.warm import DEFAULT_RESERVE
        self.assertEqual(request_reserve(None), DEFAULT_RESERVE)
        # Without a usable capture the reply limit of the next request is unknown: keep a quarter of the window,
        # at most 65,536 tokens (the Hermes default of a native Gemini route), at least the default.
        self.assertEqual(request_reserve(None, 200_000), 50_000)
        self.assertEqual(request_reserve(None, 1_000_000), 65_536)
        self.assertEqual(request_reserve(None, 8_000), DEFAULT_RESERVE)
        self.assertEqual(request_reserve({"body": {"messages": [], "max_tokens": 1_000}}, 200_000), 1_000)
        self.assertEqual(request_reserve({"body": {"messages": [], "max_tokens": 1_000}}), 1_000)
        self.assertEqual(request_reserve({"body": {"messages": [], "max_completion_tokens": 2_000}}), 2_000)

    def test_request_overhead_comes_from_the_capture(self):
        from warm_compaction.engine import request_overhead
        from warm_compaction.rows import estimate_tokens
        rows = old_turns(2)
        self.seed(rows, assistant("final"))
        capture = self.store.latest("s1")
        self.assertEqual(request_overhead(capture), estimate_tokens({"messages": [SYSTEM]}))
        # Without a captured body the overhead is unknown: the host's token count and the plugin estimate do not
        # use the same tokenizer, so their difference is not a measurement.
        self.assertIsNone(request_overhead(None))
        self.assertIsNone(request_overhead({"body": None}, rows))

    def test_request_overhead_counts_text_that_a_middleware_added_to_a_captured_row(self):
        from warm_compaction.engine import request_overhead
        from warm_compaction.rows import estimate_tokens
        rows = [user("hi"), assistant("ok")]
        added = "hi\n\n" + "[recalled] " * 400
        capture = {"digests": [None, None], "body": {"messages": [SYSTEM, {"role": "user", "content": added},
                                                                   {"role": "assistant", "content": "ok"}]}}
        base = estimate_tokens({"messages": [SYSTEM]})
        self.assertEqual(request_overhead(capture, rows), base + estimate_tokens(added) - estimate_tokens("hi"))
        self.assertEqual(request_overhead(capture), base)

    def test_request_overhead_counts_every_structured_request_field(self):
        # The legacy functions field (and tools, response schemas) comes again with the next ordinary request.
        from warm_compaction.engine import request_overhead
        from warm_compaction.rows import estimate_tokens
        functions = [{"name": "f", "description": "d " * 4_000, "parameters": {"type": "object"}}]
        capture = {"digests": [], "body": {"model": "m", "temperature": 0.2, "messages": [SYSTEM],
                                            "functions": functions}}
        self.assertEqual(request_overhead(capture), estimate_tokens({"messages": [SYSTEM], "functions": functions}))

    def test_a_request_middleware_that_changes_the_request_in_place_is_refused(self):
        # A host whose request chain passes the request itself: the plugin compares with its own snapshot.
        import sys
        from types import SimpleNamespace

        def in_place(request, **context):
            request["messages"].insert(0, {"role": "system", "content": "policy"})
            return SimpleNamespace(payload=request)
        module = sys.modules["hermes_cli.middleware"]
        original = module.apply_llm_request_middleware
        module.apply_llm_request_middleware = in_place
        self.addCleanup(setattr, module, "apply_llm_request_middleware", original)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        stored = json.dumps(self.store.latest("s1")["body"], sort_keys=True)
        self.engine.compress([*rows, reply])
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("fallback", "middleware_rewrite"))
        self.assertEqual(self.post.calls, [])
        self.assertEqual(json.dumps(self.store.latest("s1")["body"], sort_keys=True), stored)

    def test_request_reserve_is_the_larger_limit(self):
        from warm_compaction.engine import request_reserve
        self.assertEqual(request_reserve({"body": {"messages": [], "max_tokens": 100,
                                                   "max_completion_tokens": 5_000}}), 5_000)

    def test_a_capture_of_another_route_is_not_used_for_the_copy_budget(self):
        # After a model change, the old capture says nothing about the system prompt and tools of the new route.
        from warm_compaction.rows import estimate_tokens
        rows = old_turns(20)
        engine = self.make(threshold=0.95, tail_tokens=2_000)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        large = {"role": "system", "content": "s " * 120_000}
        self.store.on_pre_api_request(api_request_id="r1", session_id="s1", conversation_history=list(rows[:-1]),
                                      model=ROUTE[0], base_url=ROUTE[1], api_mode=ROUTE[2])
        self.store.on_llm_execution(request={"model": ROUTE[0], "messages": [large, *wire(rows[:-1])]},
                                    next_call=lambda: None, api_request_id="r1")
        self.store.on_post_api_request(api_request_id="r1", session_id="s1", finish_reason="stop",
                                       assistant_message=reply_object(rows[-1]))
        engine.update_model(model="another-model", context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows, current_tokens=estimate_tokens(rows) + 1_000)
        self.assertEqual(engine.warm_last["reason"], "route_changed")
        self.assertIsNone(engine._budget_capture(self.store.latest("s1"), rows))

    def test_a_large_prepended_user_row_is_cut_to_fit(self):
        # The newest user message is before the tail (it is larger than the tail) and goes in front of it. It
        # must fit below the threshold and the window less the reply reserve.
        from warm_compaction.rows import MIDDLE_MARK, estimate_tokens
        from warm_compaction.warm import DEFAULT_RESERVE
        rows = [*old_turns(4), user("BIG start " + "q" * 240_000 + " big end"),
                assistant("", [("c1", "read", "{}")]), tool("c1", "r1"), assistant("done")]
        engine = self.make(threshold=0.95, tail_tokens=2_000)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows, current_tokens=estimate_tokens(rows) + 1_000)
        prepended = next(row for row in new if str(row.get("content")).startswith("BIG start"))
        self.assertIn(MIDDLE_MARK, prepended["content"])
        self.assertTrue(prepended["content"].endswith(" big end"))
        self.assertLessEqual(estimate_tokens(new) + 1_000, min(engine.threshold_tokens, 64_000 - DEFAULT_RESERVE))

    def test_an_unknown_overhead_cuts_the_prepended_row_to_its_minimum(self):
        from warm_compaction.layout import MIN_COPY_CHARS
        rows = [*old_turns(4), user("BIG start " + "q" * 40_000 + " big end"),
                assistant("", [("c1", "read", "{}")]), tool("c1", "r1"), assistant("done")]
        engine = self.make(threshold=0.95, tail_tokens=2_000)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows, current_tokens=None)
        prepended = next(row for row in new if str(row.get("content")).startswith("BIG start"))
        self.assertLessEqual(len(prepended["content"]), MIN_COPY_CHARS)
        self.assertTrue(prepended["content"].endswith(" big end"))

    def test_without_a_capture_the_copies_leave_room_for_the_system_prompt_and_tools(self):
        # After a restart there is no capture. The system prompt and the tool schemas still take their space.
        from warm_compaction.rows import estimate_tokens
        rows = old_turns(20)
        engine = self.make(threshold=0.95, tail_tokens=2_000)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        # The host count is not used: a compressible history can have an estimate above it.
        for current, copied in ((estimate_tokens(rows) + 1_000, False), (estimate_tokens(rows) + 60_000, False),
                                (estimate_tokens(rows) // 2, False), (None, False)):
            with self.subTest(current=current):
                new = engine.compress(rows, current_tokens=current)
                summary = next(row["content"] for row in new if "## Copied user messages" in str(row["content"]))
                section = summary.split("## Copied user messages", 1)[1].strip()
                self.assertEqual(not section.startswith("(none)"), copied)

    def test_fixed_summary_when_the_fallback_fails(self):
        self.llm = FakeLlm(error=RuntimeError("down"))
        engine = self.make()
        with self.assertLogs("warm_compaction.fallback", level="WARNING"):
            new = engine.compress([*old_turns(), assistant("done")])
        self.assertEqual(engine.warm_last["path"], "fixed")
        self.assertIn("Summary unavailable.", new[1]["content"])

    def test_disabled_setting(self):
        engine = self.make(warm=False)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        engine.compress([*rows, reply])
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("fallback", "disabled"))
        self.assertEqual(self.post.calls, [])

    def test_cancelled_attempt_keeps_the_history(self):
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine._compression_cancelled_check = lambda: True
        messages = [*rows, reply]
        self.assertIs(self.engine.compress(messages), messages)
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]), ("cancelled", "cancelled"))
        self.assertEqual((self.post.calls, self.llm.calls, self.engine.compression_count), ([], [], 0))

    def test_nothing_before_the_tail(self):
        messages = [user("hello"), assistant("hi")]
        self.assertFalse(self.engine.has_content_to_compress(messages))
        self.assertIs(self.engine.compress(messages), messages)
        self.assertIsNone(self.engine.warm_last)

    def test_clone_shares_the_store_and_not_the_session(self):
        clone = self.engine.clone_for_agent()
        self.assertIsNot(clone, self.engine)
        self.assertIs(clone._store, self.store)
        self.assertEqual((clone.name, clone._wc_session_id), ("warm_compaction", ""))

    def test_rotation_forgets_the_old_capture(self):
        self.seed(old_turns(1), assistant("x"))
        self.engine.on_session_start("s2", boundary_reason="compression", old_session_id="s1")
        self.assertIsNone(self.store.latest("s1"))
        self.assertEqual(self.engine._wc_session_id, "s2")

    def test_focus_and_memory_reach_the_instruction(self):
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply], focus_topic="db", memory_context="  keep X  ")
        last = self.post.calls[0]["body"]["messages"][-1]["content"]
        self.assertIn("Give more detail to this topic: db", last)
        self.assertIn("Also keep this context from the memory provider:\nkeep X", last)

    def test_reset_clears_the_status(self):
        self.engine.compress([*old_turns(), assistant("done")])
        self.engine.on_session_reset()
        self.assertEqual((self.engine.warm_last, self.engine.compression_count), (None, 0))
        self.assertIn("warm_last", self.engine.get_status())

    def test_an_unknown_overhead_caps_the_tail(self):
        # Without a known overhead (no capture and no host count), the tail takes at most half of the free room:
        # the other half is for the system rows and the tool schemas.
        from warm_compaction.engine import CARRIER_TOKENS, request_reserve
        from warm_compaction.rows import estimate_tokens
        rows = [*old_turns(4), user("go"), assistant("", [("c1", "read", "{}")]),
                tool("c1", "head " + "r" * 200_000 + " tail")]
        engine = self.make(threshold=0.95, tail_tokens=9_500, warm=False)
        engine.update_model(model=ROUTE[0], context_length=20_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        limit = min(engine.threshold_tokens, 20_000 - request_reserve(None, 20_000))
        for captured, capped in ((False, True), (True, False)):
            with self.subTest(captured=captured):
                if captured:
                    self.seed(rows[:-2], rows[-2])
                new = engine.compress(list(rows), current_tokens=estimate_tokens(rows) + 1_000)
                tail = [row for row in new if not row.get("_compressed_summary")]
                self.assertEqual(estimate_tokens(tail) <= (limit - CARRIER_TOKENS) // 2, capped)

    def test_the_fallback_summary_gets_the_middles_that_the_tail_cuts(self):
        from warm_compaction.layout import CUT_NOTE
        rows = [*old_turns(4), user("go"), assistant("", [("c1", "read", "{}")]),
                tool("c1", "head " + "u" * 40_000 + " end")]
        engine = self.make(tail_tokens=2_000, warm=False)
        new = engine.compress(rows, current_tokens=None)
        self.assertEqual(engine.warm_last["path"], "fallback")
        transcript = self.llm.calls[-1][0][1]["content"]
        self.assertIn(CUT_NOTE, transcript)
        self.assertIn("u" * 100, transcript)
        self.assertTrue(any(str(row.get("content")).endswith(" end") for row in new))

    def test_a_capture_with_a_changed_source_is_not_used_for_the_budget(self):
        # A middleware removed a stored row: the captured body no longer shows which rows are the system rows.
        from warm_compaction.rows import estimate_tokens
        large = "s " * 120_000
        rows = [user("first " + "f" * 239_990), assistant("a"), *old_turns(6), assistant("final")]
        engine = self.make(threshold=0.95, tail_tokens=2_000)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        self.store.on_pre_api_request(api_request_id="r1", session_id="s1", conversation_history=list(rows[:-1]),
                                      model=ROUTE[0], base_url=ROUTE[1], api_mode=ROUTE[2])
        self.store.on_llm_execution(request={"model": ROUTE[0], "messages": [{"role": "system", "content": large},
                                                                             *wire(rows[1:-1])]},
                                    next_call=lambda: None, api_request_id="r1")
        self.store.on_post_api_request(api_request_id="r1", session_id="s1", finish_reason="stop",
                                       assistant_message=reply_object(rows[-1]))
        new = engine.compress(rows, current_tokens=estimate_tokens(rows) + estimate_tokens(large))
        self.assertEqual(engine.warm_last["reason"], "source_transform_unsupported")
        summary = next(row["content"] for row in new if "## Copied user messages" in str(row["content"]))
        self.assertTrue(summary.split("## Copied user messages", 1)[1].strip().startswith("(none)"))


class SettingsTest(unittest.TestCase):
    def setUp(self):
        wc_hermes_stub.install(self)
        from warm_compaction import engine
        self.module = engine

    def test_valid_values(self):
        values = {"threshold": 0.6, "tail_tokens": 12_000, "user_copy_chars": 0, "warm": False}
        self.assertEqual(self.module.read_settings(lambda key, default: values.get(key, default)), values)

    def test_invalid_values_use_the_defaults_with_a_warning(self):
        values = {"threshold": 2.0, "tail_tokens": -1, "user_copy_chars": "big", "warm": "no"}
        with self.assertLogs("warm_compaction.engine", level="WARNING") as logs:
            settings = self.module.read_settings(lambda key, default: values.get(key, default))
        self.assertEqual(settings, self.module.DEFAULTS)
        self.assertEqual(len(logs.output), 4)

    def test_tail_budget(self):
        self.assertEqual(self.module.tail_budget(0, 200_000), 10_000)
        self.assertEqual(self.module.tail_budget(0, 600_000), 15_000)
        self.assertEqual(self.module.tail_budget(0, 2_000_000), 25_000)
        self.assertEqual(self.module.tail_budget(7, 200_000), 7)
        # The automatic tail stays at or below half of the compaction threshold, so a small window can compact.
        self.assertEqual(self.module.tail_budget(0, 20_000, 10_000), 5_000)
        self.assertEqual(self.module.tail_budget(0, 200_000, 100_000), 10_000)
        self.assertEqual(self.module.tail_budget(7, 20_000, 10), 5)
        self.assertEqual(self.module.tail_budget(100_000, 128_000, 64_000), 32_000)
        self.assertEqual(self.module.tail_budget(7, 20_000, 10_000), 7)

    def test_hermes_value_uses_the_replacement(self):
        read = self.module.hermes_value
        self.assertEqual(read("agent.context_compressor", "SUMMARY_PREFIX", "d"), wc_hermes_stub.SUMMARY_PREFIX)
        self.assertEqual(read("agent.context_compressor", "MISSING", "d"), "d")
        self.assertEqual(read("no.such.module", "X", 1), 1)
        self.assertEqual(read("agent.context_compressor", "SUMMARY_PREFIX", 0), 0)


if __name__ == "__main__":
    unittest.main()
