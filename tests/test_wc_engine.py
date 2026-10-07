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

    def seed(self, rows, reply, session="s1", system=SYSTEM, extra=None, during=None):
        """Run one main-model request through the capture store, in the Hermes order. during runs while the
        request is open."""
        self.store.on_pre_api_request(api_request_id="r1", session_id=session, conversation_history=list(rows),
                                      model=ROUTE[0], base_url=ROUTE[1], api_mode=ROUTE[2])
        self.store.on_llm_execution(request={"model": ROUTE[0], "messages": [system, *wire(rows)], **(extra or {})},
                                    next_call=lambda: None, api_request_id="r1")
        if during is not None:
            during()
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

    def test_a_timed_out_attempt_does_not_send(self):
        # The host can give up on the attempt while an execution middleware still runs.
        def slow(request=None, next_call=None, **context):
            self.engine._compression_cancelled_check = lambda: True
            return next_call()
        wc_hermes_stub.EXECUTION_MIDDLEWARE.append(slow)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        self.engine.compress([*rows, reply])
        self.assertEqual(self.post.calls, [])
        self.assertEqual(self.engine.warm_last["path"], "cancelled")

    def test_a_route_change_during_the_attempt_does_not_send_to_the_new_route(self):
        def switch(request=None, next_call=None, **context):
            self.engine.update_model(model="other-model", context_length=200_000, base_url="https://other/v1",
                                     api_key="other-key", provider="custom", api_mode=ROUTE[2])
            return next_call()
        wc_hermes_stub.EXECUTION_MIDDLEWARE.append(switch)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        history = [*rows, reply]
        self.assertIs(self.engine.compress(history), history)
        self.assertEqual(self.post.calls, [])
        # The summary would be of the old route: the attempt is discarded, and no fallback runs.
        self.assertEqual((self.engine.warm_last["path"], self.engine.warm_last["reason"]),
                         ("cancelled", "route_changed"))
        self.assertEqual(self.llm.calls, [])

    def test_preflight_counts_the_captured_overhead_and_reply_reserve(self):
        # A large system prompt and tool schemas take space that the history estimate does not show.
        from warm_compaction.rows import estimate_tokens
        rows = old_turns(4)
        engine = self.make(threshold=0.95)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        large = {"role": "system", "content": "s " * 120_000}
        self.store.on_pre_api_request(api_request_id="r1", session_id="s1", conversation_history=list(rows[:-1]),
                                      model=ROUTE[0], base_url=ROUTE[1], api_mode=ROUTE[2])
        self.store.on_llm_execution(request={"model": ROUTE[0], "messages": [large, *wire(rows[:-1])]},
                                    next_call=lambda: None, api_request_id="r1")
        self.store.on_post_api_request(api_request_id="r1", session_id="s1", finish_reason="stop",
                                       assistant_message=reply_object(rows[-1]))
        messages = [*rows, user("next")]
        self.assertLess(estimate_tokens(messages), engine.threshold_tokens)
        self.assertTrue(engine.should_compress_preflight(messages))
        self.assertFalse(self.make(threshold=0.95).should_compress_preflight([user("hi")]))

    def test_preflight_without_a_usable_capture_keeps_half_of_the_room_for_the_overhead(self):
        # After a restart the system prompt and tool schemas are unknown: as in compress, half of the room
        # (the window less the unknown reply reserve) is for them.
        from warm_compaction.engine import request_reserve
        from warm_compaction.rows import estimate_tokens
        engine = self.make(threshold=0.95)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        half = (64_000 - request_reserve(None, 64_000)) // 2
        rows = old_turns(26)
        self.assertGreaterEqual(estimate_tokens(rows), half)
        self.assertLess(estimate_tokens(rows), engine.threshold_tokens)
        self.assertTrue(engine.should_compress_preflight(rows))
        self.assertFalse(engine.should_compress_preflight(old_turns(4)))
        # A usable capture with a small overhead: the threshold applies.
        self.seed(rows[:-1], rows[-1])
        self.assertFalse(engine.should_compress_preflight([*rows, user("next")]))

    def test_a_switch_during_the_send_discards_the_result(self):
        # The send blocks on the network; a model or session switch can occur before it returns.
        sent = fake_post()

        def post(url, data, headers, timeout_s):
            engine.update_model(model="other-model", context_length=200_000, base_url="https://other/v1",
                                api_key="other-key", provider="custom", api_mode=ROUTE[2])
            return sent(url, data, headers, timeout_s)
        self.post = post
        engine = self.make()
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        history = [*rows, reply]
        self.assertIs(engine.compress(history), history)
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("cancelled", "route_changed"))
        self.assertEqual(self.llm.calls, [])

    def test_preflight_estimates_only_the_sent_content(self):
        # Hermes sends api_content in place of content: the stored display text is not in the request.
        rows = []
        for index in range(30):
            text = f"ask {index} " + "x" * 8_000
            rows += [user(text, api_content=text + "\n\n[ctx]"), assistant(f"answer {index}")]
        self.assertFalse(self.engine.should_compress_preflight(rows))

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

    def test_a_retry_after_a_failed_send_does_not_send_again(self):
        # A timeout can come after the provider got the request: a retry middleware must not send it again.
        attempts = []

        def post(url, data, headers, timeout_s):
            attempts.append(url)
            raise TimeoutError("read timed out")

        def retry(request=None, next_call=None, **context):
            try:
                return next_call()
            except Exception:
                return next_call()
        self.post = post
        engine = self.make()
        wc_hermes_stub.EXECUTION_MIDDLEWARE.append(retry)
        rows = old_turns()
        reply = assistant("final")
        self.seed(rows, reply)
        engine.compress([*rows, reply])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(engine.warm_last["path"], "fallback")

    def test_a_warm_summary_above_the_room_goes_to_the_fallback(self):
        # A dense handoff can pass the byte gate and still not fit below a low threshold: the next request would
        # compact again at once.
        dense = HEADINGS_TEXT.replace("Finish the test task.", "\u76ee\u6807" * 3_000)
        self.post = fake_post(content=dense)
        engine = self.make(threshold=0.10)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        rows = old_turns(8)
        reply = assistant("final")
        self.seed(rows, reply)
        engine.compress([*rows, reply])
        self.assertEqual(len(self.post.calls), 1)
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("fallback", "summary_too_large"))

    def test_the_reply_of_a_first_request_sets_the_reasoning_rule(self):
        # The first request of a session has no assistant row; its reply (stored after it) is of this route.
        rows = [user("u1")]
        self.seed(rows, assistant("final"))
        policy = self.engine._policy([*rows, assistant("final", reasoning_content="r" * 4_000)])
        self.assertTrue(policy.echo)
        self.assertTrue(policy.cut_reasoning)

    def test_a_switch_from_an_empty_identity_forgets_the_capture(self):
        # A local route can start with no provider name and no key: a later key is a switch.
        engine = self.engine_class(store=self.store, llm=self.llm, post=self.post)
        engine.on_session_start("s1", platform="cli")
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="", provider="",
                            api_mode=ROUTE[2])
        rows = old_turns(1)
        self.seed(rows, assistant("x"))
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="", provider="",
                            api_mode=ROUTE[2])
        self.assertIsNotNone(self.store.latest("s1"))
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k2", provider="",
                            api_mode=ROUTE[2])
        self.assertIsNone(self.store.latest("s1"))

    def test_the_fixed_summary_keeps_the_focus_and_the_memory_context(self):
        # The last path has no model: the inputs given for this compaction must stay.
        self.llm.error = RuntimeError("down")
        engine = self.make(warm=False)
        new = engine.compress([*old_turns(), assistant("done")], focus_topic="FOCUS-ON-PARSER",
                              memory_context="MEMORY-FACT-42")
        self.assertEqual(engine.warm_last["path"], "fixed")
        summary = "\n".join(str(row["content"]) for row in new if row.get("_compressed_summary"))
        self.assertIn("FOCUS-ON-PARSER", summary)
        self.assertIn("MEMORY-FACT-42", summary)

    def test_a_fallback_summary_above_the_room_goes_to_the_fixed_summary(self):
        # Under the reserve, but a large system prompt and a low threshold leave less room than the summary needs.
        self.llm = FakeLlm(text=HEADINGS_TEXT.replace("Finish the test task.", "word " * 1_200) + "\n" + END_LINE)
        engine = self.make(threshold=0.10, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        rows = old_turns(8)
        reply = assistant("final")
        self.seed(rows, reply, system={"role": "system", "content": "s " * 10_000}, extra={"max_tokens": 4_096})
        engine.compress([*rows, reply])
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(engine.warm_last["path"], "fixed")

    def test_a_switch_before_the_result_is_used_discards_it(self):
        # The check covers the whole compaction: a switch while the new history is built also stops it.
        from warm_compaction import layout
        engine = self.make(warm=False)
        real = layout.build

        def build(*args, **kwargs):
            engine.update_model(model="other-model", context_length=200_000, base_url="https://other/v1",
                                api_key="other-key", provider="custom", api_mode=ROUTE[2])
            return real(*args, **kwargs)
        layout.build = build
        try:
            history = [*old_turns(), assistant("done")]
            self.assertIs(engine.compress(history), history)
        finally:
            layout.build = real
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("cancelled", "route_changed"))
        self.assertEqual(engine.compression_count, 0)

    def test_the_fixed_budget_is_not_above_a_small_room(self):
        self.engine._room = lambda *args, **kwargs: 100
        self.assertEqual(self.engine._fixed_budget(0, 0), 100)
        self.engine._room = lambda *args, **kwargs: -5
        self.assertEqual(self.engine._fixed_budget(0, 0), 0)

    def test_the_fallback_summary_gets_the_cut_middle_of_the_prepended_row(self):
        # The fallback transcript has only the start and end of a long row: the middle that the tail cuts from
        # the prepended request must stay in the summary (a bounded quote: its start and end).
        rows = [*old_turns(4), user("BIG start " + "q" * 3_000 + " MID-REQ " + "q" * 37_000 + " big end"),
                assistant("", [("c1", "read", "{}")]), tool("c1", "r1"), assistant("done")]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fallback")
        self.assertNotIn("MID-REQ", json.dumps([call[0] for call in self.llm.calls]))
        prepended = next(row for row in new if str(row.get("content")).startswith("BIG start"))
        self.assertNotIn("MID-REQ", prepended["content"])
        self.assertTrue(any("MID-REQ" in str(row.get("content")) for row in new if row.get("_compressed_summary")))

    def test_the_quote_after_a_fallback_summary_stays_in_the_summary_budget(self):
        # A fallback summary near the reserve: the quote of the cut request takes only what is left, so the tail
        # cap does not go below the cap that the fallback transcript had.
        from warm_compaction.engine import SUMMARY_RESERVE
        from warm_compaction.rows import estimate_tokens
        self.llm = FakeLlm(text=HEADINGS_TEXT.replace("Finish the test task.", "word " * 3_000) + "\n" + END_LINE)
        rows = [*old_turns(4), user("BIG start " + "q" * 3_000 + " MID-REQ " + "q" * 37_000 + " big end"),
                assistant("", [("c1", "read", "{}")]), tool("c1", "r1"), assistant("done")]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fallback")
        summary = next(row["content"] for row in new if row.get("_compressed_summary") and row["role"] == "assistant")
        self.assertLessEqual(estimate_tokens(summary), SUMMARY_RESERVE + 100)

    def test_a_window_change_before_the_result_is_used_discards_it(self):
        # The same route with a smaller window: the result was sized for the old window and threshold.
        from warm_compaction import layout
        engine = self.make(warm=False)
        real = layout.build

        def build(*args, **kwargs):
            engine.update_model(model=ROUTE[0], context_length=32_000, base_url=ROUTE[1], api_key="k",
                                provider="custom", api_mode=ROUTE[2])
            return real(*args, **kwargs)
        layout.build = build
        try:
            history = [*old_turns(), assistant("done")]
            self.assertIs(engine.compress(history), history)
        finally:
            layout.build = real
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("cancelled", "route_changed"))

    def callable_key_engine(self, keys):
        # Hermes can give the key as a function (a token that refreshes or rotates).
        engine = self.engine_class(store=self.store, llm=self.llm, post=self.post)
        engine.on_session_start("s1", platform="cli")
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key=lambda: keys[0],
                            provider="custom", api_mode=ROUTE[2])
        return engine

    def test_a_callable_key_that_does_not_change_keeps_the_warm_path(self):
        keys = ["tenant-a"]
        engine = self.callable_key_engine(keys)
        rows, reply = old_turns(), assistant("final")
        self.seed(rows, reply)
        self.assertNotIn("tenant-a", json.dumps(self.store.latest("s1"), default=str))
        engine.compress([*rows, reply])
        self.assertEqual(engine.warm_last["path"], "warm")
        self.assertEqual(self.post.calls[0]["headers"]["Authorization"], "Bearer tenant-a")

    def test_a_callable_key_that_changes_after_the_capture_stops_the_warm_request(self):
        # The same function object with a new value: the old body must not go out with another credential.
        keys = ["tenant-a"]
        engine = self.callable_key_engine(keys)
        rows, reply = old_turns(), assistant("final")
        self.seed(rows, reply)
        keys[0] = "tenant-b"
        engine.compress([*rows, reply])
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("fallback", "credential_changed"))
        self.assertEqual(self.post.calls, [])

    def test_a_key_change_during_the_request_keeps_no_capture(self):
        keys = ["tenant-a"]
        self.callable_key_engine(keys)
        self.seed(old_turns(1), assistant("x"), during=lambda: keys.__setitem__(0, "tenant-b"))
        self.assertIsNone(self.store.latest("s1"))

    def test_a_capture_without_a_key_stamp_is_not_sent(self):
        # A store without the stamp function of the session cannot show which key sent the capture.
        rows, reply = old_turns(), assistant("final")
        store = type(self.store)()
        engine = self.engine_class(store=store, llm=self.llm, post=self.post)
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        engine._wc_session_id = "s1"
        wc_hermes_stub.CAPTURE_CHAIN.append(store.on_llm_execution)
        self.store = store
        self.seed(rows, reply)
        engine.compress([*rows, reply])
        self.assertEqual(engine.warm_last["reason"], "credential_changed")
        self.assertEqual(self.post.calls, [])

    def test_the_native_carrier_of_the_provider_profile_keeps_the_warm_path(self):
        # The profile declares the carrier that Hermes replays: the captured body has it, and the estimate counts it.
        from warm_compaction.rows import SendPolicy, sent_tokens
        wc_hermes_stub.PROFILE_FIELDS["native_reasoning_details_type"] = "acme.native_assistant"
        rows = old_turns()
        rows[1] = {**rows[1], "reasoning_details": [{"type": "acme.native_assistant", "data": "n" * 4_000}]}
        reply = assistant("final")
        self.seed(rows, reply)
        policy = self.engine._policy([*rows, reply])
        self.assertEqual((policy.native_type, policy.details), ("acme.native_assistant", True))
        self.assertGreater(sent_tokens(rows[1], policy), sent_tokens(rows[1], SendPolicy(echo=False)) + 900)
        self.engine.compress([*rows, reply])
        self.assertEqual(self.engine.warm_last["path"], "warm")

    def test_a_fallback_summary_without_room_for_the_cut_quote_becomes_the_fixed_summary(self):
        # The fallback summary takes almost all of its budget: no quote of the cut middle of the prepended request
        # fits after it. The fixed summary then quotes that middle (its start: the marker is near it).
        rows = [*old_turns(4), user("BIG start " + "q" * 1_000 + " MID-REQ " + "q" * 40_000 + " big end"),
                assistant("", [("c1", "read", "{}")]), tool("c1", "r1"), assistant("done")]
        self.llm.text = (HEADINGS_TEXT.replace("Finish the test task.", "Fallback summary. " + "z" * 16_150)
                         + "\n" + END_LINE)
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fixed")
        self.assertTrue(any("MID-REQ" in str(row.get("content")) for row in new if row.get("_compressed_summary")))

    def test_no_fallback_request_starts_after_a_switch(self):
        # A fallback task on the auto route follows the main route: after a switch, the old transcript must not
        # go to the new provider.
        from warm_compaction import layout
        engine = self.make(warm=False)
        real = layout.bound_tail

        def bound_tail(*args, **kwargs):
            engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k2",
                                provider="other", api_mode=ROUTE[2])
            return real(*args, **kwargs)
        layout.bound_tail = bound_tail
        try:
            history = [*old_turns(), assistant("done")]
            self.assertIs(engine.compress(history), history)
        finally:
            layout.bound_tail = real
        self.assertEqual(self.llm.calls, [])
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("cancelled", "route_changed"))

    def test_an_unknown_overhead_takes_the_summary_from_half_of_the_room(self):
        # The other half is for the system rows and tool schemas: the summary and the prepended row come out of
        # the tail half.
        engine = self.make(tail_tokens=500_000)
        self.assertEqual(engine._tail_cap(0, None, 0) - engine._tail_cap(2_000, None, 0, 1_000), 3_000)

    def test_three_compactions_without_the_warm_path_tell_the_user_one_time(self):
        # No capture: each compaction uses the fallback. Each refusal is a WARNING; the third one adds one notice
        # to the next automatic compaction status that Hermes shows.
        engine = self.make()
        history = [*old_turns(), assistant("done")]
        with self.assertLogs("warm_compaction.engine", level="WARNING") as logs:
            for _ in range(3):
                engine.compress(history)
        self.assertEqual((engine.warm_last["path"], engine.warm_last["reason"]), ("fallback", "no_capture"))
        self.assertEqual(sum("warm request was not used (no_capture)" in line for line in logs.output), 3)
        self.assertEqual(sum("last 3 compactions did not use the warm cache" in line for line in logs.output), 1)
        message = engine.get_automatic_compaction_status_message(phase="compress", default_message="Compacting")
        self.assertTrue(message.startswith("Compacting\n"))
        self.assertIn("last 3 compactions did not use the warm cache (no_capture)", message)
        self.assertIn("no main-model request completed", message)
        # One time only.
        self.assertEqual(engine.get_automatic_compaction_status_message(phase="compress",
                                                                        default_message="Compacting"), "Compacting")

    def test_a_warm_compaction_ends_the_failure_streak(self):
        engine = self.make()
        rows, reply = old_turns(), assistant("final")
        for _ in range(2):
            engine.compress([*rows, reply])
        self.seed(rows, reply)
        engine.compress([*rows, reply])
        self.assertEqual(engine.warm_last["path"], "warm")
        self.store.forget(session_id="s1")
        for _ in range(2):
            engine.compress([*rows, reply])
        self.assertEqual(engine.get_automatic_compaction_status_message(phase="compress",
                                                                        default_message="Compacting"), "Compacting")

    def test_a_cancelled_attempt_does_not_end_the_failure_streak(self):
        engine = self.make()
        history = [*old_turns(), assistant("done")]
        engine.compress(history)
        engine._compression_cancelled_check = lambda: True
        engine.compress(history)
        self.assertEqual(engine.warm_last["path"], "cancelled")
        engine._compression_cancelled_check = lambda: False
        for _ in range(2):
            engine.compress(history)
        self.assertIn("did not use the warm cache",
                      engine.get_automatic_compaction_status_message(phase="compress", default_message="Compacting"))

    def test_the_warm_setting_off_is_not_a_failure(self):
        engine = self.make(warm=False)
        history = [*old_turns(), assistant("done")]
        with self.assertNoLogs("warm_compaction.engine", level="WARNING"):
            for _ in range(4):
                engine.compress(history)
        self.assertEqual(engine.get_automatic_compaction_status_message(phase="compress",
                                                                        default_message="Compacting"), "Compacting")

    def test_the_notice_shows_when_the_host_status_is_turned_off(self):
        # A user who turned the compaction status off still gets the notice: it is a warning, not progress.
        engine = self.make()
        history = [*old_turns(), assistant("done")]
        for _ in range(3):
            engine.compress(history)
        engine.emit_automatic_compaction_status = False
        message = engine.get_automatic_compaction_status_message(phase="compress", default_message="Compacting")
        self.assertTrue(message.startswith("warm_compaction: the last 3 compactions"))

    def test_an_unusable_capture_does_not_set_the_reasoning_rule(self):
        # The capture is of this route, but its rows are not the stored rows: it does not show what the next
        # request replays. The stored rows with reasoning_content do.
        rows = old_turns(2)
        self.seed(rows, assistant("x"))
        changed = [user("other"), assistant("a", reasoning_content="r" * 4_000), user("next")]
        policy = self.engine._policy(changed)
        self.assertTrue(policy.echo)
        self.assertFalse(policy.cut_reasoning)

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
        names = ("X-Title", "User-Agent", "X-Gateway-Key", "Authorization")
        self.assertEqual(tuple(headers[name] for name in names),
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
        # A captured request without a positive limit: the provider default of the next request is unknown too.
        for body in ({"messages": []}, {"messages": [], "max_tokens": 0}, {"messages": [], "max_tokens": None}):
            with self.subTest(body=body):
                self.assertEqual(request_reserve({"body": body}, 200_000), 50_000)

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
        engine.compress(rows, current_tokens=estimate_tokens(rows) + 1_000)
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

    def test_the_fixed_summary_quotes_the_cut_middle_of_the_prepended_row(self):
        # The prepended row is the latest user message; it is not copied, so its cut middle must stay in the
        # summary when no model summary is available.
        from warm_compaction.layout import MIN_COPY_CHARS
        from warm_compaction.rows import estimate_tokens
        from warm_compaction.warm import DEFAULT_RESERVE
        self.llm.error = RuntimeError("down")
        rows = [*old_turns(4), user("BIG start " + "q" * 1_000 + " MID-REQ " + "q" * 40_000 + " big end"),
                assistant("", [("c1", "read", "{}")]), tool("c1", "r1"), assistant("done")]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fixed")
        prepended = next(row for row in new if str(row.get("content")).startswith("BIG start"))
        self.assertLessEqual(len(prepended["content"]), MIN_COPY_CHARS)
        self.assertNotIn("MID-REQ", prepended["content"])
        self.assertTrue(any("MID-REQ" in str(row.get("content")) for row in new if row.get("_compressed_summary")))
        self.assertLessEqual(estimate_tokens(new), min(engine.threshold_tokens, 64_000 - DEFAULT_RESERVE))

    def test_a_provider_or_key_switch_forgets_the_capture(self):
        # Two configurations can share the model, the base URL, and the API mode; the old body must not go out
        # with the new key and headers.
        rows = old_turns(1)
        self.seed(rows, assistant("x"))
        self.engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k",
                                 provider="custom", api_mode=ROUTE[2])
        self.assertIsNotNone(self.store.latest("s1"))
        for change in ({"api_key": "k2", "provider": "custom"}, {"api_key": "k2", "provider": "other"}):
            self.seed(rows, assistant("x"))
            with self.subTest(change=change):
                self.engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1],
                                         api_mode=ROUTE[2], **change)
                self.assertIsNone(self.store.latest("s1"))

    def test_the_fixed_path_builds_the_tail_at_the_cap_that_the_summary_quotes(self):
        # Each quote makes the summary larger and the cap smaller. When the rounds stop before the cap is stable,
        # the tail must be cut at the cap whose cut the summary quotes, not at a smaller one.
        from warm_compaction import layout
        self.llm.error = RuntimeError("down")
        text = "".join(f"{index:07d}" for index in range(30_000))
        rows = [*old_turns(4), user("go"), assistant("", [("c1", "read", "{}")]), tool("c1", text)]
        engine = self.make(threshold=0.95, tail_tokens=30_000, warm=False)
        caps = iter(range(30_000, 0, -1_500))
        engine._tail_cap = lambda *args, **kwargs: next(caps)
        cuts, built = [], {}
        real_bound, real_build = layout.bound_tail, layout.build

        def bound(rows, tokens, removed=None, *args, **kwargs):
            if removed is not None:
                cuts.append(tokens)
            return real_bound(rows, tokens, removed, *args, **kwargs)

        def build(*args, **kwargs):
            built.update(kwargs)
            return real_build(*args, **kwargs)
        layout.bound_tail, layout.build = bound, build
        try:
            engine.compress(rows)
        finally:
            layout.bound_tail, layout.build = real_bound, real_build
        self.assertEqual(engine.warm_last["path"], "fixed")
        self.assertEqual(built["tail_tokens"], cuts[-1])

    def test_a_large_captured_overhead_makes_a_short_history_compact(self):
        # The history fits the nominal tail, but the system rows and tool schemas of the captured request leave no
        # room for it and the reply reserve.
        rows = old_turns(4)
        reply = assistant("final")
        engine = self.make(threshold=0.95)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        self.seed(rows, reply, system={"role": "system", "content": "s " * 112_000}, extra={"max_tokens": 4_096})
        history = [*rows, reply]
        self.assertTrue(engine.should_compress_preflight(history))
        self.assertTrue(engine.has_content_to_compress(history))
        new = engine.compress(history)
        self.assertIsNot(new, history)
        self.assertLess(len(new), len(history))

    def test_the_copied_messages_are_sized_after_the_tail_cut(self):
        # A huge newest unit fills the room before the cut; after the cut there is room for the copies, which the
        # fixed summary needs (it does not keep the earlier user text).
        self.llm.error = RuntimeError("down")
        rows = [*old_turns(4), user("go")]
        reply = assistant("", [("c1", "read", "{}")])
        self.seed(rows, reply)
        engine = self.make(threshold=0.95, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress([*rows, reply, tool("c1", "t" * 400_000)])
        self.assertEqual(engine.warm_last["path"], "fixed")
        summary = "\n".join(str(row["content"]) for row in new if row.get("_compressed_summary"))
        self.assertIn("ask 0 ", summary)

    def test_reasoning_of_an_earlier_route_does_not_reach_the_fallback_model(self):
        # No capture of this route: old rows with reasoning_content (of a thinking route before a model switch)
        # do not show that this route sends it.
        rows = [*old_turns(4), user("go"),
                assistant("c" * 40_000, reasoning_content="r" * 40_000 + " SECRET " + "r" * 40_000)]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fallback")
        sent = json.dumps([call[0] for call in self.llm.calls])
        self.assertNotIn("SECRET", sent)
        self.assertNotIn("r" * 100, sent)

    def test_the_prepended_request_is_fitted_against_the_bounded_tail(self):
        # A huge tool result is cut to the tail cap: the room after it keeps the whole active request.
        request = "REQ start " + "q" * 6_000 + " REQ end"
        rows = [*old_turns(4), user(request)]
        reply = assistant("", [("c1", "read", "{}")])
        self.seed(rows, reply)
        engine = self.make(threshold=0.95, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress([*rows, reply, tool("c1", "t" * 400_000)])
        self.assertEqual(engine.warm_last["path"], "fallback")
        self.assertTrue(any(row.get("role") == "user" and row.get("content") == request for row in new))

    def test_unsent_reasoning_does_not_reach_the_fallback_model(self):
        # The route does not send reasoning: the auxiliary model must not get the private reasoning of a cut row.
        rows = [*old_turns(4), user("go"), assistant("c" * 40_000, reasoning="r" * 40_000 + " SECRET " + "r" * 40_000)]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fallback")
        sent = json.dumps([call[0] for call in self.llm.calls])
        self.assertNotIn("SECRET", sent)
        self.assertNotIn("r" * 100, sent)

    def test_clone_keeps_the_model_thresholds(self):
        self.engine.model_thresholds = {"fake": 0.25}
        clone = self.engine.clone_for_agent()
        self.engine.model_thresholds["fake"] = 0.9
        clone.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_mode=ROUTE[2])
        self.assertEqual(clone.threshold_tokens, 50_000)

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
        self.engine.update_from_response({"prompt_tokens": 150_000, "completion_tokens": 10})
        self.engine.awaiting_real_usage_after_compression = True
        self.engine.on_session_reset()
        self.assertEqual((self.engine.warm_last, self.engine.compression_count), (None, 0))
        # Hermes uses the real prompt count of the old session as a floor unless the latch is set.
        self.assertEqual((self.engine.last_real_prompt_tokens,
                          self.engine.awaiting_real_usage_after_compression), (0, False))
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

    def test_a_fallback_summary_above_the_reserve_goes_to_the_fixed_summary(self):
        # A dense summary (CJK) can pass the byte gate and still be above the token reserve: the tail would then
        # cut more than the fallback transcript had.
        dense = HEADINGS_TEXT.replace("Finish the test task.", "\u76ee\u6807" * 2_500)
        self.llm = FakeLlm(text=dense + "\n" + END_LINE)
        engine = self.make(warm=False)
        engine.compress([*old_turns(), assistant("done")])
        self.assertEqual(engine.warm_last["path"], "fixed")

    def test_the_fixed_summary_quotes_the_middle_that_the_final_tail_cuts(self):
        # A large earlier summary (CJK) makes the fixed summary larger than the reserve: the tail is cut at the
        # final cap, and the quote starts where the kept start ends.
        from warm_compaction.rows import MIDDLE_MARK
        self.llm.error = RuntimeError("down")
        text = "".join(f"{index:07d}" for index in range(30_000))
        rows = [assistant("\u65e7" * 9_000, _compressed_summary=True), *old_turns(4), user("go"),
                assistant("", [("c1", "read", "{}")]), tool("c1", text)]
        engine = self.make(threshold=0.95, tail_tokens=30_000, warm=False)
        engine.update_model(model=ROUTE[0], context_length=64_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        new = engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fixed")
        kept = next(row for row in new if row.get("role") == "tool")["content"]
        head = kept.split(MIDDLE_MARK)[0]
        summary = "\n".join(str(row["content"]) for row in new if row.get("_compressed_summary"))
        self.assertIn("> " + text[len(head):len(head) + 40], summary)

    def test_the_fixed_summary_gets_the_middles_that_the_tail_cuts(self):
        self.llm.error = RuntimeError("down")
        rows = [*old_turns(4), user("go"), assistant("", [("c1", "read", "{}")]),
                tool("c1", "head " + "u" * 20_000 + " MIDDLE-FACT " + "u" * 20_000 + " end")]
        engine = self.make(tail_tokens=2_000, warm=False)
        new = engine.compress(rows)
        self.assertEqual(engine.warm_last["path"], "fixed")
        self.assertTrue(any("## Key facts" in str(row.get("content")) and "u" * 50 in str(row.get("content"))
                            for row in new))

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
