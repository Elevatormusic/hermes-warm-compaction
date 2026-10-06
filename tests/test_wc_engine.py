"""Tests for the warm_compaction context engine with stand-in Hermes modules."""

import json
import unittest
from types import SimpleNamespace

import wc_hermes_stub
from wc_fixtures import HEADINGS_TEXT, ROUTE, SYSTEM, assistant, tool, user, wire


class FakeLlm:
    def __init__(self, text="## Goal\nFallback summary.", error=None):
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
        self.assertEqual(self.module.tail_budget(7, 20_000, 10), 7)

    def test_hermes_value_uses_the_replacement(self):
        read = self.module.hermes_value
        self.assertEqual(read("agent.context_compressor", "SUMMARY_PREFIX", "d"), wc_hermes_stub.SUMMARY_PREFIX)
        self.assertEqual(read("agent.context_compressor", "MISSING", "d"), "d")
        self.assertEqual(read("no.such.module", "X", 1), 1)
        self.assertEqual(read("agent.context_compressor", "SUMMARY_PREFIX", 0), 0)


if __name__ == "__main__":
    unittest.main()
