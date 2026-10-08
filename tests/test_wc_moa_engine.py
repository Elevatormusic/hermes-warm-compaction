"""MOA history, fallback, cancellation, and advisory-note checks with synthetic hooks."""

import copy
import json
import os
import unittest
from unittest.mock import patch

import test_wc_engine as engine_fixtures
from test_wc_engine import old_turns, reply_object
from wc_fixtures import HEADINGS_TEXT, SYSTEM, assistant, user, wire


def route(name):
    return {"name": name, "provider": "custom", "model": name,
            "base_url": f"http://127.0.0.1:9/{name}/v1", "context_length": 200000}


class MoaEngineTest(unittest.TestCase):
    def setUp(self):
        self.helper = engine_fixtures.EngineTest()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        from warm_compaction.capture import CaptureStore
        from warm_compaction.moa import MoaStore
        self.tracker = MoaStore([route("aggregator"), route("reference")], include_references=True)
        self.helper.store = CaptureStore(moa=self.tracker)
        engine_fixtures.wc_hermes_stub.CAPTURE_CHAIN[:] = [self.helper.store.on_llm_execution]
        self.engine = self.helper.make(moa_references=True)
        self.engine.update_model("preset", 200000, base_url="moa://local", provider="moa",
                                 api_mode="chat_completions")
        self.rows = [*old_turns(), user("Write the report.")]
        self.reply = assistant("The current answer.")
        self.messages = [*self.rows, self.reply]

    def aux(self, name, task, call_id, messages, text, streaming=False):
        target = route(name)
        common = dict(aux_task=task, api_request_id=call_id, retry_count=0, session_id="s1", turn_id="turn-1",
                      provider=target["provider"], model=name, base_url=target["base_url"],
                      api_mode="chat_completions", streaming=streaming)
        body = {"model": name, "messages": messages, "temperature": 0.2, "stream": streaming}
        self.tracker.on_pre_auxiliary_call(**common, request={"method": "POST", "body": body},
                                          request_messages=messages)
        self.tracker.on_post_auxiliary_call(
            **common, error=None, finish_reason=None if streaming else "stop",
            usage=None if streaming else {"prompt_tokens": 500},
            response=None if streaming else {"assistant_message": {"role": "assistant", "content": text},
                                              "finish_reason": "stop"})

    def seed(self, reference=True):
        if reference:
            self.aux("reference", "moa_reference", "ref-1", [SYSTEM, user("Earlier advisory view.")],
                     "Earlier reference advice.")
        store = self.helper.store
        store.on_pre_api_request(api_request_id="main-1", session_id="s1", turn_id="turn-1", provider="moa",
                                 conversation_history=self.rows, model="preset", base_url="moa://local",
                                 api_mode="chat_completions")
        from warm_compaction.moa import ADVISOR_PREFIX
        messages = [SYSTEM, *wire(self.rows), user(ADVISOR_PREFIX + "Private reference guidance.")]
        store.on_llm_execution(
            api_request_id="main-1", request={"model": "preset", "messages": messages},
            next_call=lambda: self.aux("aggregator", "moa_aggregator", "agg-1", messages,
                                      self.reply["content"], streaming=True))
        store.on_post_api_request(api_request_id="main-1", session_id="s1", finish_reason="stop",
                                  assistant_message=reply_object(self.reply), usage={"prompt_tokens": 40000})

    def test_reference_advice_goes_to_aggregator_and_only_one_history_is_committed(self):
        self.seed()
        new = self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["path"], "warm")
        calls = self.helper.post.calls
        self.assertEqual([call["body"]["model"] for call in calls], ["reference", "aggregator"])
        self.assertEqual(calls[0]["body"]["messages"][:2], [SYSTEM, user("Earlier advisory view.")])
        self.assertNotIn("The current answer.", json.dumps(calls[0]["body"]))
        self.assertIn("Earlier advisory view; not current task authority.", calls[1]["body"]["messages"][-1]["content"])
        self.assertEqual(self.engine.compression_count, 1)
        self.assertIn(HEADINGS_TEXT, new[1]["content"])
        self.assertEqual(self.helper.llm.calls, [])

    def test_reference_failure_does_not_replace_the_aggregator_handoff(self):
        self.seed()
        original = self.helper.post

        def post(url, data, headers, timeout_s):
            if json.loads(data)["model"] == "reference":
                return 500, b"{}"
            return original(url, data, headers, timeout_s)

        self.engine._post = post
        new = self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["path"], "warm")
        self.assertEqual(self.engine.warm_last["moa_slots"][0]["reason"], "provider_error")
        self.assertNotIn("Earlier advisory view; not current task authority.",
                         original.calls[0]["body"]["messages"][-1]["content"])
        self.assertIn(HEADINGS_TEXT, new[1]["content"])

    def test_reference_setting_false_sends_only_aggregator(self):
        self.seed()
        self.engine._settings["moa_references"] = False
        self.engine.compress(self.messages)
        self.assertEqual([call["body"]["model"] for call in self.helper.post.calls], ["aggregator"])

    def test_aggregator_failure_uses_one_fallback(self):
        self.seed(reference=False)
        self.engine._post = lambda *_args, **_kwargs: (500, b"{}")
        self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["path"], "fallback")
        self.assertEqual(len(self.helper.llm.calls), 1)

    def test_cancel_after_reference_keeps_history_and_sends_no_aggregator(self):
        self.seed()
        cancelled = False
        original = self.helper.post

        def post(*args, **kwargs):
            nonlocal cancelled
            reply = original(*args, **kwargs)
            cancelled = True
            return reply

        self.engine._post = post
        self.engine._compression_cancelled_check = lambda: cancelled
        new = self.engine.compress(self.messages)
        self.assertTrue(new is self.messages, "Cancellation must keep the original history object.")
        self.assertEqual(self.engine.compression_count, 0)
        self.assertEqual(self.engine.warm_last["path"], "cancelled")
        self.assertEqual([call["body"]["model"] for call in original.calls], ["reference"])
        self.assertEqual(self.helper.llm.calls, [])

    def test_reference_budget_leaves_time_for_aggregator(self):
        self.seed()
        ticks = iter([0, 0, 100, 100, 101])
        self.engine._clock = lambda: next(ticks)
        self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["path"], "warm")
        self.assertEqual(self.engine.warm_last["moa_slots"][0]["reason"], "moa_time_budget")
        self.assertEqual([call["body"]["model"] for call in self.helper.post.calls], ["aggregator"])

    def test_physical_requests_go_through_both_middleware_checks(self):
        self.seed()
        seen = []
        def request_middleware(request, **context):
            seen.append(("request", context["model"], context["base_url"]))
        def execution_middleware(next_call, **context):
            seen.append(("execution", context["model"], context["base_url"]))
            return next_call()
        stub = engine_fixtures.wc_hermes_stub
        stub.REQUEST_MIDDLEWARE.append(request_middleware)
        stub.EXECUTION_MIDDLEWARE.append(execution_middleware)
        self.engine.compress(self.messages)
        self.assertEqual(seen, [(kind, name, route(name)["base_url"])
                                for name in ("reference", "aggregator") for kind in ("request", "execution")])
        self.assertEqual(self.engine._wc_route, ("preset", "moa://local", "chat_completions"))

    def test_request_middleware_cannot_rewrite_the_physical_prefix(self):
        self.seed(reference=False)
        def rewrite(request, **context):
            request["messages"][0]["content"] = "Changed instruction"
            return {"request": request}
        engine_fixtures.wc_hermes_stub.REQUEST_MIDDLEWARE.append(rewrite)
        self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["reason"], "middleware_rewrite")
        self.assertEqual(self.helper.post.calls, [])
        self.assertEqual(len(self.helper.llm.calls), 1)

    def test_execution_middleware_cannot_replace_the_physical_reply(self):
        self.seed(reference=False)
        engine_fixtures.wc_hermes_stub.EXECUTION_MIDDLEWARE.append(lambda **kwargs: {"text": "fake"})
        self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["reason"], "middleware_changed_reply")
        self.assertEqual(self.helper.post.calls, [])

    def test_execution_middleware_cannot_send_twice(self):
        self.seed(reference=False)
        def retry(next_call, **kwargs):
            next_call()
            return next_call()
        engine_fixtures.wc_hermes_stub.EXECUTION_MIDDLEWARE.append(retry)
        self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["reason"], "middleware_repeated")
        self.assertEqual(len(self.helper.post.calls), 1)

    def test_physical_tls_refusal_stops_the_request(self):
        self.seed(reference=False)
        from warm_compaction.warm import WarmRefusal
        with patch("warm_compaction.warm.route_tls", side_effect=WarmRefusal("tls_unverified")) as check:
            self.engine.compress(self.messages)
        check.assert_called_once_with(route("aggregator")["base_url"])
        self.assertEqual(self.engine.warm_last["reason"], "tls_unverified")
        self.assertEqual(self.helper.post.calls, [])

    def test_env_key_change_in_middleware_stops_the_request(self):
        self.seed(reference=False)
        capture = self.helper.store._sessions["s1"]
        capture["moa"]["aggregator"]["route_config"]["api_key_env"] = "WC_SYNTHETIC_KEY"
        def rotate(request, **kwargs):
            os.environ["WC_SYNTHETIC_KEY"] = "synthetic-second"
        engine_fixtures.wc_hermes_stub.REQUEST_MIDDLEWARE.append(rotate)
        with patch.dict(os.environ, {"WC_SYNTHETIC_KEY": "synthetic-first"}):
            self.engine.compress(self.messages)
        self.assertEqual(self.engine.warm_last["reason"], "credential_changed")
        self.assertEqual(self.helper.post.calls, [])

    def test_route_switch_after_reference_keeps_the_original_history(self):
        self.seed()
        original = self.helper.post
        def switch(*args, **kwargs):
            response = original(*args, **kwargs)
            self.engine.update_model("changed", 200000, base_url="moa://changed", provider="moa",
                                     api_mode="chat_completions")
            return response
        self.engine._post = switch
        before = copy.deepcopy(self.messages)
        result = self.engine.compress(self.messages)
        self.assertIs(result, self.messages)
        self.assertEqual(result, before)
        self.assertEqual([call["body"]["model"] for call in original.calls], ["reference"])
        self.assertEqual(self.helper.llm.calls, [])

    def test_status_does_not_expose_mutable_slot_records(self):
        self.seed()
        self.engine.compress(self.messages)
        status = self.engine.get_status()
        status["warm_last"]["moa_slots"][0]["name"] = "changed"
        self.assertEqual(self.engine.warm_last["moa_slots"][0]["name"], "reference")


if __name__ == "__main__":
    unittest.main()
