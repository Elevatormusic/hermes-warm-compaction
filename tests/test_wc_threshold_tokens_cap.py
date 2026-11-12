"""Tests: the engine keeps the live host contract for threshold_tokens_cap.

The Hermes host reads and writes threshold_tokens_cap on the active context engine, and calls
_coerce_threshold_tokens_cap on it, when live compression config is set (tui_gateway/
session_compression.py:167; the built-in compressor has both since 2026-10-11). Without them every
live config set fails on AttributeError ("compression config sync failed" in agent.log). These tests
pin the contract: coerce, the attribute, the clamp under a live cap, and the carry-over to a clone.
"""

import unittest

import wc_hermes_stub
from test_wc_engine import FakeLlm, fake_post
from wc_fixtures import ROUTE


class ThresholdTokensCapTest(unittest.TestCase):
    def setUp(self):
        wc_hermes_stub.install(self)
        from warm_compaction.capture import CaptureStore
        from warm_compaction.engine import WarmCompactionEngine
        engine = WarmCompactionEngine(store=CaptureStore(), llm=FakeLlm(), post=fake_post())
        engine.on_session_start("s1", platform="cli")
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        self.engine = engine

    def test_the_host_config_sync_can_coerce_the_cap_value(self):
        coerce = self.engine._coerce_threshold_tokens_cap
        self.assertEqual(coerce("50000"), 50_000)
        self.assertEqual(coerce(50_000), 50_000)
        self.assertIsNone(coerce(0))
        self.assertIsNone(coerce(-10))
        self.assertIsNone(coerce("not a number"))
        self.assertIsNone(coerce(None))

    def test_assigning_the_cap_clamps_the_current_trigger(self):
        self.engine.threshold_tokens_cap = 50_000
        self.assertEqual(self.engine.threshold_tokens_cap, 50_000)
        self.assertLessEqual(self.engine.threshold_tokens, 50_000)

    def test_a_model_change_recomputes_the_trigger_under_the_cap(self):
        self.engine.threshold_tokens_cap = 50_000
        self.engine.update_model(model=ROUTE[0], context_length=128_000, base_url=ROUTE[1], api_key="k",
                                 provider="custom", api_mode=ROUTE[2])
        self.assertLessEqual(self.engine.threshold_tokens, 50_000)

    def test_a_clone_carries_the_live_cap(self):
        self.engine.threshold_tokens_cap = 40_000
        clone = self.engine.clone_for_agent()
        self.assertEqual(clone.threshold_tokens_cap, 40_000)
        clone.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k",
                           provider="custom", api_mode=ROUTE[2])
        self.assertLessEqual(clone.threshold_tokens, 40_000)

    def test_no_cap_keeps_the_plain_trigger(self):
        self.assertIsNone(self.engine.threshold_tokens_cap)
        self.assertEqual(self.engine.threshold_tokens, 100_000)


if __name__ == "__main__":
    unittest.main()
