"""Tests: the fold must not remove the newest visible answer of a running turn.

The host puts a folded answer back at the end of the compressed history only when the rows around the
fold slot allow it. A turn that is still running puts an assistant row after the answer, so a fold point
after the answer makes the host drop it ("compression removed the visible reply"). These tests pin the
plugin behavior: the fold point moves back to the answer and the answer stays a real row in the tail.
"""

import unittest

import wc_hermes_stub
from test_wc_engine import FakeLlm, fake_post, old_turns
from wc_fixtures import ROUTE, assistant, tool, user

ANSWER = "The visible answer of the turn that is still running."


class TailGuardTest(unittest.TestCase):
    def setUp(self):
        wc_hermes_stub.install(self)
        from warm_compaction.capture import CaptureStore
        from warm_compaction.engine import WarmCompactionEngine
        self.store = CaptureStore()
        self.llm = FakeLlm()
        self.post = fake_post()
        self.engine_class = WarmCompactionEngine

    def make(self, **settings):
        engine = self.engine_class(store=self.store, llm=self.llm, post=self.post, settings=settings or None)
        engine.on_session_start("s1", platform="cli")
        engine.update_model(model=ROUTE[0], context_length=200_000, base_url=ROUTE[1], api_key="k",
                            provider="custom", api_mode=ROUTE[2])
        return engine

    def real_rows(self, new):
        return [row for row in new if not row.get("_compressed_summary")]

    def test_a_running_turn_keeps_its_newest_visible_answer_in_the_tail(self):
        # The tail cap cuts inside the running turn: the fold would put the answer into the summary middle.
        rows = [*old_turns(4), user("go"), assistant("", [("c1", "read", "{}")]),
                tool("c1", "m" * 3_000), assistant(ANSWER), assistant("", [("c2", "read", "{}")]),
                tool("c2", "t" * 400_000)]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        new = engine.compress(rows, current_tokens=None)
        self.assertEqual(engine.warm_last["path"], "fallback")
        self.assertTrue(any(ANSWER in str(row.get("content")) for row in self.real_rows(new)),
                        "the newest visible answer of the running turn was folded away")

    def test_a_finished_turn_keeps_the_plain_fold_point(self):
        # A new user request follows the answer: the host can put the answer back, so the guard must not move.
        rows = [*old_turns(4), user("go"), assistant(ANSWER), user("next ask"),
                assistant("a" * 3_000)]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        with self.assertLogs("warm_compaction", level="INFO") as logged:
            engine.compress(rows, current_tokens=None)
        self.assertFalse([line for line in logged.output if "tail guard" in line],
                         "the tail guard moved a fold it must leave alone")

    def test_an_answer_larger_than_the_tail_yields(self):
        # Same give-up rule as the built-in compressor: no move when the answer alone does not fit the tail.
        rows = [*old_turns(4), user("go"), assistant("", [("c1", "read", "{}")]),
                tool("c1", "m" * 3_000), assistant("z" * 40_000), assistant("", [("c2", "read", "{}")]),
                tool("c2", "t" * 400_000)]
        engine = self.make(threshold=0.95, tail_tokens=2_000, warm=False)
        with self.assertLogs("warm_compaction", level="INFO") as logged:
            engine.compress(rows, current_tokens=None)
        self.assertTrue([line for line in logged.output if "tail guard yields" in line],
                        "the tail guard did not report that it yields on fit")


if __name__ == "__main__":
    unittest.main()
