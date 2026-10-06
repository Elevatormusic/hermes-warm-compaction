"""Tests for the fallback summary and the fixed-format summary."""

import unittest
from types import SimpleNamespace

from warm_compaction.fallback import (
    FALLBACK_INSTRUCTION, MAX_TOKENS, MIDDLE_MARK, TASK, TOOL_CHARS, TRANSCRIPT_CHARS, TRANSCRIPT_TOKENS,
    fixed_summary, llm_summary, render_row, transcript,
)
from warm_compaction.rows import estimate_tokens
from warm_compaction.handoff import HEADINGS
from wc_fixtures import assistant, tool, user

PREFIXES = ("[HERMES PREFIX]", "[CONTEXT SUMMARY]:")


class FakeLlm:
    def __init__(self, text="## Goal\nG", error=None):
        self.text, self.error, self.calls = text, error, []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text, usage=SimpleNamespace(input_tokens=321))


class TranscriptTest(unittest.TestCase):
    def test_attachments_are_marked_in_the_transcript(self):
        text = transcript([user([{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]), assistant("ok")], PREFIXES)
        self.assertIn("[first user message]\nsee\n[image attachment]", text)
        self.assertIn("[user]\nsee\n[image attachment]", text)

    def test_dense_text_is_bounded_in_tokens(self):
        rows = [user("\u4f60" * 3_000), *[assistant("\u597d" * 3_000) for _ in range(20)]]
        text = transcript(rows, PREFIXES)
        self.assertLessEqual(estimate_tokens(text), TRANSCRIPT_TOKENS)
        self.assertIn("[first user message]", text)
        self.assertTrue(text.endswith("\u597d"))

    def test_think_blocks_are_removed_only_from_assistant_rows(self):
        text = "<think>keep me</think> body"
        self.assertNotIn("keep me", render_row(assistant(text)))
        self.assertIn("keep me", render_row(user(text)))
        self.assertIn("keep me", render_row(tool("c1", text)))

    def test_earlier_summary_then_first_user_message_then_newest_rows(self):
        rows = [user("[HERMES PREFIX] header", _compressed_summary=True),
                assistant("[CONTEXT SUMMARY]:\nold summary\n\n--- END OF CONTEXT SUMMARY x", _compressed_summary=True),
                user("first ask"), assistant("", [("c1", "read", "{\"p\":1}")]),
                tool("c1", "y" * (TOOL_CHARS + 100)), assistant("done")]
        text = transcript(rows, PREFIXES)
        self.assertTrue(text.startswith("[earlier summary]\nold summary\n\n[first user message]\nfirst ask"))
        self.assertIn("(tool call c1 read: {\"p\":1})", text)
        self.assertIn("[tool result c1 read]\n" + "y" * TOOL_CHARS + " [cut]", text)
        self.assertTrue(text.endswith("[assistant]\ndone"))

    def test_size_is_bounded(self):
        rows = [user("u" * 5000), *[assistant("a" * 3000) for _ in range(20)]]
        self.assertLessEqual(len(transcript(rows, PREFIXES)), TRANSCRIPT_CHARS)

    def test_one_large_newest_row_is_cut(self):
        text = transcript([user("ask"), assistant("z" * 50_000)], PREFIXES)
        self.assertLessEqual(len(text), TRANSCRIPT_CHARS)
        self.assertIn("[assistant]\nzzz", text)

    def test_every_turn_of_large_rows_keeps_its_start_and_end(self):
        rows = []
        for index in range(6):
            rows += [user(f"question {index}"), assistant(f"state {index} " + "padding " * 10_000 + f" end {index}")]
        rows += [user("Reply only: Ready."), assistant("Ready.")]
        text = transcript(rows, PREFIXES)
        self.assertLessEqual(len(text), TRANSCRIPT_CHARS)
        for index in range(6):
            with self.subTest(index=index):
                self.assertIn(f"[user]\nquestion {index}", text)
                self.assertIn(f"[assistant]\nstate {index} padding", text)
                self.assertIn(f"padding  end {index}", text)
        self.assertTrue(text.endswith("[user]\nReply only: Ready.\n\n[assistant]\nReady."))

    def test_the_oldest_row_that_does_not_fit_is_cut_to_fit(self):
        rows = [user("first"), *[assistant(f"row {index} " + "x" * 3_500) for index in range(12)]]
        text = transcript(rows, PREFIXES)
        self.assertLessEqual(len(text), TRANSCRIPT_CHARS)
        self.assertIn("[assistant]\nrow 11 ", text)
        self.assertEqual(text.count(MIDDLE_MARK), 1)


class LlmSummaryTest(unittest.TestCase):
    def test_calls_the_registered_task(self):
        llm = FakeLlm()
        text, tokens = llm_summary(llm, [user("hi"), assistant("ok")], PREFIXES, focus_topic="db", memory_context="m")
        messages, kwargs = llm.calls[0]
        self.assertEqual((text, tokens), ("## Goal\nG", 321))
        self.assertEqual((kwargs["task"], kwargs["max_tokens"], kwargs["timeout"]), (TASK, MAX_TOKENS, 120.0))
        self.assertTrue(messages[0]["content"].startswith(FALLBACK_INSTRUCTION))
        self.assertIn("Give more detail to this topic: db", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")

    def test_task_none_uses_the_main_model_route(self):
        llm = FakeLlm()
        llm_summary(llm, [user("hi")], PREFIXES, task=None)
        self.assertIsNone(llm.calls[0][1]["task"])

    def test_failures_return_none(self):
        for llm in (None, FakeLlm(text="  "), FakeLlm(text="x" * 30_000)):
            with self.subTest(llm=llm):
                self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES), (None, None))
        with self.assertLogs("warm_compaction.fallback", level="WARNING") as logs:
            self.assertEqual(llm_summary(FakeLlm(error=RuntimeError("x")), [user("hi")], PREFIXES), (None, None))
        self.assertIn("(RuntimeError)", logs.output[0])


class FixedSummaryTest(unittest.TestCase):
    def test_five_headings_and_tool_counts(self):
        rows = [assistant("", [("c1", "read", "{}"), ("c2", "read", "{}")]), assistant("", [("c3", "ls", "{}")])]
        text = fixed_summary(rows)
        for heading in HEADINGS:
            self.assertIn(heading + "\n", text)
        self.assertIn("- Tool calls: ls x1", text)
        self.assertIn("- Tool calls: read x2", text)
        self.assertIn("[OPEN] Continue from the latest user message.", text)

    def test_no_tool_calls(self):
        self.assertIn("- No tool calls.", fixed_summary([user("x")]))


if __name__ == "__main__":
    unittest.main()
