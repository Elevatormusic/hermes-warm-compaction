"""Tests for the fallback summary and the fixed-format summary."""

import unittest
from types import SimpleNamespace

from warm_compaction.fallback import (
    END_LINE, FALLBACK_INSTRUCTION, MAX_TOKENS, MIDDLE_MARK, TASK, TOOL_CHARS, TRANSCRIPT_CHARS, TRANSCRIPT_TOKENS,
    fixed_summary, llm_summary, render_row, transcript,
)
from warm_compaction.rows import estimate_tokens
from warm_compaction.handoff import HEADINGS
from wc_fixtures import assistant, tool, user

PREFIXES = ("[HERMES PREFIX]", "[CONTEXT SUMMARY]:")


SUMMARY = "## Goal\nG\n## User instructions\n- none\n## Current state\n- [OPEN] x\n## Key facts\n- y\n## Next step\nz"


class FakeLlm:
    def __init__(self, text=SUMMARY + "\n" + END_LINE, error=None, output_tokens=None):
        self.text, self.error, self.calls, self.output_tokens = text, error, [], output_tokens

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        usage = SimpleNamespace(input_tokens=321)
        if self.output_tokens is not None:
            usage.output_tokens = self.output_tokens
        return SimpleNamespace(text=self.text, usage=usage)


class TranscriptTest(unittest.TestCase):
    def test_attachments_are_marked_in_the_transcript(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        text = transcript([user([{"type": "text", "text": "see"}, image]), assistant("ok")], PREFIXES)
        self.assertIn("[first user message]\nsee\n[image attachment]", text)
        self.assertIn("[user]\nsee\n[image attachment]", text)

    def test_dense_text_is_bounded_in_tokens(self):
        rows = [user("\u4f60" * 3_000), *[assistant("\u597d" * 3_000) for _ in range(20)]]
        text = transcript(rows, PREFIXES)
        self.assertLessEqual(estimate_tokens(text), TRANSCRIPT_TOKENS)
        self.assertIn("[first user message]", text)
        self.assertTrue(text.endswith("\u597d"))

    def test_a_reused_call_id_takes_the_name_of_its_own_turn(self):
        rows = [user("go"), assistant("", [("call_0", "read", "{}")]), tool("call_0", "old"),
                assistant("", [("call_0", "write", "{}")]), tool("call_0", "new")]
        text = transcript(rows, PREFIXES)
        self.assertIn("[tool result call_0 read]\nold", text)
        self.assertIn("[tool result call_0 write]\nnew", text)

    def test_the_sent_api_content_is_in_the_transcript(self):
        text = transcript([user("hi", api_content="[ctx]\n\nhi"), assistant("", api_content="The answer is 4.")],
                          PREFIXES)
        self.assertIn("[user]\n[ctx]\n\nhi", text)
        self.assertIn("[assistant]\nThe answer is 4.", text)

    def test_the_end_of_a_long_tool_result_is_kept(self):
        text = render_row(tool("c1", "start " + "y" * 3_000 + " exit status 1"))
        self.assertTrue(text.startswith("[tool result c1]\nstart "))
        self.assertTrue(text.endswith(" exit status 1"))
        self.assertLessEqual(len(text.split("\n", 1)[1]), TOOL_CHARS)

    def test_the_first_user_slot_shows_the_sent_api_content(self):
        rows = [user("first", api_content="[ctx]\n\nfirst"), *[assistant("z" * 3_900) for _ in range(12)]]
        text = transcript(rows, PREFIXES)
        self.assertIn("[first user message]\n[ctx]\n\nfirst", text)

    def test_a_long_earlier_summary_keeps_its_end(self):
        old = "[CONTEXT SUMMARY]:\n## Goal\n" + "x" * 10_000 + "\n## Next step\nthe final step"
        text = transcript([assistant(old, _compressed_summary=True), user("go"), assistant("ok")], PREFIXES)
        self.assertIn("## Goal", text)
        self.assertIn("## Next step\nthe final step", text)

    def test_the_first_user_slot_keeps_the_end(self):
        first = "start " + "x" * 6_000 + " the real question"
        rows = [user(first), *[assistant("z" * 3_900) for _ in range(12)]]
        text = transcript(rows, PREFIXES)
        slot = text.split("[first user message]\n", 1)[1].split("\n\n", 1)[0]
        self.assertTrue(slot.startswith("start "))
        self.assertTrue(slot.endswith(" the real question"))

    def test_long_tool_call_arguments_keep_the_end(self):
        arguments = '{"patch": "' + "y" * 1_000 + '", "path": "src/final.py"}'
        text = render_row(assistant("", [("c1", "write", arguments)]))
        self.assertIn('"path": "src/final.py"})', text)
        self.assertIn('(tool call c1 write: {"patch": "', text)

    def test_names_are_in_the_transcript_labels(self):
        text = transcript([user("ship it", name="alice"), user("wait", name="bob"), assistant("ok", name="lead")],
                          PREFIXES)
        self.assertIn("[user alice]\nship it", text)
        self.assertIn("[user bob]\nwait", text)
        self.assertIn("[assistant lead]\nok", text)

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
        self.assertIn("[tool result c1 read]\n" + "y" * 600, text)
        self.assertIn(MIDDLE_MARK, text.split("[tool result c1 read]\n", 1)[1])
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
        self.assertEqual((text, tokens), (SUMMARY, 321))
        self.assertEqual((kwargs["task"], kwargs["max_tokens"], kwargs["timeout"]), (TASK, MAX_TOKENS, 120.0))
        self.assertTrue(messages[0]["content"].startswith(FALLBACK_INSTRUCTION))
        self.assertIn("Give more detail to this topic: db", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")

    def test_the_memory_context_counts_against_the_transcript_budget(self):
        # A memory provider can give a very large context. The fallback request must stay inside its budget.
        from warm_compaction.fallback import EXTRAS_CHARS
        llm = FakeLlm()
        rows = [user("first"), *[assistant(f"row {index} " + "x" * 3_500) for index in range(12)]]
        llm_summary(llm, rows, PREFIXES, memory_context="m" * 200_000 + " the end")
        messages, _kwargs = llm.calls[0]
        extra = messages[0]["content"][len(FALLBACK_INSTRUCTION):]
        self.assertLessEqual(len(extra), EXTRAS_CHARS)
        self.assertTrue(extra.rstrip().endswith("the end"))
        self.assertLessEqual(len(extra) + len(messages[1]["content"]), TRANSCRIPT_CHARS)
        self.assertLessEqual(estimate_tokens(extra) + estimate_tokens(messages[1]["content"]),
                             TRANSCRIPT_TOKENS + 16)

    def test_a_reply_without_the_end_line_is_refused(self):
        # ctx.llm reports no finish reason. A content filter or a provider limit can stop a reply below
        # MAX_TOKENS; the five headings can already be there. Only the end line shows a complete reply.
        self.assertTrue(FALLBACK_INSTRUCTION.rstrip().endswith(END_LINE))
        for text in (SUMMARY, SUMMARY + "\n" + END_LINE + "\nmore text", END_LINE + "\n" + SUMMARY):
            with self.subTest(text=text[-20:]), self.assertLogs("warm_compaction.fallback", level="WARNING"):
                self.assertEqual(llm_summary(FakeLlm(text=text, output_tokens=300), [user("hi")], PREFIXES),
                                 (None, None))
        text = "<think>plan</think>\n" + SUMMARY + "\n\n  " + END_LINE + "  \n"
        self.assertEqual(llm_summary(FakeLlm(text=text), [user("hi")], PREFIXES), (SUMMARY, 321))

    def test_a_reply_that_reached_the_token_limit_is_refused(self):
        # ctx.llm reports no finish reason. A reply at the max_tokens limit can be cut off.
        for llm in (FakeLlm(output_tokens=MAX_TOKENS),):
            with self.subTest(), self.assertLogs("warm_compaction.fallback", level="WARNING"):
                self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES), (None, None))
        self.assertEqual(llm_summary(FakeLlm(output_tokens=300), [user("hi")], PREFIXES), (SUMMARY, 321))

    def test_task_none_uses_the_main_model_route(self):
        llm = FakeLlm()
        llm_summary(llm, [user("hi")], PREFIXES, task=None)
        self.assertIsNone(llm.calls[0][1]["task"])

    def test_failures_return_none(self):
        for llm in (None, FakeLlm(text="  "), FakeLlm(text="x" * 30_000), FakeLlm(text="The answer is 4."),
                    FakeLlm(text="## Goal\nG\n## Next step\nz"), FakeLlm(text=SUMMARY + "\n[CONTEXT SUMMARY]: x")):
            with self.subTest(llm=llm):
                self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES), (None, None))
        with self.assertLogs("warm_compaction.fallback", level="WARNING") as logs:
            self.assertEqual(llm_summary(FakeLlm(error=RuntimeError("x")), [user("hi")], PREFIXES), (None, None))
        self.assertIn("(RuntimeError)", logs.output[0])


class FixedSummaryTest(unittest.TestCase):
    def test_the_fixed_summary_quotes_the_cut_middles(self):
        # Without a model summary, the middles that the tail cut are not lost: a bounded quote keeps them.
        from warm_compaction.layout import CUT_NOTE
        rows = [user("old"), assistant("a"), tool("c1", CUT_NOTE + "REQ-42 must stay " + "m" * 40_000 + " REQ-END")]
        text = fixed_summary(rows)
        self.assertIn("REQ-42 must stay", text)
        self.assertIn("REQ-END", text)
        self.assertLess(len(text), 12_000)

    def test_five_headings_and_tool_counts(self):
        rows = [assistant("", [("c1", "read", "{}"), ("c2", "read", "{}")]), assistant("", [("c3", "ls", "{}")])]
        text = fixed_summary(rows)
        for heading in HEADINGS:
            self.assertIn(heading + "\n", text)
        self.assertIn("- Tool calls: ls x1", text)
        self.assertIn("- Tool calls: read x2", text)
        self.assertIn("[OPEN] Continue from the latest user message.", text)

    def test_keeps_the_earlier_summary(self):
        # Goals and rules that only the earlier summary has must not be lost when both summary requests fail.
        old = assistant("[CONTEXT SUMMARY]:\n## Goal\nShip the old goal.\n## Next step\nold step\n\n"
                        "--- END OF CONTEXT SUMMARY x", _compressed_summary=True)
        text = fixed_summary([old, user("x")], PREFIXES)
        self.assertIn("> ## Goal\n> Ship the old goal.", text)
        self.assertNotIn("END OF CONTEXT SUMMARY", text)
        self.assertEqual([line for line in text.splitlines() if line == "## Goal"], ["## Goal"])
        self.assertNotIn("earlier summary", fixed_summary([user("x")], PREFIXES))

    def test_all_quotes_share_one_budget(self):
        # The earlier summary, the focus, the memory context, and the cut middles together stay in the budget, and
        # each keeps a part.
        from warm_compaction.layout import CUT_NOTE
        from warm_compaction.rows import estimate_tokens
        old = assistant("[CONTEXT SUMMARY]:\nOLD-GOAL " + "o" * 40_000 + "\n\n--- END OF CONTEXT SUMMARY x ---",
                        _compressed_summary=True)
        rows = [old, user("x"), tool("c1", CUT_NOTE + "CUT-START " + "m" * 40_000)]
        for budget in (4_096, 1_000):
            with self.subTest(budget=budget):
                text = fixed_summary(rows, PREFIXES, "FOCUS-START " + "f" * 40_000, "MEMORY-START " + "y" * 40_000,
                                     max_tokens=budget)
                self.assertLessEqual(estimate_tokens(text), budget)
                for mark in ("OLD-GOAL", "FOCUS-START", "MEMORY-START", "CUT-START"):
                    self.assertIn(mark, text)

    def test_many_tool_names_stay_in_the_budget(self):
        # The tool inventory is in the budget too: the most used names stay, the others become one count.
        from warm_compaction.rows import estimate_tokens
        rows = [assistant("", [(f"c{index}", f"synthetic_tool_name_{index:04d}", "{}")]) for index in range(500)]
        rows.append(assistant("", [("x1", "read", "{}"), ("x2", "read", "{}")]))
        text = fixed_summary(rows, max_tokens=1_000)
        self.assertLessEqual(estimate_tokens(text), 1_000)
        self.assertIn("- Tool calls: read x2", text)
        self.assertIn("- Other tool calls:", text)

    def test_the_cut_quote_block_fits_its_budget_with_its_label(self):
        from warm_compaction.fallback import cut_quote
        from warm_compaction.layout import CUT_NOTE
        from warm_compaction.rows import estimate_tokens
        rows = [user(CUT_NOTE + "m" * 40_000), tool("c1", CUT_NOTE + "t" * 40_000)]
        for budget in (16, 40, 200, 2_000):
            with self.subTest(budget=budget):
                block = cut_quote(rows, budget)
                self.assertLessEqual(estimate_tokens(block), budget)
        self.assertIn("m" * 50, cut_quote(rows, 2_000))

    def test_no_tool_calls(self):
        self.assertIn("- No tool calls.", fixed_summary([user("x")]))


if __name__ == "__main__":
    unittest.main()
