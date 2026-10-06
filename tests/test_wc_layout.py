"""Tests for the new history layout."""

import unittest

from warm_compaction.layout import (
    COPY_EACH, HEADER_TEXT, build, copied_user_messages, is_real_user, is_summary, tail_start, units,
)
from wc_fixtures import assistant, tool, user

PREFIXES = ("[HERMES PREFIX]", "[CONTEXT SUMMARY]:")
END = "--- END"


def build_with(rows, start, prepend=None, summary="S"):
    return build(rows, summary, start=start, prepend=prepend, copy_chars=1000, header_prefix="[HERMES PREFIX]",
                 prefixes=PREFIXES, end_marker=END)


class PredicateTest(unittest.TestCase):
    def test_real_user_excludes_scaffolding_and_summaries(self):
        self.assertTrue(is_real_user(user("hi"), PREFIXES))
        self.assertTrue(is_real_user(user("x", display_kind="steer"), PREFIXES))
        for row in (user("x", _todo_snapshot_synthetic=True), user("x", _dropped_toolcall_nudge=True),
                    user("x", display_kind="hidden"), user("  "), user("[HERMES PREFIX]\n\n" + HEADER_TEXT),
                    user("x", _compressed_summary=True), user("[System: Your previous tool call failed"),
                    assistant("hi")):
            with self.subTest(row=row):
                self.assertFalse(is_real_user(row, PREFIXES))

    def test_hermes_recovery_nudges_are_not_real_user_rows(self):
        for text in (
                "You've reached the maximum number of tool-calling iterations allowed. Please provide a final "
                "response summarizing what you've found and accomplished so far, without calling any more tools.",
                "You just executed tool calls but returned an empty response. Please process the tool results above "
                "and continue with the task.",
                "[System: Continue now. Execute the required tool calls and only send your final answer after "
                "completing the task.]",
                "Continue from the compressed conversation context above. This marker exists because no human user "
                "turn was available."):
            with self.subTest(text=text[:30]):
                self.assertFalse(is_real_user(user(text), PREFIXES))
                self.assertFalse(is_real_user(user("  " + text + "\n"), PREFIXES))
        self.assertTrue(is_real_user(user("Continue now."), PREFIXES))

    def test_summary_detection(self):
        # Without the flag (the Hermes session store drops it), only the whole carrier is a summary: a prefix and
        # the end marker, or the plugin header row.
        self.assertTrue(is_summary(assistant("[CONTEXT SUMMARY]: s\n\n--- END OF CONTEXT SUMMARY x"), PREFIXES))
        self.assertTrue(is_summary(user("[HERMES PREFIX]\n\n" + HEADER_TEXT), PREFIXES))
        self.assertTrue(is_summary(user("plain", _compressed_summary=True), PREFIXES))
        self.assertFalse(is_summary(user("plain"), PREFIXES))

    def test_a_user_message_that_starts_with_a_prefix_is_real(self):
        # "Analyze this summary" pasted by the user is the active request, not an earlier summary.
        row = user("[CONTEXT SUMMARY]: please check this text for errors.")
        self.assertFalse(is_summary(row, PREFIXES))
        self.assertTrue(is_real_user(row, PREFIXES))


class TailTest(unittest.TestCase):
    def test_units_keep_tool_groups_whole(self):
        rows = [user("u"), assistant("", [("c1", "f", "{}"), ("c2", "f", "{}")]), tool("c1", "a"), tool("c2", "b"),
                assistant("done")]
        self.assertEqual(units(rows), [(0, 1), (1, 4), (4, 5)])

    def test_tail_starts_at_the_first_real_user_row_in_the_budget(self):
        rows = [user("old " * 50), assistant("a " * 50), user("new"), assistant("b")]
        self.assertEqual(tail_start(rows, 30, PREFIXES), (2, None))

    def test_mid_task_tail_gets_a_copy_of_the_latest_user_row(self):
        rows = [user("first"), assistant("ok"), user("do it"), assistant("", [("c1", "f", "{}")]),
                tool("c1", "x" * 4000)]
        self.assertEqual(tail_start(rows, 10, PREFIXES), (3, {"role": "user", "content": "do it"}))

    def test_no_row_before_the_tail(self):
        self.assertEqual(tail_start([user("only")], 10_000, PREFIXES), (0, None))


class CopyTest(unittest.TestCase):
    def test_copies_the_newest_messages_in_order_and_cuts_each(self):
        rows = [user("a" * 10), user("b" * (COPY_EACH + 50)), assistant("x"), user("c" * 10)]
        copies = copied_user_messages(rows, COPY_EACH + 15, PREFIXES)
        self.assertEqual((len(copies[0]), copies[1]), (COPY_EACH, "c" * 10))

    def test_a_long_copy_keeps_its_start_and_its_end(self):
        text = "start " + "x" * 5_000 + " the real question"
        [copy] = copied_user_messages([user(text)], 24_000, PREFIXES)
        self.assertTrue(copy.startswith("start ") and copy.endswith(" the real question"))
        self.assertLessEqual(len(copy), COPY_EACH)

    def test_copies_stay_inside_the_token_budget(self):
        # Six 4,000-character CJK messages are about 24,000 tokens. A 5,000-token budget keeps one.
        rows = [user("\u4f60" * 4_000) for _ in range(6)]
        self.assertEqual(len(copied_user_messages(rows, 24_000, PREFIXES)), 6)
        # The second copy is cut to the remaining 1,000 tokens or less.
        copies = copied_user_messages(rows, 24_000, PREFIXES, max_tokens=5_000)
        self.assertEqual(len(copies), 2)
        self.assertLess(len(copies[0]), COPY_EACH)
        # A remainder below MIN_COPY_CHARS characters gives no copy.
        self.assertEqual(copied_user_messages(rows, 24_000, PREFIXES, max_tokens=100), [])

    def test_a_message_larger_than_the_budget_is_cut_to_the_budget(self):
        from warm_compaction.layout import MIN_COPY_CHARS, quote
        from warm_compaction.rows import estimate_tokens
        text = "start " + "x" * 2_000 + " the real question"
        [copy] = copied_user_messages([user(text)], 1_000, PREFIXES)
        self.assertEqual(len(copy), 1_000)
        self.assertTrue(copy.startswith("start ") and copy.endswith(" the real question"))
        [copy] = copied_user_messages([user(text)], 24_000, PREFIXES, max_tokens=150)
        self.assertLessEqual(estimate_tokens(quote(copy)), 150)
        self.assertTrue(copy.endswith(" the real question"))
        self.assertEqual(copied_user_messages([user(text)], MIN_COPY_CHARS - 1, PREFIXES), [])


class CopyFormTest(unittest.TestCase):
    def test_copies_use_the_sent_api_content(self):
        self.assertEqual(copied_user_messages([user("hi", api_content="[ctx]\n\nhi")], 1_000, PREFIXES),
                         ["[ctx]\n\nhi"])

    def test_the_token_budget_counts_the_quote_marks(self):
        from warm_compaction.layout import quote
        from warm_compaction.rows import estimate_tokens
        text = "a\n" * 1_000
        rows = [user(text)]
        # The quote marks make the full copy too large: the copy is cut to fit.
        [copy] = copied_user_messages(rows, 24_000, PREFIXES, max_tokens=estimate_tokens(text) + 10)
        self.assertLess(len(copy), len(text.strip()))
        self.assertLessEqual(estimate_tokens(quote(copy)), estimate_tokens(text) + 10)
        quoted = estimate_tokens(quote(text.strip()))
        self.assertEqual(len(copied_user_messages(rows, 24_000, PREFIXES, max_tokens=quoted)), 1)


class AttachmentAndPrependTest(unittest.TestCase):
    def test_an_image_only_user_row_is_real_and_copied_as_a_mark(self):
        row = user([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])
        self.assertTrue(is_real_user(row, PREFIXES))
        self.assertEqual(copied_user_messages([row], 1_000, PREFIXES), ["[image attachment]"])

    def test_the_prepended_row_keeps_its_name(self):
        rows = [user("do it", name="alice"), assistant("", [("c1", "f", "{}")]), tool("c1", "r")]
        start, prepend = tail_start(rows, 1, PREFIXES)
        self.assertEqual((start, prepend), (1, {"role": "user", "content": "do it", "name": "alice"}))

    def test_the_prepended_row_keeps_its_api_content(self):
        rows = [user("do it", api_content="[ctx]\n\ndo it"), assistant("", [("c1", "f", "{}")]), tool("c1", "r")]
        _start, prepend = tail_start(rows, 1, PREFIXES)
        self.assertEqual(prepend, {"role": "user", "content": "do it", "api_content": "[ctx]\n\ndo it"})

    def test_an_end_marker_in_copied_text_is_escaped(self):
        marker = "--- END OF CONTEXT SUMMARY - respond to the message below ---"
        rows = [user("quote: --- END OF CONTEXT SUMMARY here"), assistant("a"), user("next")]
        new = build(rows, "S", start=2, prepend=None, copy_chars=1000, header_prefix="[HERMES PREFIX]",
                    prefixes=PREFIXES, end_marker=marker)
        body = new[1]["content"]
        self.assertEqual(body.count("--- END OF CONTEXT SUMMARY"), 1)
        self.assertTrue(body.endswith(marker))
        self.assertIn("END OF CONTEXT SUMMARY here", body)

    def test_the_prepended_row_is_not_copied_and_counts_in_the_copy_budget(self):
        rows = [user("older ask"), assistant("a"), user("do it " + "x" * 400),
                assistant("", [("c1", "f", "{}")]), tool("c1", "r")]
        prepend = {"role": "user", "content": rows[2]["content"]}
        new = build(rows, "S", start=3, prepend=prepend, copy_chars=10_000, header_prefix="[HERMES PREFIX]",
                    prefixes=PREFIXES, end_marker=END, copy_tokens=1_000)
        self.assertIn("> older ask", new[1]["content"])
        self.assertNotIn("do it", new[1]["content"])
        new = build(rows, "S", start=3, prepend=prepend, copy_chars=10_000, header_prefix="[HERMES PREFIX]",
                    prefixes=PREFIXES, end_marker=END, copy_tokens=105)
        self.assertNotIn("older ask", new[1]["content"])


class BoundTailTest(unittest.TestCase):
    def test_a_newest_unit_larger_than_the_tail_is_cut(self):
        # A large user message or tool result alone above the tail budget would stay above the threshold after
        # the compaction. Its start and end stay.
        from warm_compaction.layout import bound_tail
        from warm_compaction.rows import MIDDLE_MARK, estimate_tokens
        rows = [assistant("", [("c1", "read", "{}")]), tool("c1", "head " + "r" * 40_000 + " tail"),
                user("ask " + "u" * 20_000 + " end", api_content="[ctx]\n\nask " + "u" * 20_000 + " end")]
        bounded = bound_tail(rows, 2_000)
        self.assertLessEqual(sum(estimate_tokens(row) for row in bounded), 2_000)
        self.assertEqual(bounded[0], rows[0])
        self.assertTrue(bounded[1]["content"].startswith("head ") and bounded[1]["content"].endswith(" tail"))
        self.assertIn(MIDDLE_MARK, bounded[1]["content"])
        self.assertTrue(bounded[2]["content"].startswith("[ctx]") and bounded[2]["content"].endswith(" end"))
        self.assertNotIn("api_content", bounded[2])
        self.assertIs(bound_tail(rows[:1], 2_000)[0], rows[0])

    def test_the_tail_is_sized_as_hermes_sends_it(self):
        # Hermes sends api_content in place of content: a row with both fields costs its sent text only.
        from warm_compaction.layout import bound_tail
        text = "t" * 4_000
        rows = [user("o" * 4_000), assistant("a"), user(text, api_content=text + " [ctx]"), assistant("b")]
        self.assertEqual(tail_start(rows, 1_500, PREFIXES), (2, None))
        self.assertIs(bound_tail(rows[2:], 1_500)[0], rows[2])

    def test_unsent_reasoning_does_not_move_the_tail(self):
        rows = [user("o" * 4_000), assistant("a"), user("q"), assistant("b", reasoning="r" * 40_000)]
        self.assertEqual(tail_start(rows, 900, PREFIXES), (2, None))

    def test_a_large_assistant_row_is_cut_and_keeps_its_tool_calls(self):
        from warm_compaction.layout import bound_tail
        from warm_compaction.rows import MIDDLE_MARK, estimate_tokens
        rows = [user("ask"), assistant("start " + "a" * 40_000 + " end", [("c1", "read", "{}")]), tool("c1", "r")]
        bounded = bound_tail(rows, 2_000)
        self.assertLessEqual(sum(estimate_tokens(row) for row in bounded), 2_000)
        self.assertEqual(bounded[1]["tool_calls"], rows[1]["tool_calls"])
        self.assertTrue(bounded[1]["content"].startswith("start ") and bounded[1]["content"].endswith(" end"))
        self.assertIn(MIDDLE_MARK, bounded[1]["content"])

    def test_large_tool_call_arguments_are_cut_to_valid_json(self):
        import json
        from warm_compaction.layout import CUT_NOTE, bound_tail
        from warm_compaction.rows import estimate_tokens
        arguments = json.dumps({"path": "a.txt", "text": "start " + "w" * 40_000 + " end"})
        rows = [user("write"), assistant("", [("c1", "write", arguments)]), tool("c1", "ok")]
        removed = []
        bounded = bound_tail(rows, 2_000, removed)
        self.assertLessEqual(sum(estimate_tokens(row) for row in bounded), 2_000)
        call = bounded[1]["tool_calls"][0]
        self.assertEqual((call["id"], call["function"]["name"]), ("c1", "write"))
        kept = json.loads(call["function"]["arguments"])["truncated_arguments"]
        self.assertTrue(kept.startswith('{"path": "a.txt"') and kept.endswith(' end"}'))
        self.assertEqual(rows[1]["tool_calls"][0]["function"]["arguments"], arguments)
        self.assertEqual(len(removed), 1)
        self.assertTrue(removed[0]["content"].startswith(CUT_NOTE) and "w" * 100 in removed[0]["content"])

    def test_large_reasoning_text_is_cut(self):
        from warm_compaction.layout import bound_tail
        from warm_compaction.rows import MIDDLE_MARK, estimate_tokens
        rows = [user("think"), assistant("ok", reasoning_content="r-start " + "t" * 40_000 + " r-end")]
        bounded = bound_tail(rows, 2_000)
        self.assertLessEqual(sum(estimate_tokens(row) for row in bounded), 2_000)
        self.assertIn(MIDDLE_MARK, bounded[1]["reasoning_content"])
        self.assertTrue(bounded[1]["reasoning_content"].endswith(" r-end"))

    def test_list_content_is_cut_and_a_large_image_is_replaced(self):
        from warm_compaction.layout import bound_tail
        from warm_compaction.rows import estimate_tokens
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 100_000}}
        rows = [user([{"type": "text", "text": "look " + "l" * 20_000 + " end"}, image]), assistant("ok")]
        bounded = bound_tail(rows, 2_000)
        self.assertLessEqual(sum(estimate_tokens(row) for row in bounded), 2_000)
        text, note = bounded[0]["content"]
        self.assertTrue(text["text"].startswith("look ") and text["text"].endswith(" end"))
        self.assertEqual(note, {"type": "text", "text": "[image_url removed]"})
        self.assertIs(rows[0]["content"][1], image)

    def test_raw_string_parts_and_untyped_text_parts_are_cut_as_text(self):
        # visible_text reads both shapes as text: the bound cuts them, and does not replace them with a note.
        from warm_compaction.layout import CUT_NOTE, bound_tail
        from warm_compaction.rows import estimate_tokens
        rows = [user(["raw " + "a" * 20_000 + " raw-end", {"text": "untyped " + "b" * 20_000 + " untyped-end"}]),
                assistant("ok")]
        removed = []
        bounded = bound_tail(rows, 2_000, removed)
        self.assertLessEqual(sum(estimate_tokens(row) for row in bounded), 2_000)
        raw, untyped = bounded[0]["content"]
        self.assertTrue(isinstance(raw, str) and raw.startswith("raw ") and raw.endswith(" raw-end"))
        self.assertEqual(set(untyped), {"text"})
        self.assertTrue(untyped["text"].startswith("untyped ") and untyped["text"].endswith(" untyped-end"))
        self.assertEqual(len(removed), 2)
        self.assertTrue(all(part["content"].startswith(CUT_NOTE) for part in removed))

    def test_a_cap_below_the_minimum_cut_drops_the_payloads(self):
        # Short rows, and many small parts: the minimum cut is not enough, so the payloads are dropped (the
        # fallback gets them whole). Only the row structure stays.
        from warm_compaction.layout import CUT_NOTE, DROPPED, bound_tail
        from warm_compaction.rows import estimate_tokens
        for rows, tokens, whole in (([user("x" * 150), assistant("y" * 150)], 20, "x" * 150),
                                    ([user(["p" * 100] * 20), assistant("ok")], 100, "p" * 100)):
            with self.subTest(tokens=tokens):
                removed = []
                bounded = bound_tail(rows, tokens, removed)
                self.assertLessEqual(sum(estimate_tokens(row) for row in bounded), tokens)
                self.assertIn(CUT_NOTE + whole, [part["content"] for part in removed])
                self.assertTrue(any(DROPPED in str(row["content"]) for row in bounded))

    def test_the_cut_middles_are_given_back(self):
        # The fallback summary gets the parts that the tail cuts: they are not lost.
        from warm_compaction.layout import CUT_NOTE, bound_tail
        from warm_compaction.rows import MIDDLE_MARK
        text = "head " + "".join(f"{index:06d}" for index in range(8_000)) + " tail"
        rows = [assistant("", [("c1", "read", "{}")]), tool("c1", text), user("short")]
        removed = []
        bounded = bound_tail(rows, 1_000, removed)
        self.assertEqual(len(removed), 1)
        self.assertEqual((removed[0]["role"], removed[0]["tool_call_id"]), ("tool", "c1"))
        self.assertTrue(removed[0]["content"].startswith(CUT_NOTE))
        head, end = bounded[1]["content"].split(MIDDLE_MARK)
        self.assertEqual(head + removed[0]["content"][len(CUT_NOTE):] + end, text)
        self.assertEqual(bound_tail(rows[2:], 1_000, removed), rows[2:])
        self.assertEqual(len(removed), 1)

    def test_build_bounds_the_tail(self):
        from warm_compaction.rows import estimate_tokens
        rows = [user("old"), assistant("a"), user("huge " + "z" * 40_000 + " end")]
        new = build(rows, "S", start=2, prepend=None, copy_chars=1000, header_prefix="[HERMES PREFIX]",
                    prefixes=PREFIXES, end_marker=END, tail_tokens=1_000)
        self.assertLessEqual(estimate_tokens(new[-1]), 1_000)
        self.assertTrue(new[-1]["content"].endswith(" end"))


class BuildTest(unittest.TestCase):
    def test_two_summary_rows_then_the_tail_without_the_marker(self):
        rows = [user("goal"), assistant("a1"), user("next", _db_persisted=True), assistant("a2", _db_persisted=True)]
        new = build_with(rows, 2, summary="## Goal\nG")
        self.assertEqual([row["role"] for row in new], ["user", "assistant", "user", "assistant"])
        self.assertEqual(new[0]["content"], "[HERMES PREFIX]\n\n" + HEADER_TEXT)
        self.assertTrue(new[1]["content"].startswith(
            "[CONTEXT SUMMARY]:\n## Goal\nG\n\n## Copied user messages\n\n> goal"))
        self.assertTrue(new[1]["content"].endswith("\n\n" + END))
        self.assertTrue(new[0]["_compressed_summary"] and new[1]["_compressed_summary"])
        self.assertNotIn("_db_persisted", new[2])
        self.assertIn("_db_persisted", rows[2])

    def test_prepend_row_follows_the_summary(self):
        rows = [user("do it"), assistant("", [("c1", "f", "{}")]), tool("c1", "r")]
        new = build_with(rows, 1, prepend={"role": "user", "content": "do it"})
        self.assertEqual([row["role"] for row in new], ["user", "assistant", "user", "assistant", "tool"])

    def test_one_summary_row_when_the_tail_starts_without_a_user_row(self):
        rows = [assistant("a0"), assistant("", [("c1", "f", "{}")]), tool("c1", "r")]
        new = build_with(rows, 1)
        self.assertEqual([row["role"] for row in new], ["user", "assistant", "tool"])
        self.assertIn(HEADER_TEXT, new[0]["content"])
        self.assertIn("[CONTEXT SUMMARY]:", new[0]["content"])
        self.assertIn("(none)", new[0]["content"])

    def test_returns_none_without_rows_before_the_tail(self):
        self.assertIsNone(build_with([user("x")], 0))


if __name__ == "__main__":
    unittest.main()
