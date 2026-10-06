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
                    user("x", display_kind="hidden"), user("  "), user("[HERMES PREFIX] old"),
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
        self.assertTrue(is_summary(assistant("[CONTEXT SUMMARY]: s"), PREFIXES))
        self.assertTrue(is_summary(user("plain", _compressed_summary=True), PREFIXES))
        self.assertFalse(is_summary(user("plain"), PREFIXES))


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
        self.assertEqual(copied_user_messages(rows, COPY_EACH + 15, PREFIXES), ["b" * COPY_EACH, "c" * 10])

    def test_copies_stay_inside_the_token_budget(self):
        # Six 4,000-character CJK messages are about 24,000 tokens. A 5,000-token budget keeps one.
        rows = [user("\u4f60" * 4_000) for _ in range(6)]
        self.assertEqual(len(copied_user_messages(rows, 24_000, PREFIXES)), 6)
        self.assertEqual(len(copied_user_messages(rows, 24_000, PREFIXES, max_tokens=5_000)), 1)
        self.assertEqual(copied_user_messages(rows, 24_000, PREFIXES, max_tokens=100), [])


class AttachmentAndPrependTest(unittest.TestCase):
    def test_an_image_only_user_row_is_real_and_copied_as_a_mark(self):
        row = user([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])
        self.assertTrue(is_real_user(row, PREFIXES))
        self.assertEqual(copied_user_messages([row], 1_000, PREFIXES), ["[image attachment]"])

    def test_the_prepended_row_keeps_its_name(self):
        rows = [user("do it", name="alice"), assistant("", [("c1", "f", "{}")]), tool("c1", "r")]
        start, prepend = tail_start(rows, 1, PREFIXES)
        self.assertEqual((start, prepend), (1, {"role": "user", "content": "do it", "name": "alice"}))

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
