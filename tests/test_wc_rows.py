"""Tests for the row helpers."""

import unittest
from types import SimpleNamespace

from warm_compaction.rows import (
    attr, compact_json, estimate_tokens, plain_text, reply_text, row_digest, strip_think, tool_calls_of,
    visible_text,
)


class RowsTest(unittest.TestCase):
    def test_visible_text_marks_each_non_text_part(self):
        content = [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                   {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
                   {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                   {"type": "file", "file": {"filename": "report.pdf", "file_data": "AAAA"}}]
        self.assertEqual(visible_text(content), "see\n[image attachment]\n[image attachment: "
                         "https://example.invalid/a.png]\n[audio attachment]\n[file attachment: report.pdf]")
        self.assertEqual(visible_text("plain"), "plain")

    def test_attr_reads_mappings_and_objects(self):
        self.assertEqual(attr({"a": 1}, "a"), 1)
        self.assertEqual(attr(SimpleNamespace(a=2), "a"), 2)
        self.assertIsNone(attr(SimpleNamespace(), "a"))

    def test_tool_calls_of_reads_rows_and_reply_objects(self):
        row = {"tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{\"p\":1}"}}]}
        reply = SimpleNamespace(tool_calls=[SimpleNamespace(id="c2", function=SimpleNamespace(name="ls", arguments="{}"))])
        self.assertEqual(tool_calls_of(row), [("c1", "read", "{\"p\":1}")])
        self.assertEqual(tool_calls_of(reply), [("c2", "ls", "{}")])
        self.assertEqual(tool_calls_of({"role": "user"}), [])

    def test_plain_text_joins_text_parts(self):
        self.assertEqual(plain_text([{"type": "text", "text": "a"}, {"type": "image_url"}, "b"]), "a\nb")
        self.assertEqual(plain_text(None), "")
        self.assertEqual(plain_text("x"), "x")

    def test_strip_think_removes_one_leading_block(self):
        self.assertEqual(strip_think("<think>x</think>\n\nanswer"), "answer")
        self.assertEqual(strip_think("answer <think>x</think>"), "answer <think>x</think>")
        self.assertEqual(reply_text("  <think>a\nb</think> done \n"), "done")

    def test_estimate_tokens_uses_utf8_bytes_of_compact_json(self):
        self.assertEqual(compact_json({"a": [1, 2]}), "{\"a\":[1,2]}")
        self.assertEqual(estimate_tokens("abcd"), 2)
        self.assertEqual(estimate_tokens("\u00e9"), 2)

    def test_estimate_tokens_counts_each_non_ascii_character_as_a_token(self):
        # CJK text and emoji have about one token or more for each character, not one for each 4 bytes.
        self.assertEqual(estimate_tokens("\u4f60" * 100), 101)
        self.assertEqual(estimate_tokens("\U0001f600" * 10), 11)

    def test_row_digest_ignores_private_keys_and_sees_calls(self):
        base = {"role": "assistant", "content": "x",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]}
        marked = dict(base, _db_persisted=True, reasoning="hidden")
        other = dict(base, tool_calls=[{"id": "c2", "function": {"name": "f", "arguments": "{}"}}])
        self.assertEqual(row_digest(base), row_digest(marked))
        self.assertNotEqual(row_digest(base), row_digest(other))
        self.assertEqual(len(row_digest(base)), 64)


if __name__ == "__main__":
    unittest.main()
