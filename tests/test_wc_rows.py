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


class SentRowsTest(unittest.TestCase):
    def test_the_estimate_uses_the_fields_that_hermes_sends(self):
        # wire_row: no stored reasoning field, no metadata, reasoning_details only on a route that replays it.
        from warm_compaction.rows import SendPolicy, sent_tokens
        base = {"role": "assistant", "content": "ok"}
        small = sent_tokens(base, SendPolicy(echo=False))
        quiet = SendPolicy(echo=False)
        self.assertEqual(sent_tokens({**base, "reasoning": "r" * 40_000, "timestamp": "t" * 4_000}, quiet), small)
        # reasoning_content only on a route that needs it back (apply_reasoning_content_policy); there a stored
        # reasoning field without tool calls goes as reasoning_content.
        self.assertEqual(sent_tokens({**base, "reasoning_content": "r" * 4_000}, quiet), small)
        self.assertGreater(sent_tokens({**base, "reasoning_content": "r" * 4_000}), small + 900)
        self.assertGreater(sent_tokens({**base, "reasoning": "r" * 4_000}), small + 900)
        details = {**base, "reasoning_details": [{"type": "reasoning.text", "text": "d" * 4_000}]}
        self.assertGreater(sent_tokens(details), small + 900)
        from warm_compaction.rows import SendPolicy
        self.assertEqual(sent_tokens(details, SendPolicy(details=False, echo=False)), small)
        # The private native-assistant carriers are not replayed (warm._replay_details).
        native = {**base, "reasoning_details": [{"type": "anthropic.native_assistant", "text": "n" * 8_000}]}
        self.assertEqual(sent_tokens(native, SendPolicy(echo=False)), small)

    def test_the_estimate_uses_the_tool_call_fields_that_hermes_sends(self):
        # wire_row: id, type, and function name and arguments; the thought signature only for a model that reads it.
        from warm_compaction.rows import SendPolicy, sent_tokens
        call = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        base = {"role": "assistant", "content": "", "tool_calls": [call]}
        small = sent_tokens(base)
        extra = {"google": {"thought_signature": "s" * 8_000}}
        noisy = {**base, "tool_calls": [{**call, "index": 0, "response_meta": "m" * 8_000,
                                         "function": {**call["function"], "trace": "t" * 8_000},
                                         "extra_content": extra}]}
        self.assertEqual(sent_tokens(noisy, SendPolicy(signatures=False)), small)
        self.assertGreater(sent_tokens(noisy, SendPolicy(signatures=True)), small + 1_900)
        self.assertLess(sent_tokens(noisy, SendPolicy(signatures=True)), small + 2_100)
        # extra_content without a usable thought signature is not sent (warm._signature).
        for unsigned in ({"meta": "m" * 8_000}, {"google": {"thought_signature": " "}, "meta": "m" * 8_000}):
            with self.subTest(extra=unsigned):
                row = {**base, "tool_calls": [{**call, "extra_content": unsigned}]}
                self.assertEqual(sent_tokens(row, SendPolicy(signatures=True)), small)

