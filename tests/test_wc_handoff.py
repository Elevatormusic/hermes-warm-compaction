"""Tests for the handoff instruction and the reply gate."""

import unittest

from warm_compaction.handoff import HEADINGS, INSTRUCTION, MAX_REPLY_BYTES, build_instruction, gate
from wc_fixtures import HEADINGS_TEXT


def reply(content=HEADINGS_TEXT, **extra):
    return {"content": content, "finish_reason": "stop", "tool_calls": False, "refusal": False, **extra}


class HandoffTest(unittest.TestCase):
    def test_instruction_starts_with_the_stop_line_and_lists_the_headings(self):
        self.assertTrue(INSTRUCTION.startswith("Stop the current task now. This request comes from the host program"))
        for heading in HEADINGS:
            self.assertIn("\n" + heading + "\n", INSTRUCTION)

    def test_build_instruction_adds_focus_and_memory(self):
        text = build_instruction("the parser", "remember X")
        self.assertIn("\nGive more detail to this topic: the parser\n", text)
        self.assertIn("\nAlso keep this context from the memory provider:\nremember X\n", text)
        self.assertEqual(build_instruction(None, ""), INSTRUCTION)

    def test_gate_accepts_a_complete_handoff_after_a_think_block(self):
        text, reason = gate(reply("<think>plan</think>\n" + HEADINGS_TEXT))
        self.assertIsNone(reason)
        self.assertTrue(text.startswith("## Goal"))

    def test_gate_reasons(self):
        cases = {
            "finish_not_stop": reply(finish_reason="length"),
            "tool_call": reply(tool_calls=True),
            "refusal": reply(refusal=True),
            "content_required": reply("<think>only</think>  "),
            "byte_bound": reply(HEADINGS_TEXT + "x" * MAX_REPLY_BYTES),
            "carrier_marker": reply("[CONTEXT SUMMARY]:\n" + HEADINGS_TEXT),
            "heading_missing": reply(HEADINGS_TEXT.replace("## Key facts", "Key facts")),
            "heading_order": reply("## Next step\nx\n## Goal\ng\n## User instructions\n## Current state\n- s\n"
                                   "## Key facts\n"),
            "section_empty": reply("## Goal\n## User instructions\n## Current state\n## Key facts\n## Next step"),
            # Text before the first heading can be an answer or an action that did not occur, not a summary.
            "heading_preamble": reply("I ran the tests and they pass.\n\n" + HEADINGS_TEXT),
            "heading_repeated": reply("## Goal\n## Goal\n## User instructions\n## Current state\n## Current state\n"
                                      "## Key facts\n## Next step\n## Next step"),
        }
        for expected, value in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(gate(value), (None, expected))

    def test_a_leading_think_block_is_not_a_preamble(self):
        text, reason = gate(reply("<think>plan</think>\n\n" + HEADINGS_TEXT))
        self.assertIsNone(reason)
        self.assertTrue(text.startswith("## Goal"))

    def test_gate_needs_text_in_goal_state_and_next_step(self):
        sections = {"## Goal": "g", "## User instructions": "- u", "## Current state": "- [OPEN] s",
                    "## Key facts": "- k", "## Next step": "n"}
        for heading in ("## Goal", "## Current state", "## Next step"):
            text = "\n".join(f"{name}\n{'' if name == heading else body}" for name, body in sections.items())
            with self.subTest(heading=heading):
                self.assertEqual(gate(reply(text)), (None, "section_empty"))
        # Empty User instructions and Key facts are allowed: there can be none.
        text = "## Goal\ng\n## User instructions\n## Current state\n- [OPEN] s\n## Key facts\n## Next step\nn"
        self.assertEqual(gate(reply(text)), (text, None))

    def test_gate_refuses_a_hermes_summary_prefix(self):
        self.assertEqual(gate(reply("[HERMES] x\n" + HEADINGS_TEXT), ("[HERMES]",)), (None, "carrier_marker"))


if __name__ == "__main__":
    unittest.main()
