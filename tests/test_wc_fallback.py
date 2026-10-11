"""Tests for the fallback model summary and its bounded quotes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from warm_compaction.fallback import (
    END_LINE, FALLBACK_INSTRUCTION, MAX_TOKENS, REQUEST_MAX_TOKENS, MIDDLE_MARK, TASK, TOOL_CHARS,
    TRANSCRIPT_CHARS, TRANSCRIPT_TOKENS,
    llm_summary, render_row, transcript,
)
from warm_compaction.rows import estimate_tokens
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
        marker = messages[0]["content"].rstrip().splitlines()[-1]
        text = (self.text(messages, kwargs, len(self.calls)) if callable(self.text)
                else self.text.replace(END_LINE, marker))
        return SimpleNamespace(text=text, usage=usage)


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
    def marker(self, messages):
        return messages[0]["content"].rstrip().splitlines()[-1]

    def test_calls_the_registered_task(self):
        llm = FakeLlm()
        text, tokens = llm_summary(llm, [user("hi"), assistant("ok")], PREFIXES, focus_topic="db", memory_context="m")
        messages, kwargs = llm.calls[0]
        self.assertEqual((text, tokens), (SUMMARY, 321))
        self.assertEqual((kwargs["task"], kwargs["max_tokens"]), (TASK, REQUEST_MAX_TOKENS))
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 120.0)
        self.assertIn("Give more detail to this topic: db", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertRegex(self.marker(messages), r"^\[END OF SUMMARY [0-9a-f]{32}\]$")
        self.assertEqual(len(llm.calls), 1)

    def test_the_fallback_instruction_asks_for_the_set_word_count(self):
        llm = FakeLlm()
        llm_summary(llm, [user("hi"), assistant("ok")], PREFIXES, summary_words=1200)
        self.assertIn("Use at most 1200 words.", llm.calls[0][0][0]["content"])
        self.assertIn("Use at most 600 words.", FALLBACK_INSTRUCTION)

    def test_the_memory_context_counts_against_the_transcript_budget(self):
        from warm_compaction.fallback import EXTRAS_CHARS
        llm = FakeLlm()
        rows = [user("first"), *[assistant(f"row {index} " + "x" * 3_500) for index in range(12)]]
        llm_summary(llm, rows, PREFIXES, memory_context="m" * 200_000 + " the end")
        messages, _kwargs = llm.calls[0]
        content = messages[0]["content"]
        extra = content[content.index("\nAlso keep this context"):content.index("End the summary with this line:")]
        self.assertLessEqual(len(extra), EXTRAS_CHARS)
        self.assertTrue(extra.rstrip().endswith("the end"))
        self.assertLessEqual(len(extra) + len(messages[1]["content"]), TRANSCRIPT_CHARS)
        self.assertLessEqual(estimate_tokens(extra) + estimate_tokens(messages[1]["content"]), TRANSCRIPT_TOKENS + 16)

    def test_reported_output_counts_are_telemetry_only(self):
        for output in (2048, 8192, 7281, 300, None, True, -1, "fake-count", 2 ** 64):
            with self.subTest(output=output):
                self.assertEqual(llm_summary(FakeLlm(output_tokens=output), [user("hi")], PREFIXES), (SUMMARY, 321))

    def test_the_exact_fresh_marker_is_required_once_at_the_end(self):
        cases = (
            (lambda marker: SUMMARY, "missing_end_marker"),
            (lambda marker: SUMMARY + "\n[END OF SUMMARY]", "missing_end_marker"),
            (lambda marker: SUMMARY + "\n[END OF SUMMARY " + "0" * 32 + "]", "missing_end_marker"),
            (lambda marker: SUMMARY + "\n" + marker + "\nmore text", "missing_end_marker"),
            (lambda marker: SUMMARY + "\n" + marker + "\n" + marker, "duplicate_end_marker"),
            (lambda marker: SUMMARY + " embedded " + marker + "\n" + marker, "duplicate_end_marker"),
            (lambda marker: SUMMARY + "\n " + marker + " ", "missing_end_marker"),
        )
        for make, reason in cases:
            with self.subTest(reason=reason), self.assertLogs("warm_compaction.fallback", level="WARNING") as logs:
                llm = FakeLlm(text=lambda messages, kwargs, count, make=make: make(self.marker(messages)))
                failure = {}
                self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, failure=failure), (None, None))
            self.assertEqual(failure["reason"], "gate:" + reason)
            self.assertEqual(failure["attempts"], 2)
            self.assertEqual(len(llm.calls), 2)
            self.assertNotIn(self.marker(llm.calls[-1][0]), str(logs.output) + str(failure))

    def test_thinking_is_removed_before_the_content_and_body_checks(self):
        llm = FakeLlm(text="<think>" + "private-plan" * 4_000 + "</think>\n" + SUMMARY + "\n" + END_LINE + "\n")
        self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES), (SUMMARY, 321))

    def test_oversized_body_is_refused_with_any_usage(self):
        for suffix in ("word " * 2_000, "\u76ee" * 2_200, "\U0001f600" * 2_200):
            for output in (None, 1, 2048, 7281, 8192):
                with self.subTest(output=output, suffix=suffix[:5]):
                    body = SUMMARY + "\n" + suffix
                    failure = {}
                    llm = FakeLlm(text=body + "\n" + END_LINE, output_tokens=output)
                    self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, failure=failure), (None, None))
                    self.assertEqual(failure["reason"], "gate:body_token_limit")
                    self.assertGreater(failure["body_tokens"], MAX_TOKENS)
                    self.assertEqual(len(llm.calls), 2)

    def test_content_checks_stay_in_force(self):
        for text, reason in (("answer", "heading_missing"), (SUMMARY + "\n[CONTEXT SUMMARY]: x", "carrier_marker"),
                             (SUMMARY + "\n" + "x" * 30_000, "byte_bound")):
            with self.subTest(reason=reason):
                failure = {}
                self.assertEqual(llm_summary(FakeLlm(text=text + "\n" + END_LINE), [user("hi")], PREFIXES,
                                             failure=failure), (None, None))
                self.assertEqual(failure["reason"], "gate:" + reason)

    def test_refusal_diagnostics_have_counts_and_fixed_codes_only(self):
        private = "synthetic-private-reply"
        for output in (7281, 814, None, True, -1, "synthetic-private-count", 2 ** 64):
            with self.subTest(output=output), self.assertLogs("warm_compaction.fallback", level="WARNING") as logs:
                failure = {}
                llm = FakeLlm(text=SUMMARY + "\n" + private, output_tokens=output)
                self.assertEqual(llm_summary(llm, [user("synthetic request")], PREFIXES, failure=failure), (None, None))
            safe_output = output if type(output) is int and 0 <= output <= 2 ** 63 - 1 else None
            self.assertEqual(failure, {"reason": "gate:missing_end_marker", "output_tokens": safe_output,
                                      "requested_max_tokens": REQUEST_MAX_TOKENS, "end_marker": "missing",
                                      "attempts": 2, "body_tokens": estimate_tokens(SUMMARY + "\n" + private)})
            logged = "\n".join(logs.output)
            self.assertNotIn(private, logged + str(failure))
            self.assertNotIn(SUMMARY, logged + str(failure))
            self.assertNotIn("synthetic-private-count", logged + str(failure))

    def test_marker_metadata_has_bounded_codes(self):
        cases = ((lambda marker: "<think>fake</think>\n", "empty"),
                 (lambda marker: SUMMARY + "\n" + marker + "\nlater", "not_last"),
                 (lambda marker: SUMMARY + "\n" + marker + "\n" + marker, "duplicate"),
                 (lambda marker: "bad body\n" + marker, "present"))
        for make, expected in cases:
            with self.subTest(expected=expected):
                failure = {}
                llm_summary(FakeLlm(text=lambda messages, kwargs, count, make=make: make(self.marker(messages))),
                            [user("hi")], PREFIXES, failure=failure)
                self.assertEqual(failure["end_marker"], expected)

    def test_retry_has_fresh_nonce_original_transcript_shorter_instruction_and_remaining_timeout(self):
        clock = [10.0]

        def reply(messages, kwargs, count):
            clock[0] += 7.0
            return "synthetic partial reply" if count == 1 else SUMMARY + "\n" + self.marker(messages)

        llm, failure = FakeLlm(text=reply), {"reason": "old"}
        with patch("warm_compaction.fallback.time.monotonic", side_effect=lambda clock=clock: clock[0]):
            self.assertEqual(llm_summary(llm, [user("original synthetic input")], PREFIXES,
                                         summary_words=1200, timeout_s=20.0, failure=failure), (SUMMARY, 321))
        self.assertEqual(failure, {})
        first, second = llm.calls
        self.assertNotEqual(self.marker(first[0]), self.marker(second[0]))
        self.assertEqual(first[0][1], second[0][1])
        self.assertNotIn("synthetic partial reply", str(second))
        self.assertIn("Use at most 1200 words.", first[0][0]["content"])
        self.assertIn("Use at most 300 words.", second[0][0]["content"])
        self.assertEqual((first[1]["timeout"], second[1]["timeout"]), (20.0, 13.0))
        self.assertEqual(second[1]["max_tokens"], REQUEST_MAX_TOKENS)

    def test_retry_word_count_does_not_raise_a_small_user_limit(self):
        llm = FakeLlm(text=SUMMARY)
        llm_summary(llm, [user("hi")], PREFIXES, summary_words=100)
        self.assertIn("Use at most 100 words.", llm.calls[1][0][0]["content"])

    def test_retry_refuses_the_nonce_from_the_previous_attempt(self):
        previous = []

        def reply(messages, kwargs, count):
            marker = self.marker(messages)
            if count == 1:
                previous.append(marker)
                return SUMMARY
            return SUMMARY + "\n" + previous[0]

        failure = {}
        self.assertEqual(llm_summary(FakeLlm(text=reply), [user("hi")], PREFIXES, failure=failure), (None, None))
        self.assertEqual(failure["reason"], "gate:missing_end_marker")

    def test_no_second_call_after_deadline_and_late_valid_reply_is_refused(self):
        for valid in (False, True):
            with self.subTest(valid=valid):
                clock = [0.0]

                def reply(messages, kwargs, count, clock=clock, valid=valid):
                    clock[0] = 21.0
                    return SUMMARY + "\n" + self.marker(messages) if valid else SUMMARY

                llm, failure = FakeLlm(text=reply), {}
                with patch("warm_compaction.fallback.time.monotonic", side_effect=lambda clock=clock: clock[0]):
                    self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, timeout_s=20, failure=failure),
                                     (None, None))
                self.assertEqual(failure, {"reason": "deadline"})
                self.assertEqual(len(llm.calls), 1)

    def test_preparation_time_reduces_the_first_call_budget(self):
        clock = [0.0]
        original = transcript

        def slow_transcript(*args, **kwargs):
            clock[0] = 7.0
            return original(*args, **kwargs)

        llm = FakeLlm()
        with patch("warm_compaction.fallback.time.monotonic", side_effect=lambda: clock[0]), \
                patch("warm_compaction.fallback.transcript", side_effect=slow_transcript):
            self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, timeout_s=20), (SUMMARY, 321))
        self.assertEqual(llm.calls[0][1]["timeout"], 13.0)

    def test_deadline_before_retry_prevents_a_second_call(self):
        llm, failure = FakeLlm(text=SUMMARY), {}
        with patch("warm_compaction.fallback.time.monotonic", side_effect=(0, 0, 1, 20)):
            self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, timeout_s=20, failure=failure),
                                     (None, None))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(failure, {"reason": "deadline"})

    def test_ready_checks_before_every_call_and_after_each_reply(self):
        for states, text, count in (((False,), SUMMARY, 0), ((True, False), SUMMARY, 1),
                                   ((True, True, False), SUMMARY, 1),
                                   ((True, False), SUMMARY + "\n" + END_LINE, 1),
                                   ((True, True, True, False), SUMMARY, 2)):
            with self.subTest(states=states):
                values = iter(states)
                llm, failure = FakeLlm(text=text), {}
                self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES,
                                             ready=lambda values=values: next(values), failure=failure), (None, None))
                self.assertEqual(failure, {"reason": "cancelled"})
                self.assertEqual(len(llm.calls), count)

    def test_task_none_uses_the_main_model_route(self):
        llm = FakeLlm()
        llm_summary(llm, [user("hi")], PREFIXES, task=None)
        self.assertIsNone(llm.calls[0][1]["task"])

    def test_unavailable_and_errors_do_not_retry(self):
        for llm, reason in ((None, "unavailable"), (FakeLlm(error=RuntimeError("fake-private")), "error:RuntimeError")):
            failure = {"reason": "old"}
            self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, failure=failure), (None, None))
            self.assertEqual(failure, {"reason": reason})
            if llm is not None:
                self.assertEqual(len(llm.calls), 1)

    def test_second_call_exception_does_not_make_a_third_call(self):
        def reply(messages, kwargs, count):
            if count == 2:
                raise TimeoutError("fake private exception")
            return SUMMARY
        llm, failure = FakeLlm(text=reply), {}
        self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, failure=failure), (None, None))
        self.assertEqual(failure, {"reason": "error:TimeoutError"})
        self.assertEqual(len(llm.calls), 2)

    def test_a_success_clears_old_failure_metadata(self):
        failure = {"reason": "old"}
        self.assertEqual(llm_summary(FakeLlm(), [user("synthetic request")], PREFIXES, failure=failure), (SUMMARY, 321))
        self.assertEqual(failure, {})

    def test_an_unsafe_exception_class_name_is_not_published(self):
        unsafe = type("synthetic provider body / secret", (Exception,), {})
        failure = {}
        with self.assertLogs("warm_compaction.fallback", level="WARNING") as logs:
            self.assertEqual(llm_summary(FakeLlm(error=unsafe("synthetic private message")),
                                         [user("synthetic request")], PREFIXES, failure=failure), (None, None))
        self.assertEqual(failure, {"reason": "error:Exception"})
        self.assertNotIn("secret", str(logs.output))
        self.assertNotIn("private message", str(logs.output))


class CutQuoteTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
