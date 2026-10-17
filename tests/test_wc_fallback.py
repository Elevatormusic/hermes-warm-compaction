"""Tests for the fallback model summary and its bounded quotes."""

import itertools
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from warm_compaction.fallback import (
    END_LINE, FALLBACK_INSTRUCTION, MAX_STATE_TOKENS, MAX_TOKENS, REQUEST_MAX_TOKENS, MIDDLE_MARK, TASK, TOOL_CHARS,
    TIMEOUT_S, TRANSCRIPT_CHARS, TRANSCRIPT_TOKENS,
    llm_summary, render_row, transcript,
)
from warm_compaction.rows import estimate_tokens
from wc_fixtures import assistant, tool, user

PREFIXES = ("[HERMES PREFIX]", "[CONTEXT SUMMARY]:")


SUMMARY = ("## Goal\n- goal unknown\n## User instructions\n- instructions unknown\n"
           "## Current state\n- progress unknown\n## Key facts\n- facts unknown\n## Next step\n- next step unknown")


class FakeLlm:
    def __init__(self, text=None, error=None, output_tokens=None):
        self.text, self.error, self.calls, self.output_tokens = text, error, [], output_tokens

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        usage = SimpleNamespace(input_tokens=321)
        if self.output_tokens is not None:
            usage.output_tokens = self.output_tokens
        marker = messages[0]["content"].rstrip().splitlines()[-1]
        if self.text is None:
            lines = messages[1]["content"].split("\n")
            value = next(line for index, line in enumerate(lines) if index and line.strip() and "[cut]" not in line
                         and not line.startswith("(tool call ")
                         and lines[index - 1].startswith(
                             ("[first user message]", "[user", "[assistant", "[tool result"))
                         and estimate_tokens("- " + json.dumps(line) + "\n- facts unknown") <= MAX_STATE_TOKENS)
            text = useful_summary(value) + "\n" + marker
        else:
            text = (self.text(messages, kwargs, len(self.calls)) if callable(self.text)
                    else self.text.replace(END_LINE, marker))
        return SimpleNamespace(text=text, usage=usage)


def useful_summary(value):
    return SUMMARY.replace("- facts unknown", "- " + json.dumps(value))


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
        self.assertEqual((text, tokens), (useful_summary("hi"), 321))
        self.assertEqual((kwargs["task"], kwargs["max_tokens"]), (TASK, REQUEST_MAX_TOKENS))
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 120.0)
        self.assertIn("Give more detail to this topic: db", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertRegex(self.marker(messages), r"^\[END OF SUMMARY [0-9a-f]{32}\]$")
        self.assertEqual(len(llm.calls), 1)

    def test_request_timeout_does_not_exceed_the_configured_limit(self):
        llm = FakeLlm()
        with patch("warm_compaction.fallback.time.monotonic", side_effect=(0.0, -3e-14, 0.0)):
            self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES), (useful_summary("hi"), 321))
        self.assertEqual(llm.calls[0][1]["timeout"], TIMEOUT_S)

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
        for output in (2048, 8192, 7281, 300, 0, None, True, -1, "fake-count", 2 ** 64):
            with self.subTest(output=output):
                self.assertEqual(llm_summary(FakeLlm(output_tokens=output), [user("hi")], PREFIXES),
                                 (useful_summary("hi"), 321))

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
            self.assertEqual(failure["attempts"], 3)
            self.assertEqual(len(llm.calls), 3)
            self.assertNotIn(self.marker(llm.calls[-1][0]), str(logs.output) + str(failure))

    def test_missing_marker_retry_repeats_the_marker_requirement(self):
        def reply(messages, kwargs, count):
            body = useful_summary("hi")
            return body if count == 1 else body + "\n" + self.marker(messages)
        llm, failure = FakeLlm(text=reply), {}
        self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, failure=failure), (useful_summary("hi"), 321))
        self.assertEqual(len(llm.calls), 2)
        first, second = llm.calls
        self.assertNotIn("did not end with the marker line", first[0][0]["content"])
        self.assertIn("did not end with the marker line", second[0][0]["content"])
        self.assertEqual(failure, {})

    def test_third_attempt_passes_after_two_missing_markers(self):
        def reply(messages, kwargs, count):
            body = useful_summary("hi")
            return body + ("\n" + self.marker(messages) if count == 3 else "")
        llm, failure = FakeLlm(text=reply), {}
        self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, failure=failure), (useful_summary("hi"), 321))
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(failure, {})

    def test_thinking_is_removed_before_the_content_and_body_checks(self):
        body = useful_summary("hi")
        llm = FakeLlm(text="<think>" + "private-plan" * 4_000 + "</think>\n" + body + "\n" + END_LINE + "\n")
        self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES), (body, 321))

    def test_oversized_body_is_refused_with_any_usage(self):
        for suffix in ("word " * 2_000, "\u76ee" * 2_200, "\U0001f600" * 2_200):
            for output in (None, 0, 1, 2048, 7281, 8192):
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
        for output in (7281, 814, 0, None, True, -1, "synthetic-private-count", 2 ** 64):
            with self.subTest(output=output), self.assertLogs("warm_compaction.fallback", level="WARNING") as logs:
                failure = {}
                llm = FakeLlm(text=SUMMARY + "\n" + private, output_tokens=output)
                self.assertEqual(llm_summary(llm, [user("synthetic request")], PREFIXES, failure=failure), (None, None))
            safe_output = output if type(output) is int and 0 < output <= 2 ** 63 - 1 else None
            self.assertEqual(failure, {"reason": "gate:missing_end_marker", "output_tokens": safe_output,
                                      "requested_max_tokens": REQUEST_MAX_TOKENS, "end_marker": "missing",
                                      "attempts": 3, "body_tokens": estimate_tokens(SUMMARY + "\n" + private)})
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
            return ("synthetic partial reply" if count == 1 else
                    useful_summary("original synthetic input") + "\n" + self.marker(messages))

        llm, failure = FakeLlm(text=reply), {"reason": "old"}
        with patch("warm_compaction.fallback.time.monotonic", side_effect=lambda clock=clock: clock[0]):
            self.assertEqual(llm_summary(llm, [user("original synthetic input")], PREFIXES,
                                         summary_words=1200, timeout_s=20.0, failure=failure),
                             (useful_summary("original synthetic input"), 321))
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
            self.assertEqual(llm_summary(llm, [user("hi")], PREFIXES, timeout_s=20), (useful_summary("hi"), 321))
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
        self.assertEqual(llm_summary(FakeLlm(), [user("synthetic request")], PREFIXES, failure=failure),
                         (useful_summary("synthetic request"), 321))
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


def state_summary(state):
    """Put a synthetic Current state section in the test summary."""
    head, rest = SUMMARY.split("## Current state\n", 1)
    _old, facts = rest.split("\n## Key facts\n", 1)
    return head + "## Current state\n" + state + "\n## Key facts\n" + facts


class CurrentStateSourceTest(unittest.TestCase):
    def assert_repaired(self, state, rows, expected="- progress unknown"):
        failure = {}
        llm = FakeLlm(text=state_summary(state) + "\n" + END_LINE)
        body = state_summary(expected)
        result = llm_summary(llm, rows, PREFIXES, failure=failure)
        if expected == "- progress unknown":
            self.assertEqual(result, (None, None))
            self.assertEqual(failure["reason"], "gate:no_useful_evidence")
            self.assertEqual(len(llm.calls), 2)
        else:
            self.assertEqual(result, (body, 321))
            self.assertEqual(failure, {})
            self.assertEqual(len(llm.calls), 1)
        return llm, failure

    def test_accepted_alternative_heading_separators_do_not_raise_during_construction(self):
        from warm_compaction.fallback import _construct_current_state
        from warm_compaction.handoff import gate_content
        for separator in ("\r", "\r\n", "\u2028"):
            with self.subTest(separator=separator):
                body = state_summary("- progress unknown").replace("\n", separator)
                text, reason = gate_content(body, PREFIXES)
                self.assertIsNone(reason)
                self.assertIsNotNone(text)
                repaired = _construct_current_state(text, "[user]\nCheck progress.", PREFIXES)
                self.assertEqual(gate_content(repaired, PREFIXES), (text, None))

    def test_unreported_japanese_incomplete_state_is_not_retained_as_evidence(self):
        state = "- " + json.dumps("PKG-17 \u306f\u672a\u5b8c\u4e86\u3067\u3059\u3002")
        self.assert_repaired(state, [user("PKG-17 \u3092\u78ba\u8a8d\u3057\u3066\u304f\u3060\u3055\u3044\u3002")])

    def test_completion_fragment_restores_the_whole_negative_source_line(self):
        self.assert_repaired('- "complete"', [assistant("not complete")], '- "not complete"')

    def test_exact_reported_unfinished_state_is_accepted(self):
        reported = "PKG-17 \u306f\u672a\u5b8c\u4e86\u3067\u3059\u3002"
        body = state_summary("- " + json.dumps(reported))
        self.assertEqual(llm_summary(FakeLlm(text=body + "\n" + END_LINE), [assistant(reported)], PREFIXES),
                         (body, 321))

    def test_unknown_state_needs_no_reported_progress(self):
        body = SUMMARY.replace("- goal unknown", '- "Check PKG-17."')
        self.assertEqual(llm_summary(FakeLlm(text=body + "\n" + END_LINE), [user("Check PKG-17.")], PREFIXES),
                         (body, 321))

    def test_calibration_result_keeps_its_scope_without_a_parent_completion_label(self):
        reported = "Calibration call_read_7 is complete."
        rows = [user("Verify package PKG-17."), tool("call_read_7", reported)]
        body = state_summary("- " + json.dumps(reported))
        self.assertEqual(llm_summary(FakeLlm(text=body + "\n" + END_LINE), rows, PREFIXES), (body, 321))
        self.assert_repaired('- [COMPLETE] Verify package PKG-17: ' + json.dumps(reported), rows)
        self.assert_repaired("- " + json.dumps("Verify package PKG-17 is complete."), rows)

    def test_json_escapes_and_multiline_source_quotes_are_kept_encoded(self):
        reports = ('The result is "unfinished".', "C:\\synthetic\\report is unfinished.",
                   "State one.\nState two.", "PKG-17 \u306f\u672a\u5b8c\u4e86\u3067\u3059\u3002", "A\tB")
        for reported in reports:
            for ascii_only in (True, False):
                with self.subTest(reported=reported, ascii_only=ascii_only):
                    body = state_summary("- " + json.dumps(reported, ensure_ascii=ascii_only))
                    self.assertEqual(llm_summary(FakeLlm(text=body + "\n" + END_LINE),
                                                 [assistant(reported)], PREFIXES), (body, 321))

    def test_json_round_trip_preserves_complete_source_lines_across_escape_pairs(self):
        # Include escapes next to one another, with fixed text at each source boundary.
        for pair in itertools.product(('a', '"', "\\", "\n", "\t", "\u65e5", "\U0001f600"), repeat=2):
            reported = "State: " + "".join(pair) + ": end."
            for ascii_only in (True, False):
                with self.subTest(pair=pair, ascii_only=ascii_only):
                    body = state_summary("- " + json.dumps(reported, ensure_ascii=ascii_only))
                    self.assertEqual(llm_summary(FakeLlm(text=body + "\n" + END_LINE),
                                                 [assistant(reported)], PREFIXES), (body, 321))

    def test_malformed_extra_prose_and_nonstring_values_become_unknown(self):
        for state in ('- "reported" later', '- [OPEN] "reported"', '- "reported', '- null', '- true', '- 1',
                      '- []', '- {}', '- ""', '- "   "', 'reported', '- progress unknown later',
                      '- ' + '[' * 1100 + '0' + ']' * 1100, '- ' + '1' * 5000):
            with self.subTest(state=state[:30]):
                self.assert_repaired(state, [assistant("reported")])

    def test_unique_partial_quotes_expand_to_complete_source_lines(self):
        for reported, quote in (("not complete", "complete"), ("complete later", "complete"),
                                ("scope complete later", "scope complete")):
            with self.subTest(reported=reported, quote=quote):
                self.assert_repaired("- " + json.dumps(quote), [assistant(reported)], "- " + json.dumps(reported))
        self.assert_repaired('- "first\\nsecond"', [assistant("first\nsecond line")])

    def test_cut_source_and_cut_markers_become_unknown(self):
        for reported in ("reported [cut] state", "reported" + MIDDLE_MARK + "state"):
            with self.subTest(reported=reported):
                self.assert_repaired("- " + json.dumps(reported), [assistant(reported)])
        rows = [user("Check progress."), assistant("start " + "p" * 15_000 + "\nMiddle state.\n" + "q" * 15_000)]
        llm, _failure = self.assert_repaired('- "Middle state."', rows)
        self.assertNotIn("Middle state.", llm.calls[0][0][1]["content"])

    def test_blank_lines_and_multiple_supported_bullets_are_accepted(self):
        rows = [assistant("First state.\nSecond state.")]
        body = state_summary('\n- "First state."\n\n- "Second state."\n- progress unknown\n')
        self.assertEqual(llm_summary(FakeLlm(text=body + "\n" + END_LINE), rows, PREFIXES), (body, 321))

    def test_ambiguous_case_changed_and_paraphrased_quotes_become_unknown(self):
        self.assert_repaired('- "complete"', [assistant("not complete"), tool("c1", "Calibration is complete.")])
        self.assert_repaired('- "Complete"', [assistant("not complete")])
        self.assert_repaired('- "complete."', [assistant("not complete")])
        self.assert_repaired('- "unfinished"', [assistant("not complete")])

    def test_duplicate_first_user_lines_do_not_create_false_ambiguity(self):
        reported = "The task is not complete."
        llm, _failure = self.assert_repaired('- "complete"', [user(reported)], "- " + json.dumps(reported))
        self.assertEqual(llm.calls[0][0][1]["content"].count(reported), 2)

    def test_a_unique_fragment_keeps_the_entire_long_first_user_line(self):
        reported = "Check PKG-17, keep all prior constraints, and report that only calibration call_read_7 is complete."
        self.assert_repaired('- "call_read_7 is complete"', [user(reported)], "- " + json.dumps(reported))

    def test_a_calibration_fragment_does_not_become_parent_task_completion(self):
        reported = "Calibration call_read_7 is complete."
        rows = [user("Verify package PKG-17."), tool("call_read_7", reported)]
        self.assert_repaired('- "complete"', rows, "- " + json.dumps(reported))

    def test_a_cut_marker_on_the_expanded_source_line_prevents_its_use(self):
        self.assert_repaired('- "complete"', [assistant("not complete [cut] later scope")])

    def test_supported_state_survives_and_any_dropped_state_adds_unknown(self):
        state = '- "reported state"\n- "invented progress"'
        expected = '- "reported state"\n- progress unknown'
        self.assert_repaired(state, [assistant("reported state")], expected)

    def test_unicode_source_line_separators_are_escaped_when_fragments_expand(self):
        for separator in ("\u0085", "\u2028", "\u2029"):
            with self.subTest(separator=separator):
                reported = "The task is not complete" + separator + "pending review."
                expected = "- " + json.dumps(reported, ensure_ascii=True)
                self.assert_repaired('- "not complete"', [assistant(reported)], expected)

    def test_state_budget_includes_json_encoding_and_reserved_unknown(self):
        reported = "State: " + '\\"' * 400 + ": end."
        encoded = "- " + json.dumps(reported, ensure_ascii=False)
        self.assertLess(estimate_tokens(reported), MAX_STATE_TOKENS)
        self.assertGreater(estimate_tokens(encoded), MAX_STATE_TOKENS)
        self.assert_repaired("- " + json.dumps('\\"'), [assistant(reported)])
        rows = [assistant("First state " + "a" * 1000), assistant("Second state " + "b" * 1000)]
        first, second = (row["content"] for row in rows)
        third = "Third state " + "c" * 1000
        rows.append(assistant(third))
        state = "\n".join("- " + json.dumps(line) for line in (first, second, third))
        llm = FakeLlm(text=state_summary(state) + "\n" + END_LINE)
        text, _tokens = llm_summary(llm, rows, PREFIXES)
        self.assertIsNotNone(text)
        section = text.split("## Current state\n", 1)[1].split("\n## Key facts", 1)[0]
        self.assertLessEqual(estimate_tokens(section), MAX_STATE_TOKENS)
        self.assertIn("- progress unknown", section)
        self.assertLessEqual(estimate_tokens(text), MAX_TOKENS)
        self.assertEqual(len(llm.calls), 1)

    def test_unsupported_long_goal_is_repaired_before_state_expansion(self):
        reported = "not complete " + "x" * 1900
        raw = state_summary('- "not complete"').replace("- goal unknown", "k" * 5700)
        expected = state_summary("- " + json.dumps(reported))
        self.assertLessEqual(estimate_tokens(raw), MAX_TOKENS)
        llm = FakeLlm(text=raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [assistant(reported)], PREFIXES), (expected, 321))
        self.assertLessEqual(estimate_tokens(expected), MAX_TOKENS)
        self.assertEqual(len(llm.calls), 1)

    def test_an_oversized_raw_state_still_fails_the_original_body_check(self):
        body = state_summary("- " + json.dumps("x" * 9000))
        llm, failure = FakeLlm(text=body + "\n" + END_LINE), {}
        self.assertEqual(llm_summary(llm, [user("Check progress.")], PREFIXES, failure=failure), (None, None))
        self.assertEqual(failure["reason"], "gate:body_token_limit")
        self.assertEqual(len(llm.calls), 2)

    def test_an_actual_protocol_failure_retries_with_the_same_source(self):
        reported = "not complete"
        good = state_summary("- " + json.dumps(reported))
        bad = state_summary('- "unsupported"')

        def reply(messages, kwargs, count):
            return bad if count == 1 else good + "\n" + messages[0]["content"].rstrip().splitlines()[-1]

        llm, failure = FakeLlm(text=reply), {}
        self.assertEqual(llm_summary(llm, [assistant(reported)], PREFIXES, failure=failure), (good, 321))
        self.assertEqual(failure, {})
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0][0][1], llm.calls[1][0][1])
        self.assertNotEqual(llm.calls[0][0][0]["content"].splitlines()[-1],
                            llm.calls[1][0][0]["content"].splitlines()[-1])

    def test_repaired_state_does_not_publish_the_quote_or_source(self):
        state = '- "synthetic-private-state"'
        with patch("warm_compaction.fallback.logger") as logger:
            llm, failure = self.assert_repaired(state, [assistant("synthetic-private-source")])
        published = str(failure) + str(logger.method_calls)
        self.assertNotIn("synthetic-private-state", published)
        self.assertNotIn("synthetic-private-source", published)
        self.assertNotIn(llm.calls[-1][0][0]["content"].splitlines()[-1], published)
        logger.warning.assert_called_once()


def facts_summary(facts):
    head, rest = SUMMARY.split("## Key facts\n", 1)
    _old, step = rest.split("\n## Next step\n", 1)
    return head + "## Key facts\n" + facts + "\n## Next step\n" + step


def field_summary(heading, content, body=SUMMARY):
    headings = ("## Goal", "## User instructions", "## Current state", "## Key facts", "## Next step")
    head, rest = body.split(heading + "\n", 1)
    index = headings.index(heading)
    following = "\n" + headings[index + 1] + "\n" if index + 1 < len(headings) else None
    tail = following + rest.split(following, 1)[1] if following else ""
    return head + heading + "\n" + content + tail


class KeyFactsSourceTest(unittest.TestCase):
    def test_unreported_approval_condition_is_dropped_without_losing_supported_values(self):
        reported = "JOB-X3 output is 28 V."
        facts = "- " + json.dumps(reported) + "\n- approval for JOB-X3 is required before proceeding"
        raw = facts_summary(facts)
        expected = facts_summary("- " + json.dumps(reported) + "\n- facts unknown")
        llm = FakeLlm(text=raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [assistant(reported)], PREFIXES), (expected, 321))
        self.assertEqual(len(llm.calls), 1)

    def test_exact_fact_fragment_keeps_full_scope_and_negation(self):
        reported = "JOB-X3 does not require approval; output is 28 V."
        raw = facts_summary('- "require approval"')
        expected = facts_summary("- " + json.dumps(reported))
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [tool("c1", reported)], PREFIXES),
                         (expected, 321))

    def test_facts_loss_and_section_budget_are_explicit(self):
        first, second = "Fact one " + "a" * 1400, "Fact two " + "b" * 1400
        raw = facts_summary("\n".join("- " + json.dumps(value) for value in (first, second))
                            + '\n- "invented condition"')
        expected = facts_summary("- " + json.dumps(first) + "\n- facts unknown")
        text, tokens = llm_summary(FakeLlm(raw + "\n" + END_LINE), [assistant(first + "\n" + second)], PREFIXES)
        self.assertEqual((text, tokens), (expected, 321))
        section = text.split("## Key facts\n", 1)[1].split("\n## Next step", 1)[0]
        self.assertLessEqual(estimate_tokens(section), MAX_STATE_TOKENS)

    def test_blank_lines_count_against_each_source_section_budget(self):
        for section, unknown in ((state_summary, "- progress unknown"), (facts_summary, "- facts unknown")):
            with self.subTest(section=section.__name__):
                raw = section(unknown + "\n" * 1100).replace("- goal unknown", '- "Check task."')
                self.assertLessEqual(estimate_tokens(raw), MAX_TOKENS)
                self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [user("Check task.")], PREFIXES),
                                 (SUMMARY.replace("- goal unknown", '- "Check task."'), 321))


class AllFieldsSourceTest(unittest.TestCase):
    def test_recovery_goal_drops_unconditional_publish_restriction(self):
        raw = SUMMARY.replace("- goal unknown", "Finish JOB-X3 without publishing.")
        rows = [user("Finish JOB-X3. Do not publish without written approval.")]
        raw = raw.replace("- instructions unknown", "- " + json.dumps(rows[0]["content"]))
        llm = FakeLlm(raw + "\n" + END_LINE)
        text, tokens = llm_summary(llm, rows, PREFIXES)
        self.assertEqual(text.split("## Goal\n", 1)[1].split("\n## User instructions", 1)[0], "- goal unknown")
        self.assertEqual(tokens, 321)
        self.assertEqual(len(llm.calls), 1)

    def test_next_step_discards_added_condition_and_unchecked_tail(self):
        raw = field_summary("## Next step", '- "Review JOB-X3 after approval."\nDo not perform it yet.')
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [user("Review JOB-X3.")], PREFIXES), (None, None))
        self.assertEqual(len(llm.calls), 2)

    def test_next_step_keeps_supported_action_and_records_lost_tail_condition(self):
        raw = field_summary("## Next step", '- "Review JOB-X3."\nDo not perform it yet.')
        expected = field_summary("## Next step", '- "Review JOB-X3."\n- next step unknown')
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [user("Review JOB-X3.")], PREFIXES),
                         (expected, 321))

    def test_complete_goal_source_quote_keeps_the_original_approval_condition(self):
        reported = "Finish JOB-X3. Do not publish without written approval."
        raw = field_summary("## Goal", '- "Do not publish"')
        expected = field_summary("## Goal", "- " + json.dumps(reported))
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [user(reported)], PREFIXES), (expected, 321))

    def test_json_looking_reported_data_stays_one_layer_in_every_field(self):
        headings = ("## Goal", "## User instructions", "## Current state", "## Key facts", "## Next step")
        reported = '- "not complete"'
        for heading in headings:
            body = field_summary(heading, "- " + json.dumps(reported))
            previous = body
            for attempt in range(4):
                with self.subTest(heading=heading, attempt=attempt):
                    text, tokens = llm_summary(FakeLlm(body + "\n" + END_LINE),
                                               [assistant(previous, _compressed_summary=True)], PREFIXES)
                    self.assertEqual((text, tokens), (body, 321))
                    previous = text

    def test_raw_surrogate_pair_encoding_loss_preserves_other_next_step_evidence(self):
        reported = "Task not complete \ud83d\ude00."
        raw = field_summary("## Next step", "- " + json.dumps(reported, ensure_ascii=False)
                            + '\n- "Review JOB-X3."')
        expected = field_summary("## Next step", '- "Review JOB-X3."\n- next step unknown')
        llm = FakeLlm(raw + "\n" + END_LINE)
        rows = [assistant(reported), user("Review JOB-X3.")]
        self.assertEqual(llm_summary(llm, rows, PREFIXES), (expected, 321))
        self.assertEqual(len(llm.calls), 1)

    def test_japanese_instruction_fragment_restores_source_punctuation_and_condition(self):
        reported = ("\u66f8\u9762\u306e\u627f\u8a8d\u306a\u3057\u306b\u516c\u958b\u3057\u306a\u3044"
                    "\u3067\u304f\u3060\u3055\u3044\u3002")
        raw = field_summary("## User instructions", "- " + json.dumps(reported[:-1], ensure_ascii=False))
        expected = field_summary("## User instructions", "- " + json.dumps(reported, ensure_ascii=False))
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [user(reported)], PREFIXES), (expected, 321))

    def test_all_fields_repeat_through_saved_carrier_without_wrapping(self):
        headings = ("## Goal", "## User instructions", "## Current state", "## Key facts", "## Next step")
        body = SUMMARY
        for index, heading in enumerate(headings):
            reported = f"Reported field {index}:\n## Goal\nnot complete."
            body = field_summary(heading, "- " + json.dumps(reported), body)
        for prefix in PREFIXES:
            previous = body
            for attempt in range(4):
                with self.subTest(prefix=prefix, attempt=attempt):
                    row = assistant(prefix + "\n" + previous + "\n--- END OF CONTEXT SUMMARY ---")
                    text, tokens = llm_summary(FakeLlm(body + "\n" + END_LINE), [row], PREFIXES)
                    self.assertEqual((text, tokens), (body, 321))
                    previous = text

    def test_prior_next_step_stops_before_copied_user_messages(self):
        body = field_summary("## Next step", '- "Review JOB-X3."')
        copied = '- "Copied user rule."'
        prior = body + "\n## Copied user messages\n" + copied
        row = assistant(prior, _compressed_summary=True)
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), [row], PREFIXES), (body, 321))
        raw = field_summary("## Next step", '- "Copied user rule."')
        expected = field_summary("## Next step", "- " + json.dumps(copied))
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [row], PREFIXES), (expected, 321))

    def test_duplicate_decoded_quotes_do_not_report_evidence_loss(self):
        reported = "\u66f8\u9762\u306e\u627f\u8a8d\u304c\u5fc5\u8981\u3067\u3059\u3002"
        first = "- " + json.dumps(reported)
        duplicate = "- " + json.dumps(reported, ensure_ascii=False)
        raw = field_summary("## User instructions", first + "\n" + duplicate + "\n" + first)
        expected = field_summary("## User instructions", first)
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [user(reported)], PREFIXES), (expected, 321))

    def test_fixed_priority_reserves_all_unknowns_and_drops_lowest_priority_fact(self):
        headings = ("## Goal", "## User instructions", "## Current state", "## Key facts", "## Next step")
        values = {heading: f"Field {index}: " + chr(97 + index) * 1900 for index, heading in enumerate(headings)}
        raw = expected = SUMMARY
        for index, heading in enumerate(headings):
            raw = field_summary(heading, "- " + json.dumps(f"Field {index}:"), raw)
            if heading != "## Key facts":
                expected = field_summary(heading, "- " + json.dumps(values[heading]), expected)
        rows = [assistant(value) for value in values.values()]
        llm = FakeLlm(raw + "\n" + END_LINE)
        text, tokens = llm_summary(llm, rows, PREFIXES)
        self.assertEqual((text, tokens), (expected, 321))
        self.assertLessEqual(estimate_tokens(text), MAX_TOKENS)
        self.assertEqual(len(llm.calls), 1)

    def test_raw_unicode_heading_data_is_canonicalized_in_every_field(self):
        headings = ("## Goal", "## User instructions", "## Current state", "## Key facts", "## Next step")
        reported = "Reported\u2028## Goal\u2028not complete."
        for heading in headings:
            with self.subTest(heading=heading):
                raw = field_summary(heading, "- " + json.dumps(reported, ensure_ascii=False))
                expected = field_summary(heading, "- " + json.dumps(reported))
                self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [assistant(reported)], PREFIXES),
                                 (expected, 321))


class SourceContextTest(unittest.TestCase):
    def test_memory_provider_unique_fact_is_kept_from_the_sent_context(self):
        reported = "Memory setting for JOB-X3 is 28 V."
        raw = facts_summary('- "JOB-X3 is 28 V."')
        expected = facts_summary("- " + json.dumps(reported))
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [user("Continue JOB-X3.")], PREFIXES, memory_context=reported),
                         (expected, 321))
        self.assertIn(reported, llm.calls[0][0][0]["content"])
        self.assertNotIn(reported, llm.calls[0][0][1]["content"])
        self.assertEqual(len(llm.calls), 1)

    def test_embedded_ordinary_prior_wrapper_is_ambiguous(self):
        wrapper = '- "not complete"'
        prior = state_summary(wrapper)
        raw = state_summary("- " + json.dumps(wrapper))
        ordinary = 'Preserve the literal - "not complete" exactly.'
        for row in (user(ordinary), tool("c1", ordinary)):
            with self.subTest(role=row["role"]):
                llm = FakeLlm(raw + "\n" + END_LINE)
                self.assertEqual(llm_summary(llm, [assistant(prior, _compressed_summary=True), row], PREFIXES),
                                 (None, None))
                self.assertEqual(len(llm.calls), 2)

    def test_bounded_context_keeps_tail_fact_but_drops_missing_middle(self):
        from warm_compaction.fallback import EXTRAS_CHARS, EXTRAS_TOKENS, _bound
        from warm_compaction.handoff import extras
        missing, kept = "Missing middle setting is 91 V.", "Tail setting is 28 V."
        memory = "First setting.\n" + "x" * 7000 + "\n" + missing + "\n" + "y" * 7000 + "\n" + kept
        bounded = _bound(extras(None, memory), EXTRAS_CHARS, EXTRAS_TOKENS, middle=True)
        raw = facts_summary("- " + json.dumps(missing) + "\n- " + json.dumps(kept))
        expected = facts_summary("- " + json.dumps(kept) + "\n- facts unknown")
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [user("Check settings.")], PREFIXES, memory_context=memory),
                         (expected, 321))
        sent = llm.calls[0][0][0]["content"]
        self.assertIn(bounded, sent)
        self.assertIn(kept, sent)
        self.assertNotIn(missing, sent)
        self.assertLessEqual(len(bounded), EXTRAS_CHARS + len(MIDDLE_MARK))
        self.assertLessEqual(estimate_tokens(bounded), EXTRAS_TOKENS)

    def test_expansion_to_a_cut_memory_line_is_dropped(self):
        raw = facts_summary('- "Memory prefix"')
        memory = "Memory prefix " + "x" * 10000 + " ending."
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [user("Check memory.")], PREFIXES, memory_context=memory), (None, None))
        self.assertIn("[cut]", llm.calls[0][0][0]["content"])
        self.assertEqual(len(llm.calls), 2)

    def test_context_and_transcript_do_not_make_new_source_adjacency(self):
        quoted = "Transcript report.\nMemory report."
        raw = facts_summary("- " + json.dumps(quoted))
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [assistant("Transcript report.")], PREFIXES,
                                     memory_context="Memory report."), (None, None))

    def test_focus_and_memory_rendered_block_keeps_the_existing_request_reserve(self):
        from warm_compaction.fallback import EXTRAS_CHARS, EXTRAS_TOKENS, _bound
        from warm_compaction.handoff import extras
        focus, memory = "JOB-X3", "Memory result is 28 V."
        bounded = _bound(extras(focus, memory), EXTRAS_CHARS, EXTRAS_TOKENS, middle=True)
        rendered_focus = "Give more detail to this topic: JOB-X3"
        raw = facts_summary('- "JOB-X3"\n- "Memory result is 28 V."')
        expected = facts_summary("- " + json.dumps(rendered_focus) + '\n- "Memory result is 28 V."')
        rows = [assistant("Row " + str(index) + " " + "r" * 3900) for index in range(20)]
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, rows, PREFIXES, focus_topic=focus, memory_context=memory), (expected, 321))
        self.assertIn(bounded, llm.calls[0][0][0]["content"])
        original = llm.calls[0][0][1]["content"]
        self.assertLessEqual(estimate_tokens(original) + estimate_tokens(bounded), TRANSCRIPT_TOKENS)
        self.assertLessEqual(len(original) + len(bounded), TRANSCRIPT_CHARS)

    def test_prior_report_duplicates_and_exact_ordinary_reports_stay_supported(self):
        reported = "not complete"
        body = state_summary("- " + json.dumps(reported))
        rows = [assistant(body, _compressed_summary=True), assistant(reported), user("not complete later")]
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), rows, PREFIXES), (body, 321))

    def test_memory_wrapper_collision_records_unknown_without_logging_context(self):
        wrapper = '- "not complete"'
        prior = state_summary(wrapper)
        raw = state_summary("- " + json.dumps(wrapper))
        memory = 'Preserve the literal - "not complete" exactly; synthetic-private-memory.'
        llm, failure = FakeLlm(raw + "\n" + END_LINE), {}
        with patch("warm_compaction.fallback.logger") as logger:
            self.assertEqual(llm_summary(llm, [assistant(prior, _compressed_summary=True)], PREFIXES,
                                         memory_context=memory, failure=failure), (None, None))
        self.assertEqual(failure["reason"], "gate:no_useful_evidence")
        self.assertNotIn(memory, str(logger.method_calls) + str(failure))
        logger.warning.assert_called_once()


class UsefulEvidenceTest(unittest.TestCase):
    def test_all_unknown_retries_then_refuses_without_provider_text(self):
        llm, failure = FakeLlm(SUMMARY + "\n" + END_LINE), {}
        self.assertEqual(llm_summary(llm, [user("Check JOB-X3.")], PREFIXES, failure=failure), (None, None))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(failure["reason"], "gate:no_useful_evidence")
        self.assertEqual(failure["attempts"], 2)
        self.assertIn("failed a summary check", llm.calls[1][0][0]["content"])
        self.assertNotIn("Check JOB-X3.", str(failure))

    def test_generated_labels_alone_do_not_count_as_useful_evidence(self):
        for label in ("[user]", "[first user message]", "## Goal"):
            with self.subTest(label=label):
                body = facts_summary("- " + json.dumps(label))
                rows = [assistant(SUMMARY, _compressed_summary=True), user("Check JOB-X3.")]
                llm, failure = FakeLlm(body + "\n" + END_LINE), {}
                self.assertEqual(llm_summary(llm, rows, PREFIXES, failure=failure), (None, None))
                self.assertEqual(failure["reason"], "gate:no_useful_evidence")
                self.assertEqual(len(llm.calls), 2)

    def test_copied_appendix_cannot_cut_a_core_fact_that_fits_the_existing_share(self):
        fact = "JOB-X3 output is 28 V."
        core = facts_summary("- " + json.dumps(fact)).replace("- goal unknown", "g" * 5200)
        prior = core + "\n## Copied user messages\n" + "c" * 30000
        sent = transcript([assistant(prior, _compressed_summary=True)], PREFIXES)
        self.assertIn("## Key facts\n- " + json.dumps(fact), sent)
        self.assertIn("## Next step\n- next step unknown", sent)

    def test_all_unknown_first_reply_can_retry_to_one_useful_quote(self):
        reported = "Check JOB-X3."
        body = useful_summary(reported)

        def reply(messages, _kwargs, count):
            return (SUMMARY if count == 1 else body) + "\n" + messages[0]["content"].rstrip().splitlines()[-1]

        llm, failure = FakeLlm(reply), {}
        self.assertEqual(llm_summary(llm, [user(reported)], PREFIXES, failure=failure), (body, 321))
        self.assertEqual(failure, {})
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0][0][1], llm.calls[1][0][1])
        self.assertNotEqual(llm.calls[0][0][0]["content"].splitlines()[-1],
                            llm.calls[1][0][0]["content"].splitlines()[-1])
        self.assertIn("## Current state\n- progress unknown", body)

    def test_empty_tool_call_is_scaffold_but_the_same_user_literal_is_payload(self):
        label = "(tool call c1 read: )"
        body = useful_summary(label)
        llm, failure = FakeLlm(body + "\n" + END_LINE), {}
        self.assertEqual(llm_summary(llm, [assistant("", [("c1", "read", "")])], PREFIXES, failure=failure),
                         (None, None))
        self.assertEqual(failure["reason"], "gate:no_useful_evidence")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), [user(label)], PREFIXES), (body, 321))

    def test_nonempty_supplied_tool_argument_can_supply_evidence(self):
        label = '(tool call c1 read: {"path":"synthetic.txt"})'
        body = useful_summary(label)
        rows = [assistant("", [("c1", "read", '{"path":"synthetic.txt"}')])]
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), rows, PREFIXES), (body, 321))

    def test_multiline_argument_prefix_does_not_count_but_real_argument_line_does(self):
        arguments = '\n{"path":"synthetic.txt"}\n '
        rows = [assistant("", [("c1", "read", arguments)])]
        for label in ("(tool call c1 read: ", " )"):
            with self.subTest(label=label):
                body = useful_summary(label)
                llm, failure = FakeLlm(body + "\n" + END_LINE), {}
                self.assertEqual(llm_summary(llm, rows, PREFIXES, failure=failure), (None, None))
                self.assertEqual(failure["reason"], "gate:no_useful_evidence")
        body = useful_summary('{"path":"synthetic.txt"}')
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), rows, PREFIXES), (body, 321))

    def test_empty_attachment_marks_do_not_count_but_real_literals_do(self):
        from warm_compaction.rows import visible_text
        for part in ({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                     {"type": "input_audio", "input_audio": {}}, {"type": "file", "file": {}},
                     {"type": "unknown", "unknown": {}}):
            with self.subTest(kind=part["type"]):
                label = visible_text([part])
                body = useful_summary(label)
                llm, failure = FakeLlm(body + "\n" + END_LINE), {}
                self.assertEqual(llm_summary(llm, [user([part])], PREFIXES, failure=failure), (None, None))
                self.assertEqual(failure["reason"], "gate:no_useful_evidence")
                self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), [user(label)], PREFIXES), (body, 321))

    def test_attachment_filename_or_url_is_real_retained_data(self):
        from warm_compaction.rows import visible_text
        for part in ({"type": "image_url", "image_url": {"url": "https://example.invalid/synthetic.png"}},
                     {"type": "file", "file": {"filename": "synthetic.txt"}}):
            with self.subTest(kind=part["type"]):
                body = useful_summary(visible_text([part]))
                self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), [user([part])], PREFIXES), (body, 321))
        for reference, label in (("   ", "[file attachment:    ]"), ("\nsynthetic.txt", "[file attachment: ")):
            with self.subTest(reference=reference):
                part = {"type": "file", "file": {"filename": reference}}
                body = useful_summary(label)
                llm, failure = FakeLlm(body + "\n" + END_LINE), {}
                self.assertEqual(llm_summary(llm, [user([part])], PREFIXES, failure=failure), (None, None))
                self.assertEqual(failure["reason"], "gate:no_useful_evidence")
        part = {"type": "file", "file": {"filename": "\nsynthetic.txt"}}
        body = useful_summary("synthetic.txt]")
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), [user([part])], PREFIXES), (body, 321))

    def test_cut_actual_marker_cannot_bless_a_later_generated_identical_marker(self):
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        content = [{"type": "text", "text": "start " + "x" * 6000 + "\n[image attachment]\n" + "y" * 6000}, part]
        body = useful_summary("[image attachment]")
        rows = [user("Check the attachment."), assistant(content)]
        llm, failure = FakeLlm(body + "\n" + END_LINE), {}
        self.assertEqual(llm_summary(llm, rows, PREFIXES, failure=failure), (None, None))
        self.assertEqual(failure["reason"], "gate:no_useful_evidence")
        self.assertEqual(llm.calls[0][0][1]["content"].count("[image attachment]"), 1)

    def test_ignored_multipart_values_do_not_change_payload_offsets(self):
        content = [17, [], {"type": "text", "text": "JOB-X3 output is 28 V."}]
        body = useful_summary("JOB-X3 output is 28 V.")
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), [user(content)], PREFIXES), (body, 321))

    def test_empty_copied_quote_is_scaffold_but_copied_literal_and_user_literal_are_data(self):
        prior = SUMMARY + "\n## Copied user messages\n>"
        body = useful_summary(">")
        llm, failure = FakeLlm(body + "\n" + END_LINE), {}
        self.assertEqual(llm_summary(llm, [assistant(prior, _compressed_summary=True)], PREFIXES, failure=failure),
                         (None, None))
        self.assertEqual(failure["reason"], "gate:no_useful_evidence")
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE), [user(">")], PREFIXES), (body, 321))
        copied = SUMMARY + "\n## Copied user messages\n> >"
        body = useful_summary("> >")
        self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE),
                                     [assistant(copied, _compressed_summary=True)], PREFIXES), (body, 321))

    def test_padded_prior_structure_is_not_payload_but_user_literals_remain_payload(self):
        prior = SUMMARY.replace("##", "  ##").replace("- ", "  - ")
        for label in ("  ## Key facts", "  - facts unknown"):
            with self.subTest(label=label):
                body = useful_summary(label)
                llm, failure = FakeLlm(body + "\n" + END_LINE), {}
                self.assertEqual(llm_summary(llm, [assistant(prior, _compressed_summary=True)], PREFIXES,
                                             failure=failure), (None, None))
                self.assertEqual(failure["reason"], "gate:no_useful_evidence")
                self.assertEqual(llm_summary(FakeLlm(body + "\n" + END_LINE),
                                             [user("First line.\n" + label + "\nLast line.")], PREFIXES), (body, 321))

    def test_real_label_like_literal_stays_usable_in_known_prior_quotes(self):
        body = useful_summary("[user]")
        rows = [user("[user]")]
        for attempt in range(4):
            with self.subTest(attempt=attempt):
                text, tokens = llm_summary(FakeLlm(body + "\n" + END_LINE), rows, PREFIXES)
                self.assertEqual((text, tokens), (body, 321))
                rows = [assistant(text, _compressed_summary=True)]

    def test_core_can_use_more_than_three_quarters_and_dense_appendix_uses_only_remainder(self):
        from warm_compaction.fallback import EARLIER_SUMMARY_CHARS, EARLIER_SUMMARY_TOKENS
        for goal in ("g" * 6800, "\u65e5" * 1200):
            with self.subTest(goal=goal[:10]):
                core = useful_summary("JOB-X3 is not complete.").replace("- goal unknown", goal)
                prior = core + "\n## Copied user messages\n" + "\u5831" * 12000
                sent = transcript([assistant(prior, _compressed_summary=True)], PREFIXES)
                earlier = sent.removeprefix("[earlier summary]\n")
                self.assertTrue(earlier.startswith(core))
                self.assertLessEqual(len(earlier), EARLIER_SUMMARY_CHARS)
                self.assertLessEqual(estimate_tokens(earlier), EARLIER_SUMMARY_TOKENS)

    def test_unknown_legacy_summary_keeps_the_existing_whole_bound(self):
        from warm_compaction.fallback import EARLIER_SUMMARY_CHARS, EARLIER_SUMMARY_TOKENS, _bound
        core = "Legacy text " + "x" * 9000 + "\n## Copied user messages\n" + "tail " * 6000
        for prefix in PREFIXES:
            with self.subTest(prefix=prefix):
                row = assistant(prefix + "\n" + core + "\n--- END OF CONTEXT SUMMARY ---")
                self.assertEqual(transcript([row], PREFIXES), "[earlier summary]\n"
                                 + _bound(core.strip(), EARLIER_SUMMARY_CHARS, EARLIER_SUMMARY_TOKENS, middle=True))

    def test_literal_copy_heading_inside_json_is_not_an_appendix_boundary(self):
        reported = "Reported line.\n## Copied user messages\nJOB-X3 is not complete."
        core = useful_summary(reported)
        row = assistant(core + "\n## Copied user messages\n> copied text", _compressed_summary=True)
        self.assertEqual(llm_summary(FakeLlm(core + "\n" + END_LINE), [row], PREFIXES), (core, 321))



class PriorStateTest(unittest.TestCase):
    def assert_stable(self, reported, section=state_summary, prefix=None):
        body = section("- " + json.dumps(reported))
        previous = body
        for attempt in range(4):
            with self.subTest(attempt=attempt):
                llm = FakeLlm(text=body + "\n" + END_LINE)
                rows = ([assistant(previous, _compressed_summary=True)] if prefix is None else
                        [assistant(prefix + "\n" + previous + "\n--- END OF CONTEXT SUMMARY ---")])
                text, tokens = llm_summary(llm, rows, PREFIXES)
                previous = text
                self.assertEqual((text, tokens), (body, 321))
                self.assertEqual(len(llm.calls), 1)

    def test_reported_state_stays_stable_across_repeated_compactions(self):
        self.assert_stable("Task PKG-17 is not complete.")

    def test_json_looking_reported_data_is_unwrapped_only_once_per_summary(self):
        self.assert_stable('- "not complete"')

    def test_both_sections_survive_saved_summary_carriers_without_private_flags(self):
        for section in (state_summary, facts_summary):
            for prefix in PREFIXES:
                with self.subTest(section=section.__name__, prefix=prefix):
                    self.assert_stable("JOB-X3 is not complete; output is 28 V.", section, prefix)

    def test_multiline_heading_like_reported_data_stays_encoded_across_compactions(self):
        for section in (state_summary, facts_summary):
            with self.subTest(section=section.__name__):
                self.assert_stable("Reported state:\n## Goal\nnot complete\n## Key facts\n28 V.", section)

    def test_model_quote_of_prior_wrapper_recovers_one_layer(self):
        for section in (state_summary, facts_summary):
            with self.subTest(section=section.__name__):
                reported = '- "not complete"'
                wrapper = "- " + json.dumps(reported)
                prior = section(wrapper)
                raw = section("- " + json.dumps(wrapper))
                self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE),
                                             [assistant(prior, _compressed_summary=True)], PREFIXES), (prior, 321))

    def test_prior_wrapper_collision_with_ordinary_text_is_ambiguous(self):
        reported = "Task not complete."
        wrapper = "- " + json.dumps(reported)
        prior = state_summary(wrapper)
        raw = state_summary("- " + json.dumps(wrapper))
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [assistant(prior, _compressed_summary=True), user(wrapper)], PREFIXES),
                         (None, None))
        self.assertEqual(len(llm.calls), 2)

    def test_no_one_layer_decode_from_arbitrary_user_or_tool_rows(self):
        prior = state_summary('- "Task not complete."')
        raw = state_summary('- "Task not complete."')
        expected = state_summary("- " + json.dumps('- "Task not complete."'))
        for row in (user(prior), tool("c1", prior)):
            with self.subTest(role=row["role"]):
                self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [row], PREFIXES), (expected, 321))

    def test_cut_earlier_quote_does_not_supply_missing_evidence(self):
        reported = "Missing middle report."
        prior = state_summary('- "first"\n' + 'x' * 12000 + '\n- ' + json.dumps(reported)
                              + '\n' + 'y' * 12000 + '\n- "last"')
        llm = FakeLlm(state_summary("- " + json.dumps(reported)) + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [assistant(prior, _compressed_summary=True)], PREFIXES), (None, None))
        self.assertNotIn(reported, llm.calls[0][0][1]["content"])

    def test_malformed_prior_heading_structure_does_not_decode_wrappers(self):
        prior = state_summary('- "not complete"')
        for malformed in (prior.replace("## Current state\n", "## Current state\n## Goal\n"),
                          prior.replace("## Current state", "Current state"),
                          prior.replace("## Current state", "## Key facts")):
            with self.subTest(malformed=malformed):
                expected = state_summary("- " + json.dumps('- "not complete"'))
                raw = state_summary('- "not complete"')
                self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE),
                                             [assistant(malformed, _compressed_summary=True)], PREFIXES),
                                 (expected, 321))

    def test_intact_facts_span_survives_uncertain_state_boundaries(self):
        prior = facts_summary('- "Output is 28 V."').replace("## Current state", "[cut] State")
        raw = facts_summary('- "28 V."')
        expected = facts_summary('- "Output is 28 V."')
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE),
                                     [assistant(prior, _compressed_summary=True)], PREFIXES), (expected, 321))

    def test_prior_alias_collision_drops_only_ambiguous_evidence(self):
        quote = '- "not complete"'
        prior = state_summary(quote + "\n- " + json.dumps(quote) + '\n- "Other task is complete."')
        raw = state_summary("- " + json.dumps("- " + json.dumps(quote)) + '\n- "Other task is complete."')
        expected = state_summary('- "Other task is complete."\n- progress unknown')
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [assistant(prior, _compressed_summary=True)], PREFIXES), (expected, 321))
        self.assertEqual(len(llm.calls), 1)

    def test_raw_unicode_json_with_oversized_whitespace_fails_before_canonicalization(self):
        for padding, reason in ((9000, "body_token_limit"), (25000, "byte_bound")):
            with self.subTest(padding=padding):
                reported = "Task\u2028not complete"
                raw = state_summary("- " + json.dumps(reported, ensure_ascii=False) + " " * padding)
                llm, failure = FakeLlm(raw + "\n" + END_LINE), {}
                self.assertEqual(llm_summary(llm, [assistant(reported)], PREFIXES, failure=failure), (None, None))
                self.assertEqual(failure["reason"], "gate:" + reason)
                self.assertEqual(len(llm.calls), 2)

    def test_lone_surrogate_in_expanded_source_is_stored_as_json_escape(self):
        reported = "Task not complete \ud800."
        raw = state_summary('- "not complete"')
        expected = state_summary("- " + json.dumps(reported))
        text, tokens = llm_summary(FakeLlm(raw + "\n" + END_LINE), [assistant(reported)], PREFIXES)
        self.assertEqual((text, tokens), (expected, 321))
        text.encode("utf-8")

    def test_full_match_raw_lone_surrogate_is_stored_as_json_escape(self):
        reported = "Task not complete \ud800."
        raw = state_summary("- " + json.dumps(reported, ensure_ascii=False))
        expected = state_summary("- " + json.dumps(reported))
        text, tokens = llm_summary(FakeLlm(raw + "\n" + END_LINE), [assistant(reported)], PREFIXES)
        self.assertEqual((text, tokens), (expected, 321))
        text.encode("utf-8")

    def test_surrogate_pair_encoding_loss_drops_only_that_source_candidate(self):
        reported = "Task not complete \ud83d\ude00."
        raw = state_summary('- "not complete"\n- "Other task is complete."')
        expected = state_summary('- "Other task is complete."\n- progress unknown')
        llm = FakeLlm(raw + "\n" + END_LINE)
        self.assertEqual(llm_summary(llm, [assistant(reported + "\nOther task is complete.")], PREFIXES),
                         (expected, 321))
        self.assertEqual(len(llm.calls), 1)

    def test_arbitrary_user_and_tool_json_bullets_stay_unchanged(self):
        body = state_summary('- "reported state"')
        for row in (user(body), tool("c1", body)):
            with self.subTest(role=row["role"]):
                text = transcript([row], PREFIXES)
                self.assertIn('## Current state\n- "reported state"\n', text)

    def test_malformed_prior_state_and_unknown_are_preserved_as_data(self):
        for state in ('- "unfinished', '- {"result": "unfinished"}', '- progress unknown'):
            with self.subTest(state=state):
                text = transcript([assistant(state_summary(state), _compressed_summary=True)], PREFIXES)
                self.assertIn("## Current state\n" + state + "\n", text)

    def test_raw_json_unicode_separators_and_heading_data_survive_reply_parsing(self):
        for section in (state_summary, facts_summary):
            for separator in ("\u0085", "\u2028", "\u2029"):
                with self.subTest(section=section.__name__, separator=separator):
                    reported = "Task not complete" + separator + "## Goal" + separator + "same scope."
                    raw = section("- " + json.dumps(reported, ensure_ascii=False))
                    expected = section("- " + json.dumps(reported, ensure_ascii=True))
                    llm = FakeLlm(text=raw + "\n" + END_LINE)
                    self.assertEqual(llm_summary(llm, [assistant(reported)], PREFIXES), (expected, 321))
                    self.assertEqual(len(llm.calls), 1)

    def test_japanese_fragment_expansion_keeps_visible_names(self):
        reported = "PKG-17 \u306f\u672a\u5b8c\u4e86\u3067\u3059\u3002"
        raw = state_summary("- " + json.dumps("PKG-17"))
        expected = state_summary("- " + json.dumps(reported, ensure_ascii=False))
        self.assertEqual(llm_summary(FakeLlm(raw + "\n" + END_LINE), [assistant(reported)], PREFIXES),
                         (expected, 321))


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
