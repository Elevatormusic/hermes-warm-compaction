"""Unit tests for the work ledger. All rows, commands, paths, and outputs are SYNTHETIC."""

import json
import unittest

from warm_compaction import ledger
from warm_compaction.layout import COPY_HEADING
from warm_compaction.rows import estimate_tokens
from wc_fixtures import assistant, tool, user

PREFIXES = ("[HERMES PREFIX]", "[CONTEXT SUMMARY]:")


def terminal(call_id, command):
    return assistant("", [(call_id, "terminal", json.dumps({"command": command}))])


def result(call_id, output="", exit_code=0, error=None):
    return tool(call_id, json.dumps({"output": output, "exit_code": exit_code, "error": error}))


SESSION = [
    user("Synthetic task: fix tests/test_dates.py."),
    terminal("c1", "pytest tests/test_dates.py -q"),
    result("c1", "F.\nFAILED tests/test_dates.py::test_parse_dates - ValueError: unconverted data remains: Z\n"
                 "1 failed, 1 passed", exit_code=1),
    assistant("", [("c2", "read_file", json.dumps({"path": "src/dates.py"}))]),
    tool("c2", "def parse_date(text): ..."),
    assistant("", [("c3", "patch", json.dumps({"path": "src/dates.py", "old_string": "a", "new_string": "b"}))]),
    tool("c3", json.dumps({"success": True})),
    terminal("c4", "pytest tests/test_dates.py -q"),
    result("c4", "..\n2 passed in 0.04s", exit_code=0),
    assistant("", [("c5", "write_file", json.dumps({"path": "CHANGELOG.md", "content": "- Accept Z."}))]),
    tool("c5", json.dumps({"success": True})),
]


class LedgerTest(unittest.TestCase):
    def test_commands_keep_the_exit_code_and_the_first_error_line(self):
        text = ledger.build(SESSION, PREFIXES)
        self.assertIn("- `pytest tests/test_dates.py -q`: exit 1; error lines:\n  - FAILED "
                      "tests/test_dates.py::test_parse_dates - ValueError: unconverted data remains: Z", text)
        self.assertIn("- `pytest tests/test_dates.py -q`: exit 0", text)
        self.assertLess(text.index("exit 1"), text.index("exit 0"))

    def test_changed_files_are_listed_and_reads_are_not(self):
        text = ledger.build(SESSION, PREFIXES)
        files = text.split(ledger.FILES_HEADING, 1)[1]
        self.assertIn("- src/dates.py: changed", files)
        self.assertIn("- CHANGELOG.md: written", files)
        self.assertEqual(files.count("src/dates.py"), 1)

    def test_the_block_has_the_heading_and_the_reference_note(self):
        lines = ledger.build(SESSION, PREFIXES).splitlines()
        self.assertEqual(lines[:2], [ledger.HEADING, ledger.NOTE])

    def test_no_tool_calls_give_no_block(self):
        self.assertEqual(ledger.build([user("hi"), assistant("hello")], PREFIXES), "")

    def test_a_failed_edit_is_a_tool_line_not_a_changed_file(self):
        edit = json.dumps({"path": "src/x.py", "old_string": "a", "new_string": "b"})
        rows = [assistant("", [("e1", "patch", edit)]),
                tool("e1", json.dumps({"success": False, "error": "old_string not found in src/x.py"}))]
        text = ledger.build(rows, PREFIXES)
        self.assertIn("- tool patch src/x.py: error; old_string not found in src/x.py", text)
        self.assertNotIn(ledger.FILES_HEADING, text)

    def test_an_edit_without_a_result_is_a_tool_line_not_a_changed_file(self):
        edit = json.dumps({"path": "src/x.py", "old_string": "a", "new_string": "b"})
        patch = json.dumps({"mode": "patch", "patch": "*** Begin Patch\n*** Update File: y.py\n@@\n-x\n+y\n"})
        text = ledger.build([assistant("", [("u1", "patch", edit), ("u2", "patch", patch)])], PREFIXES)
        self.assertIn("- tool patch src/x.py: no result recorded", text)
        self.assertIn("- tool patch y.py: no result recorded", text)
        self.assertNotIn(ledger.FILES_HEADING, text)

    def test_v4a_patch_headers_give_each_file_and_action(self):
        patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** Add File: b.py\n+z\n*** Delete File: c.py\n"
        rows = [assistant("", [("p1", "patch", json.dumps({"mode": "patch", "patch": patch}))]),
                tool("p1", json.dumps({"success": True}))]
        files = ledger.build(rows, PREFIXES).split(ledger.FILES_HEADING, 1)[1]
        self.assertIn("- a.py: changed", files)
        self.assertIn("- b.py: written", files)
        self.assertIn("- c.py: deleted", files)

    def test_plain_text_results_and_terminal_errors(self):
        cases = (("make build\nexit code 2\nerror: missing header", "exit 2; error lines:\n  - error: missing header"),
                 ("Error: command not allowed", "error; Error: command not allowed"),
                 ("all good", "exit unknown"))
        for output, expected in cases:
            with self.subTest(output=output):
                rows = [terminal("t1", "make build"), tool("t1", output)]
                self.assertIn(f"- `make build`: {expected}", ledger.build(rows, PREFIXES))
        timeout = [terminal("t2", "sleep 999"),
                   result("t2", exit_code=124, error="Command timed out after 180 seconds")]
        self.assertIn("- `sleep 999`: exit 124; Command timed out after 180 seconds", ledger.build(timeout, PREFIXES))

    def test_a_reused_call_id_keeps_the_result_of_each_turn(self):
        # A provider can use the same call id again in a later turn (synthetic ids).
        rows = [terminal("call_0", "pytest -q"), result("call_0", "FAILED tests/test_a.py::test_x", exit_code=1),
                terminal("call_0", "pytest -q"), result("call_0", "3 passed", exit_code=0)]
        commands = ledger.build(rows, PREFIXES).split(ledger.COMMANDS_HEADING + "\n", 1)[1]
        self.assertEqual(commands.splitlines(), ["- `pytest -q`: exit 1; error lines:",
                                                 "  - FAILED tests/test_a.py::test_x",
                                                 "- `pytest -q`: exit 0; output:", "  - 3 passed"])

    def test_a_result_before_its_call_is_not_used(self):
        rows = [result("r1", exit_code=1), terminal("r1", "make")]
        self.assertIn("- `make`: no result recorded", ledger.build(rows, PREFIXES))

    def test_a_call_without_a_result_says_so(self):
        self.assertIn("- `ls`: no result recorded", ledger.build([terminal("n1", "ls")], PREFIXES))

    def test_commands_are_one_bounded_line_without_backticks(self):
        rows = [terminal("b1", "echo `date`\n" + "x" * 500), result("b1")]
        line = next(line for line in ledger.build(rows, PREFIXES).splitlines() if line.startswith("- `"))
        self.assertNotIn("\n", line)
        self.assertLessEqual(len(line), ledger.COMMAND_CHARS + 40)
        self.assertEqual(line.count("`"), 2)

    def test_the_oldest_commands_go_first_when_the_block_is_too_large(self):
        rows = []
        for index in range(60):
            rows += [terminal(f"k{index}", f"run step {index}"), result(f"k{index}", exit_code=0)]
        rows += [assistant("", [("w1", "write_file", json.dumps({"path": "out.txt", "content": "x"}))]),
                 tool("w1", json.dumps({"success": True}))]
        text = ledger.build(rows, PREFIXES, max_tokens=200)
        self.assertLessEqual(estimate_tokens(text), 200)
        self.assertIn("run step 59", text)
        self.assertNotIn("run step 0`", text)
        self.assertIn("- out.txt: written", text)

    def test_a_budget_below_the_heading_gives_no_block(self):
        self.assertEqual(ledger.build(SESSION, PREFIXES, max_tokens=5), "")

    def test_identifiers_come_from_text_arguments_and_tool_output(self):
        write = json.dumps({"path": "src/money.py", "content": "def to_cents(x): ..."})
        rows = [user("Synthetic task: fix BILL-2291 before v2.4.1."),
                assistant("The new helper is round_half_even_cents; I keep ROUND_HALF_EVEN.",
                          [("i1", "write_file", write)]),
                tool("i1", json.dumps({"success": True})),
                terminal("i2", "python -m preview --check"),
                result("i2", "listening on 127.0.0.1:8743\nhidden_helper_name in a file read"),
                assistant("Commit c4e3b4d raised a KeyError.")]
        identifiers, _reasons = ledger.notes(rows, PREFIXES)
        self.assertIn("round_half_even_cents", identifiers["names"])
        self.assertIn("ROUND_HALF_EVEN", identifiers["names"])
        self.assertIn("to_cents", identifiers["names"])
        # Argument keys and names in tool results are not names of the work.
        self.assertNotIn("file_path", identifiers["names"])
        self.assertNotIn("hidden_helper_name", identifiers["names"])
        self.assertEqual(identifiers["addresses"], ["127.0.0.1:8743"])
        self.assertEqual(set(identifiers["references"]), {"BILL-2291", "v2.4.1", "c4e3b4d"})
        self.assertEqual(identifiers["errors"], ["KeyError"])
        text = ledger.build(rows, PREFIXES)
        self.assertIn(ledger.IDENTIFIERS_HEADING + "\n- names: ", text)

    def test_a_failed_command_keeps_its_diagnostic_lines(self):
        # Synthetic Go test and vet output: the diagnostics have a file and line, not the word "error".
        cases = (("--- FAIL: TestQuerySince (0.01s)\n    store_test.go:57: got 14 records, want 9\nFAIL\n"
                  "FAIL\tsynth/internal/store\t0.188s",
                  "exit 1; error lines:\n  - --- FAIL: TestQuerySince (0.01s)\n  - store_test.go:57: got 14 "
                  "records, want 9\n  - FAIL synth/internal/store 0.188s"),
                 ("# synth/cmd/tool\ncmd/tool/export.go:41:2: unreachable code",
                  "exit 1; error lines:\n  - cmd/tool/export.go:41:2: unreachable code"),
                 ("--- FAIL: TestExport (0.00s)\n    --- FAIL: TestExport/utc_boundary (0.00s)\n"
                  "        export_test.go:88: record was exported\nFAIL",
                  "exit 1; error lines:\n  - --- FAIL: TestExport (0.00s)\n  - --- FAIL: TestExport/utc_boundary "
                  "(0.00s)\n  - export_test.go:88: record was exported"))
        for output, expected in cases:
            with self.subTest(expected=expected):
                rows = [terminal("g1", "go test ./..."), result("g1", output, exit_code=1)]
                self.assertIn(f"- `go test ./...`: {expected}\n", ledger.build(rows, PREFIXES) + "\n")
        # A passing command keeps an output note, not error lines.
        rows = [terminal("g2", "go test ./..."), result("g2", "ok  synth/a.go:3: note", exit_code=0)]
        self.assertIn("- `go test ./...`: exit 0; output:\n  - ok synth/a.go:3: note\n",
                      ledger.build(rows, PREFIXES) + "\n")

    def test_a_successful_command_keeps_a_short_output_note(self):
        # Synthetic outputs: a short output stays whole; a long one keeps its last line.
        short = "read docs/synth.yaml (41 keys)\nwrote docs/synth.toml"
        long = "\n".join(f"case {index} ok" for index in range(30)) + "\nTest Suites: 48 passed, 48 total"
        server = "preview config ok\nlistening on 127.0.0.1:8743\n" + "\n".join(f"step {i}" for i in range(9))
        for output, expected in ((short, "exit 0; output:\n  - read docs/synth.yaml (41 keys)\n"
                                         "  - wrote docs/synth.toml"),
                                 ("preview config ok\nlistening on 127.0.0.1:8743\nstopped after check",
                                  "exit 0; output:\n  - preview config ok\n  - listening on 127.0.0.1:8743\n"
                                  "  - stopped after check"),
                                 (server, "exit 0; output:\n  - listening on 127.0.0.1:8743\n  - step 8"),
                                 (long, "exit 0; output:\n  - Test Suites: 48 passed, 48 total"),
                                 ("..\n", "exit 0")):
            with self.subTest(expected=expected):
                rows = [terminal("o1", "node convert.mjs"), result("o1", output)]
                self.assertIn(f"- `node convert.mjs`: {expected}\n", ledger.build(rows, PREFIXES) + "\n")

    def test_output_notes_go_before_commands_when_the_block_is_too_large(self):
        rows = []
        for index in range(12):
            rows += [terminal(f"n{index}", f"step {index}"), result(f"n{index}", f"step {index} wrote {'w' * 150}")]
        text = ledger.build(rows, PREFIXES, max_tokens=300)
        lines = [line for line in text.splitlines() if line.startswith("- `")]
        self.assertEqual(len(lines), 12)
        self.assertNotIn("output:", lines[0])
        self.assertIn("output:", lines[-1])
        self.assertTrue(text.endswith("\n  - step 11 wrote " + "w" * 150))

    def test_environment_variables_come_also_from_a_file_read(self):
        # A setting that only a file read shows (synthetic file text).
        source = ('MODE = os.environ.get("SYNTH_ROUNDING_MODE", "half_even")\nKEY = os.getenv(\'SYNTH_KEY\')\n'
                  'const url = process.env.SYNTH_URL;\nrun --home ${SYNTH_HOME}\nnot_env = environ_text("lower")\n')
        expected = {"SYNTH_ROUNDING_MODE", "SYNTH_KEY", "SYNTH_URL", "SYNTH_HOME"}
        # As plain text, and as the JSON output of a terminal tool (the quotes have backslashes).
        for output in (source, json.dumps({"output": source, "exit_code": 0})):
            with self.subTest(json=output is not source):
                rows = [assistant("", [("r1", "read_file", json.dumps({"path": "settings.py"}))]), tool("r1", output)]
                identifiers, _reasons = ledger.notes(rows, PREFIXES)
                self.assertEqual(set(identifiers["environment"]), expected)
                self.assertIn("- environment: ", ledger.build(rows, PREFIXES))

    def test_reasons_are_exact_assistant_sentences(self):
        rows = [user("Do not change the global context because I said so."),
                assistant("Read the file.\n\nI will not change the global context, because the reports service "
                          "shares the process. Next, run the tests."),
                assistant("I chose Decimal instead of float."),
                assistant("I put the parser in internal/timeparse, so the import command can reuse it. So be it.")]
        _identifiers, reasons = ledger.notes(rows, PREFIXES)
        self.assertEqual(reasons, ["I will not change the global context, because the reports service shares the "
                                   "process.", "I chose Decimal instead of float.",
                                   "I put the parser in internal/timeparse, so the import command can reuse it."])

    def test_notes_without_tool_work_give_a_block_only_without_require_work(self):
        rows = [user("hi"), assistant("I use round_half_even because the audit needs it.")]
        self.assertIn(ledger.REASONS_HEADING, ledger.build(rows, PREFIXES))
        self.assertEqual(ledger.build(rows, PREFIXES, require_work=True), "")

    def test_a_later_compaction_keeps_the_notes_of_the_earlier_summary(self):
        first = ledger.build([assistant("I keep float_total because the PDF export imports it.")], PREFIXES)
        content = f"[CONTEXT SUMMARY]:\n## Goal\n- x\n\n{first}\n\n{COPY_HEADING}\n\n(none)"
        later = [{"role": "user", "content": content, "_compressed_summary": True}, assistant("Port is 8743 now.")]
        identifiers, reasons = ledger.notes(later, PREFIXES)
        self.assertIn("float_total", identifiers["names"])
        self.assertEqual(reasons, ["I keep float_total because the PDF export imports it."])

    def test_a_later_compaction_keeps_the_ledger_of_the_earlier_summary(self):
        first = ledger.build(SESSION, PREFIXES)
        content = f"[CONTEXT SUMMARY]:\n## Goal\n- x\n\n{first}\n\n{COPY_HEADING}\n\n(none)"
        summary_row = {"role": "user", "content": content, "_compressed_summary": True}
        later = [summary_row, terminal("z1", "pytest -q"),
                 result("z1", "FAILED tests/test_api.py::test_timeout - AssertionError: 30 != 60", exit_code=1),
                 assistant("", [("z2", "patch", json.dumps({"path": "src/dates.py", "old_string": "b",
                                                             "new_string": "c"}))]),
                 tool("z2", json.dumps({"success": True}))]
        text = ledger.build(later, PREFIXES)
        self.assertIn("exit 1; error lines:\n  - FAILED tests/test_dates.py::test_parse_dates", text)
        self.assertIn("- `pytest -q`: exit 1; error lines:\n  - FAILED tests/test_api.py::test_timeout", text)
        files = text.split(ledger.FILES_HEADING, 1)[1].split("\n###")[0].splitlines()
        self.assertEqual([line for line in files if line],
                         ["- CHANGELOG.md: written", "- src/dates.py: changed", "  - c"])
        self.assertEqual(text.count(ledger.HEADING), 1)

    def test_error_lines_are_separate_exact_lines(self):
        # A diagnostic that starts with "error:" must not mix with the "exit 101;" label (synthetic clippy output).
        output = ("error: this `MutexGuard` is held across an `await` point\n  --> src/synth.rs:198:13\n"
                  "error: could not compile `synth`")
        rows = [terminal("x1", "cargo clippy"), result("x1", output, exit_code=101)]
        commands = ledger.build(rows, PREFIXES).split(ledger.COMMANDS_HEADING + "\n", 1)[1].splitlines()
        self.assertEqual(commands, ["- `cargo clippy`: exit 101; error lines:",
                                    "  - error: this `MutexGuard` is held across an `await` point",
                                    "  - --> src/synth.rs:198:13", "  - error: could not compile `synth`"])

    def test_exception_names_are_error_lines(self):
        # Synthetic pytest output: the exception name is the only sign of the error on its line.
        cases = (("F\nE   AssertionError: uploaded 974 of 1000 objects\n1 failed in 41.33s",
                  ["E AssertionError: uploaded 974 of 1000 objects", "1 failed in 41.33s"]),
                 ('E   synth.exceptions.EndpointConnectionError: Could not connect to "http://127.0.0.1:9/synth"',
                  ['E synth.exceptions.EndpointConnectionError: Could not connect to "http://127.0.0.1:9/synth"']),
                 ("mirror ready\nterror level 3\nKeyError: 'synth'", ["KeyError: 'synth'"]))
        for output, expected in cases:
            with self.subTest(expected=expected):
                rows = [terminal("p1", "pytest -q"), result("p1", output, exit_code=1)]
                commands = ledger.build(rows, PREFIXES).split(ledger.COMMANDS_HEADING + "\n", 1)[1]
                self.assertEqual(commands.split("\n###")[0].splitlines()[1:], [f"  - {line}" for line in expected])
                identifiers, _reasons = ledger.notes(rows, PREFIXES)
                for address in identifiers.get("addresses", []):
                    self.assertFalse(address.endswith("\\"), address)

    def test_changed_files_keep_their_first_added_lines(self):
        # A V4A patch for two files and a replacement (synthetic files).
        patch = ("*** Begin Patch\n*** Update File: synth/Cargo.toml\n@@\n-tokio = { version = \"1.4.0\" }\n"
                 "+tokio = { version = \"1.4.1\" }\n*** Update File: synth/a.rs\n@@\n-let a = 1;\n+let a = 2;\n"
                 "+let b = 3;\n+let c = 4;\n*** Delete File: synth/old.rs\n*** End Patch")
        replace = json.dumps({"path": "synth/b.py", "old_string": "x = 1\ny = 2", "new_string": "x = 1\ny = 3\n"})
        rows = [assistant("", [("v1", "patch", json.dumps({"mode": "patch", "patch": patch}))]),
                tool("v1", json.dumps({"success": True})),
                assistant("", [("v2", "patch", replace)]), tool("v2", json.dumps({"success": True}))]
        files = ledger.build(rows, PREFIXES).split(ledger.FILES_HEADING + "\n", 1)[1].splitlines()
        self.assertEqual(files, ["- synth/Cargo.toml: changed", '  - tokio = { version = "1.4.1" }',
                                 "- synth/a.rs: changed", "  - let a = 2;", "  - let b = 3;",
                                 "- synth/old.rs: deleted", "- synth/b.py: changed", "  - y = 3"])

    def test_changed_files_keep_the_step_and_only_new_lines(self):
        # A line that a patch removes and adds again is not new; a line that moves is (synthetic files).
        same = ("*** Begin Patch\n*** Update File: synth/en.json\n@@\n-  \"title\": \"Choose\",\n"
                "+  \"title\": \"Choose\",\n+  \"previous\": \"Previous month\",\n*** End Patch")
        swap = ("*** Begin Patch\n*** Update File: synth/lock.rs\n@@\n-let cart = cart_lock();\n"
                "-let inv = inv_lock();\n+let inv = inv_lock();\n+let cart = cart_lock();\n*** End Patch")
        rows = [assistant("Now add the label strings. Then run the tests.",
                          [("s1", "patch", json.dumps({"mode": "patch", "patch": same}))]),
                tool("s1", json.dumps({"success": True})),
                assistant("Now take the locks in a fixed order.",
                          [("s2", "patch", json.dumps({"mode": "patch", "patch": swap}))]),
                tool("s2", json.dumps({"success": True})),
                assistant("Now fix the second label.", [("s3", "patch", json.dumps(
                    {"path": "synth/en.json", "old_string": "a", "new_string": "\"next\": \"Next month\""}))]),
                tool("s3", json.dumps({"success": True}))]
        files = ledger.build(rows, PREFIXES).split(ledger.FILES_HEADING + "\n", 1)[1].split("\n###")[0].splitlines()
        self.assertEqual(files, ["- synth/lock.rs: changed; step: Now take the locks in a fixed order.",
                                 "  - let inv = inv_lock();",
                                 "- synth/en.json: changed; step: Now add the label strings. | "
                                 "Now fix the second label.",
                                 '  - "next": "Next month"'])
        # A later compaction keeps the steps.
        content = f"[CONTEXT SUMMARY]:\n## Goal\n- x\n\n{ledger.build(rows, PREFIXES)}\n\n{COPY_HEADING}\n\n(none)"
        later = [{"role": "user", "content": content, "_compressed_summary": True},
                 assistant("Now name the third label.", [("s4", "patch", json.dumps(
                     {"path": "synth/en.json", "old_string": "b", "new_string": "\"third\": \"3\""}))]),
                 tool("s4", json.dumps({"success": True}))]
        self.assertIn("- synth/en.json: changed; step: Now fix the second label. | Now name the third label.\n"
                      '  - "third": "3"', ledger.build(later, PREFIXES))

    def test_changed_lines_go_before_commands_when_the_block_is_too_large(self):
        rows = []
        for index in range(8):
            edit = json.dumps({"path": f"synth/f{index}.py", "old_string": "a", "new_string": f"value_{index} = "
                                                                                              + "v" * 150})
            rows += [terminal(f"t{index}", f"step {index}"), result(f"t{index}", exit_code=1),
                     assistant("", [(f"e{index}", "patch", edit)]), tool(f"e{index}", json.dumps({"success": True}))]
        text = ledger.build(rows, PREFIXES, max_tokens=350)
        self.assertLessEqual(estimate_tokens(text), 350)
        self.assertIn("`step 0`", text)
        self.assertNotIn("  - value_0 =", text)
        self.assertIn("  - value_7 =", text)

    def test_task_status_keeps_the_latest_state_of_each_task(self):
        rows = [assistant("Progress:\n- reproduce the synthetic flake: done (23 of 500 runs failed)\n"
                          "- add the lock-order test: open (not started)\n- Port: 8743\n\nNext, read the file."),
                assistant("Plan:\n1. [x] update the synth docs\n2. [ ] write the postmortem"),
                assistant("", [("d1", "todo", json.dumps({"todos": [
                    {"id": "1", "content": "check the release notes", "status": "in_progress"},
                    {"id": "2", "content": "remove the old flag", "status": "cancelled"}]}))]),
                tool("d1", json.dumps({"todos": []})),
                assistant("- add the lock-order test: done (it fails on the old code)"),
                user("- do not count this: done")]
        self.assertEqual(ledger.tasks(rows, PREFIXES), [
            "reproduce the synthetic flake: done (23 of 500 runs failed)", "update the synth docs: done",
            "write the postmortem: open", "check the release notes: in progress", "remove the old flag: cancelled",
            "add the lock-order test: done (it fails on the old code)"])
        text = ledger.build(rows, PREFIXES)
        self.assertIn(ledger.TASKS_HEADING + "\n- reproduce the synthetic flake: done", text)
        self.assertEqual(ledger.build(rows, PREFIXES, require_work=True), "")

    def test_a_later_compaction_keeps_tasks_and_exact_lines(self):
        rows = [assistant("- write the synth ADR: open"), terminal("y1", "make lint"),
                result("y1", "error: line too long", exit_code=2),
                assistant("", [("y2", "patch", json.dumps({"path": "synth/c.py", "old_string": "a",
                                                            "new_string": "limit = 750"}))]),
                tool("y2", json.dumps({"success": True}))]
        first = ledger.build(rows, PREFIXES)
        content = f"[CONTEXT SUMMARY]:\n## Goal\n- x\n\n{first}\n\n{COPY_HEADING}\n\n(none)"
        later = [{"role": "user", "content": content, "_compressed_summary": True},
                 assistant("- write the synth ADR: done")]
        text = ledger.build(later, PREFIXES)
        self.assertIn("- `make lint`: exit 2; error lines:\n  - error: line too long", text)
        self.assertIn("- synth/c.py: changed\n  - limit = 750", text)
        self.assertIn("- write the synth ADR: done", text)
        self.assertNotIn("ADR: open", text)

    def test_a_file_and_line_is_not_an_address(self):
        rows = [terminal("a1", "cargo test"),
                result("a1", "panicked at crates/synth/src/mod.rs:212:9:\nlistening on db.synth:5432", exit_code=1)]
        identifiers, _reasons = ledger.notes(rows, PREFIXES)
        self.assertEqual(identifiers["addresses"], ["db.synth:5432"])


if __name__ == "__main__":
    unittest.main()
