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
        self.assertIn("- `pytest tests/test_dates.py -q`: exit 1; FAILED tests/test_dates.py::test_parse_dates - "
                      "ValueError: unconverted data remains: Z", text)
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

    def test_v4a_patch_headers_give_each_file_and_action(self):
        patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** Add File: b.py\n+z\n*** Delete File: c.py\n"
        rows = [assistant("", [("p1", "patch", json.dumps({"mode": "patch", "patch": patch}))]),
                tool("p1", json.dumps({"success": True}))]
        files = ledger.build(rows, PREFIXES).split(ledger.FILES_HEADING, 1)[1]
        self.assertIn("- a.py: changed", files)
        self.assertIn("- b.py: written", files)
        self.assertIn("- c.py: deleted", files)

    def test_plain_text_results_and_terminal_errors(self):
        cases = (("make build\nexit code 2\nerror: missing header", "exit 2; error: missing header"),
                 ("Error: command not allowed", "error; Error: command not allowed"),
                 ("all good", "exit unknown"))
        for output, expected in cases:
            with self.subTest(output=output):
                rows = [terminal("t1", "make build"), tool("t1", output)]
                self.assertIn(f"- `make build`: {expected}", ledger.build(rows, PREFIXES))
        timeout = [terminal("t2", "sleep 999"),
                   result("t2", exit_code=124, error="Command timed out after 180 seconds")]
        self.assertIn("- `sleep 999`: exit 124; Command timed out after 180 seconds", ledger.build(timeout, PREFIXES))

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
        self.assertIn("exit 1; FAILED tests/test_dates.py::test_parse_dates", text)
        self.assertIn("- `pytest -q`: exit 1; FAILED tests/test_api.py::test_timeout", text)
        files = text.split(ledger.FILES_HEADING, 1)[1].splitlines()
        self.assertEqual([line for line in files if line], ["- CHANGELOG.md: written", "- src/dates.py: changed"])
        self.assertEqual(text.count(ledger.HEADING), 1)


if __name__ == "__main__":
    unittest.main()
