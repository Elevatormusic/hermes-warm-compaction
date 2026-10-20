"""Tests for the compaction benchmark harness (benchmarks/compaction). No model and no Hermes code run here."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from benchmarks.compaction import cli_worker, fixtures, run, scoring, worker
from benchmarks.compaction.scenarios import SCENARIOS

GOLD_DIR = Path(run.__file__).resolve().parent / "fixtures"
# A fake runtime identity (fake hashes) for the resume and merge checks.
FAKE_RUNTIME = {"public": {"host": {"os": "FakeOS", "machine": "x", "python": "3.11.0"},
                           "hermes_python": {"version": "3.11.14", "implementation": "cpython"}},
                "sha256": {"runner_python": "a" * 64, "hermes_python": "b" * 64, "hermes_prefix": "c" * 64,
                           "hermes_packages": "d" * 64, "cli.codex_cli": "e" * 64}}
# A fake harness (fake hash): the synthetic records carry its ID, as the records of a run carry the harness ID.
FAKE_HARNESS = {"run.py": "0" * 64}
# The source identity of a record whose meta.json has no plugin, Hermes, or gold values (synthetic).
FAKE_SOURCES = {"plugin_files": run.digest_id(None), "hermes_commit": None, "gold": run.digest_id(None)}
FAKE_RUNTIME_META = {"host": FAKE_RUNTIME["public"]["host"],
                     "runtime": {"hermes_python": FAKE_RUNTIME["public"]["hermes_python"]},
                     "runtime_sha256": FAKE_RUNTIME["sha256"]}


def planned_batches(gold: dict, answer) -> list[dict]:
    """Synthetic raw QA batches in the gold plan, with the prompt hash that the worker records."""
    return [{"numbers": batch, "text": json.dumps({str(n): answer(n) for n in batch}),
             "prompt_sha256": scoring.prompt_sha256(scoring.qa_prompt(
                 [(n, gold["questions"][n - 1]["question"]) for n in batch]))} for batch in gold["qa_batches"]]


def question(kind, gold, text="Question?", **extra):
    return {"kind": kind, "gold": gold, "question": text, "retention": fixtures.retention_for(kind, gold), **extra}


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {spec["id"]: fixtures.build(spec) for spec in SCENARIOS}

    def test_build_matches_the_committed_gold_files(self):
        with tempfile.TemporaryDirectory() as folder:
            fixtures.write_all(Path(folder), Path(folder) / "gold")
            for spec in SCENARIOS:
                name = f"{spec['id']}.gold.json"
                committed = (GOLD_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")
                self.assertEqual((Path(folder) / "gold" / name).read_text(encoding="utf-8"), committed)
                fixtures.load(Path(folder) / f"{spec['id']}.history.json", GOLD_DIR / name)

    def test_load_refuses_a_changed_history(self):
        with tempfile.TemporaryDirectory() as folder:
            history, gold = self.built[SCENARIOS[0]["id"]]
            history = {**history, "rows": history["rows"][:-1]}
            (Path(folder) / "h.json").write_text(json.dumps(history), encoding="utf-8")
            (Path(folder) / "g.json").write_text(json.dumps(gold), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "history_mismatch"):
                fixtures.load(Path(folder) / "h.json", Path(folder) / "g.json")

    def test_each_scored_category_has_thirty_or_more_questions(self):
        counts = {}
        for _history, gold in self.built.values():
            self.assertEqual(len(gold["questions"]), 46)
            for item in gold["questions"]:
                counts[item["category"]] = counts.get(item["category"], 0) + 1
        for category in run.CATEGORIES:
            self.assertGreaterEqual(counts[category], 12 if category == "recent" else 30, category)

    def test_sizes_cover_the_planned_range(self):
        sizes = sorted(gold["est_tokens"] for _history, gold in self.built.values())
        self.assertLess(sizes[0], 25_000)
        self.assertGreater(sizes[-1], 95_000)

    def test_rows_are_labeled_and_well_formed(self):
        for history, _gold in self.built.values():
            rows = history["rows"] + [row for chunk in history["continuations"] for row in chunk]
            self.assertTrue(history["rows"][0]["content"].startswith(fixtures.LABEL))
            for index, row in enumerate(rows):
                if row["role"] == "tool":
                    calls = rows[index - 1].get("tool_calls") or []
                    self.assertEqual(calls[0]["id"], row["tool_call_id"])
                if index and row["role"] == "assistant" and rows[index - 1]["role"] == "assistant":
                    self.assertIn("tool_calls", rows[index - 1])

    def test_each_fact_is_in_its_evidence_rows(self):
        for history, gold in self.built.values():
            rows = history["rows"]
            for item in gold["questions"]:
                # A tool result is read with its call: the command or path is in the call row before it.
                indexes = sorted({j for i in item["evidence_rows"] for j in ((i - 1, i) if rows[i]["role"] == "tool"
                                                                              else (i,))})
                text = scoring.history_text([rows[index] for index in indexes])
                self.assertTrue(scoring.retained(item, text), item["id"])

    def test_an_echo_of_the_question_is_not_a_correct_answer(self):
        # The gold keyword must come from the history, not from the question text.
        for _history, gold in self.built.values():
            for item in gold["questions"]:
                self.assertFalse(scoring.score(item, item["question"]), item["id"])

    def test_no_batch_prompt_passes_a_gold_rule(self):
        # Each gold rule is checked against the full prompt of its QA request: the prompt, each line, and each word
        # of the instruction and of the other questions.
        for name, (_history, gold) in self.built.items():
            numbers = [n for batch in gold["qa_batches"] for n in batch]
            self.assertEqual(sorted(numbers), list(range(1, len(gold["questions"]) + 1)), name)
            for batch in gold["qa_batches"]:
                self.assertLessEqual(len(batch), scoring.QA_BATCH_MAX, name)
                items = [(n, gold["questions"][n - 1]) for n in batch]
                self.assertEqual(scoring.prompt_leaks(items), [], name)
        # The check finds the leaks of consecutive batches of 12 (the plan before this check).
        _history, gold = self.built["s4-orchard"]
        found = []
        for start in range(0, len(gold["questions"]), 12):
            found += scoring.prompt_leaks(list(enumerate(gold["questions"][start:start + 12], start + 1)))
        for leak in ("s4-orchard:12", "s4-orchard:40", "s4-orchard:42"):
            self.assertIn(leak, found)

    def test_questioned_facts_are_before_the_tail_guard(self):
        for history, gold in self.built.values():
            for item in gold["questions"]:
                if item["category"] == "recent":
                    continue
                after = fixtures.estimate_tokens(history["rows"][max(item["evidence_rows"]):])
                self.assertGreaterEqual(after, fixtures.TAIL_GUARD_TOKENS - 2_000, item["id"])


class ScoringTest(unittest.TestCase):
    def test_number_skips_the_numbers_of_the_question(self):
        item = question("number", "1", "What exit code did `pytest tests/test_part_3.py` return?")
        self.assertTrue(scoring.score(item, "test_part_3 returned 1"))
        self.assertTrue(scoring.score(item, "1"))
        self.assertFalse(scoring.score(item, "0"))
        self.assertFalse(scoring.score(item, "unknown"))

    def test_text_needs_whole_words(self):
        item = question("text", "37")
        self.assertTrue(scoring.score(item, "37 invoices"))
        self.assertFalse(scoring.score(item, "374"))
        self.assertTrue(scoring.score(question("text", "AssertionError: Decimal('1.0')"),
                                      'E   AssertionError: Decimal("1.0")'))

    def test_path_ignores_quotes_and_backslashes(self):
        item = question("path", "src/app/main.py")
        self.assertTrue(scoring.score(item, "`src\\app\\main.py`"))
        self.assertTrue(scoring.score(item, "/workspace/x/src/app/main.py"))
        self.assertFalse(scoring.score(item, "src/app/main.pyc"))

    def test_verbatim_ignores_case_and_punctuation(self):
        item = question("verbatim", "Do not edit anything under vendor/.")
        self.assertTrue(scoring.score(item, "\"do not edit anything under vendor\""))
        self.assertFalse(scoring.score(item, "Do not edit vendor."))

    def test_choice_needs_one_class(self):
        item = question("choice", "blocked")
        self.assertTrue(scoring.score(item, "Blocked (waiting for rates)"))
        self.assertFalse(scoring.score(item, "open"))
        self.assertFalse(scoring.score(question("choice", "done"), "done or blocked"))
        self.assertTrue(scoring.score(question("choice", "open"), "not done"))
        self.assertTrue(scoring.score(question("choice", "done"), "Completed"))
        # A negated completion phrase names the open class (synthetic answers).
        for answer in ("not completed", "Not complete yet.", "not finished", "It isn't finished.", "incomplete",
                       "unfinished", "It has not been completed.", "not yet done"):
            with self.subTest(answer=answer):
                self.assertTrue(scoring.score(question("choice", "open"), answer))
                self.assertFalse(scoring.score(question("choice", "done"), answer))
        # A negated phrase does not hide another class: two classes are no answer.
        self.assertFalse(scoring.score(question("choice", "open"), "not completed, blocked"))
        self.assertTrue(scoring.score(question("choice", "done"), "Finished and complete."))

    def test_keywords_match_word_starts_and_short_words(self):
        self.assertTrue(scoring.score(question("keywords", [["round"], ["cent"]]), "Fix rounding by one-cent"))
        self.assertFalse(scoring.score(question("keywords", [["round"], ["cent"]]), "Fix rounding"))
        self.assertFalse(scoring.score(question("keywords", [["rc"]]), "the source"))
        self.assertTrue(scoring.score(question("keywords", [["rc"]]), "tag the RC"))
        self.assertFalse(scoring.score(question("keywords", [["50"]]), "1500 objects"))
        self.assertTrue(scoring.score(question("keywords", [["50"]]), "a 50% failure rate"))

    def test_retention_needs_all_groups_in_one_window(self):
        item = {"kind": "number", "gold": "2", "retention": [["make test"], fixtures.exit_patterns(2)]}
        self.assertTrue(scoring.retained(item, "- `make test`: exit 2; error"))
        self.assertFalse(scoring.retained(item, "make test" + " x" * 400 + " exit 2"))
        self.assertFalse(scoring.retained(item, "- `make test`: exit 0"))

    def test_prompt_leaks_checks_the_full_batch_prompt(self):
        def item(id_, kind, gold, text):
            return {**question(kind, gold, text), "id": id_}

        status = item("a", "keywords", [["billing"]], "What did the user ask for next?")
        task = item("b", "choice", "open", 'What is the status of "update the billing docs"? done, open, or blocked.')
        self.assertEqual(scoring.prompt_leaks([(1, status), (2, task)]), ["a"])
        # A number in another question, a question number, and a word of the instruction.
        code = item("c", "number", "0", "What exit code did `make` return?")
        other = item("d", "number", "7", "What exit code did `lint --max-warnings 0` return?")
        self.assertEqual(scoring.prompt_leaks([(1, code), (2, other)]), ["c"])
        self.assertEqual(scoring.prompt_leaks([(7, code), (9, other)]), ["c", "d"])
        rule = item("e", "keywords", [["exactly"]], "How must the agent copy the names?")
        self.assertEqual(scoring.prompt_leaks([(1, rule)]), ["e"])
        # Keyword groups can match across two other questions (the full prompt).
        goal = item("f", "keywords", [["upload"], ["retry"]], "What is the main goal?")
        parts = [item("g", "text", "x", "Which upload failed?"), item("h", "text", "y", "Which retry ran?")]
        self.assertEqual(scoring.prompt_leaks([(1, goal), (2, parts[0])]), [])
        self.assertEqual(scoring.prompt_leaks([(1, goal), (2, parts[0]), (3, parts[1])]), ["f"])

    def test_parse_answers_reads_json_and_numbered_lines(self):
        answers, ok = scoring.parse_answers("```json\n{\"1\": \"a\", \"2\": 3}\n```", [1, 2])
        self.assertTrue(ok)
        self.assertEqual(answers, {"1": "a", "2": "3"})
        answers, ok = scoring.parse_answers("<think>x</think>1. first\n2: second", [1, 2])
        self.assertEqual(answers, {"1": "first", "2": "second"})
        self.assertEqual(scoring.parse_answers("no answers", [1]), ({}, False))

    def test_statistics(self):
        low, high = scoring.wilson(0, 12)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2425, places=3)
        low, high = scoring.wilson(2, 8)
        self.assertAlmostEqual(low, 0.0715, places=3)
        self.assertAlmostEqual(high, 0.5907, places=3)
        self.assertEqual(scoring.wilson(0, 0), (None, None))
        self.assertAlmostEqual(scoring.mcnemar(0, 6), 0.03125)
        self.assertIsNone(scoring.mcnemar(0, 0))
        interval = scoring.mean_interval([1.0, 2.0, 3.0])
        self.assertAlmostEqual(interval["mean"], 2.0)
        self.assertAlmostEqual(interval["high"] - interval["mean"], 4.303 / 3 ** 0.5, places=3)

    def test_statistics_are_exact_at_edge_sizes(self):
        # t: the exact quantile at each df, also between the keys of a printed table (two-sided 95%, 3 decimals).
        printed = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 20: 2.086, 21: 2.080, 22: 2.074,
                   24: 2.064, 25: 2.060, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
        for df, value in printed.items():
            self.assertAlmostEqual(scoring.t_quantile(df), value, places=3, msg=df)
        # Between table keys the quantile is larger than at the next key (a smaller df gives a wider interval).
        self.assertGreater(scoring.t_quantile(21), scoring.t_quantile(25))
        self.assertGreater(scoring.t_quantile(100000), 1.95996)
        self.assertTrue(scoring.t_quantile(0) != scoring.t_quantile(0))  # NaN: no interval for df 0
        # mean_interval at n = 0, 1, 2.
        self.assertEqual(scoring.mean_interval([])["mean"], None)
        self.assertEqual((scoring.mean_interval([4.0])["mean"], scoring.mean_interval([4.0])["low"]), (4.0, None))
        two = scoring.mean_interval([1.0, 3.0])
        self.assertAlmostEqual(two["high"] - two["mean"], 12.7062 * (2 ** 0.5) / 2 ** 0.5, places=3)
        # Wilson at n = 0, 1, 2 (z = 1.959964, at least the exact 97.5% normal quantile).
        self.assertGreaterEqual(1.959964, 1.95996398454)
        self.assertEqual(scoring.wilson(0, 0), (None, None))
        self.assertEqual(scoring.wilson(0, 1)[0], 0.0)
        self.assertAlmostEqual(scoring.wilson(0, 1)[1], 0.79345, places=4)
        self.assertAlmostEqual(scoring.wilson(1, 1)[0], 0.20655, places=4)
        self.assertEqual(scoring.wilson(1, 1)[1], 1.0)
        low, high = scoring.wilson(1, 2)
        self.assertAlmostEqual(low + high, 1.0)
        # McNemar (exact binomial) at 0, 1, and 2 discordant pairs, and a tie.
        self.assertIsNone(scoring.mcnemar(0, 0))
        self.assertEqual(scoring.mcnemar(0, 1), 1.0)
        self.assertEqual(scoring.mcnemar(1, 1), 1.0)
        self.assertEqual(scoring.mcnemar(0, 2), 0.5)
        self.assertEqual(scoring.mcnemar(3, 3), 1.0)
        self.assertAlmostEqual(scoring.mcnemar(0, 5), 0.0625)


class RunTest(unittest.TestCase):
    def test_method_settings_follow_the_plugin_settings(self):
        old = {"settings_known": ["warm"]}
        self.assertIsNone(run.method_settings("warm_ledger", old, 0.95))
        self.assertEqual(run.method_settings("warm_no_ledger", old, 0.95), {"threshold": 0.95, "warm": True})
        self.assertIsNone(run.method_settings("fallback_relaxed_ledger", {"settings_known": ["warm", "ledger"]}, 0.9))
        hybrid = {"settings_known": ["warm", "ledger", "fallback_format", "ledger_on_failure"]}
        self.assertEqual(run.method_settings("fallback_quoted_ledger", hybrid, 0.95),
                         {"threshold": 0.95, "warm": False, "ledger": True, "fallback_format": "quoted",
                          "ledger_on_failure": False})
        self.assertEqual(run.method_settings("stock_hermes", hybrid, 0.95), {})

    def test_plan_rotates_methods_and_adds_drift_jobs(self):
        config = {"threshold": 0.95, "replicates": 2, "fixtures": ["s1-ledgerline"],
                  "methods": ["full_context", "stock_hermes", "warm_ledger"],
                  "drift": {"fixtures": ["s1-ledgerline"], "methods": ["warm_ledger", "stock_hermes"], "cycles": 3}}
        jobs, skipped = run.plan(config, {"settings_known": ["warm", "ledger"]})
        self.assertEqual(skipped, [])
        self.assertEqual([job["method"] for job in jobs[:3]], ["full_context", "stock_hermes", "warm_ledger"])
        self.assertEqual([job["method"] for job in jobs[3:6]], ["stock_hermes", "warm_ledger", "full_context"])
        # The drift jobs rotate with the same helper (in the order of "methods"), so no method always runs first.
        self.assertEqual([(job["method"], job["rep"], job["cycles"]) for job in jobs[6:]],
                         [("stock_hermes", 1, 3), ("warm_ledger", 1, 3), ("warm_ledger", 2, 3), ("stock_hermes", 2, 3)])
        # Each list names each item once, the drift cycle count is 2 or more (also when it is missing), and every
        # job ID is unique.
        for changed in ({"methods": ["stock_hermes", "stock_hermes"]}, {"fixtures": ["s1-ledgerline"] * 2},
                        {"drift": {**config["drift"], "fixtures": ["s1-ledgerline"] * 2}},
                        {"drift": {**config["drift"], "methods": ["warm_ledger", "warm_ledger"]}},
                        {"drift": {**config["drift"], "cycles": 1}}, {"drift": {**config["drift"], "cycles": 2}},
                        {"drift": {**config["drift"], "cycles": 4}},
                        {"drift": {k: v for k, v in config["drift"].items() if k != "methods"}},
                        {"drift": {k: v for k, v in config["drift"].items() if k != "fixtures"}},
                        {"drift": {k: v for k, v in config["drift"].items() if k != "cycles"}},
                        {"replicates": 0}, {"replicates": "2"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                run.plan({**config, **changed}, {"settings_known": ["warm", "ledger"]})
        self.assertEqual(len(run.plan({**config, "drift": {}}, {"settings_known": ["warm", "ledger"]})[0]), 6)
        # Each drift method gets a job or is refused: not in "methods", or the control (it never compacts).
        for methods in (["warm_ledger", "fallback_relaxed_ledger"], ["warm_ledger", "full_context"]):
            with self.subTest(methods=methods), self.assertRaisesRegex(ValueError, "drift methods without a drift job"):
                run.plan({**config, "drift": {**config["drift"], "methods": methods}}, {"settings_known": ["warm"]})
        # A CLI method in "methods" cannot be a drift method: the CLI worker refuses a job of more than one cycle.
        for cli in ("codex_cli", "claude_code_cli"):
            with self.subTest(cli=cli), self.assertRaisesRegex(ValueError, "drift methods without a drift job"):
                run.plan({**config, "methods": [*config["methods"], cli],
                          "drift": {**config["drift"], "methods": ["warm_ledger", cli]}}, {"settings_known": ["warm"]})
        self.assertTrue(run.plan({**config, "methods": [*config["methods"], "codex_cli"]},
                                 {"settings_known": ["warm", "ledger"]})[0])
        # A drift method that the plugin cannot run is listed as skipped (it is a main method too).
        jobs, skipped = run.plan({**config, "methods": ["stock_hermes", "warm_ledger"]}, {"settings_known": []})
        self.assertEqual(skipped, [{"method": "warm_ledger", "reason": "plugin_setting_missing"}])
        self.assertEqual({job["method"] for job in jobs}, {"stock_hermes"})

    def test_plan_refuses_unknown_names(self):
        config = {"threshold": 0.95, "replicates": 1, "fixtures": ["s1-ledgerline"], "methods": ["stock_hermes"]}
        for changed in ({"fixtures": ["fake-name"]}, {"methods": ["fake_method"]},
                        {"drift": {"fixtures": ["fake-name"], "methods": ["stock_hermes"], "cycles": 3}}):
            with self.assertRaisesRegex(ValueError, "unknown fixture or method names"):
                run.plan({**config, **changed}, {"settings_known": []})

    def test_classify(self):
        self.assertEqual(run.classify({}, "full_context"), "none")
        self.assertEqual(run.classify({"compacted": True, "outcome": {"path": "warm"}}, "warm_ledger"), "warm")
        self.assertEqual(run.classify({"compress_error": "CompactionAborted", "outcome": {}}, "warm_ledger"),
                         "aborted")
        self.assertEqual(run.classify({"compacted": False, "outcome": {}}, "stock_hermes"), "no_change")
        static = {"compacted": True, "history": [{"role": "user", "content": run.STATIC_FALLBACK_TEXT}]}
        self.assertEqual(run.classify(static, "stock_hermes"), "static_fallback")
        preflight = {"compacted": True, "preflight": True, "outcome": {"path": "warm", "trigger": "preflight"}}
        self.assertEqual(run.classify(preflight, "warm_ledger"), "warm")
        self.assertEqual(run.classify({**static, "preflight": True}, "stock_hermes"), "static_fallback")
        self.assertEqual(run.classify({"main_failed": True}, "full_context"), "main_failed")

    def test_preflight_compaction_keeps_the_path_of_its_summary(self):
        # A ledger-only summary or the static fallback during the main turn is not a model summary.
        ledger = {"compacted": True, "preflight": True, "outcome": {"path": "ledger", "trigger": "preflight"}}
        self.assertEqual(run.classify(ledger, "warm_ledger"), "ledger")
        self.assertNotIn("preflight", run.ACCEPTED_PATHS)
        self.assertNotIn("ledger", run.ACCEPTED_PATHS)
        # Records made before this rule: the plugin path comes from the outcome; others need --rescore.
        self.assertEqual(run.stored_path({"path": "preflight", "outcome": {"path": "ledger"}}, "warm_ledger"), "ledger")
        self.assertEqual(run.stored_path({"path": "preflight", "outcome": {}}, "stock_hermes"), "preflight_unchecked")
        runs = self.synthetic_runs()
        runs[0]["cycles"][0].update(path="preflight", preflight=True, outcome={"path": "ledger"})
        entry = run.aggregate(runs, {"config": {}, "plugin": {}, "hermes": {}})["methods"]["warm_ledger"]
        self.assertEqual((entry["success"]["k"], entry["paths"], entry["preflight_runs"]), (0, {"ledger": 1}, 1))

    def synthetic_runs(self):
        """Fake worker output for one fixture: the control answers all, the ledger run all but one."""
        history, gold = fixtures.build(SCENARIOS[0])
        answers = {str(n): str(item["gold"]) if item["kind"] != "keywords" else " ".join(
            group[0] for group in item["gold"]) for n, item in enumerate(gold["questions"], 1)}
        numbers = list(range(1, len(gold["questions"]) + 1))
        records = []
        for method, wrong in (("warm_ledger", "1"), ("warm_no_ledger", None), ("stock_hermes", None)):
            given = {key: ("unknown" if key == wrong else value) for key, value in answers.items()}
            cycle = {"cycle": 1, "compacted": True, "outcome": {"path": "warm", "reason": "accepted"},
                     "history": history["rows"][-4:], "est_tokens_before": 100, "est_tokens_after": 10,
                     "rows_before": 10, "rows_after": 2, "removed": 8, "main_s": 1.0, "compress_s": 2.0,
                     "qa": {"batches": [{"numbers": numbers, "text": json.dumps(given)}],
                            "calls": [{"elapsed_s": 1.0, "prompt_tokens": 50}]}}
            scored = run.score_cycle(cycle, gold, history, method)
            records.append({"job": method, "fixture": SCENARIOS[0]["id"], "method": method, "rep": 1,
                            "cycles_requested": 1, "status": "ok", "cycles": [scored],
                            "harness": run.harness_id(FAKE_HARNESS), "sources": dict(FAKE_SOURCES)})
        return records

    def test_report_names_the_sources_of_each_run_folder(self):
        # Synthetic metadata: a stock-only first folder, a warm-only folder and a CLI-only folder added with --also.
        harness = {"run.py": "0" * 64}
        meta = {"config": {"model": "m", "methods": ["stock_hermes"]}, "plugin": {"commit": "a" * 40},
                "hermes": {"commit": "1" * 40, "clean": True}, "harness_sha256": harness,
                "merged_runs": [
                    {"config": {"methods": ["warm_ledger"]}, "plugin": {"commit": "b" * 40},
                     "hermes": {"commit": "1" * 40, "clean": True}, "harness_sha256": harness},
                    {"config": {"methods": ["codex_cli"]}, "plugin": {"commit": "c" * 40},
                     "hermes": {"commit": "2" * 40, "clean": True}, "harness_sha256": harness}]}
        lines = run.source_lines(meta)
        self.assertEqual(len(lines), 3)
        self.assertIn("methods stock_hermes;", lines[0])
        self.assertIn("plugin commit not used; Hermes commit `111111111111`", lines[0])
        self.assertIn("plugin commit `bbbbbbbbbbbb`; Hermes commit `111111111111`", lines[1])
        self.assertIn("plugin commit not used; Hermes commit not used", lines[2])
        self.assertIn(f"harness `{run.harness_id(harness)}`", lines[2])
        # The report header never labels a method with the commit of a folder that did not run it.
        text = run.markdown(run.aggregate(self.synthetic_runs(), meta), meta)
        for line in lines:
            self.assertIn(line, text)
        for unused in ("a" * 12, "c" * 12, "2" * 12):
            self.assertNotIn(unused, text)

    def test_failure_reasons_keep_the_compaction_error_code(self):
        runs = self.synthetic_runs()
        # A synthetic stock failure (a timeout) and a synthetic Codex failure (a failed turn with an error text).
        stock = runs[2]
        stock["cycles"][0].update(path="aborted", compacted=False, compress_error="TimeoutError")
        codex = json.loads(json.dumps(stock))
        codex.update(job="codex_cli", method="codex_cli")
        codex["cycles"][0].update(compress_error="turn error/x")
        result = run.aggregate(runs + [codex], {"config": {}, "plugin": {}, "hermes": {}})
        stock_entry, codex_entry = result["methods"]["stock_hermes"], result["methods"]["codex_cli"]
        self.assertEqual(stock_entry["failure_reasons"], {"aborted:TimeoutError": 1})
        self.assertEqual(codex_entry["failure_reasons"], {"aborted:turn_error_x": 1})
        self.assertEqual([item["compress_error"] for item in stock_entry["per_run_failed"]], ["TimeoutError"])
        self.assertEqual([item["compress_error"] for item in codex_entry["per_run_failed"]], ["turn_error_x"])
        self.assertIsNone(run.error_code(None))
        self.assertEqual(len(run.error_code("x" * 200)), 80)

    def test_hermes_and_cli_times_are_in_separate_columns(self):
        runs = self.synthetic_runs()
        # A synthetic CLI record: its main time is the wall time of the CLI command, its compaction time comes from
        # the session file. These are other intervals than the Hermes times.
        codex = json.loads(json.dumps(runs[-1]))
        codex.update(job="codex_cli", method="codex_cli")
        codex["cycles"][0].update(main_s=30.0, compress_s=4.0)
        result = run.aggregate(runs + [codex], {"config": {}, "plugin": {}, "hermes": {}})
        self.assertEqual(result["methods"]["codex_cli"]["latency_s"]["scope"], "cli")
        self.assertEqual(result["methods"]["warm_ledger"]["latency_s"]["scope"], "hermes")
        text = run.markdown(result, {"config": {}, "plugin": {}, "hermes": {}})
        self.assertIn("| Hermes main request s | CLI compaction s (all runs) | CLI main command s |", text)
        rows = {line.split(" | ")[0][2:]: line.rstrip(" |").split(" | ")[-4:] for line in text.splitlines()
                if line.startswith("| ") and line.count(" | ") == 8}
        def time(name, key):
            return run.num(result["methods"][name]["latency_s"][key])

        self.assertEqual(rows[run.METHODS["codex_cli"]["label"]],
                         ["-", "-", time("codex_cli", "compress"), time("codex_cli", "main_request")])
        self.assertIn("30.0", rows[run.METHODS["codex_cli"]["label"]][3])
        self.assertEqual(rows[run.METHODS["warm_ledger"]["label"]],
                         [time("warm_ledger", "compress"), time("warm_ledger", "main_request"), "-", "-"])
        self.assertIn(run.TIME_SCOPES["cli"]["main_request"], text)
        # A merged run folder ran at another time: the report states the limit.
        self.assertNotIn(run.MERGED_LIMIT, text)
        self.assertIn(run.MERGED_LIMIT, run.markdown(result, {"config": {}, "plugin": {}, "hermes": {},
                                                              "merged_runs": [{}]}))

    def test_rescore_uses_the_current_gold_and_counts_missing_raw_output(self):
        spec = next(item for item in SCENARIOS if item["id"] == "s4-orchard")
        history, gold = fixtures.build(spec)
        number = next(n for n, item in enumerate(gold["questions"], 1) if item["id"] == "s4-orchard:40")
        # Synthetic raw output in the gold batch plan: answer 40 repeats the question text only.
        def answer(n: int) -> str:
            return gold["questions"][n - 1]["question"] if n == number else "unknown"

        cycle = {"cycle": 1, "compacted": True, "history": history["rows"][-4:],
                 "qa": {"batches": planned_batches(gold, answer)}}
        stale = run.score_cycle(cycle, gold, history, "warm_ledger")
        stale["marks"][number - 1] = "1" + stale["marks"][number - 1][1:]
        records = [{"job": f"{spec['id']}|warm_ledger|r{rep}|c1", "fixture": spec["id"], "method": "warm_ledger",
                    "cycles": [json.loads(json.dumps(stale))]} for rep in (1, 2)]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "raw").mkdir()
            (Path(folder) / "raw" / f"{spec['id']}__warm_ledger__r1__c1.json").write_text(
                json.dumps({"cycles": [cycle]}), encoding="utf-8")
            counts = run.rescore(Path(folder), records)
        self.assertEqual(counts, {"jobs_rescored": 1, "jobs_without_raw": 1, "marks_changed": 1,
                                  "jobs_failed_incomplete": 0, "jobs_other_qa_prompt": 0})
        self.assertEqual(records[0]["cycles"][0]["marks"][number - 1][0], "0")
        # Raw answers to other batch prompts (consecutive batches of 12) are not rescored, and the report refuses.
        consecutive = [list(range(start, min(start + 12, len(gold["questions"])) + 1))
                       for start in range(1, len(gold["questions"]) + 1, 12)]
        other = {**cycle, "qa": {"batches": [{"numbers": batch, "text": json.dumps({str(n): "unknown" for n in batch})}
                                             for batch in consecutive]}}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "raw").mkdir()
            (Path(folder) / "raw" / f"{spec['id']}__warm_ledger__r1__c1.json").write_text(
                json.dumps({"cycles": [other]}), encoding="utf-8")
            counts = run.rescore(Path(folder), records[:1])
        self.assertEqual((counts["jobs_other_qa_prompt"], counts["jobs_rescored"]), (1, 0))
        with self.assertRaisesRegex(SystemExit, "other batches"):
            run.refuse_stale({"folder": "f", **counts})
        self.assertEqual(records[1]["cycles"][0]["marks"][number - 1][0], "1")
        # The same batch layout with another question text (an old prompt), or with no prompt hash: not rescored.
        old_gold = json.loads(json.dumps(gold))
        old_gold["questions"][number - 1]["question"] = "An older text of this question?"
        stale_prompt = {**cycle, "qa": {"batches": planned_batches(old_gold, lambda _n: "unknown")}}
        no_hash = {**cycle, "qa": {"batches": [{k: v for k, v in batch.items() if k != "prompt_sha256"}
                                               for batch in cycle["qa"]["batches"]]}}
        for raw_cycle in (stale_prompt, no_hash):
            with tempfile.TemporaryDirectory() as folder:
                (Path(folder) / "raw").mkdir()
                (Path(folder) / "raw" / f"{spec['id']}__warm_ledger__r1__c1.json").write_text(
                    json.dumps({"cycles": [raw_cycle]}), encoding="utf-8")
                counts = run.rescore(Path(folder), json.loads(json.dumps(records[:1])))
            self.assertEqual((counts["jobs_other_qa_prompt"], counts["jobs_rescored"]), (1, 0))

    def test_size_check_has_no_data_when_no_ledger_run_compacts(self):
        runs = self.synthetic_runs()
        runs[0]["cycles"][0]["compacted"] = False
        meta = {"config": {}, "plugin": {}, "hermes": {}}
        result = run.aggregate(runs, meta)
        self.assertIsNone(result["methods"]["warm_ledger"]["tokens"]["est_after"]["mean"])
        self.assertIsNone(result["decision"]["checks"]["H4_smaller_than_stock"])
        self.assertIn("comparison with the built-in compressor is unavailable",
                      " ".join(result["decision"]["verdict"]))

    def test_run_stops_before_jobs_when_hermes_is_not_clean(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "hermes"
            source.mkdir()
            self.assertEqual(run.hermes_state(source), {"commit": None, "clean": False})
            config = {"hermes_source": str(source), "plugin_source": str(Path(run.__file__).parents[2]),
                      "history_dir": str(Path(folder) / "fixtures"), "threshold": 0.95, "methods": []}
            (Path(folder) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not a clean Git checkout"):
                run.main(["run", "--config", str(Path(folder) / "config.json"), "--out", str(Path(folder) / "out")])
            self.assertFalse((Path(folder) / "out" / "runs.jsonl").exists())
            # The gold files plan the QA batches: another batch size is refused before any job.
            (Path(folder) / "config.json").write_text(json.dumps({**config, "qa_batch": 46}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "qa_batch is fixed"):
                run.main(["run", "--config", str(Path(folder) / "config.json"), "--out", str(Path(folder) / "out")])
            self.assertFalse((Path(folder) / "out" / "runs.jsonl").exists())

    def test_failed_and_missing_jobs_count_as_failures(self):
        runs = self.synthetic_runs()
        fixture = SCENARIOS[0]["id"]
        timed_out = {"job": "warm_ledger-r2", "fixture": fixture, "method": "warm_ledger", "rep": 2,
                     "cycles_requested": 1, "status": "failed", "error": "job_timeout"}
        first_try = {**timed_out, "job": "warm_ledger-r3", "rep": 3}
        retried = {**runs[0], "job": "warm_ledger-r3", "rep": 3}
        planned = {job: {"id": job, "fixture": fixture, "method": "warm_ledger", "cycles": 1}
                   for job in ("warm_ledger", "warm_ledger-r2", "warm_ledger-r3", "warm_ledger-r4")}
        latest = runs + [timed_out, retried]
        result = run.aggregate(latest, {"config": {}, "plugin": {}, "hermes": {}},
                               attempts=latest + [first_try], planned=planned)
        entry = result["methods"]["warm_ledger"]
        # 2 accepted of 4 planned: the timeout and the job that never ran are failures, not left out.
        self.assertEqual((entry["success"]["k"], entry["success"]["n"]), (2, 4))
        self.assertEqual(entry["planned_runs"], 4)
        self.assertEqual(entry["not_run"], 1)
        self.assertEqual(entry["failure_reasons"], {"job:job_timeout": 1, "not_run": 1})
        self.assertEqual([item["job"] for item in entry["failed_attempts"]], ["warm_ledger-r2", "warm_ledger-r3"])
        self.assertEqual(entry["attempts"], 4)
        self.assertEqual(entry["categories_failures_wrong"]["goal"]["n"], 4 * entry["categories"]["goal"]["n"] // 2)
        self.assertLess(entry["overall_failures_wrong"]["rate"], entry["overall"]["rate"])
        text = run.markdown(result, {"config": {}, "plugin": {}, "hermes": {}})
        self.assertIn("| Warm path + ledger | 2 / 4 | 4 (2) |", text)

    def test_timing_uses_every_completed_compaction(self):
        runs = self.synthetic_runs()
        ok = runs[0]
        # The compaction completed, then the questions failed: no accuracy, but the time counts.
        qa_failed = {**ok, "job": "warm_ledger-r2", "rep": 2, "status": "failed", "error": "qa_request:no_content",
                     "cycles": [{**ok["cycles"][0], "compress_s": 8.0, "main_s": 3.0}]}
        # An earlier attempt of a job that a resume ran again, and an attempt that the server check marked invalid.
        earlier = {**ok, "job": "warm_ledger-r3", "rep": 3, "status": "failed", "error": "job_timeout",
                   "cycles": [{**ok["cycles"][0], "compress_s": 5.0, "compacted": False, "main_s": 2.0}]}
        retried = {**ok, "job": "warm_ledger-r3", "rep": 3}
        invalid = {**ok, "job": "warm_ledger-r4", "rep": 4, "status": "invalid",
                   "cycles": [{**ok["cycles"][0], "compress_s": 100.0}]}
        main_failed = {**ok, "job": "warm_ledger-r5", "rep": 5, "status": "failed", "error": "main_turn_failed",
                       "cycles": [{**ok["cycles"][0], "compress_s": None, "main_s": 50.0, "main_failed": True}]}
        latest = runs + [qa_failed, retried, invalid, main_failed]
        result = run.aggregate(latest, {"config": {}, "plugin": {}, "hermes": {}},
                               attempts=latest + [earlier])
        entry = result["methods"]["warm_ledger"]
        latency = entry["latency_s"]
        # compress: 2.0 (ok), 8.0 (questions failed), 2.0 (retried), 5.0 (earlier attempt); not the invalid one.
        self.assertEqual((latency["compress"]["n"], latency["compress"]["mean"]), (4, 17.0 / 4))
        self.assertEqual((latency["compress_compacted"]["n"], latency["compress_compacted"]["mean"]), (3, 4.0))
        # main_request: every completed main turn, not a failed one.
        self.assertEqual((latency["main_request"]["n"], latency["main_request"]["mean"]), (4, 7.0 / 4))
        self.assertEqual(latency["timed_attempts"], 5)
        # Accuracy uses the scored runs only: the ok run and the retried run.
        self.assertEqual(entry["compacted_runs"], 2)

    def test_rescore_reads_only_raw_files_of_a_verified_run(self):
        spec = next(item for item in SCENARIOS if item["id"] == "s4-orchard")
        history, gold = fixtures.build(spec)
        cycle = {"cycle": 1, "compacted": True, "history": history["rows"][-4:],
                 "qa": {"batches": planned_batches(gold, lambda _n: "unknown")}}
        data = json.dumps({"cycles": [cycle]}).encode("utf-8")
        name = f"{spec['id']}__warm_ledger__r1__c1.json"
        meta = {"schema": "s", **FAKE_RUNTIME_META, "config": {"model": "m", "methods": ["warm_ledger"]},
                "hermes": {"commit": "a"}, "plugin": {"files_sha256": {"x": "1"}}}

        def record(**extra):
            return {"job": f"{spec['id']}|warm_ledger|r1|c1", "fixture": spec["id"], "method": "warm_ledger",
                    "status": "ok", "started_utc": "2026-01-01T00:00:00Z",
                    "cycles": [run.score_cycle(cycle, gold, history, "warm_ledger")], **extra}

        def folder_with(base: Path, label: str, meta_value: dict, attempts: list[dict], raw: bytes) -> Path:
            folder = base / label
            (folder / "raw").mkdir(parents=True)
            (folder / "meta.json").write_text(json.dumps(meta_value), encoding="utf-8")
            (folder / "runs.jsonl").write_text("".join(json.dumps(item) + "\n" for item in attempts),
                                               encoding="utf-8")
            (folder / "raw" / name).write_bytes(raw)
            return folder

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / "out"
            (out / "raw").mkdir(parents=True)
            # A record with a hash reads only the file with that hash.
            (out / "raw" / name).write_bytes(data)
            good = hashlib.sha256(data).hexdigest()
            self.assertEqual(run.rescore(out, [record(raw_sha256=good)], meta=meta)["jobs_rescored"], 1)
            (out / "raw" / name).write_bytes(data + b" ")
            self.assertEqual(run.rescore(out, [record(raw_sha256=good)], meta=meta)["jobs_without_raw"], 1)
            # A --raw-dir of the same run identity, with the same attempt (job and start time).
            (out / "raw" / name).unlink()
            same = folder_with(base, "same", meta, [record()], data)
            self.assertEqual(run.rescore(out, [record()], [same / "raw"], meta)["jobs_rescored"], 1)
            self.assertEqual(run.rescore(out, [record(raw_sha256=good)], [same / "raw"], meta)["jobs_rescored"], 1)
            # The same job name from another attempt is not the raw file of this record.
            other_attempt = folder_with(base, "other", meta, [record(started_utc="2026-02-02T00:00:00Z")], data)
            self.assertEqual(run.rescore(out, [record()], [other_attempt / "raw"], meta)["jobs_without_raw"], 1)
            # A --raw-dir of another run, or a folder that is not the raw folder of a run, stops the report.
            changed = folder_with(base, "changed", {**meta, "hermes": {"commit": "b"}}, [record()], data)
            with self.assertRaisesRegex(SystemExit, "differs in Hermes commit"):
                run.rescore(out, [record()], [changed / "raw"], meta)
            with self.assertRaisesRegex(SystemExit, "not the raw folder of a run folder"):
                run.rescore(out, [record()], [same], meta)

    def test_run_job_names_its_raw_file_by_hash(self):
        spec = SCENARIOS[0]
        with tempfile.TemporaryDirectory() as folder:
            history_dir = Path(folder) / "fixtures"
            fixtures.write_all(history_dir, history_dir / "gold")
            out_dir = Path(folder) / "out"
            out_dir.mkdir()

            def fake_worker(argv, _timeout, **_kwargs):
                # A fake worker report (synthetic): one job without cycles.
                Path(argv[argv.index("--out") + 1]).write_text(
                    json.dumps({"status": "ok", "error": None, "cycles": []}), encoding="utf-8")
                return types.SimpleNamespace(returncode=0, stderr=b"")

            job = {"id": f"{spec['id']}|stock_hermes|r1|c1", "fixture": spec["id"], "method": "stock_hermes",
                   "rep": 1, "cycles": 1, "settings": {}}
            config = {"hermes_python": "fake-python", "hermes_source": "h", "plugin_source": "p",
                      "endpoint": "http://127.0.0.1:9/v1", "model": "m", "context_length": 8, "threshold": 0.95,
                      "keep_text": True}
            with mock.patch.object(run, "run_tree", side_effect=fake_worker):
                record = run.run_job(job, config, out_dir, history_dir)
            raw = out_dir / "raw" / f"{spec['id']}__stock_hermes__r1__c1.json"
            self.assertEqual(record["raw_sha256"], run.sha256_file(raw))

    def test_failed_job_with_a_partial_cycle_is_not_scored(self):
        runs = self.synthetic_runs()
        # A failed main turn: the worker sends the partial cycle (no compaction step, no rows_after).
        partial = {**runs[0]["cycles"][0], "main_failed": True, "path": "main_failed", "compacted": False,
                   "rows_after": None, "est_tokens_after": None}
        runs[0].update(status="failed", error="main_turn_failed:empty_reply", cycles=[partial])
        entry = run.aggregate(runs, {"config": {}, "plugin": {}, "hermes": {}})["methods"]["warm_ledger"]
        self.assertEqual((entry["success"]["k"], entry["compacted_runs"]), (0, 0))
        self.assertEqual(entry["failure_reasons"], {"job:main_turn_failed:empty_reply": 1})
        self.assertEqual(entry["paths"], {})

    def test_a_good_compaction_counts_when_a_later_step_fails(self):
        # The compaction completed (path warm); the questions then failed the job. Success and paths count the
        # compaction; accuracy leaves the cycle out; the job is still a job failure.
        runs = self.synthetic_runs()
        runs[0].update(status="failed", error="qa_request:TimeoutError")
        result = run.aggregate(runs, {"config": {}, "plugin": {}, "hermes": {}})
        entry = result["methods"]["warm_ledger"]
        self.assertEqual((entry["success"]["k"], entry["success"]["n"]), (1, 1))
        # Sizes use every completed compaction that compacted, so H4 has data; accuracy stays scored-only.
        self.assertEqual((entry["tokens"]["sized_runs"], entry["tokens"]["est_after"]["mean"]), (1, 10))
        self.assertEqual(entry["tokens"]["before_not_measured_runs"], 1)
        self.assertIsNotNone(result["decision"]["checks"]["H4_smaller_than_stock"])
        self.assertIsNone(entry["latency_s"]["questions"]["mean"])
        self.assertEqual(entry["paths"], {"warm": 1})
        self.assertEqual(entry["failure_reasons"], {})
        self.assertEqual((entry["compacted_runs"], entry["overall"]["n"], entry["job_failures"]), (0, 0, 1))
        self.assertEqual(entry["per_run_failed"][0]["error"], "qa_request:TimeoutError")
        # The same rule in the drift section: cycle 3 completed, its questions failed.
        cycle = self.synthetic_runs()[0]["cycles"][0]
        drift = {"job": "d1", "fixture": SCENARIOS[0]["id"], "method": "warm_ledger", "rep": 1,
                 "cycles_requested": 3, "status": "failed", "error": "qa_request:TimeoutError",
                 "cycles": [cycle, cycle, cycle]}
        result = run.aggregate([drift], {"config": {}, "plugin": {}, "hermes": {}})
        cycles = result["drift"]["warm_ledger"]["cycles"]
        self.assertEqual([item["paths"] for item in cycles], [{"warm": 1}] * 3)
        self.assertEqual([item["scored_runs"] for item in cycles], [0, 0, 0])
        self.assertEqual([item["est_tokens_after"]["mean"] for item in cycles], [10, 10, 10])
        # An invalid attempt (changed server) is in no population.
        runs = self.synthetic_runs()
        runs[0].update(status="invalid")
        entry = run.aggregate(runs, {"config": {}, "plugin": {}, "hermes": {}})["methods"]["warm_ledger"]
        self.assertEqual((entry["success"]["k"], entry["paths"], entry["latency_s"]["timed_attempts"]), (0, {}, 0))

    def test_report_works_on_a_run_folder_with_zero_records(self):
        plugin = run.plugin_state(Path(run.__file__).parents[2])
        meta = {"schema": "s", "plugin": plugin, "hermes": {"commit": None, "clean": True}, "started_utc": "2026",
                "config": {"model": "m", "context_length": 8, "threshold": 0.95, "replicates": 1,
                           "fixtures": [SCENARIOS[0]["id"]], "methods": ["stock_hermes", "warm_ledger"],
                           "drift": {"fixtures": [SCENARIOS[0]["id"]], "methods": ["warm_ledger"], "cycles": 3}},
                "gold_sha256": run.current_gold_sha256(), **FAKE_RUNTIME_META}
        with tempfile.TemporaryDirectory() as folder:
            first, raw_run = Path(folder) / "a", Path(folder) / "b"
            for path in (first, raw_run):
                path.mkdir()
                (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (raw_run / "raw").mkdir()
            # No runs.jsonl in either folder: a stop before the first record.
            self.assertEqual(run.load_attempts(first), [])
            args = types.SimpleNamespace(out=str(first), json=str(first / "r.json"), md=str(first / "r.md"),
                                         also=[str(raw_run)], rescore=False, raw_dir=None)
            with contextlib.redirect_stdout(io.StringIO()):
                run.command_report(args)
            result = json.loads((first / "r.json").read_text(encoding="utf-8"))["results"]
            self.assertEqual(result["methods"]["warm_ledger"]["not_run"], 1)
            self.assertEqual(result["methods"]["warm_ledger"]["success"]["k"], 0)
            self.assertEqual(result["drift"]["warm_ledger"]["runs"], 0)
            self.assertIn("unavailable", (first / "r.md").read_text(encoding="utf-8"))
            # --rescore with a --raw-dir of a run folder without records.
            args = types.SimpleNamespace(out=str(first), json=str(first / "r2.json"), md=None, also=None,
                                         rescore=True, raw_dir=[str(raw_run / "raw")])
            with contextlib.redirect_stdout(io.StringIO()):
                run.command_report(args)
            self.assertTrue((first / "r2.json").exists())
            # A run with a key: the --raw-dir check needs the --same-credential confirmation, and accepts it.
            keyed = {**meta, "config": {**meta["config"], "api_key_env": "WC_FAKE_KEY_ENV"}}
            for path in (first, raw_run):
                (path / "meta.json").write_text(json.dumps(keyed), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "--same-credential"):
                run.command_report(types.SimpleNamespace(**{**vars(args), "json": str(first / "r3.json")}))
            with contextlib.redirect_stdout(io.StringIO()):
                run.command_report(types.SimpleNamespace(**{**vars(args), "json": str(first / "r4.json"),
                                                            "same_credential": True}))
            self.assertTrue((first / "r4.json").exists())

    def test_partial_last_record_is_recovered_and_earlier_damage_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            path = out / "runs.jsonl"
            first = json.dumps({"job": "a", "status": "ok"}) + "\n"
            # A stop during the append of the second record: a partial line without a newline (synthetic records).
            path.write_bytes((first + '{"job": "b", "sta').encode("utf-8"))
            self.assertEqual(run.load_runs(out), [{"job": "a", "status": "ok"}])
            self.assertTrue(run.repair_runs(out))
            self.assertEqual(path.read_text(encoding="utf-8"), first)
            self.assertFalse(run.repair_runs(out))
            # The next append starts on a new line, so the file stays readable.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"job": "b", "status": "ok"}) + "\n")
            self.assertEqual([item["job"] for item in run.load_attempts(out)], ["a", "b"])
            # A complete line that is not a JSON record is damage before the last line: stop, change nothing.
            for bad in ('{"job": "b", "sta\n', "[1]\n"):
                with self.subTest(bad=bad):
                    path.write_text(first + bad + first, encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, "line 2 is not a JSON record"):
                        run.load_attempts(out)
                    self.assertFalse(run.repair_runs(out))

    def test_planned_jobs_include_the_plan_of_each_resume(self):
        config = {"threshold": 0.95, "replicates": 1, "fixtures": ["s1-ledgerline"], "methods": ["stock_hermes"]}
        meta = {"config": config, "plugin": {"settings_known": []}}
        self.assertEqual(len(run.planned_jobs(meta)), 1)
        # A resume with five replicates plans four more jobs, also before they run.
        meta["resumes"] = [{"plan": {**config, "replicates": 5}}, {"plan": {**config, "methods": ["full_context"]}}]
        planned = run.planned_jobs(meta)
        self.assertEqual(sorted({job["method"] for job in planned.values()}), ["full_context", "stock_hermes"])
        self.assertEqual(len(planned), 6)

    def test_meta_and_report_keep_no_override_values(self):
        # FAKE values only: a route override with a key and a header, and a URL with a password and a query.
        marker = "fake-value-123"
        config = {"model": "m", "threshold": 0.95, "endpoint": f"http://user:{marker}@127.0.0.1:9/v1?key={marker}",
                  "hermes_config": {"model": {"api_key": marker, "headers": {"X-Token": marker}}},
                  "qa_extra_body": {"token": marker}, "unknown_key": marker, "hermes_source": "/private"}
        public = run.public_config(config)
        self.assertNotIn(marker, json.dumps(public))
        self.assertNotIn("hermes_source", public)
        self.assertEqual(public["endpoint"], "http://127.0.0.1:9")
        self.assertEqual(public["hermes_config"]["types"], {"model.api_key": "str", "model.headers.X-Token": "str"})
        self.assertEqual(public["hermes_config"], {"redacted": True, "types": public["hermes_config"]["types"]})
        # A resume compares the digests of the raw values; a changed value changes the hash.
        changed = {**config, "hermes_config": {"model": {"api_key": "other", "headers": {"X-Token": marker}}}}
        self.assertNotEqual(run.config_hash(changed), run.config_hash(config))
        self.assertEqual(run.digest_hash(run.config_digests(config)), run.config_hash(config))
        # A user value with a "redacted" key is not trusted as a redaction record.
        forged = {"hermes_config": {"redacted": True, "types": {}, "api_key": marker},
                  "drift": {"fixtures": [], "methods": [], "cycles": 3, "note": marker}}
        self.assertNotIn(marker, json.dumps(run.public_config(forged)))
        # A run folder made before the rule keeps raw values in meta.json: the report redacts them.
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            meta = {"schema": "s", "config": config, "plugin": {}, "hermes": {}}
            (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (out / "runs.jsonl").write_text("", encoding="utf-8")
            args = types.SimpleNamespace(out=str(out), json=str(out / "r.json"), md=None, also=None, rescore=False,
                                         raw_dir=None)
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                run.command_report(args)
            report = json.loads((out / "r.json").read_text(encoding="utf-8"))
            self.assertNotIn(marker, json.dumps(report))
            # The hash of a redacted value stays in the local meta.json only: a short credential can be guessed.
            self.assertEqual(report["meta"]["config"]["hermes_config"],
                             {"redacted": True, "types": public["hermes_config"]["types"]})

    def test_endpoint_keeps_no_path_and_no_remote_host(self):
        marker = "fake-path-token-789"
        public = run.public_config({"endpoint": f"https://gateway.example/token/{marker}/v1",
                                    "anthropic_base_url": "http://localhost:1234/v1"})
        self.assertEqual(public["endpoint"], "https://<remote host>")
        self.assertEqual(public["anthropic_base_url"], "http://localhost:1234")
        self.assertNotIn(marker, json.dumps(public))
        self.assertNotIn("gateway.example", json.dumps(public))

    def test_rescore_refuses_jobs_without_raw_output(self):
        runs = self.synthetic_runs()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "meta.json").write_text(json.dumps({"schema": "s", "config": {}, "plugin": {}, "hermes": {},
                                                       "harness_sha256": FAKE_HARNESS}), encoding="utf-8")
            (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")
            args = types.SimpleNamespace(out=str(out), json=str(out / "r.json"), md=None, also=None, rescore=True,
                                         raw_dir=None)
            with self.assertRaisesRegex(SystemExit, "3 jobs .* without a verified raw output"):
                run.command_report(args)
            self.assertFalse((out / "r.json").exists())

    def test_every_job_uses_the_sources_of_the_run_start(self):
        plugin = {"commit": "p", "files_sha256": {"x": "1"}, "settings_known": []}
        with tempfile.TemporaryDirectory() as folder:
            # The generated session files are the same for every folder (fixed seeds).
            fixtures.write_all(Path(folder), Path(folder) / "gold-check")
            history = run.history_hashes({"history_dir": folder})
        self.assertNotIn(None, history.values())
        start = {"harness": run.harness_hashes(), "plugin": plugin["files_sha256"],
                 "hermes": {"commit": "a", "clean": True}, "gold": run.current_gold_sha256(), "history": history,
                 "cli": {}, "runtime": FAKE_RUNTIME["sha256"]}

        def fake_job(job, *_args):
            return {"job": job["id"], "fixture": job["fixture"], "method": job["method"], "rep": job["rep"],
                    "cycles_requested": 1, "status": "ok", "error": None, "cycles": [], "elapsed_s": 1.0,
                    "harness": run.harness_id(run.harness_hashes())}

        # Snapshots: (before job 1, after job 1, before job 2). A change stops the run or marks the job invalid.
        edited = {**history, "s1-ledgerline": "0" * 64}
        cases = (([start, start, {**start, "plugin": {"x": "2"}}], "sources changed before .*r2.*: plugin", "ok"),
                 ([start, start, {**start, "history": edited}], "sources changed before .*r2.*: history", "ok"),
                 ([start, {**start, "hermes": {"commit": "b", "clean": True}}], "check failed after .*r1", "invalid"))
        for snapshots, stop, status in cases:
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "out"
                config = {"hermes_source": "h", "plugin_source": "p", "history_dir": str(Path(folder) / "fixtures"),
                          "threshold": 0.95, "methods": ["stock_hermes"], "fixtures": ["s1-ledgerline"],
                          "replicates": 2, "endpoint": "http://127.0.0.1:9/v1", "model": "m"}
                (Path(folder) / "config.json").write_text(json.dumps(config), encoding="utf-8")
                with mock.patch.object(run, "hermes_state", return_value={"commit": "a", "clean": True}), \
                        mock.patch.object(run, "plugin_state", return_value=plugin), \
                        mock.patch.object(run, "cli_state", return_value={}), \
                        mock.patch.object(run, "runtime_state", return_value=FAKE_RUNTIME), \
                        mock.patch.object(run, "source_snapshot", side_effect=snapshots), \
                        mock.patch.object(run, "run_job", side_effect=fake_job), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, stop):
                        run.main(["run", "--config", str(Path(folder) / "config.json"), "--out", str(out)])
                records = run.load_attempts(out)
                self.assertEqual([record["status"] for record in records], [status])
                self.assertEqual(records[0]["sources"], {"plugin_files": run.digest_id({"x": "1"}),
                                                         "hermes_commit": "a", "gold": run.digest_id(start["gold"])})
                if status == "invalid":
                    self.assertEqual(records[0]["error"], "sources_changed:hermes")
                # The report accepts the records that meta.json accounts for.
                meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
                self.assertEqual(run.record_mismatches(meta, records), [])

    def test_snapshot_hashes_the_generated_session_files(self):
        with tempfile.TemporaryDirectory() as folder:
            fixtures.write_all(Path(folder), Path(folder) / "gold-check")
            config = {"history_dir": folder, "plugin_source": str(Path(run.__file__).parents[2]),
                      "hermes_source": folder, "methods": ["stock_hermes"]}
            with mock.patch.object(run, "runtime_state", return_value=FAKE_RUNTIME):
                before = run.source_snapshot(config)
                # A synthetic edit of one session file during a run changes the snapshot (the next job stops).
                path = Path(folder) / "s1-ledgerline.history.json"
                path.write_text(path.read_text(encoding="utf-8").replace("Continue", "Proceed", 1), encoding="utf-8")
                after = run.source_snapshot(config)
            self.assertEqual([key for key in before if before[key] != after[key]], ["history"])

    def test_report_refuses_records_that_meta_does_not_account_for(self):
        meta = {"schema": "s", "config": {}, "hermes": {"commit": "a"}, "plugin": {"files_sha256": {"x": "1"}},
                "gold_sha256": {"g": "1"}, "harness_sha256": FAKE_HARNESS,
                "resumes": [{"harness_sha256": {"run.py": "1" * 64}}]}
        sources = {"plugin_files": run.digest_id({"x": "1"}), "hermes_commit": "a", "gold": run.digest_id({"g": "1"})}
        runs = self.synthetic_runs()
        for record in runs:
            record["sources"] = dict(sources)
        runs[1]["harness"] = run.harness_id({"run.py": "1" * 64})
        self.assertEqual(run.record_mismatches(meta, runs), [])
        # A missing field is a mismatch, not "nothing to check": no sources, or a missing key.
        for changed in ({"sources": None}, {"sources": {k: v for k, v in sources.items() if k != "gold"}},
                        {"harness": None}):
            with self.subTest(changed=changed):
                self.assertEqual(len(run.record_mismatches(meta, [{**runs[0], **changed}])), 1)
        runs[1]["harness"] = run.harness_id({"run.py": "2" * 64})
        runs[2]["sources"] = {**sources, "hermes_commit": "b"}
        self.assertEqual(len(run.record_mismatches(meta, runs)), 2)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")
            args = types.SimpleNamespace(out=str(out), json=str(out / "r.json"), md=None, also=None, rescore=False,
                                         raw_dir=None)
            with self.assertRaisesRegex(SystemExit, "2 records .* sources that meta.json does not record"):
                run.command_report(args)
            self.assertFalse((out / "r.json").exists())

    def test_every_aggregate_publishes_its_items(self):
        runs = self.synthetic_runs()
        runs[0]["started_utc"] = "2026-01-02T00:00:00Z"
        # An earlier attempt of the same job (a resume ran it again): its time is in the mean and in the list.
        earlier = json.loads(json.dumps(runs[0]))
        earlier.update(status="failed", error="qa_request:TimeoutError", started_utc="2026-01-01T00:00:00Z")
        earlier["cycles"][0]["compress_s"] = 4.0
        fixture = SCENARIOS[0]["id"]
        planned = {"warm_ledger": {"id": "warm_ledger", "fixture": fixture, "method": "warm_ledger", "cycles": 1},
                   "w2": {"id": "w2", "fixture": fixture, "method": "warm_ledger", "cycles": 1}}
        result = run.aggregate(runs, {"config": {}, "plugin": {}, "hermes": {}}, attempts=[earlier] + runs,
                               planned=planned)
        entry = result["methods"]["warm_ledger"]
        timing = entry["timing_attempts"]
        self.assertEqual([item["compress_s"] for item in timing], [4.0, 2.0])
        self.assertEqual(entry["latency_s"]["compress"]["mean"], 3.0)
        self.assertEqual(len(timing), entry["latency_s"]["timed_attempts"])
        self.assertEqual([item["started_utc"] for item in timing], ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"])
        self.assertEqual([item["est_tokens_after"] for item in entry["sizes_per_run"]], [10])
        self.assertEqual(len(entry["sizes_per_run"]), entry["tokens"]["sized_runs"])
        self.assertEqual(entry["not_run_jobs"], ["w2"])
        self.assertEqual(entry["failed_attempts"][0]["started_utc"], "2026-01-01T00:00:00Z")
        # Each item carries the job, the attempt, and its source identity.
        for item in timing + entry["sizes_per_run"] + entry["per_run"] + entry["failed_attempts"]:
            self.assertTrue({"job", "started_utc", "status", "harness", "sources"} <= set(item), item)
            self.assertEqual(item["harness"], run.harness_id(FAKE_HARNESS))
        # Drift: the items of each cycle number.
        cycle = self.synthetic_runs()[0]["cycles"][0]
        drift = {"job": "d1", "fixture": fixture, "method": "warm_ledger", "rep": 1, "cycles_requested": 3,
                 "status": "ok", "cycles": [cycle, cycle, cycle], "harness": run.harness_id(FAKE_HARNESS)}
        drift_entry = run.aggregate([drift], {"config": {}, "plugin": {}, "hermes": {}})["drift"]["warm_ledger"]
        self.assertEqual([c["path"] for c in drift_entry["per_run"][0]["cycles"]], ["warm"] * 3)
        self.assertEqual([c["est_tokens_after"] for c in drift_entry["per_run"][0]["cycles"]], [10] * 3)
        self.assertTrue(drift_entry["per_run"][0]["in_cycle_1_and_3"])
        self.assertEqual(sum(c["right"] for c in drift_entry["per_run"][0]["cycles"][:1]),
                         drift_entry["cycles"][0]["accuracy"]["k"])
        json.dumps(result)

    def test_drift_leaves_out_cycles_without_a_compaction(self):
        cycle = self.synthetic_runs()[0]["cycles"][0]
        failed = {**cycle, "compacted": False, "path": "aborted", "marks": ["0000"] * len(cycle["marks"])}
        drift = {"job": "d1", "fixture": SCENARIOS[0]["id"], "method": "warm_ledger", "rep": 1,
                 "cycles_requested": 3, "status": "ok", "cycles": [cycle, cycle, failed]}
        result = run.aggregate([drift], {"config": {}, "plugin": {}, "hermes": {}})
        cycles = result["drift"]["warm_ledger"]["cycles"]
        self.assertEqual((cycles[2]["scored_runs"], cycles[2]["accuracy"]["n"]), (0, 0))
        self.assertEqual(cycles[2]["paths"], {"aborted": 1})
        # Cycle 3 has no answers: D1 has no data (it does not pass), it is not a false failure.
        self.assertEqual(result["drift"]["warm_ledger"]["cycle_1_and_3"]["runs"], 0)
        self.assertIsNone(result["decision"]["checks"]["D1_drift_within_10_points"])
        second = {**drift, "job": "d2", "cycles": [cycle, failed, cycle]}
        result = run.aggregate([drift, second], {"config": {}, "plugin": {}, "hermes": {}})
        self.assertEqual(result["drift"]["warm_ledger"]["cycle_1_and_3"]["runs"], 1)
        self.assertTrue(result["decision"]["checks"]["D1_drift_within_10_points"])

    def test_every_report_section_lists_each_planned_method(self):
        # Only warm_ledger has records. warm_no_ledger and a warm_ledger drift job are planned but have no record.
        runs = [run for run in self.synthetic_runs() if run["method"] == "warm_ledger"]
        fixture = SCENARIOS[0]["id"]
        planned = {"a": {"id": "a", "fixture": fixture, "method": "warm_no_ledger", "rep": 1, "cycles": 1},
                   "d": {"id": "d", "fixture": fixture, "method": "warm_ledger", "rep": 1, "cycles": 3}}
        meta = {"config": {"model": "fake", "context_length": 1, "threshold": 0.95, "replicates": 1},
                "plugin": {"commit": None}, "hermes": {"commit": None, "clean": True}, "started_utc": "2026"}
        result = run.aggregate(runs, meta, planned=planned)
        self.assertEqual(result["methods"]["warm_no_ledger"]["runs"], 0)
        # Paired: each planned pair is listed with 0 pairs and no difference.
        pairs = [item for item in result["paired"] if (item["first"], item["second"]) ==
                 ("warm_ledger", "warm_no_ledger")]
        self.assertEqual(len(pairs), len(run.CATEGORIES))
        self.assertTrue(all(item["pairs"] == 0 and item["difference"] is None for item in pairs))
        # Drift: the planned method is listed with 0 runs; D1 is unavailable.
        self.assertEqual((result["drift"]["warm_ledger"]["runs"], result["drift"]["warm_ledger"]["planned_runs"]),
                         (0, 1))
        checks = result["decision"]["checks"]
        self.assertIsNone(checks["D1_drift_within_10_points"])
        # H checks without data are unavailable (None), not a false failure.
        self.assertIsNone(checks["H1_ledger_helps"])
        self.assertIsNone(checks["H2_no_regression"])
        self.assertIn("The ledger result is unavailable (no data for a check).", result["decision"]["verdict"])
        text = run.markdown(result, meta)
        self.assertIn("| warm_ledger | warm_no_ledger | goal | 0 | unavailable | n/a |", text)
        self.assertIn(f"| {run.METHODS['warm_ledger']['label']} | 0 / 1 | - | unavailable | - | n/a |", text)
        # With no planned drift job, the drift section has no row for the method.
        self.assertNotIn("warm_ledger", run.aggregate(runs, meta, planned={"a": planned["a"]})["drift"])

    def test_report_drops_the_local_config_digests(self):
        marker = "fake-value-456"
        config = {"model": "m", "threshold": 0.95, "hermes_config": {"model": {"headers": {"X-Fake": marker}}}}
        meta = {"schema": "s", "config": run.public_config(config), "config_sha256": run.config_digests(config),
                "plugin": {}, "hermes": {}, "runtime_sha256": {"runner_python": "0" * 64}}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (out / "runs.jsonl").write_text("", encoding="utf-8")
            args = types.SimpleNamespace(out=str(out), json=str(out / "r.json"), md=None, also=None, rescore=False,
                                         raw_dir=None)
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                run.command_report(args)
            report = json.loads((out / "r.json").read_text(encoding="utf-8"))
        self.assertNotIn("config_sha256", report["meta"])
        self.assertNotIn("runtime_sha256", report["meta"])
        self.assertEqual(report["meta"]["config"]["hermes_config"],
                         {"redacted": True, "types": {"model.headers.X-Fake": "str"}})

    def test_report_lists_the_methods_that_each_resume_skipped(self):
        first = {"method": "fallback_quoted_ledger", "reason": "plugin_setting_missing"}
        later = {"method": "warm_no_ledger", "reason": "plugin_setting_missing"}
        meta = {"schema": "s", "config": {"model": "m"}, "plugin": {}, "hermes": {}, "skipped_methods": [first],
                "resumes": [{"plan": {"methods": ["warm_no_ledger"]}, "skipped_methods": [later, first]}]}
        self.assertEqual(run.skipped_methods(meta), [first, later])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (out / "runs.jsonl").write_text("", encoding="utf-8")
            args = types.SimpleNamespace(out=str(out), json=str(out / "r.json"), md=str(out / "r.md"), also=None,
                                         rescore=False, raw_dir=None)
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                run.command_report(args)
            report = json.loads((out / "r.json").read_text(encoding="utf-8"))
            text = (out / "r.md").read_text(encoding="utf-8")
        self.assertEqual([item["method"] for item in report["meta"]["skipped_methods"]],
                         ["fallback_quoted_ledger", "warm_no_ledger"])
        self.assertIn("Skipped methods: fallback_quoted_ledger, warm_no_ledger.", text)

    def test_plugin_hash_covers_every_loaded_file(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder) / "warm_compaction"
            package.mkdir()
            (package / "engine.py").write_text('"warm"\nTHRESHOLD_RANGE = 1\n', encoding="utf-8")
            (package / "plugin.yaml").write_text("name: warm_compaction\n", encoding="utf-8")
            first = run.plugin_state(Path(folder))["files_sha256"]
            (package / "plugin.yaml").write_text("name: warm_compaction\nhooks: []\n", encoding="utf-8")
            second = run.plugin_state(Path(folder))["files_sha256"]
        self.assertEqual(sorted(first), ["engine.py", "plugin.yaml"])
        self.assertNotEqual(first["plugin.yaml"], second["plugin.yaml"])

    def test_decision_thresholds_are_exact_at_the_boundary(self):
        def pair(category, first_only, second_only, pairs=100):
            return {"first": "warm_ledger", "second": "warm_no_ledger", "category": category, "pairs": pairs,
                    "first_only": first_only, "second_only": second_only,
                    "difference": (first_only - second_only) / pairs, "p_mcnemar": 0.5}

        def result(guard_loss):
            paired = [pair(c, 12, 0) for c in run.LEDGER_CATEGORIES] + [
                pair(c, 0, guard_loss) for c in run.GUARD_CATEGORIES]
            methods = {"warm_ledger": {"success": run.rate(27, 30)}, "warm_no_ledger": {"success": None}}
            drift = {"warm_ledger": {"cycle_1_and_3": {"runs": 1, "cycle_1": run.rate(9, 10),
                                                       "cycle_3": run.rate(8, 10)}}}
            return {"methods": methods, "paired": paired, "drift": drift}

        # Exactly 5 points lower (not significant) passes H2; 6 points lower fails it.
        self.assertTrue(run.decide(result(5))["checks"]["H2_no_regression"])
        self.assertFalse(run.decide(result(6))["checks"]["H2_no_regression"])
        checks = run.decide(result(5))["checks"]
        self.assertTrue(checks["S1_warm_success_at_least_90"])  # exactly 90%
        self.assertTrue(checks["D1_drift_within_10_points"])  # exactly 10 points lower

    def test_config_overrides_cannot_change_the_method(self):
        job = {"model": "m", "endpoint": "http://127.0.0.1:9/v1", "api_key": "", "context_length": 8,
               "threshold": 0.9, "engine": "warm_compaction", "settings": {"ledger": True}}
        # An allow-list: only model.headers. Every other key is refused, also a key that the benchmark does not set
        # (an auxiliary model of the built-in compressor, another engine key, an unknown section).
        for override in ({"context": {"engine": "compressor"}}, {"compression": {"threshold": 0.5}},
                         {"plugins": {"entries": {"warm_compaction": {"settings": {"ledger": False}}}}},
                         {"model": {"base_url": "http://127.0.0.1:8/v1"}}, {"auxiliary": {"warm_compaction": {}}},
                         {"auxiliary": {"compression": {"provider": "other", "model": "other"}}},
                         {"model": {"headers_extra": 1}}, {"model": {}}, {"new_section": {"key": 1}}, {"x": 1}):
            self.assertTrue(worker.override_conflicts(override), override)
        self.assertEqual(worker.override_conflicts({"auxiliary": {"compression": {"model": "other"}}}),
                         ["auxiliary"])
        self.assertEqual(worker.override_conflicts({"model": {"headers": {"X-Fake": "1"}}}), [])
        self.assertEqual(worker.override_conflicts({"model": {"headers": {}, "provider": "x"}}), ["model.provider"])
        self.assertEqual(worker.qa_body_conflicts({"max_tokens": 1, "top_p": 1}), ["max_tokens"])
        self.assertEqual(worker.qa_body_conflicts({"logit_bias": {}, "response_format": {}, "seed": 1}),
                         ["logit_bias", "response_format"])
        self.assertEqual(worker.qa_body_conflicts({key: 1 for key in worker.QA_EXTRA_ALLOWED}), [])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(worker.JobFailed, "hermes_config_protected:1"):
                worker.write_config(Path(folder), {**job, "hermes_config": {"context": {"engine": "compressor"}}})
            config = {"model": "m", "endpoint": "http://127.0.0.1:9/v1", "context_length": 8, "threshold": 0.9,
                      "hermes_config": {"compression": {"threshold": 0.5}}, "qa_extra_body": {"model": "other"}}
            (Path(folder) / "c.json").write_text(json.dumps(config), encoding="utf-8")
            args = types.SimpleNamespace(config=str(Path(folder) / "c.json"), out=str(Path(folder) / "out"))
            with self.assertRaisesRegex(SystemExit, "hermes_config.compression.*qa_extra_body.model"):
                run.command_run(args)
            self.assertFalse((Path(folder) / "out" / "meta.json").exists())

    def test_route_headers_cannot_replace_a_header_of_another_source(self):
        # Synthetic header names and fake values. A reserved name (any case), two names that differ only in case,
        # and a malformed name or value are refused before any request.
        refused = ({"Authorization": "Bearer other"}, {"authorization": "x"}, {"CONTENT-TYPE": "text/plain"},
                   {" Content-Type ": "x"}, {"X-Api-Key": "x"}, {"anthropic-version": "x"}, {"Host": "x"},
                   {"Content-Length": "1"}, {"X-Gateway": "a", "x-gateway": "b"}, {"X-Gateway": 1},
                   {"X-Gateway": "a\r\nAuthorization: b"}, {"": "x"}, {1: "x"}, ["X-Gateway"],
                   # Names that are not HTTP tokens, a value with a control character, and non-ASCII values (urllib
                   # cannot send a value outside Latin-1; other clients encode non-ASCII values differently).
                   {"Bad Header": "x"}, {"X:Y": "x"}, {"X-Gatéway": "x"}, {"X-Gateway ": "x"},
                   {"X-Gateway": "a\x01b"}, {"X-Gateway": "a\x7fb"}, {"X-Gateway": "a\u20acb"},
                   {"X-Gateway": "a\U0001f600b"}, {"X-Gateway": "caf\u00e9"})
        for headers in refused:
            with self.subTest(headers=headers):
                self.assertTrue(worker.override_conflicts({"model": {"headers": headers}}))
                self.assertTrue(run.override_conflicts({"hermes_config": {"model": {"headers": headers}}}))
                with self.assertRaisesRegex(worker.JobFailed, "^route_headers_refused:"):
                    worker.route_headers({"hermes_config": {"model": {"headers": headers}}})
        accepted = {"X-Gateway": "synthetic", "X-Trace": "1", "x-gw_1.v2~!#$%&'*+^`|": "a\tb"}
        self.assertEqual(worker.override_conflicts({"model": {"headers": accepted}}), [])
        self.assertEqual(worker.route_headers({"hermes_config": {"model": {"headers": accepted}}}), accepted)
        # The QA request fails before it is sent, so the harness key is the only Authorization value.
        job = {"questions": [{"n": 1, "text": "Q?"}], "qa_batches": [[1]], "model": "m",
               "endpoint": "http://127.0.0.1:9/v1", "api_key": "fake-key",
               "hermes_config": {"model": {"headers": {"authorization": "Bearer other"}}}}
        with mock.patch.object(worker.urllib.request, "urlopen", side_effect=AssertionError("no request")):
            with self.assertRaisesRegex(worker.JobFailed, "^route_headers_refused:1$"):
                worker.ask(job, "system", [])
        with tempfile.TemporaryDirectory() as folder:
            job = {"model": "m", "endpoint": "http://127.0.0.1:9/v1", "api_key": "", "context_length": 8,
                   "threshold": 0.9, "engine": "compressor", "hermes_config": {"model": {"headers": {
                       "Content-Type": "text/plain"}}}}
            with self.assertRaisesRegex(worker.JobFailed, "hermes_config_protected:1"):
                worker.write_config(Path(folder), job)

    def test_report_refuses_an_incompatible_run_folder(self):
        gold = {path.name: run.sha256_file(path) for path in sorted(GOLD_DIR.glob("*.json"))}
        base = {"schema": "s", "gold_sha256": gold, "hermes": {"commit": "a"}, "plugin": {"files_sha256": {"x": "1"}},
                "config": {"model": "m", "context_length": 8, "threshold": 0.95, "methods": ["warm_ledger"]},
                **FAKE_RUNTIME_META}
        cli = {**base, "hermes": {"commit": "b"}, "plugin": {},
               "config": {**base["config"], "methods": ["codex_cli"]}}
        self.assertEqual(run.merge_mismatches(base, cli, False), [])
        self.assertEqual(run.merge_mismatches(base, {**cli, "config": {**cli["config"], "model": "other"}}, False),
                         ["config model"])
        self.assertEqual(run.merge_mismatches(base, {**base, "hermes": {"commit": "b"},
                                                     "plugin": {"files_sha256": {"x": "2"}}}, False),
                         ["Hermes commit", "plugin files"])
        self.assertEqual(run.merge_mismatches(base, {**cli, "gold_sha256": {}}, False), ["gold files (use --rescore)"])
        self.assertEqual(run.merge_mismatches(base, {**cli, "gold_sha256": {}}, True), [])
        # Harness files, server model, CLI version of a shared CLI method, and other result settings.
        self.assertEqual(run.merge_mismatches(base, {**cli, "harness_sha256": {"run.py": "0"}}, False),
                         ["harness files"])
        model = {"id": "m", "quantization": "Q4_K_M"}
        self.assertEqual(run.merge_mismatches({**base, "server_model": model},
                                              {**cli, "server_model": {**model, "quantization": "Q8_0"}}, False),
                         ["server model"])
        # A folder that checked the server model, merged with one that did not record it.
        self.assertEqual(run.merge_mismatches({**base, "server_model": model}, cli, False),
                         ["server model (not recorded)"])
        checked = {**base, "config": {**base["config"], "server_check": True}}
        self.assertEqual(run.merge_mismatches(checked, {**cli, "server_model": model}, False),
                         ["server model (not recorded)"])
        # The methods of a folder are those of the first plan and of every resume plan: a resume that added a
        # plugin method makes the plugin files part of the merge check.
        resumed = {**cli, "resumes": [{"methods": ["codex_cli", "warm_ledger"],
                                       "plan": {"methods": ["codex_cli", "warm_ledger"]}}]}
        self.assertEqual(run.plan_methods(resumed), {"codex_cli", "warm_ledger"})
        self.assertEqual(run.plan_methods({**cli, "resumes": [{"plan": {"methods": ["full_context"]}}]}),
                         {"codex_cli", "full_context"})
        self.assertEqual(run.merge_mismatches(base, {**resumed, "plugin": {"files_sha256": {"x": "2"}}}, False),
                         ["Hermes commit", "plugin files"])
        codex = {**cli, "cli": {"codex_cli": {"version": "0.147.0"}}}
        self.assertEqual(run.merge_mismatches(codex, {**codex, "cli": {"codex_cli": {"version": "0.148.0"}}}, False),
                         ["codex_cli version"])
        self.assertEqual(run.merge_mismatches(base, {**cli, "config": {**cli["config"], "job_timeout_s": 5}}, False),
                         ["config job_timeout_s"])
        # The runtime: the parts that both folders use must be the same, and both folders must record it.
        def changed(meta: dict, key: str) -> dict:
            return {**meta, "runtime_sha256": {**meta["runtime_sha256"], key: "1" * 64}}

        self.assertEqual(run.merge_mismatches(base, changed(base, "hermes_packages"), False),
                         ["runtime hermes_packages"])
        self.assertEqual(run.merge_mismatches(base, changed(base, "hermes_python"), False), ["runtime hermes_python"])
        self.assertEqual(run.merge_mismatches(base, changed(cli, "hermes_packages"), False), [])
        self.assertEqual(run.merge_mismatches(base, changed(cli, "runner_python"), False), ["runtime runner_python"])
        self.assertEqual(run.merge_mismatches(base, {**base, "runtime": {"hermes_python": {"version": "3.12.0"}}},
                                              False), ["runtime hermes_python version"])
        self.assertEqual(run.merge_mismatches(base, {**cli, "host": {"os": "Other"}}, False), ["runtime host"])
        self.assertEqual(run.merge_mismatches(codex, changed(codex, "cli.codex_cli"), False), ["runtime cli.codex_cli"])
        self.assertEqual(run.merge_mismatches(base, {k: v for k, v in cli.items() if k != "runtime_sha256"}, False),
                         ["runtime (not recorded)"])
        # The key: the same api_key_env name, and with a key the user confirms the same key and route (a run folder
        # keeps only the name, never the key or a hash of it).
        keyed = {**base, "config": {**base["config"], "api_key_env": "WC_FAKE_KEY_ENV"}}
        keyed_cli = {**cli, "config": {**cli["config"], "api_key_env": "WC_FAKE_KEY_ENV"}}
        self.assertEqual(run.merge_mismatches(base, keyed_cli, False), ["api_key_env"])
        self.assertEqual(run.merge_mismatches(keyed, {**cli, "config": {**cli["config"], "api_key_env": "OTHER"}},
                                              False, True), ["api_key_env"])
        self.assertEqual(run.merge_mismatches(keyed, keyed_cli, False),
                         ["credential (add --same-credential to confirm the same key and route)"])
        self.assertEqual(run.merge_mismatches(keyed, keyed_cli, False, True), [])
        self.assertEqual(run.merge_mismatches(base, {**cli, "config": {**cli["config"], "api_key_env": ""}}, False),
                         [])
        # The job plan and a CLI timeout of a run without CLI methods can differ.
        self.assertEqual(run.merge_mismatches(base, {**cli, "config": {**cli["config"], "cli_timeout_s": 5,
                                                                       "replicates": 2}}, False), [])
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a", Path(folder) / "b"
            for path, meta in ((first, base), (second, {**cli, "config": {**cli["config"], "context_length": 9}})):
                path.mkdir()
                (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                (path / "runs.jsonl").write_text("", encoding="utf-8")
            args = types.SimpleNamespace(out=str(first), json=str(first / "r.json"), md=None, also=[str(second)],
                                         rescore=False, raw_dir=None)
            with self.assertRaisesRegex(SystemExit, "config context_length"):
                run.command_report(args)
            self.assertFalse((first / "r.json").exists())

    def test_report_checks_each_added_folder_against_every_accepted_folder(self):
        gold = {path.name: run.sha256_file(path) for path in sorted(GOLD_DIR.glob("*.json"))}
        config = {"model": "m", "context_length": 8, "threshold": 0.95}
        # The first folder runs a CLI method only; the added folders run Hermes methods (synthetic metadata).
        cli = {"schema": "s", "gold_sha256": gold, "hermes": {}, "plugin": {},
               "config": {**config, "methods": ["codex_cli"]}, **FAKE_RUNTIME_META}
        warm = {**cli, "hermes": {"commit": "a"}, "plugin": {"files_sha256": {"x": "1"}},
                "config": {**config, "methods": ["warm_ledger"]}}
        stock = {**cli, "hermes": {"commit": "b"}, "config": {**config, "methods": ["stock_hermes"]}}
        # Each added folder alone matches the first folder.
        self.assertEqual(run.merge_mismatches(cli, warm, False), [])
        self.assertEqual(run.merge_mismatches(cli, stock, False), [])
        other_plugin = {**warm, "plugin": {"files_sha256": {"x": "2"}}}
        cases = ((stock, "Hermes commit"), (other_plugin, "plugin files"),
                 ({**stock, "hermes": {"commit": "a"}}, None))
        for second, refused in cases:
            with self.subTest(refused=refused), tempfile.TemporaryDirectory() as folder:
                paths = [Path(folder) / name for name in ("cli", "warm", "second")]
                for path, meta in zip(paths, (cli, warm, second)):
                    path.mkdir()
                    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                    (path / "runs.jsonl").write_text("", encoding="utf-8")
                args = types.SimpleNamespace(out=str(paths[0]), json=str(paths[0] / "r.json"), md=None,
                                             also=[str(paths[1]), str(paths[2])], rescore=False, raw_dir=None)
                if refused:
                    with self.assertRaisesRegex(SystemExit, f"differs from .*warm in {refused}"):
                        run.command_report(args)
                    self.assertFalse((paths[0] / "r.json").exists())
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        run.command_report(args)
                    self.assertTrue((paths[0] / "r.json").exists())

    def test_plugin_state_records_links_and_never_follows_them(self):
        def link(target: Path, path: Path) -> None:
            try:
                os.symlink(target, path, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    raise
                import _winapi  # A junction needs no symlink privilege on Windows.
                _winapi.CreateJunction(str(target), str(path))

        with tempfile.TemporaryDirectory() as folder:
            source, outside = Path(folder) / "plugin", Path(folder) / "outside"
            (source / "warm_compaction").mkdir(parents=True)
            (source / "warm_compaction" / "engine.py").write_text("x = 1\n", encoding="utf-8")
            outside.mkdir()
            (outside / "extra.py").write_text("y = 2\n", encoding="utf-8")  # synthetic external code
            plain = run.plugin_state(source)
            subprocess.run(["git", "-C", str(source), "init", "-q"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                            "commit", "-q", "-m", "synthetic"], check=True, capture_output=True)
            self.assertIsNotNone(run.plugin_state(source)["commit"])
            link(outside, source / "warm_compaction" / "ext")
            subprocess.run(["git", "-C", str(source), "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                            "commit", "-q", "-m", "link"], check=False, capture_output=True)
            state = run.plugin_state(source)
            # The link is recorded and its target files are not: no commit labels these files.
            self.assertIsNone(state["commit"])
            self.assertNotIn("ext/extra.py", state["files_sha256"])
            self.assertEqual(state["files_sha256"].get("ext"), getattr(run, "SYMLINK", "symlink"))
            self.assertEqual((run.plugin_links(plain), run.plugin_links(state)), ([], ["ext"]))
            # A link added during a run changes the source identity (the next job stops).
            self.assertNotEqual(state["files_sha256"], plain["files_sha256"])
            # The run stops before any job.
            config = {"plugin_source": str(source), "hermes_source": str(source), "history_dir": str(outside / "h")}
            with mock.patch.object(run, "hermes_state", return_value={"commit": "abc", "clean": True}), \
                    mock.patch.object(run, "plan", side_effect=AssertionError("planned")):
                config_path = Path(folder) / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                args = types.SimpleNamespace(config=str(config_path), out=str(Path(folder) / "out"), limit=None,
                                             dry_run=True, allow_harness_change=False)
                with self.assertRaisesRegex(SystemExit, "symbolic links or junctions \\(ext\\)"):
                    run.command_run(args)

    def test_hermes_checkout_with_an_ignored_file_is_not_clean(self):
        def git(folder, *args):
            subprocess.run(["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                            *args], check=True, capture_output=True)

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "hermes"
            source.mkdir()
            (source / "agent.py").write_text("x = 1\n", encoding="utf-8")  # synthetic Hermes source
            (source / ".gitignore").write_text("*.gen.py\n.venv/\n__pycache__/\n", encoding="utf-8")
            git(source, "init", "-q")
            git(source, "add", "-A")
            git(source, "commit", "-q", "-m", "synthetic")
            self.assertTrue(run.hermes_state(source)["clean"])
            # A virtual environment and __pycache__ do not count (the packages are in the runtime hash).
            (source / ".venv" / "lib").mkdir(parents=True)
            (source / ".venv" / "pyvenv.cfg").write_text("home = fake\n", encoding="utf-8")
            (source / ".venv" / "lib" / "site.py").write_text("y = 1\n", encoding="utf-8")
            (source / "__pycache__").mkdir()
            (source / "__pycache__" / "agent.cpython-312.pyc").write_bytes(b"fake")
            self.assertTrue(run.hermes_state(source)["clean"])
            # An ignored module that Python can load from the checkout makes it not clean.
            (source / "patch.gen.py").write_text("import agent\nagent.x = 2\n", encoding="utf-8")
            self.assertFalse(run.hermes_state(source)["clean"])
            self.assertEqual(run.ignored_files(source), ["patch.gen.py"])

    def test_hermes_checkout_with_a_tracked_link_is_not_clean(self):
        def git(folder, *args, data=None):
            return subprocess.run(["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                                   *args], check=True, capture_output=True, input=data).stdout

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "hermes"
            source.mkdir()
            (source / "agent.py").write_text("x = 1\n", encoding="utf-8")  # synthetic Hermes source
            git(source, "init", "-q")
            git(source, "config", "core.symlinks", "false")
            git(source, "add", "-A")
            git(source, "commit", "-q", "-m", "synthetic")
            self.assertTrue(run.hermes_state(source)["clean"])
            # A tracked symbolic link (mode 120000) to a module outside the checkout. With core.symlinks false the
            # work tree has the target path as text, so Git reports a clean status on every platform.
            target = b"../outside/agent_patch.py"
            blob = git(source, "hash-object", "-w", "--stdin", data=target).decode().strip()
            git(source, "update-index", "--add", "--cacheinfo", f"120000,{blob},agent_patch.py")
            (source / "agent_patch.py").write_bytes(target)
            git(source, "commit", "-q", "-m", "link")
            self.assertEqual(git(source, "status", "--porcelain"), b"")
            self.assertFalse(run.hermes_state(source)["clean"])
            self.assertEqual(run.tracked_links(source), ["agent_patch.py"])

    def test_runner_python_identity_follows_its_files(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "Lib" / "__pycache__").mkdir(parents=True)
            (base / "Lib" / "site-packages").mkdir()
            executable = base / "python.exe"
            executable.write_bytes(b"synthetic interpreter 1")
            (base / "Lib" / "os.py").write_text("x = 1\n", encoding="utf-8")

            def identity():
                return run.runtime_state({})["sha256"]["runner_python"]

            with mock.patch.object(run.sys, "executable", str(executable)), \
                    mock.patch.object(run.sys, "base_prefix", str(base)), \
                    mock.patch("sysconfig.get_path", lambda name, *_a, **_k: str(base / "Lib")):
                first = identity()
                # Bytecode caches and installed packages do not change the identity.
                (base / "Lib" / "__pycache__" / "os.cpython-314.pyc").write_bytes(b"cache")
                (base / "Lib" / "site-packages" / "extra.py").write_text("y = 1\n", encoding="utf-8")
                self.assertEqual(identity(), first)
                # An upgrade in place: the same path, other executable bytes of the same size.
                executable.write_bytes(b"synthetic interpreter 2")
                second = identity()
                self.assertNotEqual(second, first)
                # A changed standard library module.
                (base / "Lib" / "os.py").write_text("x = 22\n", encoding="utf-8")
                self.assertNotEqual(identity(), second)

    def test_resume_with_a_credential_needs_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out"
            out.mkdir()
            (out / "meta.json").write_text("{}", encoding="utf-8")  # an existing run folder (synthetic)
            config = {"threshold": 0.95, "methods": ["stock_hermes"], "fixtures": ["s1-ledgerline"],
                      "replicates": 1, "api_key_env": "WC_FAKE_KEY_ENV", "hermes_source": "h",
                      "plugin_source": str(Path(run.__file__).parents[2]), "history_dir": str(Path(folder) / "h")}
            config_path = Path(folder) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            def resume(**flags):
                args = types.SimpleNamespace(config=str(config_path), out=str(out), limit=None, dry_run=True,
                                             allow_harness_change=False, **flags)
                with mock.patch.object(run, "hermes_state", return_value={"commit": "abc", "clean": True}), \
                        mock.patch.dict(os.environ, {"WC_FAKE_KEY_ENV": "fake-key-value"}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run.command_run(args)

            # A configured key variable that is not set stops the run before any job (no placeholder key).
            with mock.patch.dict(os.environ, {"WC_FAKE_KEY_ENV": ""}), \
                    self.assertRaisesRegex(SystemExit, "WC_FAKE_KEY_ENV, which is not set"):
                run.command_run(types.SimpleNamespace(config=str(config_path), out=str(out), limit=None,
                                                      dry_run=True, allow_harness_change=False))
            # A dry run writes no meta.json, so the next run of the folder is not a resume.
            fresh = Path(folder) / "fresh"
            with mock.patch.object(run, "hermes_state", return_value={"commit": "abc", "clean": True}), \
                    mock.patch.dict(os.environ, {"WC_FAKE_KEY_ENV": "fake-key-value"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                run.command_run(types.SimpleNamespace(config=str(config_path), out=str(fresh), limit=None,
                                                      dry_run=True, allow_harness_change=False))
            self.assertFalse((fresh / "meta.json").exists())
            # The folder keeps only the variable name: without the confirmation the resume stops.
            with self.assertRaisesRegex(SystemExit, "--same-credential"):
                resume()
            with self.assertRaisesRegex(SystemExit, "--same-credential"):
                resume(same_credential=False)
            # With the confirmation the other resume checks run (this synthetic folder records nothing).
            with self.assertRaises(SystemExit) as stop:
                resume(same_credential=True)
            self.assertNotIn("--same-credential", str(stop.exception))
            self.assertIn("runtime (not recorded)", str(stop.exception))

    def test_commits_label_only_the_exact_source_folder(self):
        def git(folder, *args):
            subprocess.run(["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                            *args], check=True, capture_output=True)

        def plugin_files(folder: Path):
            (folder / "warm_compaction").mkdir(parents=True)
            (folder / "warm_compaction" / "engine.py").write_text("x = 1\n", encoding="utf-8")
            (folder / "warm_compaction" / "plugin.yaml").write_text("name: synthetic\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as folder:
            outer = Path(folder) / "outer"
            copy = outer / "plugin-copy"
            plugin_files(copy)
            git(outer, "init", "-q")
            git(outer, "add", "-A")
            git(outer, "commit", "-q", "-m", "synthetic")
            # A copied folder inside another checkout: Git reads the enclosing checkout. Unknown, not clean.
            self.assertIsNone(run.plugin_state(copy)["commit"])
            self.assertEqual(run.hermes_state(copy), {"commit": None, "clean": False})
            # The top folder of a clean checkout that tracks every hashed file: the commit is known.
            head = subprocess.run(["git", "-C", str(outer), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
            source = Path(folder) / "plugin"
            plugin_files(source)
            git(source, "init", "-q")
            git(source, "add", "-A")
            git(source, "commit", "-q", "-m", "synthetic")
            commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True,
                                    text=True, check=True).stdout.strip()
            self.assertEqual(run.plugin_state(source)["commit"], commit)
            self.assertEqual(run.hermes_state(outer), {"commit": head, "clean": True})
            # An untracked, an ignored, or a changed file in the hashed folder: the commit does not label it.
            (source / "warm_compaction" / "extra.py").write_text("y = 2\n", encoding="utf-8")
            self.assertIsNone(run.plugin_state(source)["commit"])
            (source / ".gitignore").write_text("extra.py\n", encoding="utf-8")
            git(source, "add", ".gitignore")
            git(source, "commit", "-q", "-m", "ignore")
            self.assertIsNone(run.plugin_state(source)["commit"])
            (source / "warm_compaction" / "extra.py").unlink()
            self.assertIsNotNone(run.plugin_state(source)["commit"])
            (source / "warm_compaction" / "engine.py").write_text("x = 2\n", encoding="utf-8")
            self.assertIsNone(run.plugin_state(source)["commit"])
        # A configured commit is kept apart and shown as not verified.
        self.assertEqual(run.plugin_commit_text({"commit": "f" * 40}), "`ffffffffffff`")
        self.assertEqual(run.plugin_commit_text({"commit": None, "commit_configured": "abc123"}),
                         "unknown (configured, not verified: `abc123`)")
        self.assertEqual(run.plugin_commit_text({}), "unknown")

    def test_resume_refuses_a_changed_source(self):
        plugin = run.plugin_state(Path(run.__file__).parents[2])
        hermes = {"commit": "abc", "clean": True}
        config = {"model": "m", "threshold": 0.95, "methods": ["stock_hermes"], "plugin_source": "/private"}
        gold = {path.name: run.sha256_file(path) for path in sorted(GOLD_DIR.glob("*.json"))}
        meta = {"config": {k: v for k, v in config.items() if k != "plugin_source"}, "plugin": plugin,
                "hermes": hermes, "gold_sha256": gold, "harness_sha256": run.harness_hashes(), **FAKE_RUNTIME_META}
        runtime = FAKE_RUNTIME
        self.assertEqual(run.resume_mismatches(meta, config, plugin, hermes, {}, runtime), ([], []))
        # A resume runs the jobs of the first plan in the same order: an added, removed, or reordered method,
        # fixture, replicate, or drift job is refused (the new jobs would run alone at the end, after time drift).
        # The server check is not part of the plan.
        for changed_plan in ({"methods": ["stock_hermes", "full_context"]}, {"methods": ["full_context"]},
                             {"fixtures": ["s1-ledgerline"]}, {"replicates": 6},
                             {"drift": {"fixtures": ["s1-ledgerline"], "methods": ["stock_hermes"], "cycles": 3}}):
            with self.subTest(changed_plan=changed_plan):
                self.assertEqual(run.resume_mismatches(meta, {**config, **changed_plan}, plugin, hermes, {},
                                                       runtime), (["job plan"], []))
        two = {**config, "methods": ["stock_hermes", "full_context"]}
        two_meta = {**meta, "config": {k: v for k, v in two.items() if k != "plugin_source"}}
        self.assertEqual(run.resume_mismatches(two_meta, two, plugin, hermes, {}, runtime), ([], []))
        self.assertEqual(run.resume_mismatches(two_meta, {**two, "methods": ["full_context", "stock_hermes"]},
                                               plugin, hermes, {}, runtime), (["job plan"], []))
        self.assertEqual(run.resume_mismatches(meta, {**config, "replicates": 5}, plugin, hermes, {}, runtime),
                         ([], []))
        # A folder in which an earlier resume (of an earlier harness) changed the plan cannot resume again.
        self.assertEqual(run.resume_mismatches({**meta, "resumes": [{"plan": {"methods": ["full_context"]}}]},
                                               config, plugin, hermes, {}, runtime), (["job plan"], []))
        # Result settings, plugin, and Hermes cannot change either.
        changed = {**plugin, "files_sha256": {**plugin["files_sha256"], "engine.py": "0"}}
        self.assertEqual(run.resume_mismatches(meta, config, changed, hermes, {}, runtime)[0], ["plugin files"])
        self.assertEqual(run.resume_mismatches(meta, config, plugin, {"commit": "def"}, {}, runtime)[0],
                         ["Hermes commit"])
        self.assertEqual(run.resume_mismatches(meta, {**config, "threshold": 0.5}, plugin, hermes, {}, runtime)[0],
                         ["config"])
        # The runtime: the Python of Hermes (path, prefix, packages, version), the runner, the host, the CLI argv.
        for key in FAKE_RUNTIME["sha256"]:
            other = {**runtime, "sha256": {**runtime["sha256"], key: "0" * 64}}
            self.assertEqual(run.resume_mismatches(meta, config, plugin, hermes, {}, other)[0], [f"runtime {key}"])
        other = {**runtime, "sha256": {**runtime["sha256"], "cli.claude_code": "0" * 64}}
        self.assertEqual(run.resume_mismatches(meta, config, plugin, hermes, {}, other)[0],
                         ["runtime cli.claude_code"])
        other = {"public": {**runtime["public"], "hermes_python": {"version": "3.12.0", "implementation": "cpython"}},
                 "sha256": runtime["sha256"]}
        self.assertEqual(run.resume_mismatches(meta, config, plugin, hermes, {}, other)[0],
                         ["runtime hermes_python version"])
        other = {"public": {**runtime["public"], "host": {"os": "Other"}}, "sha256": runtime["sha256"]}
        self.assertEqual(run.resume_mismatches(meta, config, plugin, hermes, {}, other)[0], ["runtime host"])
        unrecorded = {k: v for k, v in meta.items() if k != "runtime_sha256"}
        self.assertEqual(run.resume_mismatches(unrecorded, config, plugin, hermes, {}, runtime)[0],
                         ["runtime (not recorded)"])
        old = {**meta, "harness_sha256": {**meta["harness_sha256"], "worker.py": "0"}}
        self.assertEqual(run.resume_mismatches(old, config, plugin, hermes, {}, runtime), ([], ["worker.py"]))
        # A recorded resume with --allow-harness-change becomes the new reference.
        old["resumes"] = [{"harness_sha256": run.harness_hashes()}]
        self.assertEqual(run.resume_mismatches(old, config, plugin, hermes, {}, runtime), ([], []))
        # The server model: a failed probe or a folder without the model stops a checked resume.
        model = {"id": "m", "quantization": "Q8_K_XL"}
        checked = {**config, "server_check": True}
        with mock.patch.object(run, "server_model", return_value=None):
            self.assertEqual(run.resume_mismatches({**meta, "server_model": model}, checked, plugin, hermes, {},
                                                   runtime)[0], ["server model (probe failed)"])
        with mock.patch.object(run, "server_model", return_value=model):
            self.assertEqual(run.resume_mismatches(meta, checked, plugin, hermes, {}, runtime)[0],
                             ["server model (not recorded)"])
            self.assertEqual(run.resume_mismatches({**meta, "server_model": model}, checked, plugin, hermes, {},
                                                   runtime)[0], [])
            # A folder that recorded the model is checked also when the resume turns server_check off.
            self.assertEqual(run.resume_mismatches({**meta, "server_model": {**model, "id": "x"}}, config, plugin,
                                                   hermes, {}, runtime)[0], ["server model"])

    def test_a_failed_server_model_probe_stops_the_run_before_meta(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out"
            config = {"hermes_source": "h", "plugin_source": "p", "history_dir": str(Path(folder) / "fixtures"),
                      "threshold": 0.95, "methods": [], "server_check": True,
                      "endpoint": "http://127.0.0.1:9/v1", "model": "m"}
            (Path(folder) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.object(run, "hermes_state", return_value={"commit": "a", "clean": True}), \
                    mock.patch.object(run, "plugin_state", return_value={"commit": "p", "files_sha256": {}}), \
                    mock.patch.object(run, "runtime_state", return_value=FAKE_RUNTIME), \
                    mock.patch.object(run, "server_model", return_value=None):
                with self.assertRaisesRegex(SystemExit, "model probe failed"):
                    run.main(["run", "--config", str(Path(folder) / "config.json"), "--out", str(out)])
            self.assertFalse((out / "meta.json").exists())
            self.assertFalse((out / "runs.jsonl").exists())

    def test_runtime_state_hashes_paths_and_argv(self):
        config = {"hermes_python": sys.executable, "cli": {"codex_cli": {"command": ["fake-codex", "--flag"]}}}
        state = run.runtime_state(config)
        self.assertEqual(set(state["sha256"]), {"runner_python", "hermes_python", "hermes_prefix", "hermes_packages",
                                                "hermes_files", "cli.codex_cli"})
        self.assertEqual(state["public"]["hermes_python"]["version"], sys.version.split()[0])
        # The public part has no path and no argv.
        public = json.dumps(state["public"])
        for private in (sys.executable, sys.prefix, "fake-codex"):
            self.assertNotIn(json.dumps(private)[1:-1], public)
        other = run.runtime_state({**config, "cli": {"codex_cli": {"command": ["fake-codex", "--other"]}}})
        self.assertNotEqual(state["sha256"]["cli.codex_cli"], other["sha256"]["cli.codex_cli"])
        self.assertEqual({k: v for k, v in state["sha256"].items() if k != "cli.codex_cli"},
                         {k: v for k, v in other["sha256"].items() if k != "cli.codex_cli"})
        with self.assertRaisesRegex(SystemExit, "runtime probe"):
            run.runtime_state({"hermes_python": str(Path(tempfile.gettempdir()) / "no-such-python.exe")})

    def test_cli_hash_covers_the_files_that_the_command_runs(self):
        # A synthetic npm layout: a CLI package with a script and a hoisted dependency package (fake code).
        with tempfile.TemporaryDirectory() as folder:
            modules = Path(folder) / "node_modules"
            cli, dep = modules / "@fake" / "cli", modules / "fake-dep"
            for package, name, extra in ((cli, "@fake/cli", {"optionalDependencies": {"fake-dep": "1.0.0"}}),
                                         (dep, "fake-dep", {})):
                (package / "bin").mkdir(parents=True)
                (package / "package.json").write_text(json.dumps({"name": name, "version": "1.0.0", **extra}),
                                                      encoding="utf-8")
            (cli / "bin" / "cli.js").write_text("console.log('1.0.0')\n", encoding="utf-8")
            (dep / "bin" / "native.bin").write_bytes(b"fake-binary-1")
            (Path(folder) / "unrelated.txt").write_text("a\n", encoding="utf-8")
            command = [str(cli / "bin" / "cli.js")]
            config = {"methods": ["codex_cli"], "cli": {"codex_cli": {"command": command}}}
            first = run.runtime_state(config)["sha256"]["cli.codex_cli"]
            # A file outside the packages does not count; the same argv with changed code does.
            (Path(folder) / "unrelated.txt").write_text("b\n", encoding="utf-8")
            self.assertEqual(run.runtime_state(config)["sha256"]["cli.codex_cli"], first)
            for path, changed in ((cli / "bin" / "cli.js", b"console.log('1.0.0') // edited\n"),
                                  (dep / "bin" / "native.bin", b"fake-binary-2")):
                with self.subTest(path=path.name):
                    before = run.runtime_state(config)["sha256"]["cli.codex_cli"]
                    path.write_bytes(changed)
                    self.assertNotEqual(run.runtime_state(config)["sha256"]["cli.codex_cli"], before)
            # A CLI that the plan does not use keeps the argv hash only (no file read).
            with mock.patch.object(run, "cli_files_sha256", side_effect=AssertionError("hashed")):
                run.runtime_state({**config, "methods": ["stock_hermes"]})

    def test_example_config_checks_the_server_and_plans_the_full_run(self):
        example = json.loads((Path(run.__file__).parent / "config.example.json").read_text(encoding="utf-8"))
        # The example is an LM Studio plan of several hours: the server check must be on.
        self.assertIs(example.get("server_check"), True)
        self.assertEqual(run.public_url(example["endpoint"]), "http://localhost:1234")
        jobs, skipped = run.plan(example, {"settings_known": ["warm", "ledger", "fallback_format"]})
        self.assertEqual((len(jobs), skipped), (270, []))

    def test_method_order_is_balanced_over_all_blocks(self):
        for count in (2, 3, 4, 5, 8):
            items = list(range(count))
            rows = count if count % 2 == 0 else 2 * count
            orders = [run.balanced_order(items, block) for block in range(rows)]
            with self.subTest(count=count):
                self.assertTrue(all(sorted(order) == items for order in orders))
                # Each item runs once in each position per set of rows (twice for an odd count).
                for position in range(count):
                    self.assertEqual(sorted(order[position] for order in orders), sorted(items * (rows // count)))
                # Each item runs directly after each other item equally often.
                pairs = Counter((order[i], order[i + 1]) for order in orders for i in range(count - 1))
                self.assertEqual((len(pairs), set(pairs.values())), (count * (count - 1), {rows // count}))
        # The block number counts over all fixtures: with 4 methods and 2 replicates, the second fixture continues
        # the design, so each method starts one block.
        config = {"threshold": 0.95, "replicates": 2, "fixtures": ["s1-ledgerline", "s2-tidepool"],
                  "methods": ["full_context", "stock_hermes", "warm_ledger", "fallback_relaxed_ledger"]}
        jobs, _skipped = run.plan(config, {"settings_known": ["warm", "ledger", "fallback_format"]})
        self.assertEqual(sorted(jobs[index]["method"] for index in range(0, len(jobs), 4)), sorted(config["methods"]))

    def test_cli_timeout_stops_the_whole_process_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            beat = Path(folder) / "beat.txt"
            # A synthetic CLI that starts a child; the child writes a heartbeat for about 40 seconds and holds the
            # output pipes of its parent.
            child = ("import sys, time\n"
                     "for _ in range(400):\n"
                     "    open(sys.argv[1], 'a').write('x')\n"
                     "    time.sleep(0.1)\n")
            parent = ("import subprocess, sys, time\n"
                      f"subprocess.Popen([sys.executable, '-c', {child!r}, {str(beat)!r}])\n"
                      "time.sleep(60)\n")
            started = time.monotonic()
            result = cli_worker.run_cli([sys.executable, "-c", parent], Path(folder), dict(os.environ), 3)
            self.assertTrue(result["timeout"])
            self.assertLess(time.monotonic() - started, 20)
            self.assertTrue(beat.exists())
            time.sleep(1)
            size = beat.stat().st_size
            time.sleep(1.5)
            # The child stopped with its parent: no request of this job runs on during the next job.
            self.assertEqual(beat.stat().st_size, size)

    def test_cli_hash_records_and_follows_folder_links(self):
        def link(target: Path, path: Path) -> None:
            try:
                os.symlink(target, path, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    raise
                import _winapi  # A junction needs no symlink privilege on Windows.
                _winapi.CreateJunction(str(target), str(path))

        with tempfile.TemporaryDirectory() as folder:
            cli, generated = Path(folder) / "node_modules" / "fake-cli", Path(folder) / "generated"
            (cli / "bin").mkdir(parents=True)
            generated.mkdir()
            (cli / "package.json").write_text(json.dumps({"name": "fake-cli", "version": "1.0.0"}), encoding="utf-8")
            (cli / "bin" / "cli.js").write_text("require('../dist/main.js')\n", encoding="utf-8")
            (generated / "main.js").write_text("// fake generated code 1\n", encoding="utf-8")
            command = [str(cli / "bin" / "cli.js")]
            before = run.cli_files_sha256(command)
            # A linked dist/ folder is recorded (with its target) and read through the link.
            link(generated, cli / "dist")
            linked = run.cli_files_sha256(command)
            self.assertNotEqual(linked, before)
            (generated / "main.js").write_text("// fake generated code 2\n", encoding="utf-8")
            self.assertNotEqual(run.cli_files_sha256(command), linked)
            # A link loop ends: each real folder is read once.
            link(cli, generated / "loop")
            run.cli_files_sha256(command)

    def test_runtime_probe_covers_the_files_of_each_distribution(self):
        # A synthetic site folder: one installed distribution with a module, and one editable install whose source
        # folder is elsewhere (fake packages, no real code).
        with tempfile.TemporaryDirectory() as folder:
            site, source = Path(folder) / "site", Path(folder) / "source"
            info = site / "fake_pkg-1.0.dist-info"
            info.mkdir(parents=True)
            (site / "fake_pkg.py").write_text("x = 1\n", encoding="utf-8")
            (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-pkg\nVersion: 1.0\n", encoding="utf-8")
            (info / "RECORD").write_text("fake_pkg.py,,\nfake_pkg-1.0.dist-info/METADATA,,\n", encoding="utf-8")
            editable = site / "fake_edit-1.0.dist-info"
            editable.mkdir()
            source.mkdir()
            (source / "fake_edit.py").write_text("y = 1\n", encoding="utf-8")
            (editable / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-edit\nVersion: 1.0\n",
                                               encoding="utf-8")
            (editable / "direct_url.json").write_text(json.dumps(
                {"url": source.as_uri(), "dir_info": {"editable": True}}), encoding="utf-8")

            def probe() -> dict:
                code = f"import sys; sys.path.insert(0, {str(site)!r})\n" + run.RUNTIME_PROBE
                done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
                return json.loads(done.stdout)

            self.assertIn("fake-pkg==1.0", probe()["packages"])
            # The same name and version with patched code: the package list stays, the file identity changes.
            for path in (site / "fake_pkg.py", source / "fake_edit.py"):
                with self.subTest(path=path.name):
                    before = probe()
                    path.write_text(path.read_text(encoding="utf-8") + "patched = True\n", encoding="utf-8")
                    after = probe()
                    self.assertEqual(after["packages"], before["packages"])
                    self.assertNotEqual(after.get("files_sha256"), before.get("files_sha256"))

    def test_server_check_reads_the_loaded_context_length(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        state = {"state": "loaded", "loaded_context_length": 131072}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                body = json.dumps(state).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            config = {"server_check": True, "endpoint": f"http://127.0.0.1:{server.server_port}/v1", "model": "m",
                      "context_length": 131072}
            self.assertIsNone(run.server_state(config))
            state["loaded_context_length"] = 8192
            self.assertEqual(run.server_state(config), "loaded_context_length:8192")
            state["state"] = "not-loaded"
            self.assertEqual(run.server_state(config), "model_state:not-loaded")
            self.assertIsNone(run.server_state({**config, "server_check": False}))
        finally:
            server.shutdown()

    def test_score_cycle_and_report(self):
        runs = self.synthetic_runs()
        self.assertEqual(sum(mark[0] == "1" for mark in runs[0]["cycles"][0]["marks"]), 45)
        self.assertEqual(sum(mark[0] == "1" for mark in runs[1]["cycles"][0]["marks"]), 46)
        meta = {"config": {"model": "fake", "context_length": 1, "threshold": 0.95, "replicates": 1},
                "plugin": {"commit": None}, "hermes": {"commit": None, "clean": True}, "started_utc": "2026"}
        result = run.aggregate(runs, meta)
        self.assertEqual(result["methods"]["warm_ledger"]["categories"]["goal"]["k"], 4)
        self.assertEqual(result["methods"]["warm_ledger"]["success"]["k"], 1)
        self.assertIn("H1_ledger_helps", result["decision"]["checks"])
        self.assertFalse(result["decision"]["checks"]["H1_ledger_helps"])
        text = run.markdown(result, meta)
        self.assertIn("## Pre-registered decision rule", text)
        self.assertNotIn("1284.37", text)


class CliWorkerTest(unittest.TestCase):
    """Session file conversion and context read-back for the Codex CLI and Claude Code methods (synthetic data)."""

    @classmethod
    def setUpClass(cls):
        cls.history, _gold = fixtures.build(SCENARIOS[0])
        cls.base = cli_worker.datetime(2026, 1, 2, 3, 4, 5, tzinfo=cli_worker.timezone.utc)

    def test_codex_rollout_keeps_every_row(self):
        lines = cli_worker.codex_rollout(self.history, "sid", Path("cwd"), "0.0.0", "local", self.base)
        self.assertEqual(lines[0]["type"], "session_meta")
        items = [line["payload"] for line in lines if line["type"] == "response_item"]
        rows = self.history["rows"]
        calls = [call for row in rows for call in row.get("tool_calls") or []]
        self.assertEqual(sum(item["type"] == "function_call" for item in items), len(calls))
        self.assertEqual(sum(item["type"] == "function_call_output" for item in items),
                         sum(row["role"] == "tool" for row in rows))
        self.assertEqual([item["arguments"] for item in items if item["type"] == "function_call"],
                         [call["function"]["arguments"] for call in calls])

    def test_codex_context_reads_the_last_replacement_history(self):
        stamp = "2026-01-02T03:04:05.000Z"
        lines = [{"timestamp": stamp, "type": "event_msg", "payload": {"type": "task_started"}},
                 {"timestamp": "2026-01-02T03:04:07.500Z", "type": "compacted", "payload": {
                     "message": "SUMMARY", "replacement_history": [
                         {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "goal"}]},
                         {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "x"}]},
                         {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "SUMMARY"}]},
                     ]}}]
        rows, meta = cli_worker.codex_context(lines)
        self.assertEqual(rows, [{"role": "user", "content": "goal"}, {"role": "user", "content": "[developer] x"},
                                {"role": "user", "content": "SUMMARY"}])
        self.assertEqual(meta["compress_model_s"], 2.5)
        self.assertEqual(cli_worker.codex_context(lines[:1]), (None, {}))

    def test_cli_contexts_keep_every_item_kind(self):
        # Codex: tool calls and outputs, custom tools, local shell, web search, and reasoning text become rows.
        items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "goal"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I run it."}]},
            {"type": "function_call", "name": "shell", "arguments": '{"cmd": "make"}', "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "exit_code: 2"},
            {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch", "call_id": "c2"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": {"content": "Done"}},
            {"type": "local_shell_call", "call_id": "c3", "action": {"command": ["ls"]}},
            {"type": "web_search_call", "id": "w1", "action": {"query": "x"}},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Plan: fix the build."}]},
            {"type": "reasoning", "summary": [], "encrypted_content": "e"}]
        rows = cli_worker.codex_rows(items)
        self.assertEqual([row["role"] for row in rows],
                         ["user", "assistant", "tool", "assistant", "tool", "assistant", "assistant"])
        self.assertEqual([call["function"]["name"] for call in rows[1]["tool_calls"]], ["shell"])
        self.assertEqual(rows[2], {"role": "tool", "tool_call_id": "c1", "content": "exit_code: 2"})
        self.assertEqual([call["function"]["name"] for call in rows[5]["tool_calls"]], ["local_shell", "web_search"])
        self.assertEqual(rows[4]["content"], "Done")
        self.assertEqual(rows[6]["content"], "Plan: fix the build.")
        text = scoring.history_text(rows)
        for fact in ("make", "exit_code: 2", "apply_patch", "Plan: fix the build."):
            self.assertIn(fact, text)
        with self.assertRaisesRegex(cli_worker.JobFailed, "codex_context_item:new_kind"):
            cli_worker.codex_rows([{"type": "new_kind"}])
        # Claude Code: the context keeps thinking text; a reply text does not; an unknown block fails the job.
        content = [{"type": "thinking", "thinking": "Check the cache key."}, {"type": "text", "text": "Done."}]
        self.assertEqual(cli_worker.claude_text(content, thinking=True), "Check the cache key.\nDone.")
        self.assertEqual(cli_worker.claude_text(content), "Done.")
        with self.assertRaisesRegex(cli_worker.JobFailed, "claude_context_block:new_block"):
            cli_worker.claude_text([{"type": "new_block"}])
        entries = [{"type": "system", "subtype": "compact_boundary"},
                   {"type": "assistant", "message": {"role": "assistant", "content": content}}]
        self.assertEqual(cli_worker.claude_context(entries)[0],
                         [{"role": "assistant", "content": "Check the cache key.\nDone."}])

    def test_claude_transcript_links_entries_and_pairs_tool_results(self):
        entries = cli_worker.claude_transcript(self.history, "sid", Path("cwd"), "0.0.0", "m", self.base)
        self.assertEqual(len(entries), len(self.history["rows"]))
        self.assertIsNone(entries[0]["parentUuid"])
        for before, after in zip(entries, entries[1:]):
            self.assertEqual(after["parentUuid"], before["uuid"])
        uses = {block["id"] for entry in entries if entry["type"] == "assistant"
                for block in entry["message"]["content"] if block["type"] == "tool_use"}
        results = {block["tool_use_id"] for entry in entries if isinstance(entry["message"]["content"], list)
                   for block in entry["message"]["content"] if block["type"] == "tool_result"}
        self.assertEqual(uses, results)

    def test_claude_context_keeps_the_summary_and_drops_command_rows(self):
        entries = [
            {"type": "user", "message": {"role": "user", "content": "old"}},
            {"type": "system", "subtype": "compact_boundary", "compactMetadata": {"trigger": "manual",
                                                                                   "durationMs": 1500}},
            {"type": "user", "isCompactSummary": True, "message": {"role": "user", "content": "SUMMARY"}},
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "caveat"}},
            {"type": "user", "message": {"role": "user", "content": "<command-name>/compact</command-name>"}},
            {"type": "user", "message": {"role": "user", "content": "<local-command-stdout>ok</local-command-stdout>"}},
        ]
        rows, meta = cli_worker.claude_context(entries)
        self.assertEqual(rows, [{"role": "user", "content": "SUMMARY"}])
        self.assertEqual(meta["compress_model_s"], 1.5)
        self.assertEqual(meta["summary_chars"], len("SUMMARY"))
        self.assertEqual(cli_worker.claude_context(entries[:1]), (None, {}))

    def test_cli_environment_has_no_user_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            env = cli_worker.cli_environment(Path(folder), {"CODEX_HOME": folder})
        self.assertTrue(all(env[key] == folder for key in ("HOME", "USERPROFILE", "APPDATA", "CODEX_HOME")))
        self.assertFalse([key for key in env if key.upper().startswith(("ANTHROPIC", "OPENAI", "CLAUDE"))])

    def test_cli_methods_need_a_pinned_version(self):
        with self.assertRaisesRegex(SystemExit, "pinned"):
            run.cli_state({"cli": {}}, ["codex_cli"])
        self.assertEqual(run.cli_state({}, ["stock_hermes"]), {})
        self.assertEqual(run.anthropic_base_url({"endpoint": "http://localhost:1234/v1/"}), "http://localhost:1234")


# Synthetic main-turn replies of the fake CLIs.
CODEX_REPLY = {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                   "content": [{"type": "output_text", "text": "Done."}]}}
CLAUDE_REPLY = {"type": "assistant", "message": {"role": "assistant", "content": "Done."}}


class MainTurnFailureTest(unittest.TestCase):
    """A failed main turn stops the job before the compaction. Fake Hermes and CLI parts; no model runs."""

    def fake_hermes(self, calls, reply=None, compression_enabled=None, compress=None):
        class Agent:
            def __init__(self, **_kwargs):
                self.context_compressor = types.SimpleNamespace(name="compressor")
                self.session_id = "synthetic"
                self.compression_enabled = compression_enabled

            def run_conversation(self, *_args, **kwargs):
                return reply(kwargs["conversation_history"]) if reply else {"failed": True, "messages": []}

            def close(self):
                pass

        class SessionDB:
            def __init__(self, _path):
                pass

            def create_session(self, *_args):
                pass

            def close(self):
                pass

        def module(name, **items):
            item = types.ModuleType(name)
            item.__dict__.update(items)
            return item

        manager = types.SimpleNamespace(discover_and_load=lambda: None, _plugins={})
        plugins = module("hermes_cli.plugins", get_plugin_manager=lambda: manager)
        server = module("tui_gateway.server", _compress_session_history=compress or (
            lambda *_a: calls.append("compress")))
        notify = module("agent.conversation_compression",
                        finalize_context_engine_compression_notification=lambda *_a, **_k: None)
        return {"hermes_cli": module("hermes_cli", plugins=plugins), "hermes_cli.plugins": plugins,
                "hermes_state": module("hermes_state", SessionDB=SessionDB),
                "run_agent": module("run_agent", AIAgent=Agent), "tui_gateway": module("tui_gateway", server=server),
                "tui_gateway.server": server, "agent": module("agent", conversation_compression=notify),
                "agent.conversation_compression": notify}

    def run_hermes(self, engine, reply, compression_enabled=None, compress=None):
        history, _gold = fixtures.build(SCENARIOS[0])
        job = {"hermes_source": "synthetic", "engine": engine, "endpoint": "http://127.0.0.1:9/v1",
               "api_key": "fake", "model": "fake", "cycles": 1, "questions": [], "qa_batches": []}
        out = {"cycles": []}
        calls = []
        modules = self.fake_hermes(calls, reply, compression_enabled, compress)
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, modules), \
                mock.patch.object(worker, "write_config"), mock.patch.object(sys, "path", list(sys.path)):
            worker.run_cycles(job, history, Path(folder), out)
        return history, out

    def test_control_cannot_compact(self):
        job = {"model": "m", "endpoint": "http://127.0.0.1:9/v1", "api_key": "", "context_length": 8,
               "threshold": 0.9, "engine": "none"}
        self.assertFalse(worker.base_config(job)["compression"]["enabled"])
        self.assertTrue(worker.base_config({**job, "engine": "compressor"})["compression"]["enabled"])
        with self.assertRaisesRegex(worker.JobFailed, "control_auto_compaction_on"):
            self.run_hermes("none", lambda rows: {"messages": rows, "final_response": "ok"}, compression_enabled=True)
        # A control turn whose history shrinks (a compaction) fails; it is not labeled "none".
        with self.assertRaisesRegex(worker.JobFailed, "control_history_shrank"):
            self.run_hermes("none", lambda rows: {"messages": rows[-3:], "final_response": "ok"},
                            compression_enabled=False)

    def test_preflight_sizes_cover_the_same_rows_before_and_after(self):
        resume = fixtures.build(SCENARIOS[0])[0]["resume"]
        summary = [{"role": "user", "content": "[synthetic summary]"}]

        def reply(_rows, tail=({"role": "assistant", "content": "x" * 4000},)):
            # A preflight compaction: the main turn ran on the summary, the user row, and then the reply.
            return {"messages": summary + [{"role": "user", "content": resume}, *tail], "final_response": "x"}

        fixture, out = self.run_hermes("compressor", reply)
        cycle = out["cycles"][0]
        self.assertTrue(cycle["preflight"])
        # Before: the history and the user row. After: the summary and the user row, without the reply.
        compacted = summary + [{"role": "user", "content": resume}]
        self.assertEqual(cycle["est_tokens_before"], worker.estimate_tokens(worker.wire_rows(
            fixture["rows"] + [{"role": "user", "content": resume}])))
        self.assertEqual((cycle["rows_after"], cycle["est_tokens_after"]),
                         (2, worker.estimate_tokens(worker.wire_rows(compacted))))
        self.assertEqual(cycle["removed"], len(fixture["rows"]) + 1 - 2)
        # The questions see the history after the main turn (with the reply).
        self.assertEqual(len(cycle["history"]), 3)
        with self.assertRaisesRegex(worker.JobFailed, "preflight_user_row_missing"):
            self.run_hermes("compressor", lambda _rows: {"messages": summary + [
                {"role": "assistant", "content": "x"}], "final_response": "x"})

    def test_sizes_measure_the_exact_compaction_input(self):
        def reply(rows):
            return {"messages": list(rows) + [{"role": "user", "content": "resume"},
                                              {"role": "assistant", "content": "x" * 4000}],
                    "final_response": "x" * 4000}

        def compress(session, _focus):
            session["history"] = session["history"][-2:]
            return 10, None

        fixture, out = self.run_hermes("compressor", reply, compress=compress)
        cycle = out["cycles"][0]
        after_main = reply(fixture["rows"])["messages"]
        self.assertEqual(cycle["est_tokens_before"], worker.estimate_tokens(worker.wire_rows(after_main)))
        self.assertEqual(cycle["est_tokens_before_main"], worker.estimate_tokens(worker.wire_rows(fixture["rows"])))
        self.assertEqual(cycle["rows_before"], len(after_main))
        self.assertGreater(cycle["est_tokens_before"], cycle["est_tokens_before_main"] + 900)
        scored = run.score_cycle(cycle, fixtures.build(SCENARIOS[0])[1], fixtures.build(SCENARIOS[0])[0],
                                 "stock_hermes")
        self.assertEqual(scored["size_basis"], "compaction_input")

    def test_hermes_worker_stops_after_a_failed_main_turn(self):
        calls = []
        history, _gold = fixtures.build(SCENARIOS[0])
        job = {"hermes_source": "synthetic", "engine": "compressor", "endpoint": "http://127.0.0.1:9/v1",
               "api_key": "fake", "model": "fake", "cycles": 1, "questions": [], "qa_batches": []}
        out = {"cycles": []}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, self.fake_hermes(calls)), \
                mock.patch.object(worker, "write_config"), mock.patch.object(sys, "path", list(sys.path)):
            with self.assertRaises(worker.MainTurnFailed):
                worker.run_cycles(job, history, Path(folder), out)
        self.assertEqual(calls, [])
        self.assertTrue(out["cycles"][0]["main_failed"])
        self.assertNotIn("compacted", out["cycles"][0])

    def run_cli_job(self, engine, main):
        history, _gold = fixtures.build(SCENARIOS[0])
        job = {"engine": engine, "cli_command": ["fake-cli"], "cli_version": "0.0.0", "model": "fake",
               "api_key": "fake",
               "endpoint": "http://127.0.0.1:9/v1", "anthropic_base_url": "http://127.0.0.1:9",
               "context_length": 131072, "cli_timeout_s": 5}
        out = {"cycles": []}
        commands = []

        def fake_run(command, *_args):
            commands.append(command)
            return {"elapsed_s": 1.0, "login_prompt": False, **main}

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(cli_worker, "run_cli", fake_run), \
                mock.patch.object(cli_worker, "codex_app_server_compact",
                                  side_effect=AssertionError("compaction after a failed main turn")):
            function = cli_worker.run_codex if engine == "codex_cli" else cli_worker.run_claude
            with self.assertRaises(cli_worker.MainTurnFailed) as caught:
                function(job, history, Path(folder), out)
        self.assertEqual(len(commands), 1)
        self.assertTrue(out["cycles"][0]["main_failed"])
        return str(caught.exception)

    def test_failed_qa_request_fails_the_job(self):
        job = {"questions": [{"n": 1, "text": "Q?"}], "model": "m", "endpoint": "http://127.0.0.1:9/v1",
               "api_key": "", "qa_batches": [[1]]}
        with mock.patch.object(worker, "post_chat", side_effect=TimeoutError()):
            with self.assertRaisesRegex(worker.JobFailed, "qa_request:TimeoutError"):
                worker.ask(job, "system", [])
        with mock.patch.object(worker, "post_chat", return_value={"error": {"message": "overloaded"}}):
            with self.assertRaisesRegex(worker.JobFailed, "qa_request:no_choices"):
                worker.ask(job, "system", [])
        # Both workers report the failure as the job error (status failed), not as wrong answers.
        for module, runner in ((worker, "run_cycles"), (cli_worker, "run_codex")):
            with tempfile.TemporaryDirectory() as folder:
                fixture = Path(folder) / "f.json"
                fixture.write_text("{}", encoding="utf-8")
                (Path(folder) / "job.json").write_text(json.dumps(
                    {"fixture_path": str(fixture), "engine": "codex_cli", "cycles": 1}), encoding="utf-8")
                argv = ["w", "--job", str(Path(folder) / "job.json"), "--out", str(Path(folder) / "out.json")]
                with mock.patch.object(module, runner, side_effect=module.JobFailed("qa_request:TimeoutError")), \
                        mock.patch.object(sys, "argv", argv), mock.patch.dict("os.environ", {"HERMES_HOME": folder}):
                    module.main()
                out = json.loads((Path(folder) / "out.json").read_text(encoding="utf-8"))
            self.assertEqual((out["status"], out["error"]), ("failed", "qa_request:TimeoutError"))

    def test_every_qa_reply_that_is_not_a_full_answer_fails_the_request(self):
        job = {"questions": [{"n": 1, "text": "Q1?"}, {"n": 2, "text": "Q2?"}], "model": "m",
               "endpoint": "http://127.0.0.1:9/v1", "api_key": "", "qa_batches": [[1, 2]]}

        def reply(content: object = '{"1": "a", "2": "b"}', finish: object = "stop") -> dict:
            return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}]}

        cases = {
            "no_choices": [{"error": {"message": "overloaded"}}, {"choices": []}, {"choices": "x"}, [], None],
            "no_content": [reply(None), reply(""), reply("  \n"), reply(["1. a"]),
                           {"choices": [{"finish_reason": "stop"}]}, {"choices": [{"message": None}]},
                           {"choices": ["text"]}],
            "finish_length": [reply(finish="length")],
            "finish_tool_calls": [reply(finish="tool_calls")],
            "finish_content_filter": [reply(finish="content_filter")],
            "finish_None": [reply(finish=None)],
            "incomplete_answers": [reply("I cannot answer."), reply('{"1": "a"}'), reply('{"1": "a", "2": ""}'),
                                   reply('{"1": "a", "2": null}'), reply("1. a"), reply('{"3": "a", "4": "b"}'),
                                   reply("<think>1. a\n2. b</think>")],
        }
        for error, replies in cases.items():
            for item in replies:
                with self.subTest(error=error, reply=item), \
                        mock.patch.object(worker, "post_chat", return_value=item):
                    with self.assertRaisesRegex(worker.JobFailed, f"^qa_request:{error}$"):
                        worker.ask(job, "system", [])
        for text in ('{"1": "a", "2": "unknown"}', "1. a\n2. b", '```json\n{"1": "a", "2": "b"}\n```'):
            with mock.patch.object(worker, "post_chat", return_value=reply(text)):
                self.assertEqual(worker.ask(job, "system", [])["batches"][0]["text"], text)

    def test_worker_asks_the_batches_of_the_gold_plan(self):
        _history, gold = fixtures.build(next(item for item in SCENARIOS if item["id"] == "s4-orchard"))
        job = {"questions": [{"n": n, "text": q["question"]} for n, q in enumerate(gold["questions"], 1)],
               "qa_batches": gold["qa_batches"], "model": "m", "endpoint": "http://127.0.0.1:9/v1", "api_key": ""}
        sent = []

        def fake_post(_endpoint, _key, body, _timeout, _headers=None):
            sent.append(body["messages"][-1]["content"])
            numbers = gold["qa_batches"][len(sent) - 1]
            return {"choices": [{"message": {"content": json.dumps({str(n): "unknown" for n in numbers})},
                                 "finish_reason": "stop"}]}

        with mock.patch.object(worker, "post_chat", side_effect=fake_post):
            result = worker.ask(job, "system", [])
        self.assertEqual(sent, [scoring.qa_prompt([(n, gold["questions"][n - 1]["question"]) for n in batch])
                                for batch in gold["qa_batches"]])
        self.assertEqual([batch["numbers"] for batch in result["batches"]], gold["qa_batches"])
        # Each raw batch names the exact prompt that the model saw.
        self.assertEqual([batch["prompt_sha256"] for batch in result["batches"]],
                         [scoring.prompt_sha256(text) for text in sent])
        # A plan that does not ask each question once fails before any request.
        for plan in ([[1]], gold["qa_batches"] + [[1]]):
            with mock.patch.object(worker, "post_chat", side_effect=AssertionError("no request")):
                with self.assertRaisesRegex(worker.JobFailed, "qa_plan_mismatch"):
                    worker.ask({**job, "qa_batches": plan}, "system", [])

    def test_a_scored_cycle_without_a_full_answer_fails_the_job(self):
        spec = next(item for item in SCENARIOS if item["id"] == "s1-ledgerline")
        history, gold = fixtures.build(spec)
        count = len(gold["questions"])
        full = json.dumps({str(n): "unknown" for n in range(1, count + 1)})
        part = json.dumps({str(n): "unknown" for n in range(1, count)})
        cycle = {"cycle": 1, "compacted": True, "history": history["rows"][-4:],
                 "qa": {"batches": [{"numbers": list(range(1, count + 1)), "text": full}]}}
        self.assertEqual(run.score_cycle(cycle, gold, history, "warm_ledger")["qa_missing"], 0)
        # A question that no batch asked is missing too.
        short = {**cycle, "qa": {"batches": [{"numbers": list(range(1, count)), "text": part}]}}
        self.assertEqual(run.score_cycle(short, gold, history, "warm_ledger")["qa_missing"], 1)
        # A cycle that did not compact asks no questions; the control asks them on each cycle.
        self.assertIsNone(run.score_cycle({**cycle, "compacted": False, "qa": {}}, gold, history,
                                          "warm_ledger")["qa_missing"])
        self.assertEqual(run.score_cycle({**cycle, "compacted": False, "qa": {}}, gold, history,
                                         "full_context")["qa_missing"], count)
        record = {"status": "ok", "error": None, "cycles": [run.score_cycle(short, gold, history, "warm_ledger")]}
        self.assertTrue(run.fail_incomplete(record))
        self.assertEqual((record["status"], record["error"]), ("failed", "qa_incomplete"))
        # --rescore applies the same rule to kept raw output (gold batch plan, the last answer left out).
        last = gold["qa_batches"][-1][-1]
        planned = {**cycle, "qa": {"batches": planned_batches(gold, lambda n: "" if n == last else "unknown")}}
        records = [{"job": f"{spec['id']}|warm_ledger|r1|c1", "fixture": spec["id"], "method": "warm_ledger",
                    "status": "ok", "error": None, "cycles": [run.score_cycle(cycle, gold, history, "warm_ledger")]}]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "raw").mkdir()
            (Path(folder) / "raw" / f"{spec['id']}__warm_ledger__r1__c1.json").write_text(
                json.dumps({"cycles": [planned]}), encoding="utf-8")
            counts = run.rescore(Path(folder), records)
        self.assertEqual(counts["jobs_failed_incomplete"], 1)
        self.assertEqual((records[0]["status"], records[0]["error"]), ("failed", "qa_incomplete"))

    def test_hermes_worker_fails_when_the_engine_is_not_the_planned_one(self):
        calls = []
        history, _gold = fixtures.build(SCENARIOS[0])
        # The fake agent reports the built-in compressor, as when the plugin does not load.
        job = {"hermes_source": "synthetic", "engine": "warm_compaction", "endpoint": "http://127.0.0.1:9/v1",
               "api_key": "fake", "model": "fake", "cycles": 1, "questions": [], "qa_batches": []}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, self.fake_hermes(calls)), \
                mock.patch.object(worker, "write_config"), mock.patch.object(sys, "path", list(sys.path)):
            with self.assertRaisesRegex(worker.JobFailed, "engine_mismatch:compressor"):
                worker.run_cycles(job, history, Path(folder), {"cycles": []})
        self.assertEqual(calls, [])

    def run_cli_session(self, engine, main_lines, compact_lines=()):
        """Run a CLI job with a fake CLI that appends lines to the session file (synthetic data)."""
        history, _gold = fixtures.build(SCENARIOS[0])
        job = {"engine": engine, "cli_command": ["fake-cli"], "cli_version": "0.0.0", "model": "fake",
               "api_key": "fake",
               "endpoint": "http://127.0.0.1:9/v1", "anthropic_base_url": "http://127.0.0.1:9",
               "context_length": 131072, "cli_timeout_s": 5, "questions": [], "qa_batches": []}
        out = {"cycles": []}
        with tempfile.TemporaryDirectory() as folder:
            pattern = "codex-home/sessions/**/*.jsonl" if engine == "codex_cli" else "claude-home/**/*.jsonl"
            calls = []

            def fake_run(command, *_args):
                session = next(Path(folder).glob(pattern))
                lines = main_lines if not calls else compact_lines
                calls.append(command)
                with open(session, "a", encoding="utf-8") as handle:
                    handle.write("".join(json.dumps(line) + "\n" for line in lines))
                return {"elapsed_s": 1.0, "login_prompt": False, "exit_code": 0, "timeout": False, "text": ""}

            with mock.patch.object(cli_worker, "run_cli", fake_run), mock.patch.object(
                    cli_worker, "codex_app_server_compact", side_effect=AssertionError("no compaction expected")):
                (cli_worker.run_codex if engine == "codex_cli" else cli_worker.run_claude)(
                    job, history, Path(folder), out)
        return history, out

    def test_cli_main_turn_may_not_compact(self):
        with self.assertRaisesRegex(cli_worker.JobFailed, "main_turn_compacted"):
            self.run_cli_session("claude_code_cli", [CLAUDE_REPLY, {"type": "system", "subtype": "compact_boundary"}])
        with self.assertRaisesRegex(cli_worker.JobFailed, "main_turn_compacted"):
            self.run_cli_session("codex_cli", [CODEX_REPLY, {"type": "compacted", "payload": {"message": "summary"}}])

    def test_every_main_turn_without_reply_text_fails_the_job(self):
        # Codex and Claude Code: a main turn that exits 0 but adds no assistant reply text. The assistant rows of
        # the seeded session do not count as the reply.
        empty_codex = {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                            "content": [{"type": "output_text", "text": " "}]}}
        empty_claude = {"type": "assistant", "message": {"role": "assistant", "content": ""}}
        for engine, lines in (("codex_cli", []), ("codex_cli", [empty_codex]), ("claude_code_cli", []),
                              ("claude_code_cli", [empty_claude])):
            with self.subTest(engine=engine, lines=lines), \
                    self.assertRaisesRegex(cli_worker.MainTurnFailed, "^empty_reply$"):
                self.run_cli_session(engine, lines)
        # Hermes: run_conversation without final response text; no compaction runs.
        for final in (None, "", " \n"):
            calls = []
            history, _gold = fixtures.build(SCENARIOS[0])
            job = {"hermes_source": "synthetic", "engine": "compressor", "endpoint": "http://127.0.0.1:9/v1",
                   "api_key": "fake", "model": "fake", "cycles": 1, "questions": [], "qa_batches": []}
            out = {"cycles": []}
            fake = self.fake_hermes(calls, lambda rows, final=final: {
                "messages": list(rows) + [{"role": "user", "content": "resume"}], "final_response": final})
            with self.subTest(final=final), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.dict(sys.modules, fake), mock.patch.object(worker, "write_config"), \
                    mock.patch.object(sys, "path", list(sys.path)):
                with self.assertRaisesRegex(worker.MainTurnFailed, "^empty_reply$"):
                    worker.run_cycles(job, history, Path(folder), out)
                self.assertEqual(calls, [])
                self.assertTrue(out["cycles"][0]["main_failed"])

    def test_qa_and_cli_requests_use_the_route_of_the_main_turn(self):
        # Synthetic route: a fake key and one allowed extra header (hermes_config.model.headers).
        route = {"api_key": "fake-key", "hermes_config": {"model": {"headers": {"X-Gateway": "synthetic"}}}}
        self.assertEqual(worker.route_headers(route), {"X-Gateway": "synthetic"})
        self.assertEqual(worker.route_headers({}), {})
        sent = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": '{"1": "a"}'},
                                                "finish_reason": "stop"}]}).encode("utf-8")

        def fake_urlopen(request, **_kwargs):
            sent.append(request)
            return Response()

        job = {**route, "questions": [{"n": 1, "text": "Q?"}], "qa_batches": [[1]], "model": "m",
               "endpoint": "http://127.0.0.1:9/v1"}
        with mock.patch.object(worker.urllib.request, "urlopen", fake_urlopen):
            worker.ask(job, "system", [])
        headers = {key.lower(): value for key, value in sent[0].header_items()}
        self.assertEqual((headers["x-gateway"], headers["authorization"]), ("synthetic", "Bearer fake-key"))
        # Codex CLI: the provider reads the same key and sends the same headers. Claude Code: the same.
        for engine in ("codex_cli", "claude_code_cli"):
            history, _gold = fixtures.build(SCENARIOS[0])
            cli_job = {**route, "engine": engine, "cli_command": ["fake-cli"], "cli_version": "0.0.0", "model": "m",
                       "endpoint": "http://127.0.0.1:9/v1", "anthropic_base_url": "http://127.0.0.1:9",
                       "context_length": 131072, "cli_timeout_s": 5}
            seen = {}

            def fake_run(_command, _cwd, env, *_args, seen=seen):
                seen["env"] = dict(env)
                config = Path(env.get("CODEX_HOME", "missing")) / "config.toml"
                seen["config"] = config.read_text(encoding="utf-8") if config.exists() else ""
                return {"elapsed_s": 1.0, "login_prompt": False, "exit_code": 1, "timeout": False, "text": ""}

            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(cli_worker, "run_cli", fake_run):
                function = cli_worker.run_codex if engine == "codex_cli" else cli_worker.run_claude
                with self.assertRaises(cli_worker.MainTurnFailed):
                    function(cli_job, history, Path(folder), {"cycles": []})
            if engine == "codex_cli":
                self.assertEqual(seen["env"]["WC_BENCH_API_KEY"], "fake-key")
                self.assertIn('env_key = "WC_BENCH_API_KEY"', seen["config"])
                self.assertIn('http_headers = {"X-Gateway" = "synthetic"}', seen["config"])
                if sys.version_info >= (3, 11):
                    import tomllib
                    provider = tomllib.loads(seen["config"])["model_providers"]["benchlocal"]
                    self.assertEqual(provider["http_headers"], {"X-Gateway": "synthetic"})
            else:
                self.assertEqual(seen["env"]["ANTHROPIC_API_KEY"], "fake-key")
                self.assertEqual(seen["env"]["ANTHROPIC_CUSTOM_HEADERS"], "X-Gateway: synthetic")

    def test_cli_questions_get_every_row_of_the_compacted_context(self):
        # A Codex context with an assistant row that has tool calls only, and the tool row of the call.
        rows = cli_worker.codex_rows([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "goal"}]},
            {"type": "function_call", "name": "shell", "arguments": '{"cmd": "make"}', "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "exit_code: 2"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "SUMMARY"}]}])
        self.assertEqual(rows[1]["content"], "")
        asked = []
        item = {"rows_before": 10}
        compact = {"elapsed_s": 1.0, "exit_code": 0, "timeout": False, "status": "completed", "error": None}

        def fake_ask(_job, _system, given):
            asked.append(given)
            return {}

        with mock.patch.object(cli_worker, "ask", fake_ask):
            cli_worker.finish(item, {}, {"system": "system"}, rows, compact, {}, path="codex")
        expected = worker.wire_rows(rows)
        self.assertEqual(item["history"], expected)
        self.assertEqual(asked, [expected])
        self.assertEqual([row["role"] for row in expected], ["user", "assistant", "tool", "user"])
        self.assertEqual(expected[1]["tool_calls"][0]["function"]["name"], "shell")
        self.assertEqual(expected[2]["tool_call_id"], "c1")
        self.assertEqual((item["rows_after"], item["est_tokens_after"]), (4, worker.estimate_tokens(expected)))

    def test_codex_app_server_wait_uses_one_deadline(self):
        class FakeServer:
            """A synthetic app server that answers the given request IDs, then stalls (it never exits)."""

            def __init__(self, answers):
                self.stdin = mock.Mock()
                self.stdout = iter([json.dumps({"id": number, "result": {}}) + "\n" for number in answers])
                self.stderr = iter([])
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        # Initialize stalls; or initialize answers and resume stalls; or both answer and the compaction stalls.
        for answers, error in (([], "initialize_failed"), ([1], "resume_failed"), ([1, 2], "no_turn_completed")):
            with self.subTest(error=error), mock.patch.object(
                    cli_worker.subprocess, "Popen", lambda *_a, answers=answers, **_k: FakeServer(answers)):
                started = time.monotonic()
                result = cli_worker.codex_app_server_compact(["codex"], "thread", Path("."), {}, 0.5)
                self.assertLess(time.monotonic() - started, 10)
                self.assertEqual(result["error"], error)
                self.assertTrue(result["timeout"])
                self.assertEqual(cli_worker.compaction_failure({}, result, "codex"), "cli_timeout")

    def test_failed_cli_compaction_command_is_not_accepted(self):
        rows = [{"role": "user", "content": "synthetic summary"}]
        codex_ok = {"elapsed_s": 1.0, "exit_code": 0, "timeout": False, "status": "completed", "error": None}
        claude_ok = {"elapsed_s": 1.0, "exit_code": 0, "timeout": False}
        # The command wrote a compacted record, then timed out, exited with an error, or reported a failed turn.
        cases = (("codex", codex_ok, {}, None),
                 ("codex", {**codex_ok, "timeout": True}, {}, "cli_timeout"),
                 ("codex", {**codex_ok, "status": "failed"}, {}, "turn_failed"),
                 ("codex", {**codex_ok, "error": "turn_error"}, {}, "turn_error"),
                 ("claude_code", claude_ok, {}, None),
                 ("claude_code", {**claude_ok, "exit_code": 1}, {}, "exit_1"),
                 ("claude_code", claude_ok, {"compact_result": {"is_error": True}}, "error_result"))
        for path, compact, extra, failure in cases:
            with self.subTest(path=path, failure=failure):
                item = {"rows_before": 10, **extra}
                with mock.patch.object(cli_worker, "ask", return_value={"batches": [], "calls": []}) as ask:
                    cli_worker.finish(item, {}, {"system": "system"}, rows, compact, {}, path=path)
                self.assertEqual(item["compacted"], failure is None)
                self.assertEqual(item["compress_error"], failure)
                self.assertEqual(item["outcome"]["path"], path if failure is None else "aborted")
                self.assertEqual(ask.called, failure is None)
                # The record keeps the reason; the scored cycle is a failed compaction of the method.
                scored = run.score_cycle({**item, "cycle": 1}, {"questions": []}, {"rows": []}, f"{path}_cli"
                                         if path == "codex" else "claude_code_cli")
                self.assertEqual(scored["compress_error"], failure)

    def test_cli_sizes_measure_the_compaction_input(self):
        reply = "y" * 4000
        thinking, call, result = "t" * 3000, {"q": "c" * 2000}, "r" * 1000  # synthetic main-turn items
        history, _gold = fixtures.build(SCENARIOS[0])
        main = [{"type": "user", "message": {"role": "user", "content": history["resume"]}},
                {"type": "assistant", "message": {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": thinking}, {"type": "tool_use", "name": "probe", "input": call}]}},
                {"type": "user", "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": result}]}},
                {"type": "assistant", "message": {"role": "assistant", "content": reply}}]
        compact = [{"type": "system", "subtype": "compact_boundary", "compactMetadata": {"durationMs": 1000}},
                   {"type": "user", "isCompactSummary": True, "message": {"role": "user", "content": "summary"}}]
        history, out = self.run_cli_session("claude_code_cli", main, compact)
        cycle = out["cycles"][0]
        # The input has the thinking, the tool call, and the tool result of the main turn, and the main user row once.
        expected = cli_worker.text_rows_all(history["rows"]) + [
            {"role": "user", "content": history["resume"]},
            {"role": "assistant", "content": f"{thinking}\n[tool call probe: {json.dumps(call)}]"},
            {"role": "user", "content": result}, {"role": "assistant", "content": reply}]
        self.assertEqual(cycle["est_tokens_before"], worker.estimate_tokens(expected))
        self.assertEqual(cycle["rows_before"], len(expected))
        # Codex: reasoning, a function call, and its output count too; an unknown item kind counts as its JSON text.
        codex = [{"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
                     {"type": "input_text", "text": history["resume"]}]}},
                 {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"type": "summary_text",
                                                                                          "text": thinking}]}},
                 {"type": "response_item", "payload": {"type": "function_call", "call_id": "c1", "name": "probe",
                                                       "arguments": json.dumps(call)}},
                 {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1",
                                                       "output": result}},
                 {"type": "response_item", "payload": {"type": "ghost_snapshot", "note": "n" * 500}},
                 {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [
                     {"type": "output_text", "text": reply}]}}]
        with self.assertRaisesRegex(AssertionError, "no compaction expected"):
            self.run_cli_session("codex_cli", codex)
        codex_rows = cli_worker.codex_added_rows(codex, history["resume"])
        self.assertEqual([row["role"] for row in codex_rows], ["assistant", "tool", "user", "assistant"])
        self.assertEqual(codex_rows[0]["content"], thinking)
        self.assertEqual(codex_rows[0]["tool_calls"][0]["function"]["arguments"], json.dumps(call))
        self.assertIn("[ghost_snapshot]", codex_rows[2]["content"])
        item = {}
        cli_worker.compaction_input(item, history, codex_rows)
        self.assertGreater(item["est_tokens_before"], worker.estimate_tokens(
            cli_worker.text_rows_all(history["rows"]) + [{"role": "user", "content": history["resume"]},
                                                          {"role": "assistant", "content": reply}]) + 1200)
        self.assertEqual(cycle["est_tokens_before_main"],
                         worker.estimate_tokens(cli_worker.text_rows_all(history["rows"])))
        self.assertTrue(cycle["compacted"])

    def test_cli_worker_stops_after_a_failed_main_command(self):
        timeout = {"exit_code": None, "timeout": True, "text": ""}
        failed = {"exit_code": 1, "timeout": False, "text": ""}
        error_result = {"exit_code": 0, "timeout": False, "text": json.dumps({"type": "result", "is_error": True})}
        self.assertEqual(self.run_cli_job("codex_cli", timeout), "timeout")
        self.assertEqual(self.run_cli_job("codex_cli", failed), "exit_1")
        self.assertEqual(self.run_cli_job("claude_code_cli", failed), "exit_1")
        self.assertEqual(self.run_cli_job("claude_code_cli", error_result), "error_result")


if __name__ == "__main__":
    unittest.main()
