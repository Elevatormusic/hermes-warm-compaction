"""Tests for the compaction benchmark harness (benchmarks/compaction). No model and no Hermes code run here."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.compaction import cli_worker, fixtures, run, scoring
from benchmarks.compaction.scenarios import SCENARIOS

GOLD_DIR = Path(run.__file__).resolve().parent / "fixtures"


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
                  "drift": {"fixtures": ["s1-ledgerline"], "methods": ["warm_ledger", "full_context"], "cycles": 3}}
        jobs, skipped = run.plan(config, {"settings_known": ["warm", "ledger"]})
        self.assertEqual(skipped, [])
        self.assertEqual([job["method"] for job in jobs[:3]], ["full_context", "stock_hermes", "warm_ledger"])
        self.assertEqual([job["method"] for job in jobs[3:6]], ["stock_hermes", "warm_ledger", "full_context"])
        self.assertEqual([(job["method"], job["cycles"]) for job in jobs[6:]], [("warm_ledger", 3)] * 2)

    def test_classify(self):
        self.assertEqual(run.classify({}, "full_context"), "none")
        self.assertEqual(run.classify({"compacted": True, "outcome": {"path": "warm"}}, "warm_ledger"), "warm")
        self.assertEqual(run.classify({"compress_error": "CompactionAborted", "outcome": {}}, "warm_ledger"),
                         "aborted")
        self.assertEqual(run.classify({"compacted": False, "outcome": {}}, "stock_hermes"), "no_change")
        static = {"compacted": True, "history": [{"role": "user", "content": run.STATIC_FALLBACK_TEXT}]}
        self.assertEqual(run.classify(static, "stock_hermes"), "static_fallback")
        preflight = {"compacted": True, "preflight": True, "outcome": {"path": "warm", "trigger": "preflight"}}
        self.assertEqual(run.classify(preflight, "warm_ledger"), "preflight")
        self.assertIn("preflight", run.ACCEPTED_PATHS)

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
                            "cycles_requested": 1, "status": "ok", "cycles": [scored]})
        return records

    def test_size_check_has_no_data_when_no_ledger_run_compacts(self):
        runs = self.synthetic_runs()
        runs[0]["cycles"][0]["compacted"] = False
        meta = {"config": {}, "plugin": {}, "hermes": {}}
        result = run.aggregate(runs, meta)
        self.assertIsNone(result["methods"]["warm_ledger"]["tokens"]["est_after"]["mean"])
        self.assertIsNone(result["decision"]["checks"]["H4_smaller_than_stock"])
        self.assertIn("Warm path + ledger does not meet the rule", " ".join(result["decision"]["verdict"]))

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

    def test_resume_refuses_a_changed_source(self):
        plugin = run.plugin_state(Path(run.__file__).parents[2])
        hermes = {"commit": "abc", "clean": True}
        config = {"model": "m", "threshold": 0.95, "methods": ["stock_hermes"], "plugin_source": "/private"}
        gold = {path.name: run.sha256_file(path) for path in sorted(GOLD_DIR.glob("*.json"))}
        meta = {"config": {k: v for k, v in config.items() if k != "plugin_source"}, "plugin": plugin,
                "hermes": hermes, "gold_sha256": gold, "harness_sha256": run.harness_hashes()}
        self.assertEqual(run.resume_mismatches(meta, config, plugin, hermes, {}), ([], []))
        # Job selection keys can change on a resume; result settings, plugin, and Hermes cannot.
        self.assertEqual(run.resume_mismatches(meta, {**config, "methods": ["full_context"]}, plugin, hermes, {}),
                         ([], []))
        changed = {**plugin, "files_sha256": {**plugin["files_sha256"], "engine.py": "0"}}
        self.assertEqual(run.resume_mismatches(meta, config, changed, hermes, {})[0], ["plugin files"])
        self.assertEqual(run.resume_mismatches(meta, config, plugin, {"commit": "def"}, {})[0], ["Hermes commit"])
        self.assertEqual(run.resume_mismatches(meta, {**config, "threshold": 0.5}, plugin, hermes, {})[0], ["config"])
        old = {**meta, "harness_sha256": {**meta["harness_sha256"], "worker.py": "0"}}
        self.assertEqual(run.resume_mismatches(old, config, plugin, hermes, {}), ([], ["worker.py"]))
        # A recorded resume with --allow-harness-change becomes the new reference.
        old["resumes"] = [{"harness_sha256": run.harness_hashes()}]
        self.assertEqual(run.resume_mismatches(old, config, plugin, hermes, {}), ([], []))

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
        self.assertEqual(rows, [{"role": "user", "content": "goal"}, {"role": "user", "content": "SUMMARY"}])
        self.assertEqual(meta["compress_model_s"], 2.5)
        self.assertEqual(cli_worker.codex_context(lines[:1]), (None, {}))

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


if __name__ == "__main__":
    unittest.main()
