"""Compaction benchmark for the warm_compaction Hermes plugin: build fixtures, run jobs, and write the report.

Commands:
    python -m benchmarks.compaction.run fixtures [--work .work/bench-fixtures]
    python -m benchmarks.compaction.run run --config <config.json> --out <run folder> [--limit N] [--dry-run]
    python -m benchmarks.compaction.run report --out <run folder> --json <results.json> --md <report.md>

Each job runs worker.py in a fresh process with the Python of a Hermes virtual environment, a clean Hermes
checkout, and a new temporary Hermes home. The run folder (keep it in .work/, which Git ignores) gets runs.jsonl
with metadata and scores only. With "keep_text": true in the config, raw answers and compacted rows of each job go
to <run folder>/raw/ for local checks; never publish that folder. All sessions are synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import fixtures, scoring

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GOLD_DIR = HERE / "fixtures"
WORKER = HERE / "worker.py"
CLI_WORKER = HERE / "cli_worker.py"
CATEGORIES = ("goal", "user_instruction", "command_outcome", "file_changed", "identifier", "decision", "task_state",
              "next_step", "recent")
ACCEPTED_PATHS = ("warm", "fallback", "model", "preflight")
STATIC_FALLBACK_TEXT = "Summary generation was unavailable"
# The compaction methods. "requires" names plugin settings that must exist in the plugin source.
METHODS: dict[str, dict[str, Any]] = {
    "full_context": {"engine": "none", "label": "No compaction (control)"},
    "stock_hermes": {"engine": "compressor", "label": "Hermes built-in compressor"},
    "warm_ledger": {"engine": "warm_compaction", "settings": {"warm": True, "ledger": True},
                    "label": "Warm path + ledger"},
    "warm_no_ledger": {"engine": "warm_compaction", "settings": {"warm": True, "ledger": False},
                       "label": "Warm path, no ledger"},
    "fallback_quoted_ledger": {"engine": "warm_compaction", "settings": {"warm": False, "ledger": True},
                               "format": "quoted", "label": "Fallback (quoted) + ledger"},
    "fallback_quoted_no_ledger": {"engine": "warm_compaction", "settings": {"warm": False, "ledger": False},
                                  "format": "quoted", "label": "Fallback (quoted), no ledger"},
    "fallback_relaxed_ledger": {"engine": "warm_compaction", "settings": {"warm": False, "ledger": True},
                                "format": "relaxed", "label": "Fallback (relaxed) + ledger"},
    "fallback_relaxed_no_ledger": {"engine": "warm_compaction", "settings": {"warm": False, "ledger": False},
                                   "format": "relaxed", "label": "Fallback (relaxed), no ledger"},
    # The compaction of other agent CLIs on the same model and fixtures (cli_worker.py). Added after the
    # pre-registration; the decision rule does not use them.
    "codex_cli": {"engine": "codex_cli", "label": "Codex CLI compaction"},
    "claude_code_cli": {"engine": "claude_code_cli", "label": "Claude Code compaction"},
}
CLI_ENGINES = ("codex_cli", "claude_code_cli")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plugin_state(plugin_root: Path) -> dict[str, Any]:
    """Hashes of the plugin files and the settings that its engine knows."""
    package = plugin_root / "warm_compaction"
    files = {path.name: sha256_file(path) for path in sorted(package.glob("*.py")) if path.is_file()}
    engine = (package / "engine.py").read_text(encoding="utf-8")
    known = [key for key in ("warm", "ledger", "fallback_format", "ledger_on_failure", "timeout_s")
             if f'"{key}"' in engine.split("THRESHOLD_RANGE", 1)[0]]
    commit = None
    try:
        commit = subprocess.run(["git", "-C", str(plugin_root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30).stdout.strip() or None
    except Exception:
        pass
    return {"files_sha256": files, "settings_known": known, "commit": commit}


def hermes_state(source: Path) -> dict[str, Any]:
    def git(*args):
        try:
            done = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, timeout=60)
        except Exception:
            return None
        # A folder that is not a Git checkout gives an error: its state is unknown, not clean.
        return done.stdout.strip() if done.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "clean": git("status", "--porcelain") == ""}


def method_settings(name: str, plugin: dict, threshold: float) -> dict[str, Any] | None:
    """The plugin settings of a method, or None when this plugin source cannot run it."""
    method = METHODS[name]
    if method["engine"] != "warm_compaction":
        return {}
    settings = {"threshold": threshold, **method.get("settings", {})}
    known = plugin["settings_known"]
    if "ledger" in settings and "ledger" not in known:
        if settings["ledger"]:
            return None
        settings.pop("ledger")
    fmt = method.get("format")
    if fmt == "relaxed" and "fallback_format" not in known:
        return None
    if fmt and "fallback_format" in known:
        settings["fallback_format"] = fmt
    if "ledger_on_failure" in known and fmt == "quoted":
        # The strict comparison: a failed quoted summary aborts, as on the plugin before the hybrid change.
        settings["ledger_on_failure"] = False
    return settings


def plan(config: dict, plugin: dict) -> tuple[list[dict], list[dict]]:
    """Return (jobs, skipped methods). Methods rotate between replicates so no method always runs first."""
    names = [spec["id"] for spec in fixtures.SCENARIOS]
    chosen = names if config.get("fixtures", "all") == "all" else list(config["fixtures"])
    methods = list(config["methods"])
    jobs, skipped = [], []
    usable = []
    for name in methods:
        settings = method_settings(name, plugin, float(config["threshold"]))
        if settings is None:
            skipped.append({"method": name, "reason": "plugin_setting_missing"})
        else:
            usable.append((name, settings))
    reps = int(config.get("replicates", 5))
    for fixture in chosen:
        for rep in range(reps):
            shift = rep % max(len(usable), 1)
            for name, settings in usable[shift:] + usable[:shift]:
                jobs.append({"id": f"{fixture}|{name}|r{rep + 1}|c1", "fixture": fixture, "method": name,
                             "rep": rep + 1, "cycles": 1, "settings": settings})
    drift = config.get("drift") or {}
    for fixture in drift.get("fixtures", []):
        for rep in range(reps):
            for name, settings in usable:
                if name in drift.get("methods", []) and METHODS[name]["engine"] != "none":
                    jobs.append({"id": f"{fixture}|{name}|r{rep + 1}|c{drift['cycles']}", "fixture": fixture,
                                 "method": name, "rep": rep + 1, "cycles": int(drift["cycles"]),
                                 "settings": settings})
    return jobs, skipped


def worker_environment(home: Path) -> dict[str, str]:
    """Runtime paths only: no credentials or profile settings of the user reach the worker."""
    names = ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC", "SYSTEMDRIVE", "LD_LIBRARY_PATH",
             "DYLD_LIBRARY_PATH")
    env = {key: value for key, value in os.environ.items() if key.upper() in names}
    env.update({key: str(home) for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME",
                                           "XDG_CACHE_HOME", "TEMP", "TMP")})
    env.update({"HERMES_HOME": str(home), "HERMES_RUNTIME_DIR": str(home / "runtime"),
                "HERMES_DISABLE_LAZY_INSTALLS": "1", "HERMES_ENABLE_PROJECT_PLUGINS": "0",
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"})
    return env


def classify(cycle: dict, method: str) -> str:
    """The summary path of one compaction: warm, fallback, ledger, model, static_fallback, no_change, aborted, or
    none (the control)."""
    engine = METHODS[method]["engine"]
    if engine == "none":
        return "none"
    if cycle.get("preflight"):
        # The engine compacted before the main request; outcome.path keeps the summary path of that compaction.
        return "preflight"
    outcome = cycle.get("outcome") or {}
    if engine == "warm_compaction":
        path = outcome.get("path")
        if cycle.get("compress_error") or path == "aborted":
            return "aborted"
        if not cycle.get("compacted"):
            return path if path in ("cancelled",) else "no_change"
        return str(path or "unknown")
    if cycle.get("compress_error"):
        return "aborted"
    if not cycle.get("compacted"):
        return "no_change"
    text = scoring.history_text(cycle.get("history") or [])
    return "static_fallback" if STATIC_FALLBACK_TEXT in text else "model"


def evidence_kept(history_rows: list, compacted: list, rows: list[int]) -> bool:
    """True when every evidence row is still in the compacted history with the same text (approximate)."""
    kept = {json.dumps([row.get("role"), row.get("content"), [
        (call.get("function") or {}).get("arguments") for call in row.get("tool_calls") or []]], ensure_ascii=False)
        for row in compacted}
    for index in rows:
        row = history_rows[index]
        key = json.dumps([row.get("role"), row.get("content"), [
            (call.get("function") or {}).get("arguments") for call in row.get("tool_calls") or []]],
            ensure_ascii=False)
        if key not in kept:
            return False
    return True


def score_cycle(cycle: dict, gold: dict, history: dict, method: str) -> dict:
    """Metadata and scores of one cycle. The answer and history text stay out of the record."""
    questions = gold["questions"]
    answers: dict[str, str] = {}
    parsed = []
    qa = cycle.get("qa") or {}
    for batch in qa.get("batches", []):
        got, ok = scoring.parse_answers(batch["text"], batch["numbers"])
        parsed.append(ok)
        answers.update({key: value for key, value in got.items() if key in {str(n) for n in batch["numbers"]}})
    compacted = cycle.get("history") or []
    text = scoring.history_text(compacted)
    marks = []
    for number, question in enumerate(questions, 1):
        answer = answers.get(str(number))
        marks.append("".join((
            "1" if answer is not None and scoring.score(question, answer) else "0",
            "1" if scoring.retained(question, text) else "0",
            "1" if evidence_kept(history["rows"], compacted, question["evidence_rows"]) else "0",
            "1" if answer is not None and not scoring.is_unknown(answer) else "0")))
    outcome = cycle.get("outcome") or {}
    keep = ("path", "reason", "fallback_reason", "elapsed_s", "prompt_tokens", "cached_tokens", "fallback_attempts",
            "fallback_end_marker", "fallback_output_tokens", "fallback_body_tokens", "compression_count", "trigger")
    calls = qa.get("calls", [])
    return {
        "cycle": cycle.get("cycle"), "path": classify(cycle, method), "compacted": bool(cycle.get("compacted")),
        "preflight": bool(cycle.get("preflight")),
        "compress_error": cycle.get("compress_error"), "outcome": {k: outcome.get(k) for k in keep if k in outcome},
        "removed": cycle.get("removed"), "rows_before": cycle.get("rows_before"), "rows_after": cycle.get("rows_after"),
        "est_tokens_before": cycle.get("est_tokens_before"), "est_tokens_after": cycle.get("est_tokens_after"),
        "main_s": cycle.get("main_s"), "compress_s": cycle.get("compress_s"),
        "main_prompt_tokens": cycle.get("main_prompt_tokens"),
        "qa_prompt_tokens": calls[0].get("prompt_tokens") if calls else None,
        "qa_s": round(sum(call.get("elapsed_s") or 0 for call in calls), 3),
        "qa_finish": [call.get("finish_reason") for call in calls], "qa_error": qa.get("error"),
        "qa_parsed": parsed,
        # One string per question: answer correct, fact retained, evidence rows kept verbatim, answer given.
        "marks": marks,
    }


def run_job(job: dict, config: dict, out_dir: Path, history_dir: Path) -> dict:
    fixture_path = history_dir / f"{job['fixture']}.history.json"
    history, gold = fixtures.load(fixture_path, GOLD_DIR / f"{job['fixture']}.gold.json")
    method = METHODS[job["method"]]
    record = {"job": job["id"], "fixture": job["fixture"], "method": job["method"], "rep": job["rep"],
              "cycles_requested": job["cycles"], "settings": job["settings"], "status": "failed", "error": None,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()
    # A CLI child process can hold a file handle for a short time after it exits (Windows): cleanup is best effort.
    with tempfile.TemporaryDirectory(prefix="wc-bench-", ignore_cleanup_errors=True) as folder:
        work = Path(folder)
        home = work / "home"
        home.mkdir()
        cli = method["engine"] in CLI_ENGINES
        if method["engine"] == "warm_compaction":
            shutil.copytree(Path(config["plugin_source"]) / "warm_compaction", home / "plugins" / "warm_compaction",
                            ignore=shutil.ignore_patterns("__pycache__"))
        spec = {
            "hermes_source": str(config["hermes_source"]), "engine": method["engine"], "settings": job["settings"],
            "fixture_path": str(fixture_path), "endpoint": config["endpoint"], "model": config["model"],
            "api_key": os.environ.get(config.get("api_key_env") or "", "") or "no-key-required",
            "context_length": int(config["context_length"]), "threshold": float(config["threshold"]),
            "cycles": job["cycles"], "qa_batch": int(config.get("qa_batch", 12)),
            "qa_max_tokens": int(config.get("qa_max_tokens", 2048)), "qa_extra_body": config.get("qa_extra_body"),
            "main_max_tokens": int(config.get("main_max_tokens", 4096)),
            "hermes_config": config.get("hermes_config") or {},
            "questions": [{"n": n, "text": q["question"]} for n, q in enumerate(gold["questions"], 1)],
        }
        if cli:
            spec.update(cli_command=list(config["cli"][method["engine"]]["command"]),
                        cli_version=config["cli"][method["engine"]]["version"],
                        anthropic_base_url=anthropic_base_url(config),
                        cli_timeout_s=float(config.get("cli_timeout_s", 1800)))
        (work / "job.json").write_text(json.dumps(spec), encoding="utf-8")
        target = work / "out.json"
        try:
            python, worker = (sys.executable, CLI_WORKER) if cli else (config["hermes_python"], WORKER)
            completed = subprocess.run(
                [python, "-I", "-B", str(worker), "--job", str(work / "job.json"), "--out",
                 str(target)], cwd=work, env=worker_environment(home), capture_output=True,
                timeout=float(config.get("job_timeout_s", 3600)))
            record["exit_code"] = completed.returncode
        except subprocess.TimeoutExpired:
            record["error"] = "job_timeout"
        out = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
        if out is None:
            record["error"] = record["error"] or "worker_no_report"
        else:
            record["status"], record["error"] = out.get("status"), out.get("error")
            record["worker_python"] = out.get("python")
            record["engine_name"] = out.get("engine_name")
            record["cycles"] = [score_cycle(cycle, gold, history, job["method"]) for cycle in out.get("cycles", [])]
            if config.get("keep_text"):
                raw = out_dir / "raw"
                raw.mkdir(exist_ok=True)
                name = job["id"].replace("|", "__")
                (raw / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    return record


def anthropic_base_url(config: dict) -> str:
    """The Anthropic Messages root of the endpoint (Claude Code adds /v1/messages)."""
    url = str(config.get("anthropic_base_url") or config["endpoint"]).rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def cli_state(config: dict, methods: list[str]) -> dict[str, Any]:
    """The version of each CLI that the run uses. Stop when it is not the pinned version."""
    from .cli_worker import cli_environment
    state = {}
    for name in methods:
        engine = METHODS[name]["engine"]
        if engine not in CLI_ENGINES:
            continue
        spec = (config.get("cli") or {}).get(engine) or {}
        if not spec.get("command") or not spec.get("version"):
            raise SystemExit(f"config cli.{engine} needs 'command' and a pinned 'version'")
        with tempfile.TemporaryDirectory(prefix="wc-cli-version-") as folder:
            home = Path(folder)
            env = cli_environment(home, {"CODEX_HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude")})
            done = subprocess.run([*spec["command"], "--version"], cwd=home, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, encoding="utf-8", errors="replace", timeout=120)
        found = (done.stdout or "").strip()
        if spec["version"] not in found.split():
            raise SystemExit(f"{engine}: found version {found!r}, pinned {spec['version']!r}; stop before any job")
        state[engine] = {"version": spec["version"], "version_output": found}
    return state


def command_run(args) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(config.get("history_dir") or ROOT / ".work" / "bench-fixtures")
    fixtures.write_all(history_dir, history_dir / "gold-check")
    for spec in fixtures.SCENARIOS:
        fixtures.load(history_dir / f"{spec['id']}.history.json", GOLD_DIR / f"{spec['id']}.gold.json")
    hermes = hermes_state(Path(config["hermes_source"]))
    if not hermes["clean"]:
        # Results must come from a clean, known Hermes source.
        raise SystemExit("hermes_source is not a clean Git checkout; stop before any job runs")
    plugin = plugin_state(Path(config["plugin_source"]))
    # A plugin folder made with git archive has no Git data: the config can name its commit.
    plugin["commit"] = plugin["commit"] or config.get("plugin_commit")
    jobs, skipped = plan(config, plugin)
    cli = cli_state(config, list(config["methods"]))
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        meta = {"schema": "warm-compaction.compaction-benchmark/v1", "synthetic": True,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": {key: value for key, value in config.items()
                           if key not in ("hermes_python", "hermes_source", "plugin_source", "history_dir", "cli")},
                "plugin": plugin, "hermes": hermes,
                "harness_sha256": {path.name: sha256_file(path) for path in sorted(HERE.glob("*.py"))},
                "gold_sha256": {path.name: sha256_file(path) for path in sorted(GOLD_DIR.glob("*.json"))},
                "host": {"os": platform.system(), "machine": platform.machine(),
                         "python": sys.version.split()[0]},
                "skipped_methods": skipped, "planned_jobs": len(jobs)}
        if cli:
            meta["cli"] = cli
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    runs_path = out_dir / "runs.jsonl"
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("status") == "ok":
                done.add(item["job"])
    todo = [job for job in jobs if job["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(jobs)} jobs planned, {len(done)} done, {len(todo)} to run; skipped methods: "
          f"{[item['method'] for item in skipped]}", flush=True)
    if args.dry_run:
        for job in todo:
            print(job["id"], json.dumps(job["settings"]))
        return 0
    for number, job in enumerate(todo, 1):
        record = run_job(job, config, out_dir, history_dir)
        with runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        paths = [cycle["path"] for cycle in record.get("cycles", [])]
        accuracy = [round(sum(m[0] == "1" for m in c["marks"]) / max(len(c["marks"]), 1), 2)
                    for c in record.get("cycles", [])]
        print(f"[{number}/{len(todo)}] {job['id']} status={record['status']} error={record['error']} "
              f"paths={paths} accuracy={accuracy} elapsed={record['elapsed_s']}s", flush=True)
        if record["error"] == "login_required":
            # The benchmark never signs in: stop and report.
            raise SystemExit(f"{job['method']} asked for a login; stop the run")
    return 0


# Report.

def load_runs(out_dir: Path) -> list[dict]:
    """The latest record of each job."""
    latest: dict[str, dict] = {}
    for line in (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            latest[item["job"]] = item
    return list(latest.values())


def rate(successes: int, total: int) -> dict[str, Any]:
    low, high = scoring.wilson(successes, total)
    return {"k": successes, "n": total, "rate": successes / total if total else None, "low": low, "high": high}


def question_table() -> dict[str, list[dict]]:
    return {spec["id"]: json.loads((GOLD_DIR / f"{spec['id']}.gold.json").read_text(encoding="utf-8"))["questions"]
            for spec in fixtures.SCENARIOS}


def aggregate(runs: list[dict], meta: dict) -> dict[str, Any]:
    gold = question_table()
    main = [run for run in runs if run["cycles_requested"] == 1]
    drift = [run for run in runs if run["cycles_requested"] > 1]
    methods = [name for name in METHODS if any(run["method"] == name for run in runs)]
    result: dict[str, Any] = {"methods": {}, "paired": [], "drift": {}}
    pairs: dict[tuple, dict[str, str]] = defaultdict(dict)
    for name in methods:
        items = [run for run in main if run["method"] == name]
        every = [(run, run["cycles"][0]) for run in items if run.get("cycles")]
        # Accuracy uses the runs that compacted: an aborted run keeps the full history and asks no questions.
        cycles = [(run, cycle) for run, cycle in every if cycle["compacted"] or METHODS[name]["engine"] == "none"]
        paths = defaultdict(int)
        reasons = defaultdict(int)
        for _run, cycle in every:
            paths[cycle["path"]] += 1
            outcome = cycle.get("outcome") or {}
            warm_refusal = outcome.get("reason") not in (None, "accepted", "disabled")
            if METHODS[name]["engine"] == "warm_compaction" and (cycle["path"] not in ACCEPTED_PATHS
                                                                  or warm_refusal):
                # The warm refusal code (or "-" when the warm path was off), then the fallback code.
                warm_code = outcome.get("reason") if warm_refusal else "-"
                reasons[f"{warm_code}/{outcome.get('fallback_reason') or '-'}"] += 1
            elif cycle["path"] not in ACCEPTED_PATHS + ("none",):
                reasons[cycle["path"]] += 1
        failed_jobs = [run["job"] for run in items if run["status"] != "ok"]
        accepted = sum(1 for _run, cycle in every if cycle["path"] in ACCEPTED_PATHS)
        entry: dict[str, Any] = {
            "label": METHODS[name]["label"], "runs": len(items), "job_failures": len(failed_jobs),
            "compacted_runs": len(cycles), "paths": dict(paths), "failure_reasons": dict(reasons),
            "success": rate(accepted, len(every)) if METHODS[name]["engine"] != "none" else None,
            "categories": {}, "categories_compacted_evidence": {}, "retention": {},
            "per_rep_accuracy": {}, "tokens": {}, "latency_s": {}, "per_run": []}
        by_category = defaultdict(lambda: [0, 0])
        by_category_compacted = defaultdict(lambda: [0, 0])
        by_retention = defaultdict(lambda: [0, 0])
        per_rep = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for run, cycle in cycles:
            questions = gold[run["fixture"]]
            for question, mark in zip(questions, cycle["marks"]):
                category = question["category"]
                correct, kept, verbatim = mark[0] == "1", mark[1] == "1", mark[2] == "1"
                by_category[category][0] += correct
                by_category[category][1] += 1
                by_retention[category][0] += kept
                by_retention[category][1] += 1
                if not verbatim:
                    by_category_compacted[category][0] += correct
                    by_category_compacted[category][1] += 1
                per_rep[run["rep"]][category][0] += correct
                per_rep[run["rep"]][category][1] += 1
                pairs[(run["fixture"], run["rep"], question["id"])][name] = mark[0]
            entry["per_run"].append({
                "fixture": run["fixture"], "rep": run["rep"], "path": cycle["path"],
                "est_tokens_before": cycle["est_tokens_before"], "est_tokens_after": cycle["est_tokens_after"],
                "qa_prompt_tokens": cycle["qa_prompt_tokens"], "main_prompt_tokens": cycle["main_prompt_tokens"],
                "compress_s": cycle["compress_s"], "main_s": cycle["main_s"], "qa_s": cycle["qa_s"],
                "accuracy": round(sum(m[0] == "1" for m in cycle["marks"]) / len(cycle["marks"]), 4),
                "marks": "".join(m[0] for m in cycle["marks"])})
        for category in CATEGORIES:
            entry["categories"][category] = rate(*by_category[category])
            entry["categories_compacted_evidence"][category] = rate(*by_category_compacted[category])
            entry["retention"][category] = rate(*by_retention[category])
            entry["per_rep_accuracy"][category] = scoring.mean_interval(
                [per_rep[rep][category][0] / per_rep[rep][category][1] for rep in sorted(per_rep)
                 if per_rep[rep][category][1]])
        scored = [c for c in CATEGORIES if c != "recent"]
        entry["overall"] = rate(sum(by_category[c][0] for c in scored), sum(by_category[c][1] for c in scored))
        # The same, with each run that did not compact scored as all wrong.
        missing = sum(len([q for q in gold[run["fixture"]] if q["category"] != "recent"])
                      for run, cycle in every if (run, cycle) not in cycles)
        entry["overall_failures_wrong"] = rate(entry["overall"]["k"], entry["overall"]["n"] + missing)
        values = [cycle for _run, cycle in cycles]
        entry["tokens"] = {
            "est_before": scoring.mean_interval([c["est_tokens_before"] for c in values]),
            "est_after": scoring.mean_interval([c["est_tokens_after"] for c in values]),
            "ratio_after_before": scoring.mean_interval([c["est_tokens_after"] / c["est_tokens_before"]
                                                         for c in values if c.get("est_tokens_before")]),
            "qa_prompt_tokens": scoring.mean_interval([c["qa_prompt_tokens"] for c in values
                                                       if c.get("qa_prompt_tokens")])}
        # Time of all runs, also the runs that did not compact: a failed attempt costs time too.
        attempts = [cycle for _run, cycle in every]
        entry["latency_s"] = {
            "compress": scoring.mean_interval([c["compress_s"] for c in attempts if c.get("compress_s") is not None]),
            "compress_compacted": scoring.mean_interval([c["compress_s"] for c in values
                                                         if c.get("compress_s") is not None]),
            "main_request": scoring.mean_interval([c["main_s"] for c in attempts if c.get("main_s") is not None]),
            "questions": scoring.mean_interval([c["qa_s"] for c in values if c.get("qa_s") is not None])}
        entry["per_run_failed"] = [{"fixture": run["fixture"], "rep": run["rep"], "path": cycle["path"],
                                    "compress_s": cycle["compress_s"], "main_s": cycle["main_s"]}
                                   for run, cycle in every if (run, cycle) not in cycles]
        result["methods"][name] = entry
    for first, second in PAIRED:
        if first not in methods or second not in methods:
            continue
        for category in CATEGORIES:
            a_only = b_only = both = neither = 0
            for (fixture, _rep, qid), marks in pairs.items():
                if first not in marks or second not in marks or not qid_in(gold[fixture], qid, category):
                    continue
                a, b = marks[first] == "1", marks[second] == "1"
                a_only += a and not b
                b_only += b and not a
                both += a and b
                neither += not a and not b
            n = a_only + b_only + both + neither
            if n:
                result["paired"].append({
                    "first": first, "second": second, "category": category, "pairs": n, "first_only": a_only,
                    "second_only": b_only, "difference": (a_only - b_only) / n,
                    "p_mcnemar": scoring.mcnemar(a_only, b_only)})
    for name in sorted({run["method"] for run in drift}, key=list(METHODS).index):
        items = [run for run in drift if run["method"] == name and run.get("cycles")]
        per_cycle = []
        for index in range(max(len(run["cycles"]) for run in items) if items else 0):
            hits = total = 0
            paths = defaultdict(int)
            for run in items:
                if index >= len(run["cycles"]):
                    continue
                cycle = run["cycles"][index]
                paths[cycle["path"]] += 1
                for question, mark in zip(gold[run["fixture"]], cycle["marks"]):
                    if question["category"] != "recent":
                        hits += mark[0] == "1"
                        total += 1
            per_cycle.append({"cycle": index + 1, "accuracy": rate(hits, total), "paths": dict(paths),
                              "est_tokens_after": scoring.mean_interval(
                                  [run["cycles"][index]["est_tokens_after"] for run in items
                                   if index < len(run["cycles"])])})
        result["drift"][name] = {"runs": len(items), "fixtures": sorted({run["fixture"] for run in items}),
                                 "cycles": per_cycle}
    result["decision"] = decide(result)
    return result


def qid_in(questions: list[dict], qid: str, category: str) -> bool:
    return any(q["id"] == qid and q["category"] == category for q in questions)


# The pre-registered contrasts (PREREGISTRATION.md).
PAIRED = (("warm_ledger", "warm_no_ledger"), ("fallback_quoted_ledger", "fallback_quoted_no_ledger"),
          ("fallback_relaxed_ledger", "fallback_relaxed_no_ledger"), ("warm_ledger", "stock_hermes"),
          ("fallback_relaxed_ledger", "fallback_quoted_ledger"), ("warm_ledger", "full_context"))
LEDGER_CATEGORIES = ("command_outcome", "file_changed")
GUARD_CATEGORIES = ("goal", "decision", "next_step", "task_state")


def paired(result: dict, first: str, second: str, category: str) -> dict | None:
    return next((item for item in result["paired"] if (item["first"], item["second"], item["category"]) ==
                 (first, second, category)), None)


def decide(result: dict) -> dict[str, Any]:
    """Apply the pre-registered decision rule (PREREGISTRATION.md). Return each check and its outcome."""
    checks: dict[str, Any] = {}
    methods = result["methods"]
    if "warm_ledger" in methods and "warm_no_ledger" in methods:
        gains = {c: paired(result, "warm_ledger", "warm_no_ledger", c) for c in LEDGER_CATEGORIES}
        checks["H1_ledger_helps"] = all(
            item is not None and item["difference"] >= 0.10 and (item["p_mcnemar"] or 1) < 0.05
            for item in gains.values())
        losses = {c: paired(result, "warm_ledger", "warm_no_ledger", c) for c in GUARD_CATEGORIES}
        checks["H2_no_regression"] = all(
            item is not None and not (item["difference"] <= -0.05) and not (
                item["difference"] < 0 and (item["p_mcnemar"] or 1) < 0.05) for item in losses.values())
    if "fallback_relaxed_ledger" in methods and "fallback_relaxed_no_ledger" in methods:
        gains = [paired(result, "fallback_relaxed_ledger", "fallback_relaxed_no_ledger", c) for c in LEDGER_CATEGORIES]
        checks["H1b_ledger_helps_fallback"] = all(
            item is not None and item["difference"] >= 0.10 and (item["p_mcnemar"] or 1) < 0.05 for item in gains)
    if "fallback_relaxed_ledger" in methods and "fallback_quoted_ledger" in methods:
        relaxed = methods["fallback_relaxed_ledger"]["success"]
        quoted = methods["fallback_quoted_ledger"]["success"]
        checks["R1_relaxed_succeeds_more_often"] = (relaxed["low"] is not None and quoted["high"] is not None
                                                    and relaxed["low"] > quoted["high"])
    if "warm_ledger" in methods and "stock_hermes" in methods:
        warm, stock = methods["warm_ledger"]["categories"], methods["stock_hermes"]["categories"]
        checks["H3a_matches_stock_on_ledger_categories"] = all(
            warm[c]["rate"] is not None and stock[c]["rate"] is not None and warm[c]["high"] >= stock[c]["rate"]
            for c in LEDGER_CATEGORIES)
        checks["H3b_no_loss_on_model_categories"] = all(
            warm[c]["rate"] is not None and stock[c]["rate"] is not None and warm[c]["high"] >= stock[c]["rate"]
            for c in ("goal", "decision", "next_step"))
        warm_after = methods["warm_ledger"]["tokens"]["est_after"]["mean"]
        stock_after = methods["stock_hermes"]["tokens"]["est_after"]["mean"]
        # None: no run of one method compacted, so the check has no data. It does not pass.
        checks["H4_smaller_than_stock"] = (None if warm_after is None or stock_after is None
                                           else warm_after <= stock_after)
    if "warm_ledger" in methods and methods["warm_ledger"]["success"]:
        checks["S1_warm_success_at_least_90"] = (methods["warm_ledger"]["success"]["rate"] or 0) >= 0.90
    if "warm_ledger" in result["drift"]:
        cycles = result["drift"]["warm_ledger"]["cycles"]
        if len(cycles) >= 3 and cycles[0]["accuracy"]["rate"] is not None:
            checks["D1_drift_within_10_points"] = (cycles[2]["accuracy"]["rate"] or 0) >= (
                cycles[0]["accuracy"]["rate"] - 0.10)
    verdict = []
    if "R1_relaxed_succeeds_more_often" in checks:
        verdict.append("The relaxed fallback format succeeds more often than the quoted format."
                       if checks["R1_relaxed_succeeds_more_often"] else
                       "The success rates of the two fallback formats are not separated by the rule.")
    if checks.get("H1_ledger_helps") and checks.get("H2_no_regression"):
        verdict.append("Keep the ledger on by default.")
    elif "H1_ledger_helps" in checks:
        verdict.append("The ledger result does not meet the rule for a default.")
    if all(checks.get(k) for k in ("H3a_matches_stock_on_ledger_categories", "H3b_no_loss_on_model_categories",
                                   "H4_smaller_than_stock")) and "H4_smaller_than_stock" in checks:
        verdict.append("Warm path + ledger is competitive with the built-in compressor on this workload.")
    elif "H4_smaller_than_stock" in checks:
        verdict.append("Warm path + ledger does not meet the rule against the built-in compressor.")
    return {"checks": checks, "verdict": verdict}


def pct(item: dict | None) -> str:
    if not item or item.get("rate") is None:
        return "n/a"
    return f"{100 * item['rate']:.0f}% [{100 * item['low']:.0f}, {100 * item['high']:.0f}] ({item['k']}/{item['n']})"


def num(item: dict | None, digits: int = 1) -> str:
    if not item or item.get("mean") is None:
        return "n/a"
    if item.get("low") is None:
        return f"{item['mean']:.{digits}f}"
    return f"{item['mean']:.{digits}f} [{item['low']:.{digits}f}, {item['high']:.{digits}f}]"


CLI_LIMITS = [
    "The Codex CLI and Claude Code methods write the synthetic history into the session file format of the pinned "
    "CLI version. These formats are internal to those versions and can change.",
    "Both CLIs are tuned for their own models. These rows measure their compaction method on the same small local "
    "model, not their behavior with GPT or Claude models.",
    "Their system prompts, tool schemas, and request formats differ from Hermes (Codex uses the Responses API with "
    "its own tools; Claude Code uses the Anthropic Messages API with no tools, as the Hermes jobs have no tools).",
    "Tool calls keep their Hermes tool names and argument text in the CLI session files, so the facts are the same "
    "in all methods.",
    "Their compaction time is the duration that each CLI records (Codex: from turn start to the compacted item; "
    "Claude Code: compactMetadata.durationMs), not the Hermes compress call time.",
    "The Codex CLI compaction uses the app server request thread/compact/start, which is marked experimental in the "
    "pinned version; the Claude Code compaction uses /compact in print mode.",
    "The CLI methods were added after the pre-registration; the pre-registered decision rule does not use them.",
]


def markdown(result: dict, meta: dict) -> str:
    methods = result["methods"]
    names = list(methods)
    lines = [
        f"# Compaction benchmark: {meta['config'].get('model')}", "",
        f"Date: {meta.get('started_utc', '')[:10]}. All sessions are synthetic. Scores use fixed rules, not a model "
        "judge. Intervals are 95% Wilson intervals for rates and 95% t intervals for means. A rate pools all "
        "questions of all fixtures and replicates, so its interval does not include the variation between "
        "replicates; see the per-replicate means in the JSON file.", "",
        f"- Plugin commit: `{(meta['plugin'].get('commit') or 'unknown')[:12]}`. Hermes commit: "
        f"`{(meta['hermes'].get('commit') or 'unknown')[:12]}` (clean: {meta['hermes'].get('clean')}).",
        f"- Model: `{meta['config'].get('model')}`; context window setting {meta['config'].get('context_length')}; "
        f"threshold {meta['config'].get('threshold')}; {meta['config'].get('replicates')} replicates per cell.",
        f"- Skipped methods: {', '.join(item['method'] for item in meta.get('skipped_methods', [])) or 'none'}.",
        *(["- Other CLIs: " + ", ".join(f"{name} {item['version']}" for name, item in meta["cli"].items()) + "."]
          if meta.get("cli") else []), "",
        "## Compaction success", "",
        "| Method | Runs | Accepted summary | Paths | Failure reasons |", "| --- | --- | --- | --- | --- |"]
    for name in names:
        entry = methods[name]
        lines.append(f"| {entry['label']} | {entry['runs']} | {pct(entry['success'])} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(entry['paths'].items()))} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(entry['failure_reasons'].items())) or '-'} |")
    if any(entry["paths"].get("preflight") for entry in methods.values()):
        lines += ["", "Path `preflight`: the engine compacted during the main turn, before the main request (for "
                  "example, the plugin preflight check with no captured request). The questions use that history, "
                  "and the main request ran on the compacted history, unlike the other runs."]
    lines += ["", "## Accuracy by category (all questions)", "",
              "| Category | " + " | ".join(methods[n]["label"] for n in names) + " |",
              "| --- | " + " | ".join("---" for _ in names) + " |"]
    for category in CATEGORIES:
        lines.append(f"| {category} | " + " | ".join(pct(methods[n]["categories"][category]) for n in names) + " |")
    lines.append("| overall (without recent) | " + " | ".join(pct(methods[n]["overall"]) for n in names) + " |")
    lines.append("| est. tokens after (mean) | " + " | ".join(num(methods[n]["tokens"]["est_after"], 0)
                                                             for n in names) + " |")
    lines += ["", "## Accuracy when the evidence rows were compacted", "",
              "Only questions whose evidence rows are not in the compacted history with the same text.", "",
              "| Category | " + " | ".join(methods[n]["label"] for n in names) + " |",
              "| --- | " + " | ".join("---" for _ in names) + " |"]
    for category in CATEGORIES:
        lines.append(f"| {category} | " + " | ".join(pct(methods[n]["categories_compacted_evidence"][category])
                                                      for n in names) + " |")
    lines += ["", "## Fact retained in the compacted text (model-free, approximate)", "",
              "| Category | " + " | ".join(methods[n]["label"] for n in names) + " |",
              "| --- | " + " | ".join("---" for _ in names) + " |"]
    for category in CATEGORIES:
        lines.append(f"| {category} | " + " | ".join(pct(methods[n]["retention"][category]) for n in names) + " |")
    lines += ["", "## Size and time", "",
              "Estimated tokens are JSON characters / 4 of the history rows (the same estimate for all methods). "
              "The question prompt tokens are the server count of the first question request, with the system row and "
              "about 400 tokens of questions. Compaction time is the wall time of the Hermes manual compress call.", "",
              "| Method | Est. tokens before | Est. tokens after | After / before | Question prompt tokens | "
              "Compaction s (all runs) | Main request s |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        entry = methods[name]
        lines.append(f"| {entry['label']} | {num(entry['tokens']['est_before'], 0)} | "
                     f"{num(entry['tokens']['est_after'], 0)} | {num(entry['tokens']['ratio_after_before'], 3)} | "
                     f"{num(entry['tokens']['qa_prompt_tokens'], 0)} | {num(entry['latency_s']['compress'])} | "
                     f"{num(entry['latency_s']['main_request'])} |")
    lines += ["", "Per-run times and token counts are in the JSON file (`per_run`).", "",
              "## Paired comparisons (McNemar, same fixture, replicate, and question)", "",
              "| First | Second | Category | Pairs | Difference (first - second) | p |",
              "| --- | --- | --- | --- | --- | --- |"]
    for item in result["paired"]:
        p = "n/a" if item["p_mcnemar"] is None else f"{item['p_mcnemar']:.3g}"
        lines.append(f"| {item['first']} | {item['second']} | {item['category']} | {item['pairs']} | "
                     f"{100 * item['difference']:+.1f} points | {p} |")
    if result["drift"]:
        lines += ["", "## Repeated compaction (drift)", "",
                  "The same session, compacted again after each new chunk of work. Accuracy is on the questions about "
                  "the first window (without the recent facts).", "",
                  "| Method | Runs | Cycle | Accuracy | Paths | Est. tokens after |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, entry in result["drift"].items():
            for cycle in entry["cycles"]:
                lines.append(f"| {METHODS[name]['label']} | {entry['runs']} | {cycle['cycle']} | "
                             f"{pct(cycle['accuracy'])} | "
                             f"{', '.join(f'{k}: {v}' for k, v in sorted(cycle['paths'].items()))} | "
                             f"{num(cycle['est_tokens_after'], 0)} |")
    if any(name in methods for name in ("codex_cli", "claude_code_cli")):
        lines += ["", "## Limits of the CLI rows", "", *[f"- {item}" for item in CLI_LIMITS]]
    lines += ["", "## Pre-registered decision rule", ""]
    for key, value in result["decision"]["checks"].items():
        state = "unavailable (no data; counts as fail)" if value is None else "pass" if value else "fail"
        lines.append(f"- {key}: {state}")
    for line in result["decision"]["verdict"]:
        lines.append(f"- Verdict: {line}")
    return "\n".join(lines) + "\n"


def command_report(args) -> int:
    out_dir = Path(args.out)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    runs = load_runs(out_dir)
    for extra in args.also or []:
        # A second run folder (for example the CLI methods): its jobs join the report, and its metadata is kept.
        extra_dir = Path(extra)
        extra_meta = json.loads((extra_dir / "meta.json").read_text(encoding="utf-8"))
        known = {run["job"] for run in runs}
        runs += [run for run in load_runs(extra_dir) if run["job"] not in known]
        meta.setdefault("merged_runs", []).append({key: extra_meta.get(key) for key in (
            "started_utc", "plugin", "hermes", "harness_sha256", "cli", "planned_jobs")})
        meta.setdefault("cli", {}).update(extra_meta.get("cli") or {})
    result = aggregate(runs, meta)
    data = {"schema": meta["schema"], "synthetic": True,
            "privacy": "Metadata only: counts, codes, booleans, times, and hashes. No message, answer, or summary "
                       "text.", "meta": meta, "results": result,
            "limits": ["Synthetic sessions from one generator; six task families.",
                       "One model per run; a small local model is used for compaction and for the answers.",
                       "Rule-based scoring can miss a correct paraphrase; the full-history control shows the ceiling.",
                       "Cache reuse is unknown when the server does not report cached tokens."]
                      + (CLI_LIMITS if any(name in result["methods"] for name in ("codex_cli", "claude_code_cli"))
                         else [])}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(result, meta), encoding="utf-8")
    print(json.dumps(result["decision"], indent=1))
    return 0


def command_fixtures(args) -> int:
    for item in fixtures.write_all(Path(args.work), GOLD_DIR):
        print(json.dumps(item))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("fixtures", help="build the synthetic histories and the gold files")
    make.add_argument("--work", default=str(ROOT / ".work" / "bench-fixtures"))
    make.set_defaults(func=command_fixtures)
    run = sub.add_parser("run", help="run the planned jobs; a second call continues an interrupted run")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=command_run)
    report = sub.add_parser("report", help="aggregate runs.jsonl into the JSON results and the Markdown report")
    report.add_argument("--out", required=True)
    report.add_argument("--json", required=True)
    report.add_argument("--md")
    report.add_argument("--also", action="append", help="another run folder to include (repeat for more)")
    report.set_defaults(func=command_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
