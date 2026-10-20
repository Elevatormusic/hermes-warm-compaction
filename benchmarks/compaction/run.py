"""Compaction benchmark for the warm_compaction Hermes plugin: build fixtures, run jobs, and write the report.

Commands:
    python -m benchmarks.compaction.run fixtures [--work .work/bench-fixtures]
    python -m benchmarks.compaction.run run --config <config.json> --out <run folder> [--limit N] [--dry-run]
    python -m benchmarks.compaction.run report --out <run folder> --json <results.json> --md <report.md> [--rescore]

Each job runs worker.py in a fresh process with the Python of a Hermes virtual environment, a clean Hermes
checkout, and a new temporary Hermes home. The run folder (keep it in .work/, which Git ignores) gets runs.jsonl
with metadata and scores only. With "keep_text": true in the config, raw answers and compacted rows of each job go
to <run folder>/raw/ for local checks; never publish that folder. All sessions are synthetic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import fixtures, scoring
from . import worker as job_worker

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GOLD_DIR = HERE / "fixtures"
WORKER = HERE / "worker.py"
CLI_WORKER = HERE / "cli_worker.py"
CATEGORIES = ("goal", "user_instruction", "command_outcome", "file_changed", "identifier", "decision", "task_state",
              "next_step", "recent")
ACCEPTED_PATHS = ("warm", "fallback", "model")
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


def git_output(folder: Path, *args: str) -> str | None:
    """The output of a Git command in folder, or None when Git fails (no checkout, no Git)."""
    try:
        done = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    return done.stdout if done.returncode == 0 else None


def checkout_root(folder: Path) -> bool:
    """True when folder is the top folder of its Git checkout. A copied folder inside another checkout is not:
    Git then reads the enclosing checkout, whose commit and state do not label the folder."""
    top = git_output(folder, "rev-parse", "--show-toplevel")
    return top is not None and bool(top.strip()) and os.path.normcase(os.path.realpath(top.strip())) == \
        os.path.normcase(os.path.realpath(folder))


def plugin_state(plugin_root: Path) -> dict[str, Any]:
    """Hashes of every file that a job copies and loads (the warm_compaction folder, plugin.yaml included) and the
    settings that its engine knows. The commit is known only when it labels these exact files: plugin_root is the
    top folder of its checkout, Git tracks every hashed file, and the folder has no changed or untracked file."""
    package = plugin_root / "warm_compaction"
    files = {path.relative_to(package).as_posix(): sha256_file(path) for path in sorted(package.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts}
    engine = (package / "engine.py").read_text(encoding="utf-8")
    known = [key for key in ("warm", "ledger", "fallback_format", "ledger_on_failure", "timeout_s")
             if f'"{key}"' in engine.split("THRESHOLD_RANGE", 1)[0]]
    commit = None
    if checkout_root(plugin_root):
        tracked = set((git_output(plugin_root, "ls-files", "-z", "--", "warm_compaction") or "").split("\0"))
        status = git_output(plugin_root, "status", "--porcelain", "--untracked-files=all", "--", "warm_compaction")
        if status is not None and not status.strip() and {f"warm_compaction/{name}" for name in files} <= tracked:
            commit = (git_output(plugin_root, "rev-parse", "HEAD") or "").strip() or None
    return {"files_sha256": files, "settings_known": known, "commit": commit}


def hermes_state(source: Path) -> dict[str, Any]:
    """The commit and state of the Hermes checkout. A folder that is not the top folder of a Git checkout (no
    checkout, or a copy inside another checkout) has an unknown commit and is not clean."""
    if not checkout_root(source):
        return {"commit": None, "clean": False}
    status = git_output(source, "status", "--porcelain")
    return {"commit": (git_output(source, "rev-parse", "HEAD") or "").strip() or None,
            "clean": status is not None and not status.strip()}


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
    drift = config.get("drift") or {}
    # Only known names: the plan values go into meta.json and the report.
    unknown = (set(chosen) | set(drift.get("fixtures", []))) - set(names) or (
        set(methods) | set(drift.get("methods", []))) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown fixture or method names in the configuration: {len(unknown)}")
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
    """The summary path of one compaction: warm, fallback, ledger, model, static_fallback, no_change, aborted,
    main_failed, or none (the control). A preflight compaction (during the main turn) gets the path of its summary,
    with the same checks as a manual compaction; the cycle keeps "preflight": true."""
    engine = METHODS[method]["engine"]
    if cycle.get("main_failed"):
        return "main_failed"
    if engine == "none":
        return "none"
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


def stored_path(cycle: dict, method: str) -> str:
    """The summary path of a stored cycle. Records made before this rule have the path "preflight" for a preflight
    compaction: the plugin keeps its summary path in outcome.path; other engines need --rescore with the raw
    output for the static fallback check."""
    if cycle.get("path") != "preflight":
        return cycle["path"]
    if METHODS[method]["engine"] == "warm_compaction":
        return str((cycle.get("outcome") or {}).get("path") or "unknown")
    return "preflight_unchecked"


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
    # A cycle that compacted (or a control cycle) must have an answer to each question. qa_missing counts the
    # questions without one: no batch, a reply that did not parse, or no answer or an empty answer in the reply.
    asked = {n for batch in qa.get("batches", []) for n in batch["numbers"]}
    missing = sum(len(scoring.missing_answers(batch["text"], batch["numbers"])) for batch in qa.get("batches", []))
    missing += sum(1 for n in range(1, len(questions) + 1) if n not in asked)
    scored = bool(cycle.get("compacted")) or METHODS[method]["engine"] == "none"
    return {
        "cycle": cycle.get("cycle"), "path": classify(cycle, method), "compacted": bool(cycle.get("compacted")),
        "preflight": bool(cycle.get("preflight")), "main_failed": bool(cycle.get("main_failed")),
        "compress_error": cycle.get("compress_error"), "outcome": {k: outcome.get(k) for k in keep if k in outcome},
        "removed": cycle.get("removed"), "rows_before": cycle.get("rows_before"), "rows_after": cycle.get("rows_after"),
        "est_tokens_before": cycle.get("est_tokens_before"), "est_tokens_after": cycle.get("est_tokens_after"),
        "rows_before_main": cycle.get("rows_before_main"),
        "est_tokens_before_main": cycle.get("est_tokens_before_main"),
        # "compaction_input": est_tokens_before is the input of the compaction (after the main turn). Records made
        # before this field measured it before the main turn.
        "size_basis": "compaction_input" if "est_tokens_before_main" in cycle else "before_main_turn",
        "main_s": cycle.get("main_s"), "compress_s": cycle.get("compress_s"),
        "main_prompt_tokens": cycle.get("main_prompt_tokens"),
        "qa_prompt_tokens": calls[0].get("prompt_tokens") if calls else None,
        "qa_s": round(sum(call.get("elapsed_s") or 0 for call in calls), 3),
        "qa_finish": [call.get("finish_reason") for call in calls], "qa_error": qa.get("error"),
        "qa_parsed": parsed, "qa_missing": missing if scored else None,
        # One string per question: answer correct, fact retained, evidence rows kept verbatim, answer given.
        "marks": marks,
    }


def run_job(job: dict, config: dict, out_dir: Path, history_dir: Path) -> dict:
    fixture_path = history_dir / f"{job['fixture']}.history.json"
    history, gold = fixtures.load(fixture_path, GOLD_DIR / f"{job['fixture']}.gold.json")
    method = METHODS[job["method"]]
    record = {"job": job["id"], "fixture": job["fixture"], "method": job["method"], "rep": job["rep"],
              "cycles_requested": job["cycles"], "settings": job["settings"], "status": "failed", "error": None,
              "harness": harness_id(harness_hashes()),
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
            "cycles": job["cycles"], "qa_batches": gold["qa_batches"],
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
        except subprocess.TimeoutExpired as error:
            record["error"] = "job_timeout"
            completed = error
        out = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
        if out is None:
            record["error"] = record["error"] or "worker_no_report"
            # The end of the worker error output, for local diagnosis only (the run folder stays in .work/).
            stderr = completed.stderr or b""
            folder = out_dir / "worker-stderr"
            folder.mkdir(exist_ok=True)
            name = job["id"].replace("|", "__") + f"-{int(time.time())}.txt"
            (folder / name).write_bytes(stderr[-16000:] if isinstance(stderr, bytes) else stderr[-16000:].encode())
            record["stderr_file"] = f"worker-stderr/{name}"
        else:
            record["status"], record["error"] = out.get("status"), out.get("error")
            record["worker_python"] = out.get("python")
            record["engine_name"] = out.get("engine_name")
            record["cycles"] = [score_cycle(cycle, gold, history, job["method"]) for cycle in out.get("cycles", [])]
            fail_incomplete(record)
            if config.get("keep_text"):
                raw = out_dir / "raw"
                raw.mkdir(exist_ok=True)
                name = job["id"].replace("|", "__")
                data = json.dumps(out, ensure_ascii=False).encode("utf-8")
                (raw / f"{name}.json").write_bytes(data)
                # The record names its raw file by hash: --rescore uses only this file for the record.
                record["raw_sha256"] = hashlib.sha256(data).hexdigest()
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    return record


def fail_incomplete(record: dict) -> bool:
    """A job with a scored cycle that has no full answer fails: a missing answer is not scored as wrong."""
    if record.get("status") == "ok" and any(cycle.get("qa_missing") for cycle in record.get("cycles") or []):
        record["status"], record["error"] = "failed", "qa_incomplete"
        return True
    return False


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


PRIVATE_CONFIG_KEYS = ("hermes_python", "hermes_source", "plugin_source", "history_dir", "cli")
# Keys that select which jobs run or only guard the run. They do not change the result of a job, so a resume can
# change them.
PLAN_KEYS = ("methods", "fixtures", "replicates", "drift", "server_check")


# Configuration keys that are safe to publish as they are: names, numbers, flags, and the job plan.
PUBLIC_CONFIG_KEYS = ("model", "api_key_env", "context_length", "threshold", "replicates", "fixtures", "methods",
                      "drift", "qa_batch", "qa_max_tokens", "main_max_tokens", "job_timeout_s", "cli_timeout_s",
                      "keep_text", "server_check", "plugin_commit")
URL_CONFIG_KEYS = ("endpoint", "anthropic_base_url")
DRIFT_KEYS = ("fixtures", "methods", "cycles")
# Keys that a report can join with other values: the job plan and local output options. A run with CLI methods
# compares cli_timeout_s only with another run that has CLI methods.
MERGE_EXEMPT_KEYS = PLAN_KEYS + ("keep_text", "plugin_commit", "api_key_env")


def shape(value: Any, prefix: str = "") -> dict[str, str]:
    """The key paths and value types of a nested value, without the values."""
    if isinstance(value, dict):
        found: dict[str, str] = {}
        for key, item in value.items():
            found.update(shape(item, f"{prefix}.{key}" if prefix else str(key)))
        return found or {prefix: "dict"}
    return {prefix: type(value).__name__}


LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def public_url(url: Any) -> Any:
    """The scheme, and the host and port of a loopback endpoint. No path (a gateway can put a credential in it), no
    user name, password, query, or fragment, and no name of another host."""
    if not isinstance(url, str):
        return {"redacted": True, "types": shape(url)}
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if host in LOOPBACK_HOSTS or host.endswith(".localhost"):
        return f"{parts.scheme}://{host}:{parts.port}" if parts.port else f"{parts.scheme}://{host}"
    return f"{parts.scheme}://<remote host>"


def public_config(config: dict) -> dict:
    """The configuration as meta.json and the report keep it: only allow-listed values. Paths and CLI commands are
    left out. URLs keep no user name, password, query, or fragment. Every other value (hermes_config, qa_extra_body,
    an unknown key) becomes its key paths and value types only. The input is always the raw configuration: a value
    in it is never trusted as a redaction record."""
    kept: dict[str, Any] = {}
    for key, value in config.items():
        if key in PRIVATE_CONFIG_KEYS:
            continue
        if key == "drift" and isinstance(value, dict):
            kept[key] = {name: value[name] for name in DRIFT_KEYS if name in value}
        elif key in PUBLIC_CONFIG_KEYS:
            kept[key] = value
        elif key in URL_CONFIG_KEYS:
            kept[key] = public_url(value)
        else:
            kept[key] = {"redacted": True, "types": shape(value)}
    return kept


def config_digests(config: dict) -> dict[str, str]:
    """A SHA-256 of each configuration value (no private keys). meta.json keeps them, apart from the configuration,
    for the resume and merge checks; the report does not get them (a hash of a short credential can be guessed)."""
    return {key: hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
            for key, value in config.items() if key not in PRIVATE_CONFIG_KEYS}


def meta_digests(meta: dict) -> dict[str, str]:
    """The configuration digests of a run folder. A folder made before the digests has raw values in its config."""
    return meta.get("config_sha256") or config_digests(meta.get("config") or {})


def report_config(meta: dict) -> dict:
    """The configuration of a run folder for the report."""
    return dict(meta.get("config") or {}) if "config_sha256" in meta else public_config(meta.get("config") or {})


def digest_hash(digests: dict[str, str], exempt: tuple = PLAN_KEYS) -> str:
    kept = {key: value for key, value in digests.items() if key not in exempt}
    return hashlib.sha256(json.dumps(kept, sort_keys=True).encode("utf-8")).hexdigest()


def config_hash(config: dict) -> str:
    """A hash of the raw settings that can change a job result: no private paths and no job selection keys."""
    return digest_hash(config_digests(config))


# Runs in the Python of the Hermes virtual environment. It prints the version, the resolved paths, and the
# installed packages; run.py keeps only the version and hashes.
RUNTIME_PROBE = """
import json, os, platform, sys
from importlib import metadata
packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in metadata.distributions())
print(json.dumps({"version": platform.python_version(), "implementation": sys.implementation.name,
                  "executable": os.path.realpath(sys.executable), "prefix": os.path.realpath(sys.prefix),
                  "packages": packages}))
"""
HERMES_RUNTIME_KEYS = ("hermes_python", "hermes_prefix", "hermes_packages")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def runtime_state(config: dict) -> dict[str, Any]:
    """The runtime that runs the jobs: the host, the Python of the runner and of Hermes, and the argv of each CLI.
    "public" goes into meta.json and the report. "sha256" has hashes of private paths, the package list, and the
    CLI argv: meta.json keeps them for the resume and merge checks, and the report does not get them."""
    public: dict[str, Any] = {"host": {"os": platform.system(), "machine": platform.machine(),
                                       "python": sys.version.split()[0]}}
    digests = {"runner_python": sha256_value(os.path.realpath(sys.executable))}
    if config.get("hermes_python"):
        try:
            done = subprocess.run([str(config["hermes_python"]), "-I", "-c", RUNTIME_PROBE], capture_output=True,
                                  encoding="utf-8", errors="replace", timeout=120)
            probe = json.loads(done.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            raise SystemExit("hermes_python did not run the runtime probe; stop before any job") from None
        public["hermes_python"] = {"version": probe["version"], "implementation": probe["implementation"]}
        digests.update(hermes_python=sha256_value(probe["executable"]), hermes_prefix=sha256_value(probe["prefix"]),
                       hermes_packages=sha256_value(probe["packages"]))
    for engine, spec in sorted((config.get("cli") or {}).items()):
        digests[f"cli.{engine}"] = sha256_value((spec or {}).get("command"))
    return {"public": public, "sha256": digests}


def runtime_mismatches(meta: dict, other: dict, keys: list[str] | None = None) -> list[str]:
    """The runtime parts that differ between two run folders (meta.json form). keys=None compares all; else only
    the runner, the host, and the given hash keys (a merge compares only what both folders use)."""
    if meta.get("runtime_sha256") is None or other.get("runtime_sha256") is None:
        return ["runtime (not recorded)"]
    found = [] if meta.get("host") == other.get("host") else ["runtime host"]
    base, extra = meta["runtime_sha256"], other["runtime_sha256"]
    names = sorted(set(base) | set(extra)) if keys is None else ["runner_python", *keys]
    found += [f"runtime {name}" for name in names if base.get(name) != extra.get(name)]
    if (keys is None or "hermes_python" in keys) and (meta.get("runtime") or {}).get("hermes_python") != (
            other.get("runtime") or {}).get("hermes_python"):
        found.append("runtime hermes_python version")
    return found


def harness_ids(meta: dict) -> set[str]:
    """The harness versions that made the records of a run folder: the first run and each resume."""
    return {harness_id(item) for item in [meta.get("harness_sha256")] + [
        resume.get("harness_sha256") for resume in meta.get("resumes") or []] if item}


def plan_methods(meta: dict) -> set[str]:
    """The methods that a run folder planned: the first configuration and each recorded resume plan."""
    found = set((meta.get("config") or {}).get("methods") or [])
    for resume in meta.get("resumes") or []:
        found |= set(resume.get("methods") or []) | set((resume.get("plan") or {}).get("methods") or [])
    return found


def merge_mismatches(meta: dict, extra: dict, rescore: bool) -> list[str]:
    """Why the run folder of `extra` cannot join the report of `meta` (empty when it can). The runs must use the
    same harness, server model, configuration values (apart from the job plan), and gold files (or --rescore). They
    must use the same Hermes commit when both run Hermes methods, the same plugin files when both run plugin
    methods, and the same version of each CLI that both run."""
    def engines(item: dict) -> set[str]:
        return {METHODS[name]["engine"] for name in plan_methods(item) if name in METHODS}

    found = []
    if meta.get("schema") != extra.get("schema"):
        found.append("schema")
    both_cli = all(item & set(CLI_ENGINES) for item in (engines(meta), engines(extra)))
    exempt = MERGE_EXEMPT_KEYS + (() if both_cli else ("cli_timeout_s",))
    base, other = meta_digests(meta), meta_digests(extra)
    found += [f"config {key}" for key in sorted(set(base) | set(other))
              if key not in exempt and base.get(key) != other.get(key)]
    if harness_ids(meta) != harness_ids(extra):
        found.append("harness files")
    if any(item.get("server_model") or (item.get("config") or {}).get("server_check") for item in (meta, extra)):
        # One folder checked the server model: both must have recorded it, and the same one.
        if not meta.get("server_model") or not extra.get("server_model"):
            found.append("server model (not recorded)")
        elif meta["server_model"] != extra["server_model"]:
            found.append("server model")
    shared_cli = engines(meta) & engines(extra) & set(CLI_ENGINES)
    found += [f"{name} version" for name in sorted(shared_cli)
              if ((meta.get("cli") or {}).get(name) or {}).get("version")
              != ((extra.get("cli") or {}).get(name) or {}).get("version")]
    if not rescore and meta.get("gold_sha256") != extra.get("gold_sha256"):
        found.append("gold files (use --rescore)")
    hermes_methods = [item - set(CLI_ENGINES) for item in (engines(meta), engines(extra))]
    found += runtime_mismatches(meta, extra, (list(HERMES_RUNTIME_KEYS) if all(hermes_methods) else []) + [
        f"cli.{name}" for name in sorted(shared_cli)])
    if all(hermes_methods) and (meta.get("hermes") or {}).get("commit") != (extra.get("hermes") or {}).get("commit"):
        found.append("Hermes commit")
    if all("warm_compaction" in item for item in (engines(meta), engines(extra))) and (
            (meta.get("plugin") or {}).get("files_sha256") != (extra.get("plugin") or {}).get("files_sha256")):
        found.append("plugin files")
    return found


def harness_hashes() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(HERE.glob("*.py"))}


def harness_id(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def resume_mismatches(meta: dict, config: dict, plugin: dict, hermes: dict, cli: dict,
                      runtime: dict) -> tuple[list, list]:
    """Compare a run folder with the current sources and runtime (runtime_state). Return (hard, harness): a hard
    mismatch always stops the run; a harness mismatch stops it unless the caller allows it and records it."""
    hard = runtime_mismatches(meta, {"host": runtime["public"]["host"], "runtime_sha256": runtime["sha256"],
                                     "runtime": {k: v for k, v in runtime["public"].items() if k != "host"}})
    if (meta.get("plugin") or {}).get("files_sha256") != plugin["files_sha256"]:
        hard.append("plugin files")
    if (meta.get("hermes") or {}).get("commit") != hermes["commit"]:
        hard.append("Hermes commit")
    if meta.get("gold_sha256") != {path.name: sha256_file(path) for path in sorted(GOLD_DIR.glob("*.json"))}:
        hard.append("gold files")
    if digest_hash(meta_digests(meta)) != config_hash(config):
        hard.append("config")
    if config.get("server_check") or meta.get("server_model"):
        # The check is on (now or when the folder was made): the model must be read and recorded, and the same.
        current_model = server_model({**config, "server_check": True})
        if current_model is None:
            hard.append("server model (probe failed)")
        elif not meta.get("server_model"):
            hard.append("server model (not recorded)")
        elif meta["server_model"] != current_model:
            hard.append("server model")
    if {name: item.get("version") for name, item in (meta.get("cli") or {}).items() if name in cli} != {
            name: item.get("version") for name, item in cli.items()} and cli:
        hard.append("CLI versions")
    latest = (meta.get("resumes") or [{}])[-1].get("harness_sha256") or meta.get("harness_sha256")
    harness = sorted(name for name in set(latest or {}) | set(harness_hashes())
                     if (latest or {}).get(name) != harness_hashes().get(name))
    return hard, harness


SERVER_MODEL_KEYS = ("id", "type", "publisher", "arch", "compatibility_type", "quantization", "max_context_length")


def server_model(config: dict) -> dict | None:
    """With "server_check": true, the model file fields that the LM Studio REST API reports (no paths)."""
    if not config.get("server_check"):
        return None
    root = str(config["endpoint"]).rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(f"{root}/api/v0/models/{config['model']}", timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return {key: data.get(key) for key in SERVER_MODEL_KEYS}


def override_conflicts(config: dict) -> list[str]:
    """The config override keys that are not on the allow-lists (worker.HERMES_CONFIG_ALLOWED and
    worker.QA_EXTRA_ALLOWED)."""
    return ([f"hermes_config.{path}" for path in job_worker.override_conflicts(config.get("hermes_config") or {})]
            + [f"qa_extra_body.{key}" for key in job_worker.qa_body_conflicts(config.get("qa_extra_body") or {})])


def skipped_methods(meta: dict) -> list[dict]:
    """The methods that the first plan and each resume plan skipped, each method once (the first reason)."""
    found: dict[str, dict] = {}
    for item in (meta.get("skipped_methods") or []) + [
            skipped for resume in meta.get("resumes") or [] for skipped in resume.get("skipped_methods") or []]:
        found.setdefault(item["method"], item)
    return list(found.values())


def server_state(config: dict) -> str | None:
    """With "server_check": true, None when the LM Studio REST API reports the model loaded with the configured
    context length, else a short reason. A changed server (for example a reload with other settings) makes the
    results of a job invalid."""
    if not config.get("server_check"):
        return None
    root = str(config["endpoint"]).rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(f"{root}/api/v0/models/{config['model']}", timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return f"server_unreachable:{type(error).__name__}"
    if data.get("state") != "loaded":
        return f"model_state:{data.get('state')}"
    if data.get("loaded_context_length") != int(config["context_length"]):
        return f"loaded_context_length:{data.get('loaded_context_length')}"
    return None


def command_run(args) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.get("qa_batch", scoring.QA_BATCH_MAX) != scoring.QA_BATCH_MAX:
        # The gold files plan the QA batches (at most 12 questions, no gold rule passes on its batch prompt).
        raise SystemExit(f"qa_batch is fixed at {scoring.QA_BATCH_MAX}: the gold files plan the QA batches")
    conflicts = override_conflicts(config)
    if conflicts:
        raise SystemExit(f"config overrides change the benchmark method: {', '.join(conflicts)}; stop before any job")
    history_dir = Path(config.get("history_dir") or ROOT / ".work" / "bench-fixtures")
    fixtures.write_all(history_dir, history_dir / "gold-check")
    for spec in fixtures.SCENARIOS:
        fixtures.load(history_dir / f"{spec['id']}.history.json", GOLD_DIR / f"{spec['id']}.gold.json")
    hermes = hermes_state(Path(config["hermes_source"]))
    if not hermes["clean"]:
        # Results must come from a clean, known Hermes source.
        raise SystemExit("hermes_source is not a clean Git checkout; stop before any job runs")
    plugin = plugin_state(Path(config["plugin_source"]))
    # A plugin folder made with git archive has no Git data: the config can name its commit. The run cannot check
    # it, so it is kept apart, as a configured (not verified) value; the verified commit stays unknown.
    if not plugin["commit"] and config.get("plugin_commit"):
        plugin["commit_configured"] = str(config["plugin_commit"])
    jobs, skipped = plan(config, plugin)
    cli = cli_state(config, list(config["methods"]))
    runtime = runtime_state(config)
    model = server_model(config)
    if config.get("server_check") and model is None:
        # A required probe that fails stops the run: meta.json never records an unknown server model.
        raise SystemExit("server_check: the LM Studio model probe failed; stop before any job")
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        meta = {"schema": "warm-compaction.compaction-benchmark/v1", "synthetic": True,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": public_config(config), "config_sha256": config_digests(config),
                "server_model": model,
                "plugin": plugin, "hermes": hermes,
                "harness_sha256": harness_hashes(),
                "gold_sha256": current_gold_sha256(),
                "host": runtime["public"]["host"],
                "runtime": {key: value for key, value in runtime["public"].items() if key != "host"},
                "runtime_sha256": runtime["sha256"],
                "skipped_methods": skipped, "planned_jobs": len(jobs)}
        if cli:
            meta["cli"] = cli
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    else:
        # A resume: the jobs of one run folder must come from the same plugin, Hermes, gold files, and settings.
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        hard, harness = resume_mismatches(meta, config, plugin, hermes, cli, runtime)
        if hard:
            raise SystemExit(f"resume refused: {', '.join(hard)} differ from {meta_path}; use a new run folder")
        if harness and not args.allow_harness_change:
            raise SystemExit(f"resume refused: harness files changed ({', '.join(harness)}); check that the change "
                             "does not change job results, then add --allow-harness-change to record it")
        if not args.dry_run:
            meta.setdefault("resumes", []).append({
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "harness_sha256": harness_hashes(),
                "harness_changed": harness, "methods": list(config["methods"]), "skipped_methods": skipped,
                "plan": public_config({key: config[key] for key in PLAN_KEYS if key in config})})
            meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    runs_path = out_dir / "runs.jsonl"
    # A job is done when its latest record is ok: a later "invalid" or failed record makes it run again.
    done = {run["job"] for run in load_runs(out_dir) if run.get("status") == "ok"} if runs_path.exists() else set()
    todo = [job for job in jobs if job["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(jobs)} jobs planned, {len(done)} done, {len(todo)} to run; skipped methods: "
          f"{[item['method'] for item in skipped]}", flush=True)
    if args.dry_run:
        for job in todo:
            print(job["id"], json.dumps(job["settings"]))
        return 0
    worker_failures = 0
    for number, job in enumerate(todo, 1):
        problem = server_state(config)
        if problem:
            raise SystemExit(f"server check failed before {job['id']}: {problem}; stop (no record written)")
        record = run_job(job, config, out_dir, history_dir)
        problem = server_state(config)
        if problem:
            # The server changed during the job: keep the record, mark it invalid so a resume runs it again.
            record["status"], record["error"] = "invalid", f"server_state_changed:{problem}"
        with runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if problem:
            raise SystemExit(f"server check failed after {job['id']}: {problem}; the job is marked invalid")
        paths = [cycle["path"] for cycle in record.get("cycles", [])]
        accuracy = [round(sum(m[0] == "1" for m in c["marks"]) / max(len(c["marks"]), 1), 2)
                    for c in record.get("cycles", [])]
        print(f"[{number}/{len(todo)}] {job['id']} status={record['status']} error={record['error']} "
              f"paths={paths} accuracy={accuracy} elapsed={record['elapsed_s']}s", flush=True)
        worker_failures = worker_failures + 1 if record["error"] == "worker_no_report" else 0
        if worker_failures >= 2:
            raise SystemExit(f"2 workers in a row gave no report (last: {job['id']}); stop. See "
                             f"{out_dir / record.get('stderr_file', 'worker-stderr')}")
        if record["error"] == "login_required":
            # The benchmark never signs in: stop and report.
            raise SystemExit(f"{job['method']} asked for a login; stop the run")
    return 0


# Report.

def load_attempts(out_dir: Path) -> list[dict]:
    """Every record of runs.jsonl, also the earlier attempts of a job that a resume ran again."""
    return [json.loads(line) for line in (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_runs(out_dir: Path) -> list[dict]:
    """The latest record of each job."""
    return list({item["job"]: item for item in load_attempts(out_dir)}.values())


def planned_jobs(meta: dict) -> dict[str, dict]:
    """The jobs that the run folder plans, by job id (empty when the plan cannot be made): the jobs of the first
    configuration and of each resume. A resume can add jobs; a job that an earlier plan had stays planned."""
    found: dict[str, dict] = {}
    configs = [meta.get("config") or {}] + [
        {**(meta.get("config") or {}), **item["plan"]} for item in meta.get("resumes") or [] if item.get("plan")]
    for config in configs:
        try:
            jobs, _skipped = plan(config, meta["plugin"])
        except (KeyError, TypeError, ValueError):
            continue
        found.update({job["id"]: job for job in jobs if job["id"] not in found})
    return found


def rate(successes: int, total: int) -> dict[str, Any]:
    low, high = scoring.wilson(successes, total)
    return {"k": successes, "n": total, "rate": successes / total if total else None, "low": low, "high": high}


def question_table() -> dict[str, list[dict]]:
    return {spec["id"]: json.loads((GOLD_DIR / f"{spec['id']}.gold.json").read_text(encoding="utf-8"))["questions"]
            for spec in fixtures.SCENARIOS}


def aggregate(runs: list[dict], meta: dict, attempts: list[dict] | None = None,
              planned: dict[str, dict] | None = None) -> dict[str, Any]:
    """Aggregate the latest record of each job. A planned job without a record, a failed job, and a job without a
    scored compaction count as failures in the success rate and in the "failures as wrong" accuracy."""
    gold = question_table()
    all_attempts = attempts if attempts is not None else runs
    planned = planned or {}
    recorded = {run["job"] for run in runs}
    main = [run for run in runs if run["cycles_requested"] == 1]
    drift = [run for run in runs if run["cycles_requested"] > 1]
    methods = [name for name in METHODS if any(run["method"] == name for run in runs)
               or any(job["method"] == name for job in planned.values())]
    result: dict[str, Any] = {"methods": {}, "paired": [], "drift": {}}
    pairs: dict[tuple, dict[str, str]] = defaultdict(dict)
    for run in runs:
        for cycle in run.get("cycles") or []:
            cycle["path"] = stored_path(cycle, run["method"])
    for name in methods:
        items = [run for run in main if run["method"] == name]
        not_run = [job for job_id, job in planned.items()
                   if job["method"] == name and job["cycles"] == 1 and job_id not in recorded]
        # A failed job (status not ok) has no scored compaction, also when the worker sent a partial cycle.
        every = [(run, run["cycles"][0]) for run in items if run.get("status") == "ok" and run.get("cycles")]
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
        # A job without cycles (a timeout, a worker error) and a planned job without a record are failures too.
        for run in items:
            if run.get("status") != "ok" or not run.get("cycles"):
                reasons[f"job:{run.get('error') or run.get('status')}"] += 1
        if not_run:
            reasons["not_run"] += len(not_run)
        failed_jobs = [run["job"] for run in items if run["status"] != "ok" or not run.get("cycles")]
        accepted = sum(1 for _run, cycle in every if cycle["path"] in ACCEPTED_PATHS)
        method_attempts = [item for item in all_attempts if item["method"] == name and item["cycles_requested"] == 1]
        total = len(items) + len(not_run)
        entry: dict[str, Any] = {
            "label": METHODS[name]["label"], "runs": len(items), "planned_runs": total, "not_run": len(not_run),
            "job_failures": len(failed_jobs), "attempts": len(method_attempts),
            "failed_attempts": [{"job": item["job"], "error": item.get("error"), "status": item.get("status")}
                                for item in method_attempts if item.get("status") != "ok" or not item.get("cycles")],
            "compacted_runs": len(cycles), "preflight_runs": sum(bool(cycle.get("preflight")) for _r, cycle in every),
            "paths": dict(paths), "failure_reasons": dict(reasons),
            "success": rate(accepted, total) if METHODS[name]["engine"] != "none" else None,
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
        # The same, with each planned run that has no scored compaction (not compacted, failed, or not run) scored
        # as all wrong.
        scored_jobs = {run["job"] for run, _cycle in cycles}
        unscored = [run["fixture"] for run in items if run["job"] not in scored_jobs]
        unscored += [job["fixture"] for job in not_run]
        missing_by_category = defaultdict(int)
        for fixture in unscored:
            for question in gold[fixture]:
                missing_by_category[question["category"]] += 1
        entry["categories_failures_wrong"] = {
            category: rate(by_category[category][0], by_category[category][1] + missing_by_category[category])
            for category in CATEGORIES}
        missing = sum(missing_by_category[c] for c in scored)
        entry["overall_failures_wrong"] = rate(entry["overall"]["k"], entry["overall"]["n"] + missing)
        values = [cycle for _run, cycle in cycles]
        # The size before compaction and the ratio use only cycles that measured the exact compaction input.
        measured = [c for c in values if c.get("size_basis") == "compaction_input" and c.get("est_tokens_before")]
        entry["tokens"] = {
            "est_before": scoring.mean_interval([c["est_tokens_before"] for c in measured]),
            "est_after": scoring.mean_interval([c["est_tokens_after"] for c in values]),
            "ratio_after_before": scoring.mean_interval([c["est_tokens_after"] / c["est_tokens_before"]
                                                         for c in measured]),
            "before_not_measured_runs": len(values) - len(measured),
            "qa_prompt_tokens": scoring.mean_interval([c["qa_prompt_tokens"] for c in values
                                                       if c.get("qa_prompt_tokens")])}
        # Time: every completed step of every attempt, also of a job that failed later (for example in the
        # questions) and of an earlier attempt that a resume ran again. Accuracy above uses the scored runs only.
        # An attempt that the server check marked invalid ran on a changed server: it is left out.
        timed = [item["cycles"][0] for item in method_attempts
                 if item.get("cycles") and item.get("status") != "invalid"]
        entry["latency_s"] = {
            "compress": scoring.mean_interval([c["compress_s"] for c in timed if c.get("compress_s") is not None]),
            "compress_compacted": scoring.mean_interval([c["compress_s"] for c in timed if c.get("compacted")
                                                         and c.get("compress_s") is not None]),
            # A preflight run compacted inside the main request: its main time includes the compaction.
            "main_request": scoring.mean_interval([c["main_s"] for c in timed if c.get("main_s") is not None
                                                   and not c.get("preflight") and not c.get("main_failed")]),
            "questions": scoring.mean_interval([c["qa_s"] for c in values if c.get("qa_s") is not None]),
            "timed_attempts": len(timed)}
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
            # A planned pair without data is listed with 0 pairs (no difference, no p-value).
            result["paired"].append({
                "first": first, "second": second, "category": category, "pairs": n, "first_only": a_only,
                "second_only": b_only, "difference": (a_only - b_only) / n if n else None,
                "p_mcnemar": scoring.mcnemar(a_only, b_only)})
    drift_methods = {run["method"] for run in drift} | {
        job["method"] for job in planned.values() if job["cycles"] > 1 and job["method"] in METHODS}
    for name in sorted(drift_methods, key=list(METHODS).index):
        recorded_drift = [run for run in drift if run["method"] == name]
        items = [run for run in recorded_drift if run.get("status") == "ok" and run.get("cycles")]
        per_cycle = []
        control = METHODS[name]["engine"] == "none"

        def scored(run: dict, index: int, control: bool = control) -> dict | None:
            # A cycle that did not compact asked no questions: it is left out, not scored as wrong.
            if index >= len(run["cycles"]):
                return None
            cycle = run["cycles"][index]
            return cycle if cycle.get("compacted") or control else None

        def hits(run: dict, cycle: dict) -> tuple[int, int]:
            marks = [mark for question, mark in zip(gold[run["fixture"]], cycle["marks"])
                     if question["category"] != "recent"]
            return sum(mark[0] == "1" for mark in marks), len(marks)

        for index in range(max(len(run["cycles"]) for run in items) if items else 0):
            right = total = 0
            paths = defaultdict(int)
            kept = []
            for run in items:
                if index < len(run["cycles"]):
                    paths[run["cycles"][index]["path"]] += 1
                cycle = scored(run, index)
                if cycle is None:
                    continue
                kept.append(cycle)
                k, n = hits(run, cycle)
                right, total = right + k, total + n
            per_cycle.append({"cycle": index + 1, "accuracy": rate(right, total), "paths": dict(paths),
                              "scored_runs": len(kept),
                              "est_tokens_after": scoring.mean_interval([c["est_tokens_after"] for c in kept])})
        # D1 compares cycle 3 with cycle 1 on the runs that have both cycles scored.
        both = [run for run in items if scored(run, 0) and scored(run, 2)]
        first = [hits(run, scored(run, 0)) for run in both]
        third = [hits(run, scored(run, 2)) for run in both]
        paired_cycles = {"runs": len(both), "cycle_1": rate(sum(k for k, _ in first), sum(n for _, n in first)),
                         "cycle_3": rate(sum(k for k, _ in third), sum(n for _, n in third))}
        drift_planned = {job_id for job_id, job in planned.items() if job["method"] == name and job["cycles"] > 1}
        drift_total = len(drift_planned | {run["job"] for run in recorded_drift})
        result["drift"][name] = {
            "runs": len(items), "planned_runs": drift_total, "failed_or_not_run": drift_total - len(items),
            "fixtures": sorted({run["fixture"] for run in items}), "cycles": per_cycle,
            "cycle_1_and_3": paired_cycles}
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
    """The paired comparison, or None when it has no pairs (no data)."""
    return next((item for item in result["paired"] if (item["first"], item["second"], item["category"]) ==
                 (first, second, category) and item["pairs"]), None)


def all_or_none(items: list, check) -> bool | None:
    """None when an item has no data (the check is unavailable), else whether the check passes for all."""
    return None if any(item is None for item in items) else all(check(item) for item in items)


def exact_difference(item: dict) -> Fraction:
    """The paired difference as an exact fraction, so a boundary value (for example exactly 5 points) is exact."""
    return Fraction(item["first_only"] - item["second_only"], item["pairs"])


def exact_rate(item: dict) -> Fraction:
    return Fraction(item["k"], item["n"]) if item.get("n") else Fraction(0)


def decide(result: dict) -> dict[str, Any]:
    """Apply the pre-registered decision rule (PREREGISTRATION.md). Return each check and its outcome."""
    checks: dict[str, Any] = {}
    methods = result["methods"]
    if "warm_ledger" in methods and "warm_no_ledger" in methods:
        gains = [paired(result, "warm_ledger", "warm_no_ledger", c) for c in LEDGER_CATEGORIES]
        checks["H1_ledger_helps"] = all_or_none(gains, lambda item: exact_difference(item) >= Fraction(10, 100)
                                                and (item["p_mcnemar"] or 1) < 0.05)
        losses = [paired(result, "warm_ledger", "warm_no_ledger", c) for c in GUARD_CATEGORIES]
        # "Not more than 5 points below": exactly 5 points below passes.
        checks["H2_no_regression"] = all_or_none(losses, lambda item: not (
            exact_difference(item) < Fraction(-5, 100)) and not (
            item["difference"] < 0 and (item["p_mcnemar"] or 1) < 0.05))
    if "fallback_relaxed_ledger" in methods and "fallback_relaxed_no_ledger" in methods:
        gains = [paired(result, "fallback_relaxed_ledger", "fallback_relaxed_no_ledger", c) for c in LEDGER_CATEGORIES]
        checks["H1b_ledger_helps_fallback"] = all_or_none(gains, lambda item: exact_difference(item) >= Fraction(
            10, 100) and (item["p_mcnemar"] or 1) < 0.05)
    if "fallback_relaxed_ledger" in methods and "fallback_quoted_ledger" in methods:
        relaxed = methods["fallback_relaxed_ledger"]["success"]
        quoted = methods["fallback_quoted_ledger"]["success"]
        checks["R1_relaxed_succeeds_more_often"] = (None if relaxed["low"] is None or quoted["high"] is None
                                                    else relaxed["low"] > quoted["high"])
    if "warm_ledger" in methods and "stock_hermes" in methods:
        warm, stock = methods["warm_ledger"]["categories"], methods["stock_hermes"]["categories"]
        def against_stock(categories) -> bool | None:
            items = [None if warm[c]["rate"] is None or stock[c]["rate"] is None else c for c in categories]
            return all_or_none(items, lambda c: warm[c]["high"] >= stock[c]["rate"])

        checks["H3a_matches_stock_on_ledger_categories"] = against_stock(LEDGER_CATEGORIES)
        checks["H3b_no_loss_on_model_categories"] = against_stock(("goal", "decision", "next_step"))
        warm_after = methods["warm_ledger"]["tokens"]["est_after"]["mean"]
        stock_after = methods["stock_hermes"]["tokens"]["est_after"]["mean"]
        # None: no run of one method compacted, so the check has no data. It does not pass.
        checks["H4_smaller_than_stock"] = (None if warm_after is None or stock_after is None
                                           else warm_after <= stock_after)
    if "warm_ledger" in methods and methods["warm_ledger"]["success"]:
        checks["S1_warm_success_at_least_90"] = exact_rate(methods["warm_ledger"]["success"]) >= Fraction(90, 100)
    if "warm_ledger" in result["drift"]:
        both = result["drift"]["warm_ledger"].get("cycle_1_and_3") or {}
        # None: no run has cycle 1 and cycle 3 scored, so the check has no data. It does not pass.
        checks["D1_drift_within_10_points"] = None if not both.get("runs") else (
            exact_rate(both["cycle_3"]) >= exact_rate(both["cycle_1"]) - Fraction(10, 100))
    verdict = []
    if "R1_relaxed_succeeds_more_often" in checks:
        verdict.append("The relaxed fallback format succeeds more often than the quoted format."
                       if checks["R1_relaxed_succeeds_more_often"] else
                       "The success rates of the two fallback formats are not separated by the rule.")
    if checks.get("H1_ledger_helps") and checks.get("H2_no_regression"):
        verdict.append("Keep the ledger on by default.")
    elif None in (checks.get("H1_ledger_helps", False), checks.get("H2_no_regression", False)):
        verdict.append("The ledger result is unavailable (no data for a check).")
    elif "H1_ledger_helps" in checks:
        verdict.append("The ledger result does not meet the rule for a default.")
    if all(checks.get(k) for k in ("H3a_matches_stock_on_ledger_categories", "H3b_no_loss_on_model_categories",
                                   "H4_smaller_than_stock")) and "H4_smaller_than_stock" in checks:
        verdict.append("Warm path + ledger is competitive with the built-in compressor on this workload.")
    elif any(checks.get(k, False) is None for k in ("H3a_matches_stock_on_ledger_categories",
                                                    "H3b_no_loss_on_model_categories", "H4_smaller_than_stock")):
        verdict.append("The comparison with the built-in compressor is unavailable (no data for a check).")
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
        f"- Plugin commit: {plugin_commit_text(meta['plugin'])}. Hermes commit: "
        f"`{(meta['hermes'].get('commit') or 'unknown')[:12]}` (clean: {meta['hermes'].get('clean')}).",
        f"- Model: `{meta['config'].get('model')}`; context window setting {meta['config'].get('context_length')}; "
        f"threshold {meta['config'].get('threshold')}; {meta['config'].get('replicates')} replicates per cell.",
        f"- Skipped methods: {', '.join(item['method'] for item in meta.get('skipped_methods', [])) or 'none'}.",
        *(["- Other CLIs: " + ", ".join(f"{name} {item['version']}" for name, item in meta["cli"].items()) + "."]
          if meta.get("cli") else []),
        *(["- Rescored with the current gold files: " + ", ".join(
            f"{item['folder']} ({item['jobs_rescored']} jobs, {item['marks_changed']} marks changed, "
            f"{item['jobs_without_raw']} without raw output, {item.get('jobs_failed_incomplete', 0)} failed for an "
            "incomplete answer)" for item in meta["rescore"]["folders"]) + "."]
          if meta.get("rescore") else []),
        *(["- Warning: the gold files changed after these runs and the marks were not rescored: "
           + ", ".join(meta["gold_changed_after_run"]) + "."] if meta.get("gold_changed_after_run") else []), "",
        "## Compaction success", "",
        "A planned job that failed, timed out, or did not run counts as a failure. Attempts include the earlier "
        "attempts of a job that a resume ran again.", "",
        "| Method | Runs ok / planned | Attempts (failed) | Accepted summary | Paths | Failure reasons |",
        "| --- | --- | --- | --- | --- | --- |"]
    for name in names:
        entry = methods[name]
        ok_runs = entry["runs"] - entry["job_failures"]
        lines.append(f"| {entry['label']} | {ok_runs} / {entry['planned_runs']} | "
                     f"{entry['attempts']} ({len(entry['failed_attempts'])}) | {pct(entry['success'])} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(entry['paths'].items()))} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(entry['failure_reasons'].items())) or '-'} |")
    if any(entry.get("preflight_runs") for entry in methods.values()):
        lines += ["", "Preflight runs: " + ", ".join(f"{entry['label']} {entry['preflight_runs']}" for entry in
                                                    methods.values() if entry.get("preflight_runs")) + ".",
                  "", "A preflight run: the engine compacted during the main turn, before the main request (for "
                  "example, the plugin preflight check with no captured request). Its path is the path of that "
                  "summary, with the same checks as a manual compaction. The questions use that history, "
                  "and the main request ran on the compacted history, unlike the other runs."]
    lines += ["", "## Accuracy by category (all questions)", "",
              "Runs with a scored compaction only (the control: all runs). The first row gives the scored runs and "
              "the planned runs of each method.", "",
              "| Category | " + " | ".join(methods[n]["label"] for n in names) + " |",
              "| --- | " + " | ".join("---" for _ in names) + " |",
              "| runs scored / planned | " + " | ".join(f"{methods[n]['compacted_runs']} / {methods[n]['planned_runs']}"
                                                      for n in names) + " |"]
    for category in CATEGORIES:
        lines.append(f"| {category} | " + " | ".join(pct(methods[n]["categories"][category]) for n in names) + " |")
    lines.append("| overall (without recent) | " + " | ".join(pct(methods[n]["overall"]) for n in names) + " |")
    lines.append("| est. tokens after (mean) | " + " | ".join(num(methods[n]["tokens"]["est_after"], 0)
                                                             for n in names) + " |")
    lines += ["", "## Accuracy by category, failures as wrong", "",
              "Each planned run without a scored compaction (not compacted, failed, timed out, or not run) counts as "
              "a wrong answer to all its questions.", "",
              "| Category | " + " | ".join(methods[n]["label"] for n in names) + " |",
              "| --- | " + " | ".join("---" for _ in names) + " |"]
    for category in CATEGORIES:
        lines.append(f"| {category} | " + " | ".join(pct(methods[n]["categories_failures_wrong"][category])
                                                      for n in names) + " |")
    lines.append("| overall (without recent) | " + " | ".join(pct(methods[n]["overall_failures_wrong"])
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
        difference = "unavailable" if item["difference"] is None else f"{100 * item['difference']:+.1f} points"
        lines.append(f"| {item['first']} | {item['second']} | {item['category']} | {item['pairs']} | "
                     f"{difference} | {p} |")
    if result["drift"]:
        lines += ["", "## Repeated compaction (drift)", "",
                  "The same session, compacted again after each new chunk of work. Accuracy is on the questions about "
                  "the first window (without the recent facts).", "",
                  "| Method | Runs ok / planned | Cycle | Accuracy | Paths | Est. tokens after |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, entry in result["drift"].items():
            if not entry["cycles"]:
                lines.append(f"| {METHODS[name]['label']} | 0 / {entry['planned_runs']} | - | unavailable | - | "
                             "n/a |")
            for cycle in entry["cycles"]:
                lines.append(f"| {METHODS[name]['label']} | {entry['runs']} / {entry['planned_runs']} | "
                             f"{cycle['cycle']} | "
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


def current_gold_sha256() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(GOLD_DIR.glob("*.json"))}


def raw_sources(out_dir: Path, meta: dict | None, raw_dirs: list[Path] | tuple) -> list[tuple[Path, set | None]]:
    """The raw folders that --rescore can read for the records of out_dir: its own raw/ folder, and each --raw-dir
    that is the raw/ folder of a run folder of the same run identity (merge_mismatches finds no difference). For a
    --raw-dir, the set has the (job, start time) of each attempt in its runs.jsonl."""
    sources: list[tuple[Path, set | None]] = [(out_dir / "raw", None)]
    for folder in raw_dirs:
        parent = Path(folder).parent
        if Path(folder).name != "raw" or not (parent / "meta.json").exists() or not (parent / "runs.jsonl").exists():
            raise SystemExit(f"report refused: --raw-dir {folder} is not the raw folder of a run folder")
        other = json.loads((parent / "meta.json").read_text(encoding="utf-8"))
        mismatches = merge_mismatches(meta or {}, other, True)
        if mismatches:
            raise SystemExit(f"report refused: --raw-dir {folder} is from a run that differs in "
                             f"{', '.join(mismatches)}")
        sources.append((Path(folder), {(item["job"], item.get("started_utc")) for item in load_attempts(parent)}))
    return sources


def find_raw(run: dict, sources: list[tuple[Path, set | None]]) -> Path | None:
    """The raw file of a record: the file whose hash the record names. A record made before the hash takes the
    file of its own run folder, or of a --raw-dir folder that has the same attempt (job and start time)."""
    name = run["job"].replace("|", "__") + ".json"
    for folder, attempts in sources:
        path = folder / name
        if not path.exists():
            continue
        if run.get("raw_sha256"):
            if sha256_file(path) == run["raw_sha256"]:
                return path
        elif attempts is None or (run["job"], run.get("started_utc")) in attempts:
            return path
    return None


def rescore(out_dir: Path, runs: list[dict], raw_dirs: list[Path] | tuple = (),
            meta: dict | None = None) -> dict[str, int]:
    """Score the kept worker output (raw/) again with the current gold files. Only the marks change.
    raw_dirs are more raw folders to search, for example of a run that gave reused jobs; meta is the meta.json of
    out_dir, which each of them must match. A job without a verified raw file keeps its marks; the result counts
    it."""
    built = {spec["id"]: fixtures.build(spec)[0] for spec in fixtures.SCENARIOS}
    counts = {"jobs_rescored": 0, "jobs_without_raw": 0, "marks_changed": 0, "jobs_failed_incomplete": 0,
              "jobs_other_qa_prompt": 0}
    sources = raw_sources(out_dir, meta, raw_dirs)
    for run in runs:
        if not run.get("cycles"):
            continue
        raw = find_raw(run, sources)
        if raw is None:
            counts["jobs_without_raw"] += 1
            continue
        out = json.loads(raw.read_text(encoding="utf-8"))
        gold = json.loads((GOLD_DIR / f"{run['fixture']}.gold.json").read_text(encoding="utf-8"))
        # Answers to other prompts (another batch plan or another question text) cannot be scored with this gold.
        if any(not same_prompts(raw_cycle, gold) for raw_cycle in out.get("cycles", [])):
            counts["jobs_other_qa_prompt"] += 1
            continue
        for cycle, raw_cycle in zip(run["cycles"], out.get("cycles", [])):
            scored = score_cycle(raw_cycle, gold, built[run["fixture"]], run["method"])
            counts["marks_changed"] += sum(a != b for a, b in zip(cycle["marks"], scored["marks"]))
            cycle["marks"], cycle["path"], cycle["qa_missing"] = scored["marks"], scored["path"], scored["qa_missing"]
        counts["jobs_failed_incomplete"] += fail_incomplete(run)
        counts["jobs_rescored"] += 1
    return counts


def same_prompts(raw_cycle: dict, gold: dict) -> bool:
    """True when the raw answers of a cycle answer the exact prompts of the current gold file: the batches of its
    plan, and for each batch the SHA-256 of the user row (instruction and question texts). A raw file without the
    hash cannot show it."""
    batches = (raw_cycle.get("qa") or {}).get("batches") or []
    if not batches:
        return True
    if [batch["numbers"] for batch in batches] != gold.get("qa_batches"):
        return False
    return all(batch.get("prompt_sha256") == scoring.prompt_sha256(scoring.qa_prompt(
        [(n, gold["questions"][n - 1]["question"]) for n in batch["numbers"]])) for batch in batches)


def refuse_stale(item: dict) -> None:
    """A rescore that cannot score every job would mix old-gold marks with the current gold files."""
    if item.get("jobs_other_qa_prompt"):
        raise SystemExit(f"report refused: --rescore found {item['jobs_other_qa_prompt']} jobs in {item['folder']} "
                         "whose raw answers do not answer the current QA prompts (other batches, other question "
                         "text, or no prompt hash). Report them with the harness and gold files that made them.")
    if item["jobs_without_raw"]:
        raise SystemExit(f"report refused: --rescore found {item['jobs_without_raw']} jobs in {item['folder']} "
                         "without a verified raw output (keep_text was off, or no raw file has the hash or the "
                         "attempt of the record); their marks use the old gold files. Add --raw-dir for them, or "
                         "report without --rescore (the report then warns).")


def plugin_commit_text(plugin: dict) -> str:
    """The plugin commit for the report: the verified commit, else unknown (with a configured value, if any)."""
    if plugin.get("commit"):
        return f"`{plugin['commit'][:12]}`"
    if plugin.get("commit_configured"):
        return f"unknown (configured, not verified: `{plugin['commit_configured'][:12]}`)"
    return "unknown"


def command_report(args) -> int:
    out_dir = Path(args.out)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    runs = load_runs(out_dir)
    attempts = load_attempts(out_dir)
    planned = planned_jobs(meta)
    gold_now = current_gold_sha256()
    rescores = []
    if args.rescore:
        raw_dirs = [Path(folder) for folder in args.raw_dir or []]
        rescores.append({"folder": out_dir.name, **rescore(out_dir, runs, raw_dirs, meta)})
        refuse_stale(rescores[-1])
    # The metadata of each accepted folder as it was read (the report changes meta below).
    accepted = [(out_dir, copy.deepcopy(meta))]
    for extra in args.also or []:
        # A second run folder (for example the CLI methods): its jobs join the report, and its metadata is kept.
        extra_dir = Path(extra)
        extra_meta = json.loads((extra_dir / "meta.json").read_text(encoding="utf-8"))
        # Each added folder must match every folder accepted so far (not only the first): two added folders can
        # share an engine that the first folder does not run.
        problems = [f"{folder} in {', '.join(found)}" for folder, item in accepted
                    if (found := merge_mismatches(item, extra_meta, args.rescore))]
        if problems:
            raise SystemExit(f"report refused: {extra_dir} differs from {'; '.join(problems)}")
        accepted.append((extra_dir, extra_meta))
        known = {run["job"] for run in runs}
        extra_runs = [run for run in load_runs(extra_dir) if run["job"] not in known]
        if args.rescore:
            rescores.append({"folder": extra_dir.name, **rescore(extra_dir, extra_runs, raw_dirs, extra_meta)})
            refuse_stale(rescores[-1])
        elif extra_meta.get("gold_sha256") != gold_now:
            meta.setdefault("gold_changed_after_run", []).append(extra_dir.name)
        runs += extra_runs
        attempts += [item for item in load_attempts(extra_dir) if item["job"] not in known]
        planned.update({job_id: job for job_id, job in planned_jobs(extra_meta).items() if job_id not in planned})
        meta.setdefault("merged_runs", []).append({**{key: extra_meta.get(key) for key in (
            "started_utc", "plugin", "hermes", "harness_sha256", "resumes", "cli", "planned_jobs", "server_model")},
            "skipped_methods": skipped_methods(extra_meta),
            "config": report_config(extra_meta)})
        meta.setdefault("cli", {}).update(extra_meta.get("cli") or {})
    if rescores:
        # The marks come from the current gold files, not from the gold files of the run.
        meta["rescore"] = {"gold_sha256": gold_now, "folders": rescores}
    elif meta.get("gold_sha256") != gold_now:
        meta.setdefault("gold_changed_after_run", []).insert(0, out_dir.name)
    if meta.get("gold_changed_after_run"):
        print("warning: gold files changed after the run; use --rescore to score the raw answers again",
              file=sys.stderr)
    result = aggregate(runs, meta, attempts, planned)
    # A run folder made before the redaction rule can have raw values in meta.json: redact them here too. The
    # digests stay in the local meta.json only.
    meta["config"] = report_config(meta)
    meta.pop("config_sha256", None)
    meta.pop("runtime_sha256", None)
    # The methods that the first plan or a resume plan skipped (also of a merged run folder).
    meta["skipped_methods"] = list({item["method"]: item for item in skipped_methods(meta) + [
        skipped for merged in meta.get("merged_runs") or [] for skipped in merged.get("skipped_methods") or []]
        }.values())
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
    run.add_argument("--allow-harness-change", action="store_true",
                     help="continue a run folder after a harness change that does not change job results (recorded)")
    run.set_defaults(func=command_run)
    report = sub.add_parser("report", help="aggregate runs.jsonl into the JSON results and the Markdown report")
    report.add_argument("--out", required=True)
    report.add_argument("--json", required=True)
    report.add_argument("--md")
    report.add_argument("--also", action="append", help="another run folder to include (repeat for more)")
    report.add_argument("--rescore", action="store_true",
                        help="score the kept raw answers (keep_text) again with the current gold files")
    report.add_argument("--raw-dir", action="append",
                        help="with --rescore: another raw folder to search, for example of a run that gave reused jobs")
    report.set_defaults(func=command_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
