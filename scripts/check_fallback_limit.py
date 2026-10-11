"""Check the fallback reply limit with real Hermes and a synthetic loopback server.

Use a Hermes dependency interpreter and a clean Hermes checkout. Each case has a fresh process and
temporary home. The check uses the public PluginContext API. It sends no real model request. The report
keeps counts and booleans only. A missing default limit is an observation of the host, not a passing limit
check. The configured case checks a custom Chat Completions workaround only.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile

from check_plugin_hermes import (
    FakeServer, Fence, MODEL, ROOT, sha, source_state, worker_environment, write_json,
)


def worker(source: Path, run_dir: Path, configured: bool) -> None:
    """Send one real PluginLlm request with fake credentials and a synthetic transcript."""
    result = {"status": "failed", "configured": configured, "checks": {}, "error_class": None}
    server = FakeServer({})
    server.start()
    platform.uname()
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(ROOT))
    home = run_dir / "home"
    task = {"provider": "custom", "model": MODEL, "base_url": server.base_url,
            "api_key": "synthetic-fake-key", "api_mode": "chat_completions"}
    if configured:
        task["extra_body"] = {"max_tokens": 2048}
    # JSON is valid YAML. No private profile or installed plugin is read.
    write_json(home / "config.yaml", {"auxiliary": {"warm_compaction": task},
                                      "model": {"provider": "custom", "model": MODEL,
                                                "base_url": server.base_url,
                                                "api_key": "synthetic-fake-key"}})
    fence = Fence(run_dir, server.httpd.server_port, source, ROOT / "warm_compaction")
    sys.addaudithook(fence)
    try:
        import hermes_bootstrap  # noqa: F401 - Select the isolated dependency runtime first.
        from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
        from warm_compaction.fallback import MAX_TOKENS, llm_summary

        context = PluginContext(PluginManifest(name="warm_compaction", version="0.0.0"), get_plugin_manager())
        context.register_auxiliary_task("warm_compaction", display_name="Synthetic fallback probe",
                                        description="Check a synthetic fallback reply limit.")
        summary, _tokens = llm_summary(context.llm, [{"role": "user", "content": "Synthetic question 1."},
                                                    {"role": "assistant", "content": "Synthetic answer 1."}],
                                      ("[CONTEXT SUMMARY]:",))
        requests = server.requests
        body = requests[0]["body"] if len(requests) == 1 else {}
        result.update(request_count=len(requests), requested_max_tokens=MAX_TOKENS,
                      wire_max_tokens=body.get("max_tokens"),
                      wire_max_completion_tokens=body.get("max_completion_tokens"))
        checks = result["checks"]
        checks["one_fallback_request"] = len(requests) == 1 and requests[0]["kind"] == "fallback"
        checks["synthetic_summary_accepted"] = summary is not None
        checks["no_other_limit_field"] = "max_completion_tokens" not in body
        # Keep optional host child processes blocked. They do not affect this wire check.
        checks["no_network_or_write_attempts"] = not set(fence.denied).difference({"child_process_blocked"})
        # Hermes catches OSError from its Git version probes. Other child calls fail this check.
        version_callers = ("child_process_blocked from hermes_cli/version_info.py:_run_git < ",
                           "child_process_blocked from hermes_cli/version_info.py:_git_version_info < ")
        checks["only_blocked_version_probes"] = (fence.programs <= {"git", "git.exe"}
                                                 and all(caller.startswith(version_callers)
                                                         for caller in fence.callers))
        if configured:
            checks["configured_limit_reaches_wire"] = body.get("max_tokens") == MAX_TOKENS
        result["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as error:
        # Do not publish exception messages, host paths, or generated text.
        name = type(error).__name__
        result["error_class"] = name if name.isascii() and name.isidentifier() and len(name) <= 80 else "Exception"
    finally:
        result["fence_denied_count"] = sum(fence.denied.values())
        result["fence_denials"] = dict(fence.denied)
        result["blocked_program_names"] = sorted(fence.programs)
        result["blocked_host_callers"] = sorted(fence.callers)
        server.stop()
        write_json(run_dir / "result.json", result)


def run(source: Path, work: Path, report_path: Path) -> int:
    """Run both isolated cases and save metadata only."""
    state = source_state(source)
    if not state["clean"]:
        raise ValueError("The Hermes checkout must be clean.")
    work.mkdir(parents=True, exist_ok=True)
    report = {"check": "custom Chat Completions fallback reply limit", "hermes": state,
              "python": sys.version.split()[0], "synthetic": True, "cases": [],
              "script_sha256": sha(Path(__file__).read_bytes()),
              "source_sha256": {name: sha((source / name).read_bytes())
                                for name in ("agent/plugin_llm.py", "agent/auxiliary_client.py")},
              "limits": "No live model, thinking behavior, summary quality, server cancellation, or speed check."}
    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as temp:
        for configured in (False, True):
            run_dir = Path(temp) / ("configured" if configured else "default")
            for folder in ("home", "runtime", "os-home", "tmp"):
                (run_dir / folder).mkdir(parents=True)
            args = [sys.executable, "-B", str(Path(__file__).resolve()), "--hermes-source", str(source),
                    "--worker", str(run_dir)]
            if configured:
                args.append("--configured")
            try:
                completed = subprocess.run(args, cwd=run_dir, env=worker_environment(run_dir),
                                           timeout=60, capture_output=True)
                path = run_dir / "result.json"
                case = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                    "configured": configured, "status": "failed", "error_class": "WorkerFailed"}
                case["worker_exit_code"] = completed.returncode
            except subprocess.TimeoutExpired:
                case = {"configured": configured, "status": "failed", "error_class": "TimeoutExpired"}
            report["cases"].append(case)
    default = report["cases"][0]
    report["default_limit_omitted"] = (default["status"] == "passed" and default.get("wire_max_tokens") is None
                                       and default.get("wire_max_completion_tokens") is None)
    report["status"] = "passed" if all(case["status"] == "passed" and case.get("worker_exit_code") == 0
                                       for case in report["cases"]) else "failed"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=ROOT / ".work" / "fallback-limit")
    parser.add_argument("--report", type=Path, default=ROOT / ".work" / "fallback-limit-report.json")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--configured", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.hermes_source.resolve(), args.worker.resolve(), args.configured)
        return 0
    return run(args.hermes_source.resolve(), args.work.resolve(), args.report.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
