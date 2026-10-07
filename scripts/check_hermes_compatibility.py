"""Check the plugin capture against a clean Hermes middleware implementation.

Each case uses a fresh process and temporary home. No model or network request runs.
The report contains versions, hashes, check results, and fixed error codes only.
This check does not prove conversation-loop hook order or provider wire behavior.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CASES = ("capture_last", "earlier_rewrite", "later_rewrite", "unknown_order")
WORKER_TIMEOUT = 60
SOURCE_MARKERS = ("hermes_cli/plugins.py", "hermes_cli/middleware.py", "agent/context_engine.py", "run_agent.py")


def plugin_hashes() -> dict[str, str]:
    """Hash the plugin source files. Do not include paths outside the plugin."""
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "warm_compaction").iterdir())
            if path.is_file() and path.suffix in (".py", ".yaml")}


def source_state(source: Path) -> dict:
    """Get the exact source commit and reject a dirty checkout."""
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("invalid_commit")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("dirty_hermes_source")
    if not all((source / name).is_file() for name in SOURCE_MARKERS):
        raise ValueError("invalid_hermes_source")
    return {"commit": commit, "clean": True}


def check_import_origins(source: Path) -> None:
    """Refuse Hermes modules from an installed runtime or another source root."""
    for name, module in tuple(sys.modules.items()):
        if name == "warm_compaction.capture":
            if Path(module.__file__).resolve() != (ROOT / "warm_compaction" / "capture.py").resolve():
                raise ValueError("source_mismatch")
            continue
        if name not in ("hermes_cli", "agent", "hermes_constants", "registration_lifecycle", "utils") and (
                not name.startswith(("hermes_cli.", "agent."))):
            continue
        path = getattr(module, "__file__", None)
        if path is None or not Path(path).resolve().is_relative_to(source):
            raise ValueError("source_mismatch")


def worker_environment(home: Path) -> dict[str, str]:
    """Keep the runtime paths only. Do not pass credentials or profile settings."""
    names = ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC", "SYSTEMDRIVE",
             "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    env = {key: value for key, value in os.environ.items() if key.upper() in names}
    env.update({key: str(home) for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
                                          "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "TEMP", "TMP")})
    env.update({"HERMES_HOME": str(home), "HERMES_RUNTIME_DIR": str(home / "runtime"),
                "HERMES_DISABLE_LAZY_INSTALLS": "1", "HERMES_ENABLE_PROJECT_PLUGINS": "0",
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def worker(source: Path, case: str) -> dict:
    """Use the real Hermes manager, registrations, hooks, and middleware chain."""
    # Windows can use a system process for this standard-library value. Cache it before the fence.
    platform.uname()
    def no_network(event, _args):
        if event in ("socket.connect", "socket.connect_ex", "socket.bind", "socket.getaddrinfo",
                     "subprocess.Popen", "os.system"):
            raise RuntimeError("external_operation_blocked")
    sys.addaudithook(no_network)
    sys.path[:0] = [str(ROOT), str(source)]
    try:
        from hermes_cli import plugins
        from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
        from warm_compaction.capture import CaptureStore
    except ImportError as error:
        raise ValueError("hermes_import_failed") from error
    check_import_origins(source)

    manager = plugins.get_plugin_manager()
    if not isinstance(manager, PluginManager):
        raise RuntimeError("manager_type_changed")
    manager.discover_and_load()
    context = PluginContext(PluginManifest(name="wc_compatibility_probe", version="0.0.0"), manager)
    store = CaptureStore()
    context.register_hook("pre_api_request", store.on_pre_api_request)
    context.register_hook("post_api_request", store.on_post_api_request)

    def request_rewrite(request, **_kwargs):
        return {"request": {**request, "temperature": 0.25}, "source": "synthetic_probe"}

    def execution_rewrite(request, next_call, **_kwargs):
        changed = copy.deepcopy(request)
        changed["extra_body"]["synthetic_marker"] = "execution_rewrite"
        return next_call(changed)

    def unknown_capture(**kwargs):
        # The registered callback cannot be identified as the store callback.
        return store.on_llm_execution(**kwargs)

    context.register_middleware("llm_request", request_rewrite)
    if case == "earlier_rewrite":
        context.register_middleware("llm_execution", execution_rewrite)
    context.register_middleware("llm_execution", unknown_capture if case == "unknown_order"
                                else store.on_llm_execution)
    if case == "later_rewrite":
        context.register_middleware("llm_execution", execution_rewrite)

    history = [{"role": "user", "content": "Synthetic probe question."}]
    request = {"model": "synthetic-model", "messages": history, "max_tokens": 128,
               "extra_body": {"synthetic_marker": "initial"},
               "extra_headers": {"x-opencode-session": "synthetic-session-header"}}
    runtime = {"api_request_id": "synthetic-request", "session_id": "synthetic-session",
               "model": "synthetic-model", "base_url": "https://example.invalid/v1",
               "api_mode": "chat_completions"}
    applied = apply_llm_request_middleware(request, **runtime)
    plugins.invoke_hook("pre_api_request", conversation_history=history, **runtime)
    calls = []
    response = object()

    def terminal(payload):
        calls.append(copy.deepcopy(payload))
        return response

    returned = run_llm_execution_middleware(applied.payload, terminal,
                                            original_request=applied.original_payload, **runtime)
    plugins.invoke_hook("post_api_request", finish_reason="stop",
                        assistant_message=SimpleNamespace(content="Synthetic reply.", tool_calls=[]),
                        usage={"prompt_tokens": 123}, **runtime)
    capture = store.latest(runtime["session_id"]) or {}
    expected_refusal = {"later_rewrite": "middleware_after_capture",
                        "unknown_order": "middleware_order_unknown"}.get(case)
    marker = "execution_rewrite" if case in ("earlier_rewrite", "later_rewrite") else "initial"
    expected_headers = {"x-opencode-session": "synthetic-session-header"}
    expected_body = {"model": "synthetic-model", "messages": history, "max_tokens": 128,
                     "temperature": 0.25, "synthetic_marker": marker}
    expected_kwargs = {"model": "synthetic-model", "messages": history, "max_tokens": 128,
                       "temperature": 0.25, "extra_body": {"synthetic_marker": marker},
                       "extra_headers": expected_headers}
    checks = {"terminal_once": len(calls) == 1, "response_unchanged": returned is response,
              "terminal_payload_exact": bool(calls) and calls[0] == expected_kwargs,
              "request_middleware_applied": bool(calls) and calls[0].get("temperature") == 0.25,
              "request_header_forwarded": bool(calls) and calls[0].get("extra_headers") == expected_headers,
              "reported_prompt_tokens": capture.get("prompt_tokens") == 123,
              "refusal_expected": capture.get("refusal") == expected_refusal}
    if expected_refusal:
        checks["capture_body_refused"] = capture.get("body") is None
        checks["capture_headers_refused"] = capture.get("request_headers") == {}
    else:
        checks["capture_body_exact"] = capture.get("body") == expected_body
        checks["capture_headers_exact"] = capture.get("request_headers") == expected_headers
    if case in ("earlier_rewrite", "later_rewrite"):
        checks["execution_rewrite_reaches_terminal"] = bool(calls) and (
            calls[0].get("extra_body", {}).get("synthetic_marker") == "execution_rewrite")
    check_import_origins(source)
    checks["import_origins_match_source"] = True
    return {"case": case, "passed": all(checks.values()), "checks": checks,
            "error": None if all(checks.values()) else "contract_check_failed"}


def run(source: Path) -> dict:
    """Run each contract case in a separate process. Discard raw child output."""
    report = {"check": "real Hermes middleware capture compatibility", "passed": False,
              "python": sys.version.split()[0], "hermes": None, "plugin_sha256": {},
              "script_sha256": None, "cases": [], "error": None}
    try:
        report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["hermes"] = source_state(source)
        report["plugin_sha256"] = plugin_hashes()
        for case in CASES:
            with tempfile.TemporaryDirectory(prefix="wc-compatibility-") as folder:
                home = Path(folder)
                (home / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
                result_path = home / "result.json"
                completed = subprocess.run(
                    [sys.executable, "-B", str(Path(__file__).resolve()), "--hermes-source", str(source),
                     "--worker-case", case, "--report", str(result_path)], cwd=home,
                    env=worker_environment(home), capture_output=True, timeout=WORKER_TIMEOUT)
                result = (json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists()
                          else {"case": case, "passed": False, "checks": {}, "error": "worker_no_report"})
                result["exit_ok"] = completed.returncode == 0
                report["cases"].append(result)
        report["checks"] = {"hermes_source_unchanged": source_state(source) == report["hermes"],
                            "plugin_source_unchanged": plugin_hashes() == report["plugin_sha256"],
                            "script_unchanged": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                            == report["script_sha256"]}
        report["passed"] = all(report["checks"].values()) and all(
            item["passed"] and item["exit_ok"] for item in report["cases"])
        if not report["passed"]:
            report["error"] = "compatibility_check_failed"
    except Exception as error:
        code = error.args[0] if type(error) is ValueError and error.args else None
        report["error"] = code if code in (
            "dirty_hermes_source", "invalid_commit", "invalid_hermes_source") else "probe_setup_failed"
        report["error_type"] = type(error).__name__
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_case:
        try:
            report = worker(args.hermes_source.resolve(), args.worker_case)
        except Exception as error:
            code = error.args[0] if type(error) is ValueError and error.args else None
            report = {"case": args.worker_case, "passed": False, "checks": {},
                      "error": code if code in ("hermes_import_failed", "source_mismatch") else "worker_runtime_failed",
                      "error_type": type(error.__cause__ or error).__name__}
    else:
        report = run(args.hermes_source.resolve())
    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            print(json.dumps({"passed": False, "error": "report_write_failed", "error_type": type(error).__name__}))
            return 1
    print(json.dumps({"passed": report["passed"], "error": report["error"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
