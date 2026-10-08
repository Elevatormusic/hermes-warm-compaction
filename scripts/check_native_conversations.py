"""Check real native Hermes conversation loops with synthetic loopback replies.

Each case runs in a fresh process and isolated home. A second process loads the saved
session. No host code is patched or wrapped. Reports contain metadata only.
These checks do not prove live provider support, summary quality, or cache reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASES = ("responses_manual", "responses_auto_tool", "anthropic_manual", "anthropic_auto_tool")
MODEL = "synthetic-model"
KEY = "synthetic-api-key"
CONTEXT_LENGTH = 200_000
SID = "synthetic-native-session"


def mode_for(case):
    return "codex_responses" if case.startswith("responses_") else "anthropic_messages"


def source_state(source):
    from check_hermes_compatibility import source_state as inspect_source
    state = inspect_source(source)
    if not state["clean"]:
        raise ValueError("hermes_source_not_clean")
    return state


def hashes():
    from check_plugin_hermes import plugin_files_sha256
    return plugin_files_sha256()


def helper_hashes():
    names = ("check_provider_apis.py", "check_plugin_hermes.py", "check_hermes_compatibility.py")
    return {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() for name in names}


def native_server(case, reload_phase):
    """Reuse the native server. Change only the synthetic response plan."""
    from check_provider_apis import FakeServer, HANDOFF, MAIN_TEXT, TOOL_ID, TOOL_NAME

    class ConversationServer(FakeServer):
        def __init__(self):
            self.main_count = 0
            super().__init__("responses_text" if case.startswith("responses_") else "anthropic_text")

        def payload(self, kind):
            payload = super().payload(kind)
            if kind != "main":
                return payload
            self.main_count += 1
            tool = case.endswith("_auto_tool") and not reload_phase and self.main_count == 1
            prompt_tokens = 150_000 if tool else 2_000
            if mode_for(case) == "codex_responses":
                payload["usage"]["input_tokens"] = prompt_tokens
                payload["usage"]["total_tokens"] = prompt_tokens + 40
                if tool:
                    payload["output"] = [{"type": "function_call", "id": "fc_native", "call_id": TOOL_ID,
                                          "status": "completed", "name": TOOL_NAME, "arguments": "{}"}]
                else:
                    payload["output"][0]["content"][0]["text"] = MAIN_TEXT
            else:
                payload["usage"]["input_tokens"] = prompt_tokens - 120
                if tool:
                    payload["content"] = [{"type": "tool_use", "id": TOOL_ID, "name": TOOL_NAME, "input": {}}]
                    payload["stop_reason"] = "tool_use"
            return payload

    return ConversationServer(), HANDOFF, MAIN_TEXT


def row_content(rows):
    """Compare persisted public conversation fields. Ignore host database markers."""
    fields = ("role", "content", "tool_calls", "tool_call_id", "codex_reasoning_items",
              "codex_message_items", "reasoning_content", "phase", "reasoning_details", "anthropic_content_blocks")
    return [{key: row[key] for key in fields if key in row and row[key] is not None} for row in rows]


def check_warm_wire(server, checks):
    """The last real main request must be the exact warm prefix."""
    from check_provider_apis import HANDOFF, TOOL_ID
    warm = [r for r in server.requests if r["kind"] == "warm"]
    checks["one_native_warm_request"] = len(warm) == 1
    if not warm:
        return
    index = server.requests.index(warm[0])
    main = [r for r in server.requests[:index] if r["kind"] == "main"][-1]
    sent, earlier = warm[0]["body"], main["body"]
    field = "input" if "input" in earlier else "messages"
    checks["native_warm_prefix_exact"] = sent[field][:len(earlier[field])] == earlier[field]
    checks["native_warm_prompt_tools_exact"] = all(sent.get(key) == earlier.get(key)
                                                  for key in ("instructions", "system", "tools"))
    checks["native_warm_endpoint_exact"] = warm[0]["path"] == main["path"]
    header = "authorization" if field == "input" else "x-api-key"
    checks["native_auth_exact"] = main["headers"].get(header) == warm[0]["headers"].get(header) == (
        f"Bearer {KEY}" if field == "input" else KEY)
    checks["no_json_headers"] = "extra_headers" not in sent
    checks["no_fallback"] = not any(r["kind"] == "fallback" for r in server.requests)
    checks["handoff_not_in_main_output"] = HANDOFF not in json.dumps(earlier)
    if any(TOOL_ID in json.dumps(r["body"]) for r in server.requests[:index + 1]):
        if field == "input":
            calls = [x for x in sent[field] if x.get("type") == "function_call" and x.get("call_id") == TOOL_ID]
            outputs = [x for x in sent[field] if x.get("type") == "function_call_output"
                       and x.get("call_id") == TOOL_ID]
        else:
            blocks = [b for row in sent[field] if isinstance(row.get("content"), list) for b in row["content"]]
            calls = [b for b in blocks if b.get("type") == "tool_use" and b.get("id") == TOOL_ID]
            outputs = [b for b in blocks if b.get("type") == "tool_result" and b.get("tool_use_id") == TOOL_ID]
        checks["native_tool_pair_once"] = len(calls) == len(outputs) == 1
        checks["real_tool_result_replayed"] = len(outputs) == 1 and "The test note says BLUE-7." in json.dumps(outputs)


def worker(source, case, dependency_paths, phase):
    """Run the host loop. Do not call plugin hooks or middleware by hand."""
    sys.path[:0] = [str(source), *map(str, dependency_paths), str(ROOT / "scripts")]
    from check_plugin_hermes import Fence, merge_config, summary_rows_ok, synthetic_history
    home = Path(os.environ["HERMES_HOME"])
    scenario = home.parent
    installed = home / "plugins" / "warm_compaction"
    server, handoff, main_text = native_server(case, phase == "reload")
    server.thread.start()
    fence = Fence(scenario, server.httpd.server_port, source, installed)

    def early_network_fence(event, args):
        if event in {"socket.getaddrinfo", "socket.connect", "socket.connect_ex"}:
            fence(event, args)

    sys.addaudithook(early_network_fence)
    result = {"case": case, "phase": phase, "status": "failed", "checks": {}, "error": None}
    checks = result["checks"]
    db, agent = None, None
    try:
        merge_config(home / "config.yaml", {
            "model": {"default": MODEL, "model": MODEL, "provider": "custom", "base_url": server.base_url,
                      "api_key": KEY, "api_mode": mode_for(case), "context_length": CONTEXT_LENGTH,
                      "streaming": False},
            "context": {"engine": "warm_compaction"},
            "tools": {"tool_search": {"enabled": "off"}},
            "compression": {"progress_notices": False},
            "auxiliary": {"warm_compaction": {"provider": "auto", "model": MODEL},
                          "title_generation": {"enabled": False, "model_upgrade_enabled": False}},
            "plugins": {"enabled": ["warm_compaction", "wc_test_tools"], "hook_callback_timeout": 0,
                        "entries": {"warm_compaction": {"settings": {"threshold": 0.4, "tail_tokens": 1024}}}},
        })
        # A fresh synthetic catalogue prevents host discovery from making an external request.
        (home / "models_dev_cache.json").write_text(json.dumps({"synthetic": {"models": {}}}), encoding="utf-8")
        platform.uname()
        platform.platform()
        platform.architecture()
        from run_agent import AIAgent
        from hermes_state import SessionDB
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from tui_gateway.server import _compress_session_history
        from agent.transports import registered_api_modes
        from hermes_cli import plugins
        manager = plugins.get_plugin_manager()
        manager.discover_and_load()
        from tools.registry import registry
        from toolsets import resolve_toolset
        result["tool_discovery"] = {"registered": registry.get_entry("wc_note") is not None,
                                    "selected": "wc_note" in resolve_toolset("wc_test"),
                                    "same_scope": registry.current_scope_key() == manager.scope_key,
                                    "loaded": [{"name": p.manifest.name, "enabled": p.enabled,
                                                "error_present": bool(p.error)} for p in manager._plugins.values()
                                               if p.manifest.name == "wc_test_tools"]}
        available = mode_for(case) in registered_api_modes()
        result["native_capability"] = "supported" if available else "unavailable"
        if not available:
            result["error"] = "native_transport_unavailable"
            return result
        origin = Path(sys.modules[AIAgent.__module__].__file__).resolve()
        checks["agent_origin_is_pinned_source"] = origin.is_relative_to(source)
        sys.addaudithook(fence)
        db = SessionDB(home / "state.db")
        if phase == "run":
            db.create_session(SID, "cli")
            history = synthetic_history(20)
            for row in history:
                row["content"] = row["content"].strip()
            db.append_messages_batch(SID, history)
            session_id = SID
        else:
            session_id = (scenario / "session-id").read_text(encoding="utf-8")
        agent = AIAgent(base_url=server.base_url, api_key=KEY, provider="custom", api_mode=mode_for(case), model=MODEL,
                        enabled_toolsets=["wc_test"] if case.endswith("_auto_tool") else [], disabled_toolsets=[],
                        quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
                        capabilities={"vision": False}, session_id=session_id, session_db=db, max_tokens=4096,
                        reasoning_config={"enabled": False}, platform="cli", cwd=str(scenario), max_iterations=5)
        engine = agent.context_compressor
        checks["plugin_selected"] = getattr(engine, "name", None) == "warm_compaction"
        checks["plugin_origin_is_temporary_install"] = Path(
            sys.modules[type(engine).__module__].__file__).resolve().is_relative_to(installed)
        history = db.get_messages_as_conversation(session_id, repair_alternation=True)
        if phase == "reload":
            checks["reloaded_summary"] = summary_rows_ok(history)
            checks["reload_has_no_process_capture"] = engine._store.latest(session_id) is None
            before = json.loads((scenario / "saved-shape.json").read_text(encoding="utf-8"))
            checks["saved_native_rows_exact_after_process_reload"] = row_content(history) == before
        reply = agent.run_conversation("Continue the synthetic task.", system_message="Keep the code word BLUE-7.",
                                       conversation_history=history)
        messages = reply.get("messages") or []
        checks["real_conversation_final_reply"] = reply.get("final_response") == main_text
        checks["real_conversation_not_failed"] = not reply.get("failed")
        auto = case.endswith("_auto_tool") and phase == "run"
        if auto:
            checks["tool_registered"] = "wc_note" in agent.valid_tool_names
            checks["automatic_summary_rows"] = summary_rows_ok(messages)
            checks["actual_loop_request_order"] = [r["kind"] for r in server.requests] == ["main", "warm", "main"]
            checks["tool_result_retained"] = any(r.get("role") == "tool" and
                "The test note says BLUE-7." in str(r.get("content")) for r in messages)
        else:
            capture = engine._store.latest(engine._wc_session_id) or {}
            main = [r for r in server.requests if r["kind"] == "main"][-1]
            # Hermes adds stream=True for Responses after capture. All other fields must match.
            wire = dict(main["body"])
            if mode_for(case) == "codex_responses":
                checks["responses_stream_added_by_host"] = wire.pop("stream", None) is True
            checks["real_hook_capture_exact_wire"] = capture.get("body") == wire
            checks["real_post_hook_usage"] = capture.get("prompt_tokens") == 2_000
            checks["real_hook_capture_no_refusal"] = capture.get("refusal") is None
            session = {"agent": agent, "history": list(messages), "history_lock": threading.Lock(),
                       "history_version": 1, "session_key": agent.session_id}
            removed, _usage = _compress_session_history(session, "")
            finalize_context_engine_compression_notification(agent, committed=True)
            checks["manual_real_entry_result_valid"] = removed >= 0 if phase == "reload" else removed > 0
            checks["manual_real_entry_compression_count"] = engine.compression_count == 1
            checks["manual_real_entry_summary_rows"] = summary_rows_ok(session["history"])
            checks["manual_request_order"] = [r["kind"] for r in server.requests] == ["main", "warm"]
            messages = session["history"]
        outcome = engine.warm_last or {}
        result["outcome"] = {k: outcome.get(k) for k in ("path", "reason", "prompt_tokens", "cached_tokens")}
        checks["warm_accepted"] = (outcome.get("path"), outcome.get("reason")) == ("warm", "accepted")
        checks["synthetic_cached_counter"] = outcome.get("cached_tokens") == 120
        checks["synthetic_summary_in_history"] = any(handoff in str(r.get("content")) for r in messages)
        checks["engine_session_follows_host"] = engine._wc_session_id == agent.session_id
        check_warm_wire(server, checks)
        saved = db.get_messages_as_conversation(agent.session_id)
        checks["native_history_saved_exact"] = row_content(saved) == row_content(messages)
        if not checks["native_history_saved_exact"]:
            result["history_mismatch"] = [{"index": i, "keys": [key for key in set(a) | set(b)
                if a.get(key) != b.get(key)]} for i, (a, b) in enumerate(zip(row_content(saved), row_content(messages)))
                if a != b]
        result["fence_callers"] = sorted(fence.callers)
        if phase == "run":
            (scenario / "session-id").write_text(agent.session_id, encoding="utf-8")
            (scenario / "saved-shape.json").write_text(json.dumps(row_content(saved)), encoding="utf-8")
        result["request_counts"] = {kind: sum(r["kind"] == kind for r in server.requests)
                                    for kind in ("main", "warm", "fallback")}
        result["history_rows"] = len(saved)
        result["compression_count"] = engine.compression_count
        result["fence_denied"] = fence.denied
        checks["plugin_made_no_blocked_call"] = fence.plugin_denied == 0
        # Host startup can probe release metadata. The fence blocks each non-loopback attempt.
        result["blocked_host_network_attempts"] = sum(v for k, v in fence.denied.items() if k.startswith("network_"))
        checks["no_private_profile_read"] = "private_profile_read_blocked" not in fence.denied
        result["status"] = "passed" if checks and all(checks.values()) else "failed"
        return result
    finally:
        if agent is not None:
            agent.close()
        if db is not None:
            db.close()
        server.httpd.shutdown()
        server.httpd.server_close()


def run(source, cases, paths):
    """Discard synthetic histories and captured worker output when each case ends."""
    from check_hermes_compatibility import worker_environment
    from check_plugin_hermes import TOOL_PLUGIN
    report = {"check": "native full Hermes conversation loops", "python": sys.version.split()[0],
              "hermes": source_state(source), "synthetic": True, "live_provider": False,
              "cache_reuse_proved": False, "privacy": "metadata only", "cases": [], "error": None,
              "plugin_sha256": hashes(), "helper_sha256": helper_hashes(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="wc-native-loop-") as folder:
            scenario = Path(folder)
            for name in ("home", "runtime", "os-home", "tmp"):
                (scenario / name).mkdir()
            home = scenario / "home"
            shutil.copytree(ROOT / "warm_compaction", home / "plugins" / "warm_compaction",
                            ignore=shutil.ignore_patterns("__pycache__"))
            tool = home / "plugins" / "wc_test_tools"
            tool.mkdir()
            for name, content in TOOL_PLUGIN.items():
                (tool / name).write_text(content, encoding="utf-8")
            result = {"case": case, "phases": [], "passed": False}
            for phase in ("run", "reload"):
                target = scenario / f"{phase}.json"
                command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--hermes-source", str(source),
                           "--worker-case", case, "--phase", phase, "--report", str(target)]
                for path in paths:
                    command += ["--dependency-path", str(path)]
                completed = subprocess.run(command, cwd=scenario, env=worker_environment(home),
                                           capture_output=True, timeout=240)
                item = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {
                    "status": "failed", "error": "worker_no_report", "checks": {}}
                item["exit_ok"] = completed.returncode == 0
                result["phases"].append(item)
                if item["status"] != "passed" or not item["exit_ok"]:
                    break
            result["passed"] = len(result["phases"]) == 2 and all(
                p["status"] == "passed" and p["exit_ok"] for p in result["phases"])
            report["cases"].append(result)
            print(f"{case}: {'passed' if result['passed'] else 'failed'}", flush=True)
    report["hermes_source_unchanged"] = source_state(source) == report["hermes"]
    report["plugin_source_unchanged"] = hashes() == report["plugin_sha256"]
    report["helpers_unchanged"] = helper_hashes() == report["helper_sha256"]
    report["script_unchanged"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == report["script_sha256"]
    guards = ("hermes_source_unchanged", "plugin_source_unchanged", "script_unchanged", "helpers_unchanged")
    report["passed"] = all(report[k] for k in guards) and all(c["passed"] for c in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--dependency-path", type=Path, action="append", default=[])
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("run", "reload"), default="run", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source, paths = args.hermes_source.resolve(), [p.resolve() for p in args.dependency_path]
    try:
        report = worker(source, args.worker_case, paths, args.phase) if args.worker_case else run(
            source, args.case or CASES, paths)
    except Exception as error:
        code = "worker_runtime_failed" if args.worker_case else "setup_failed"
        report = {"passed": False, "status": "failed", "error": code,
                  "error_type": type(error).__name__, "checks": {},
                  "error_locations": [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                                      for f in traceback.extract_tb(error.__traceback__)[-3:]]}
    passed = report.get("passed", report.get("status") == "passed")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "error": report.get("error")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
