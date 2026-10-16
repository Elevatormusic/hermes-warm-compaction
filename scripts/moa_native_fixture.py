"""Synthetic two-role Codex MOA fixture for the real Hermes conversation loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from check_plugin_hermes import (
    CONTEXT_LENGTH, FALLBACK_MARK, HANDOFF, SYSTEM, WARM_MARK, Fence, FakeServer,
    merge_config, summary_rows_ok, synthetic_history,
)
from check_native_conversations import row_content

NATIVE_CASES = ("moa_native_manual", "moa_native_auto_tool", "moa_native_references_off", "moa_native_fallback")
MODELS = {"aggregator": "gpt-5.4", "reference-a": "gpt-5.4-mini"}
PRESET = "wc-native-moa"
KEY = "synthetic-codex-access-token"
ADVICE = "Synthetic native reference advice."
FINAL = "Synthetic native acting answer."


class NativeServer:
    """Keep invented wire bodies in memory. Send complete Responses SSE records."""

    def __init__(self, case):
        self.case, self.requests, self.main_count = case, [], 0
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.reply(200, "application/json", json.dumps({"object": "list", "data": [
                    {"id": model, "object": "model", "context_length": CONTEXT_LENGTH}
                    for model in MODELS.values()]}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                slot = next((name for name, model in MODELS.items() if body.get("model") == model), "fallback")
                text = json.dumps(body)
                kind = "warm" if WARM_MARK in text else "fallback" if FALLBACK_MARK in text else (
                    "main" if slot == "aggregator" else "reference")
                with owner.lock:
                    request = {"seq": len(owner.requests), "kind": kind, "slot": slot, "body": body,
                               "path": self.path,
                               "authorization_ok": self.headers.get("Authorization") == f"Bearer {KEY}"}
                    owner.requests.append(request)
                    if kind == "main":
                        owner.main_count += 1
                    payload = owner.payload(kind, slot)
                if owner.case == "moa_native_fallback" and kind == "warm" and slot == "aggregator":
                    self.reply(500, "application/json", b'{"error":{"message":"synthetic failure"}}')
                elif self.path.endswith("/chat/completions"):
                    message = {"role": "assistant", "content": HANDOFF + "\n[END OF SUMMARY]"}
                    usage = {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}
                    data = FakeServer.sse(message, "stop", usage) if body.get("stream") else (
                        FakeServer.json_reply(message, "stop", usage))
                    self.reply(200, "text/event-stream" if body.get("stream") else "application/json", data)
                else:
                    events = []
                    for index, item in enumerate(payload["output"]):
                        done = {"type": "response.output_item.done", "output_index": index, "item": item}
                        events.append("event: response.output_item.done\ndata: " + json.dumps(done) + "\n\n")
                    events.append("event: response.completed\ndata: "
                                  + json.dumps({"type": "response.completed", "response": payload}) + "\n\n")
                    self.reply(200, "text/event-stream", "".join(events).encode())

            def reply(self, status, content_type, data):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def payload(self, kind, slot):
        text = (FINAL if slot == "aggregator" else ADVICE) if kind in ("main", "reference") else (
            HANDOFF if slot == "aggregator" else HANDOFF.replace("BLUE-7", ADVICE))
        output = [{"type": "message", "id": "msg_synthetic", "role": "assistant", "status": "completed",
                   "phase": "final_answer", "content": [{"type": "output_text", "text": text, "annotations": []}]}]
        tool = self.case == "moa_native_auto_tool" and kind == "main" and self.main_count == 1
        if tool:
            output = [{"type": "function_call", "id": "fc_native", "call_id": "call_native",
                       "status": "completed", "name": "wc_note", "arguments": "{}"}]
        if kind in ("main", "reference"):
            output.insert(0, {"type": "reasoning", "id": f"rs_synthetic_{slot}",
                              "encrypted_content": f"synthetic-sealed-{slot}",
                              "summary": [{"type": "summary_text", "text": "Synthetic reasoning."}]})
        tokens = 125_000 if tool else 2_000
        return {"id": "resp_native", "object": "response", "created_at": 0, "model": MODELS.get(slot, "synthetic"),
                "status": "completed", "output": output, "incomplete_details": None,
                "usage": {"input_tokens": tokens, "output_tokens": 40, "total_tokens": tokens + 40,
                          "input_tokens_details": {"cached_tokens": 120}}}


def config(case, server):
    routes = [{"name": name, "provider": "openai-codex", "model": model, "base_url": server.base_url,
               "api_mode": "codex_responses", "context_length": CONTEXT_LENGTH} for name, model in MODELS.items()]
    return {
        "model": {"default": PRESET, "provider": "moa", "base_url": server.base_url,
                  "context_length": CONTEXT_LENGTH},
        "moa": {"default_preset": PRESET, "presets": {PRESET: {
            "reference_models": [{"provider": "openai-codex", "model": MODELS["reference-a"], "enabled": True}],
            "aggregator": {"provider": "openai-codex", "model": MODELS["aggregator"]}, "fanout": "user_turn"}}},
        "context": {"engine": "warm_compaction"},
        "tools": {"tool_search": {"enabled": "off"}},
        "auxiliary": {"transient_retries": 0, "title_generation": {"enabled": False},
                      "warm_compaction": {"provider": "custom", "model": "synthetic-fallback",
                                          "base_url": server.base_url, "api_key": "synthetic-fallback-key"}},
        "plugins": {"enabled": ["warm_compaction", "wc_test_tools"], "hook_callback_timeout": 0,
                    "entries": {"warm_compaction": {
                        "llm": {"allow_provider_override": True, "allow_model_override": True,
                                "allowed_providers": ["openai-codex"], "allowed_models": list(MODELS.values())},
                        "settings": {"threshold": 0.4, "moa_routes": routes,
                                     "moa_references": case != "moa_native_references_off"}}}},
    }


def worker(source, case, dependency_paths):
    """Use the real MOA facade, provider renderer, hooks, and plugin native dispatch."""
    sys.path[:0] = [str(source), *map(str, dependency_paths)]
    home = Path(os.environ["HERMES_HOME"])
    scenario = home.parent
    server = NativeServer(case)
    server.thread.start()
    os.environ["HERMES_CODEX_BASE_URL"] = server.base_url
    installed = home / "plugins" / "warm_compaction"
    platform.uname()
    fence = Fence(scenario, server.httpd.server_port, source, installed)

    def early_network_fence(event, args):
        if event in {"socket.getaddrinfo", "socket.connect", "socket.connect_ex"}:
            fence(event, args)

    sys.addaudithook(early_network_fence)
    result = {"case": case, "status": "failed", "checks": {}, "error": None}
    checks = result["checks"]
    db, agent = None, None
    try:
        merge_config(home / "config.yaml", config(case, server))
        from hermes_cli.auth_codex import _save_codex_tokens
        _save_codex_tokens({"access_token": KEY, "refresh_token": "synthetic-refresh-token"}, set_active=False)
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from hermes_cli.plugins import discover_plugins
        from hermes_state import SessionDB
        from run_agent import AIAgent
        from tui_gateway.server import _compress_session_history
        discover_plugins()
        sys.addaudithook(fence)
        db = SessionDB(home / "state.db")
        sid = f"synthetic-{case}"
        db.create_session(sid, "cli")
        db.append_messages_batch(sid, synthetic_history(20))
        agent = AIAgent(base_url=server.base_url, api_key="no-key-required", provider="moa",
                        api_mode="chat_completions", model=PRESET,
                        enabled_toolsets=["wc_test"] if case == "moa_native_auto_tool" else [], disabled_toolsets=[],
                        quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
                        capabilities={"vision": False}, session_id=sid, session_db=db, max_tokens=4096,
                        platform="cli", cwd=str(scenario), max_iterations=5,
                        stream_delta_callback=lambda *_args, **_kwargs: None)
        engine = agent.context_compressor
        result["engine"] = {"context_length": engine.context_length, "threshold_tokens": engine.threshold_tokens}
        if case == "moa_native_auto_tool":
            checks["test_tool_registered"] = "wc_note" in agent.valid_tool_names
            result["tool_selection"] = {"direct_tool_exposed": "wc_note" in agent.valid_tool_names,
                                        "search_bridge_exposed": "tool_search" in agent.valid_tool_names}
        checks["plugin_selected"] = getattr(engine, "name", None) == "warm_compaction"
        checks["plugin_origin_is_temporary_install"] = Path(
            sys.modules[type(engine).__module__].__file__).resolve().is_relative_to(installed)
        history = db.get_messages_as_conversation(sid)
        reply = agent.run_conversation("Call the note tool, then answer." if case == "moa_native_auto_tool"
                                       else "Continue the synthetic task.", system_message=SYSTEM,
                                       conversation_history=history)
        checks["real_loop_final_reply"] = reply.get("final_response") == FINAL
        messages = reply.get("messages") or []
        if case != "moa_native_auto_tool":
            session = {"agent": agent, "history": list(messages), "history_lock": threading.Lock(),
                       "history_version": 1, "session_key": agent.session_id}
            removed, _usage = _compress_session_history(session, "")
            finalize_context_engine_compression_notification(agent, committed=True)
            checks["manual_rows_removed"] = removed > 0
            messages = session["history"]
        outcome = engine.warm_last or {}
        result["outcome"] = {key: outcome.get(key) for key in ("path", "reason", "prompt_tokens", "cached_tokens")}
        warm = [request for request in server.requests if request["kind"] == "warm"]
        acting = [request for request in warm if request["slot"] == "aggregator"]
        refs = [request for request in warm if request["slot"] == "reference-a"]
        checks["one_native_acting_handoff"] = len(acting) == 1
        checks["native_reference_count"] = len(refs) == int(case != "moa_native_references_off")
        checks["references_before_acting_handoff"] = bool(acting) and all(r["seq"] < acting[0]["seq"] for r in refs)
        expected = (("fallback", "moa_native_dispatch_refused") if case == "moa_native_fallback"
                    else ("warm", "accepted"))
        checks["expected_compaction_outcome"] = (outcome.get("path"), outcome.get("reason")) == expected
        checks["summary_rows"] = summary_rows_ok(messages)
        checks["acting_summary_retained"] = any(HANDOFF in str(row.get("content") or "") for row in messages)
        checks["reference_advice_not_saved_as_summary"] = not any(
            ADVICE in str(row.get("content") or "") for row in messages if row.get("_compressed_summary"))
        for request in warm:
            label = request["slot"]
            previous = [r for r in server.requests[:request["seq"]] if r["slot"] == label
                        and r["kind"] in ("main", "reference")][-1]
            before, after = previous["body"], request["body"]
            checks[f"{label}_native_prefix_exact"] = after["input"][:len(before["input"])] == before["input"]
            checks[f"{label}_native_settings_exact"] = (
                {key: value for key, value in before.items() if key != "input"}
                == {key: value for key, value in after.items() if key != "input"})
            checks[f"{label}_native_endpoint_auth_exact"] = request["path"] == previous["path"] and (
                request["authorization_ok"] and previous["authorization_ok"])
            checks[f"{label}_native_encrypted_reasoning_replayed"] = any(
                item.get("type") == "reasoning" and item.get("encrypted_content") == f"synthetic-sealed-{label}"
                for item in after["input"][len(before["input"]):])
            if case == "moa_native_auto_tool" and label == "aggregator":
                calls = [item for item in after["input"] if item.get("type") == "function_call"
                         and item.get("call_id") == "call_native"]
                outputs = [item for item in after["input"] if item.get("type") == "function_call_output"
                           and item.get("call_id") == "call_native"]
                checks["native_tool_pair_once"] = len(calls) == len(outputs) == 1
                checks["real_tool_result_replayed"] = len(outputs) == 1 and (
                    "The test note says BLUE-7." in str(outputs[0].get("output")))
                if outputs and not checks["real_tool_result_replayed"]:
                    output = outputs[0].get("output")
                    text = str(output).lower()
                    result["tool_output_metadata"] = {
                        "type": type(output).__name__, "chars": len(str(output)),
                        "failure_class": "invalid_tool_name" if "does not exist. available tools:" in text else None,
                        "markers": {mark: mark in text for mark in (
                            "unknown tool", "not found", "not available", "blocked", "error", "pending", "deferred")},
                    }
        saved = db.get_messages_as_conversation(agent.session_id)
        checks["saved_history_exact"] = row_content(saved) == row_content(messages)
        checks["plugin_made_no_blocked_call"] = fence.plugin_denied == 0
        checks["no_private_profile_read"] = "private_profile_read_blocked" not in fence.denied
        result["requests"] = [{"seq": r["seq"], "kind": r["kind"], "slot": r["slot"],
                               "stream": r["body"].get("stream"), "input_count": len(r["body"].get("input") or [])}
                              for r in server.requests]
        result["fence_denied"] = fence.denied
        result["status"] = "passed" if checks and all(checks.values()) else "failed"
        return result
    finally:
        if agent is not None:
            agent.close()
        if db is not None:
            db.close()
        server.httpd.shutdown()
        server.httpd.server_close()
