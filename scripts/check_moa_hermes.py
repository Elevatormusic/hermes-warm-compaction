"""Check real Hermes MOA with synthetic conversations and loopback replies.

Each case uses a fresh process and a temporary home. No host code is patched.
Reports contain metadata only. These checks do not prove live providers or cache reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_plugin_hermes import (
    CONTEXT_LENGTH, FALLBACK_MARK, FALLBACK_SUMMARY, HANDOFF, LOOPBACK, MODEL,
    NEXT_QUESTION, QUESTIONS, SEED_QUESTION, SYSTEM, TOOL_AGAIN_QUESTION,
    TOOL_QUESTION, WARM_MARK, Fence, by_kind, completion, contains_goal,
    merge_config, plan_for as solo_plan, summary_rows_ok, synthetic_history,
)
from check_native_conversations import source_state
from moa_native_fixture import NATIVE_CASES

ROOT = Path(__file__).resolve().parents[1]
CASES = ("moa_manual", "moa_references_off", "moa_auto_tool_loop", "moa_fallback", "moa_unconfigured")
COMPATIBILITY_CASES = ("moa_old_auto_tool_refusal",)
MOA_PRESET = "wc-moa"
MOA_SLOTS = ("aggregator", "reference-a", "reference-b")
MOA_MODELS = {name: f"fake-{name}" for name in MOA_SLOTS}
MOA_REFERENCE_MARKS = {name: f"Synthetic advice from {name}." for name in MOA_SLOTS[1:]}


def plan_for(case):
    plan = solo_plan("auto_tool_loop" if case in ("moa_auto_tool_loop", *COMPATIBILITY_CASES)
                     else "enable_manual_warm")
    plan["moa"] = True
    if case == "moa_fallback":
        plan["warm_status_by_slot"] = {"aggregator": 500}
    return plan


TOOL_PLUGIN = {
    "plugin.yaml": (
        "manifest_version: 2\nname: wc_test_tools\nversion: 0.0.1\n"
        "description: \"Test tool for the warm_compaction integration check.\"\nkind: standalone\n"
        "provides_hooks:\n  - pre_auxiliary_call\n  - post_auxiliary_call\n"
    ),
    "__init__.py": (
        '"""Test tool plugin for the warm_compaction integration check."""\n\nimport json\n\nEVENTS = []\n\n\n'
        "def _loss_paths(value, path='request'):\n"
        "    if isinstance(value, dict):\n"
        "        return [item for key, child in value.items()\n"
        "                for item in ([path + '.' + key] if key in ('_truncated', '_truncated_items')\n"
        "                             else _loss_paths(child, path + '.' + key))]\n"
        "    if isinstance(value, list):\n"
        "        return [item for index, child in enumerate(value)\n"
        "                for item in _loss_paths(child, path + '[' + str(index) + ']')]\n"
        "    return [path] if isinstance(value, str) and (' depth limit>' in value\n"
        "                                                or '...[truncated ' in value) else []\n\n\n"
        "def _event(phase, values):\n"
        "    if values.get('aux_task') in ('moa_reference', 'moa_aggregator'):\n"
        "        keys = ('aux_task', 'session_id', 'turn_id', 'api_request_id', 'retry_count', 'provider', 'model')\n"
        "        EVENTS.append({'phase': phase, **{key: values.get(key) for key in keys},\n"
        "                       'preview_loss_paths': [p for p in _loss_paths(values.get('request'))\n"
        "                                              if not p.startswith('request.body.messages[')]})\n\n\n"
        "def _note(args, **_kwargs):\n"
        '    return json.dumps({"note": "The test note says BLUE-7."})\n\n\n'
        "def register(ctx):\n"
        '    ctx.register_tool(name="wc_note", toolset="wc_test", handler=_note, schema={\n'
        '        "name": "wc_note", "description": "Return a fixed test note.",\n'
        '        "parameters": {"type": "object", "properties": {}, "required": []}})\n'
        "    ctx.register_hook('pre_auxiliary_call', lambda **kwargs: _event('pre', kwargs))\n"
        "    ctx.register_hook('post_auxiliary_call', lambda **kwargs: _event('post', kwargs))\n"
    ),
}


def classify(body) -> str:
    """Hermes rebuilds the system prompt after a compaction, so the last user question marks a main request."""
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list) or not messages:
        return "other"
    first, last = messages[0], messages[-1]
    if first.get("role") == "system" and str(first.get("content") or "").startswith(FALLBACK_MARK):
        return "fallback"
    if last.get("role") == "user" and str(last.get("content") or "").startswith(WARM_MARK):
        return "warm"
    if body.get("model") in [MOA_MODELS[name] for name in MOA_SLOTS[1:]]:
        return "reference"
    if body.get("model") == MOA_MODELS["aggregator"]:
        return "main"
    users = [row for row in messages if row.get("role") == "user"]
    if users and str(users[-1].get("content") or "") in QUESTIONS:
        return "main"
    return "other"


class FakeServer:
    """Loopback chat-completions server. It records each request and answers from a plan."""

    def __init__(self, plan):
        self.plan = plan
        self.main_replies = list(plan.get("main", []))
        self.requests = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                data = json.dumps({"object": "list", "data": [
                    {"id": model, "object": "model", "context_length": CONTEXT_LENGTH}
                    for model in (MODEL, *MOA_MODELS.values())]}).encode("utf-8")
                self.reply(200, "application/json", data)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = {}
                status, payload = server.answer(body, self.path, self.headers.get("Authorization", ""))
                if status != 200:
                    self.reply(status, "application/json", json.dumps({"error": {"message": "fake error"}}).encode())
                elif body.get("stream"):
                    self.reply(200, "text/event-stream", server.sse(*payload, model=body.get("model", MODEL)))
                else:
                    self.reply(200, "application/json", server.json_reply(*payload, model=body.get("model", MODEL)))

            def reply(self, status, content_type, data):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer((LOOPBACK, 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.httpd.server_port}/v1"

    def slot_url(self, name):
        return f"http://{LOOPBACK}:{self.httpd.server_port}/{name}/v1"

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def answer(self, body, path="/v1/chat/completions", authorization=""):
        kind = classify(body)
        messages = body.get("messages") or []
        size = len(json.dumps(messages)) // 4
        with self.lock:
            slot = next((name for name, model in MOA_MODELS.items() if body.get("model") == model), "main")
            entry = {"seq": len(self.requests), "kind": kind, "body": body, "status": 200,
                     "slot": slot, "path": path, "authorization_ok": authorization == f"Bearer fake-{slot}-key"}
            self.requests.append(entry)
            if kind == "warm":
                entry["status"] = self.plan.get("warm_status_by_slot", {}).get(slot, self.plan.get("warm_status", 200))
                text = HANDOFF if slot not in MOA_REFERENCE_MARKS else HANDOFF.replace(
                    "The code word is BLUE-7.", MOA_REFERENCE_MARKS[slot])
                cached = self.plan.get("cached", None if self.plan.get("moa") else size // 2)
                payload = completion(text, prompt_tokens=size, cached_tokens=cached)
            elif kind == "fallback":
                entry["status"] = self.plan.get("fallback_status", 200)
                payload = completion(FALLBACK_SUMMARY, prompt_tokens=size)
            elif kind == "main":
                step = self.main_replies.pop(0) if self.main_replies else {"content": "OK."}
                payload = completion(step.get("content"), step.get("tool"),
                                     prompt_tokens=step.get("prompt_tokens", size))
            elif kind == "reference":
                payload = completion(MOA_REFERENCE_MARKS[slot], prompt_tokens=size)
            else:
                payload = completion("Synthetic title", prompt_tokens=size)
        return entry["status"], payload

    @staticmethod
    def json_reply(message, finish, usage, model=MODEL) -> bytes:
        return json.dumps({"id": "fake", "object": "chat.completion", "created": 0, "model": model,
                           "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                           "usage": usage}).encode("utf-8")

    @staticmethod
    def sse(message, finish, usage, model=MODEL) -> bytes:
        def event(choices, extra=None):
            data = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": model,
                    "choices": choices, **(extra or {})}
            return b"data: " + json.dumps(data).encode("utf-8") + b"\n\n"

        delta = {"role": "assistant"}
        if message.get("content") is not None:
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(message["tool_calls"])]
        parts = [event([{"index": 0, "delta": delta, "finish_reason": None}]),
                 event([{"index": 0, "delta": {}, "finish_reason": finish}]),
                 event([], {"usage": usage}), b"data: [DONE]\n\n"]
        return b"".join(parts)


def moa_config(scenario, server):
    """Use explicit local routes. An empty reference list would select the Hermes defaults."""
    routes = [{"name": name, "provider": "custom", "model": MOA_MODELS[name],
               "base_url": server.slot_url(name), "api_key_env": slot_key_env(name),
               "context_length": CONTEXT_LENGTH, "api_mode": "chat_completions"} for name in MOA_SLOTS]
    if scenario == "moa_unconfigured":
        routes = [route for route in routes if route["name"] != "aggregator"]
    references = [{"provider": f"custom:wc-{name}", "model": MOA_MODELS[name], "enabled": True}
                  for name in MOA_SLOTS[1:]]
    providers = {f"wc-{name}": {"base_url": server.slot_url(name), "key_env": slot_key_env(name),
                                "api_mode": "chat_completions", "context_length": CONTEXT_LENGTH}
                 for name in MOA_SLOTS}
    return {
        "model": {"default": MOA_PRESET, "provider": "moa", "context_length": CONTEXT_LENGTH},
        "providers": providers,
        "moa": {"default_preset": MOA_PRESET, "presets": {MOA_PRESET: {
            "reference_models": references,
            "aggregator": {"provider": "custom:wc-aggregator", "model": MOA_MODELS["aggregator"]},
            "fanout": "user_turn", "reference_temperature": 0.25, "aggregator_temperature": 0.35}}},
        "auxiliary": {"warm_compaction": {"provider": "custom:wc-aggregator", "model": MOA_MODELS["aggregator"]}},
        "plugins": {"hook_callback_timeout": 0, "entries": {"warm_compaction": {"settings": {
            "moa_routes": routes, "moa_references": scenario != "moa_references_off"}}}},
    }


def slot_key_env(name):
    return "WC_FAKE_" + name.upper().replace("-", "_") + "_KEY"


def check_moa_request(server, request, checks, label, *, tool_rows=0):
    """Compare the warm request with the last physical request for that destination."""
    earlier = [r for r in server.requests if r["slot"] == request["slot"]
               and r["kind"] in ("main", "reference") and r["seq"] < request["seq"]]
    checks[f"{label}_capture_exists"] = bool(earlier)
    if not earlier:
        return
    previous = earlier[-1]
    sent, extended = previous["body"]["messages"], request["body"]["messages"]
    checks[f"{label}_prefix_exact"] = sent == extended[:len(sent)]
    checks[f"{label}_instruction_last"] = str(extended[-1].get("content") or "").startswith(WARM_MARK)
    checks[f"{label}_not_streamed"] = request["body"].get("stream") is False and "stream_options" not in request["body"]
    original = {key: value for key, value in previous["body"].items()
                if key not in {"messages", "stream", "stream_options"}}
    renewed = {key: value for key, value in request["body"].items()
               if key not in {"messages", "stream", "stream_options"}}
    controls = {"max_tokens", "max_completion_tokens", "stop", "web_search_options"}
    checks[f"{label}_settings_except_handoff_controls_exact"] = (
        {key: value for key, value in original.items() if key not in controls}
        == {key: value for key, value in renewed.items() if key not in controls})
    expected_limits = {key: min(max(original[key], 2048), 8192)
                       for key in ("max_tokens", "max_completion_tokens")
                       if type(original.get(key)) is int and original[key] > 0}
    if not expected_limits:
        expected_limits = {"max_tokens": 8192}
    checks[f"{label}_handoff_controls_exact"] = (
        {key: value for key, value in renewed.items() if key in controls} == expected_limits)
    if not hasattr(server, "settings_metadata"):
        server.settings_metadata = {}
    server.settings_metadata[label] = {
        "all_settings_exact": original == renewed,
        "difference_keys": sorted(key for key in set(original) | set(renewed)
                                  if original.get(key) != renewed.get(key)),
    }
    roles = [row.get("role") for row in extended[len(sent):]]
    checks[f"{label}_suffix_roles"] = roles == ["assistant", *["tool"] * tool_rows, "user"]
    if tool_rows:
        calls = extended[len(sent)].get("tool_calls") or []
        answers = extended[len(sent) + 1:-1]
        checks[f"{label}_tool_ids_match"] = [r.get("tool_call_id") for r in answers] == [c.get("id") for c in calls]


def check_moa_compaction(server, start, status, checks, label, scenario, *, tool_rows=0):
    """Check request order and each physical prefix. Save booleans, counts, and codes only."""
    attempts = server.requests[start:]
    if scenario in ("moa_auto_tool_loop", *COMPATIBILITY_CASES):
        end = next((index for index, item in enumerate(attempts)
                    if item["kind"] in ("main", "reference")), len(attempts))
        attempts = attempts[:end]
    warm = [r for r in attempts if r["kind"] == "warm"]
    aggregator = [r for r in warm if r["slot"] == "aggregator"]
    references = [r for r in warm if r["slot"] in MOA_SLOTS[1:]]
    if scenario == "moa_old_auto_tool_refusal":
        checks[f"{label}_safe_fallback_path"] = status.get("path") == "fallback"
        checks[f"{label}_safe_refusal_reason"] = status.get("reason") == "moa_payload_incomplete"
        checks[f"{label}_no_partial_warm"] = not warm
        checks[f"{label}_fallback_once"] = sum(r["kind"] == "fallback" for r in attempts) == 1
        return
    if scenario in ("moa_fallback", "moa_unconfigured"):
        checks[f"{label}_fallback_path"] = status.get("path") == "fallback"
        checks[f"{label}_fallback_once"] = len([r for r in attempts if r["kind"] == "fallback"]) == 1
        checks[f"{label}_aggregator_attempts"] = len(aggregator) == int(scenario == "moa_fallback")
        if scenario == "moa_fallback":
            checks[f"{label}_provider_error"] = status.get("reason") == "provider_error"
        else:
            checks[f"{label}_route_unconfigured"] = status.get("reason") == "moa_route_unconfigured"
            checks[f"{label}_no_partial_warm"] = not warm
        return
    checks[f"{label}_warm_accepted"] = (status.get("path"), status.get("reason")) == ("warm", "accepted")
    checks[f"{label}_aggregator_once"] = len(aggregator) == 1
    expected = [] if scenario == "moa_references_off" else list(MOA_SLOTS[1:])
    checks[f"{label}_reference_destinations"] = sorted(r["slot"] for r in references) == expected
    checks[f"{label}_no_normal_fanout"] = not any(r["kind"] == "reference" for r in attempts)
    checks[f"{label}_no_fallback"] = not any(r["kind"] == "fallback" for r in attempts)
    checks[f"{label}_references_before_aggregator"] = bool(aggregator) and all(
        r["seq"] < aggregator[0]["seq"] for r in references)
    for request in references:
        check_moa_request(server, request, checks, f"{label}_{request['slot']}")
    if aggregator:
        check_moa_request(server, aggregator[0], checks, f"{label}_aggregator", tool_rows=tool_rows)
        text = str(aggregator[0]["body"]["messages"][-1].get("content") or "")
        checks[f"{label}_reference_notes_in_instruction"] = all(MOA_REFERENCE_MARKS[name] in text for name in expected)
        if not expected:
            checks[f"{label}_no_reference_notes"] = not any(mark in text for mark in MOA_REFERENCE_MARKS.values())
    checks[f"{label}_cached_tokens_unknown"] = status.get("cached_tokens") is None
    slots = status.get("moa_slots") or []
    acting = [slot for slot in slots if slot.get("role") == "aggregator"]
    advice = [slot for slot in slots if slot.get("role") == "reference" and slot.get("included")]
    checks[f"{label}_aggregator_status"] = len(acting) == 1 and (
        acting[0].get("name"), acting[0].get("path"), acting[0].get("reason")) == ("aggregator", "warm", "accepted")
    checks[f"{label}_reference_status"] = sorted(slot.get("name") for slot in advice) == expected and all(
        (slot.get("path"), slot.get("reason")) == ("warm", "accepted") for slot in advice)


def run_moa(agent, engine, db, history, server, result, checks, scenario, compress, finalize):
    """Run real Hermes MoA with synthetic data and three local HTTP destinations."""
    if scenario in ("moa_auto_tool_loop", *COMPATIBILITY_CASES):
        checks["threshold_from_settings"] = engine.threshold_tokens == int(CONTEXT_LENGTH * 0.4)
        for index, (question, answer) in enumerate(((TOOL_QUESTION, "Done."),
                                                    (TOOL_AGAIN_QUESTION, "Done again.")), 1):
            start = len(server.requests)
            turn = agent.run_conversation(question, system_message=SYSTEM, conversation_history=list(history))
            checks[f"auto{index}_final"] = turn.get("final_response") == answer
            current = server.requests[start:]
            # Normal fan-out belongs to the user turn. Compaction starts after its streamed tool-call reply.
            kinds = ("warm", "fallback") if scenario in COMPATIBILITY_CASES else ("warm",)
            first_warm = next((r["seq"] for r in current if r["kind"] in kinds), len(server.requests))
            status = dict(engine.warm_last or {})
            result[f"warm_last_{index}"] = status
            check_moa_compaction(server, first_warm, status, checks, f"auto{index}", scenario, tool_rows=1)
            history = turn.get("messages") or []
            checks[f"auto{index}_summary_rows"] = summary_rows_ok(history)
            checks[f"auto{index}_wait_flag_cleared"] = engine.awaiting_real_usage_after_compression is False
        checks["two_compactions"] = engine.compression_count == 2
        if scenario in COMPATIBILITY_CASES:
            checks["continuation_sends_summary"] = any(
                str(row.get("content") or "").startswith("[CONTEXT SUMMARY]:")
                for request in by_kind(server, "main")[1:] for row in request["body"].get("messages") or [])
        else:
            checks["continuation_sends_summary"] = any(contains_goal(r) for r in by_kind(server, "main")[1:])
        result["compression_count"] = engine.compression_count
    else:
        seed = agent.run_conversation(SEED_QUESTION, system_message=SYSTEM, conversation_history=history)
        checks["seed_turn"] = len(seed.get("messages") or []) == len(history) + 2
        if scenario == "moa_manual":
            check_moa_session_isolation(agent, db, server, result, checks, compress, finalize)
        start = len(server.requests)
        session = {"agent": agent, "history": list(seed.get("messages") or []),
                   "history_lock": threading.Lock(), "history_version": 1, "session_key": agent.session_id}
        removed, _usage = compress(session, "")
        finalize(agent, committed=True)
        status = dict(engine.warm_last or {})
        result["warm_last"] = status
        checks["removed_rows"] = removed > 0
        checks["summary_rows"] = summary_rows_ok(session["history"])
        check_moa_compaction(server, start, status, checks, "manual", scenario)
        if status.get("path") == "warm":
            checks["acting_summary_saved"] = HANDOFF in str(session["history"][1].get("content") or "")
            checks["reference_summary_not_saved"] = not any(
                mark in str(row.get("content") or "") for row in session["history"]
                for mark in MOA_REFERENCE_MARKS.values())
        checks["engine_session_follows"] = engine._wc_session_id == agent.session_id
        count = len(server.requests)
        turn = agent.run_conversation(NEXT_QUESTION, system_message=SYSTEM,
                                      conversation_history=list(session["history"]))
        checks["continuation_reply"] = bool(str(turn.get("final_response") or "").strip())
        checks["continuation_sends_summary"] = any(
            str(row.get("content") or "").startswith("[CONTEXT SUMMARY]:")
            for request in server.requests[count:] if request["kind"] == "main"
            for row in request["body"].get("messages") or [])
        history = turn.get("messages") or []
    saved = db.get_messages_as_conversation(agent.session_id)
    checks["history_saved"] = [str(r.get("content") or "").strip() for r in saved] == [
        str(r.get("content") or "").strip() for r in history]
    checks["aggregator_streamed"] = all(r["body"].get("stream") is True for r in by_kind(server, "main"))
    checks["normal_references_both_routes"] = set(r["slot"] for r in by_kind(server, "reference")) == set(MOA_SLOTS[1:])
    result["history_rows"] = {"in_memory": len(history), "stored": len(saved)}
    result["final_session_id"] = agent.session_id
    # The fake server omits cached-token counters. It cannot prove cache reuse.
    result["cache_evidence"] = "fake server; cache reuse not tested"
    status_text = json.dumps({key: value for key, value in result.items() if key.startswith("warm_last")})
    checks["status_has_no_message_text_or_key"] = not any(
        text in status_text for text in (HANDOFF, *MOA_REFERENCE_MARKS.values(),
                                        *[f"fake-{name}-key" for name in MOA_SLOTS]))
    events = importlib.import_module("hermes_plugins.wc_test_tools").EVENTS
    result["auxiliary_observer"] = events


def helper_hashes():
    names = ("check_plugin_hermes.py", "check_native_conversations.py", "check_hermes_compatibility.py",
             "moa_native_fixture.py")
    return {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() for name in names}


def plugin_hashes(plugin_source):
    return {path.relative_to(plugin_source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((plugin_source / "warm_compaction").rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def worker(source, case, dependency_paths):
    """Run the real host loop. Keep synthetic request bodies in process memory only."""
    if case in NATIVE_CASES:
        from moa_native_fixture import worker as native_worker
        return native_worker(source, case, dependency_paths)
    sys.path[:0] = [str(source), *map(str, dependency_paths)]
    home = Path(os.environ["HERMES_HOME"])
    scenario = home.parent
    installed = home / "plugins" / "warm_compaction"
    server = FakeServer(plan_for(case))
    server.start()
    platform.uname()
    fence = Fence(scenario, server.httpd.server_port, source, installed)

    def early_network_fence(event, args):
        if event in {"socket.getaddrinfo", "socket.connect", "socket.connect_ex"}:
            fence(event, args)

    sys.addaudithook(early_network_fence)
    result = {"case": case, "python": sys.version.split()[0], "status": "failed", "checks": {}, "error": None}
    checks = result["checks"]
    db, agent = None, None
    try:
        merge_config(home / "config.yaml", {
            "context": {"engine": "warm_compaction"},
            "auxiliary": {"transient_retries": 0, "title_generation": {"enabled": False}},
            "plugins": {"enabled": ["warm_compaction", "wc_test_tools"],
                        "entries": {"warm_compaction": {"settings": {"threshold": 0.4}}}},
        })
        merge_config(home / "config.yaml", moa_config(case, server))
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from hermes_cli.plugins import discover_plugins
        from hermes_state import SessionDB
        from run_agent import AIAgent
        from tui_gateway.server import _compress_session_history

        discover_plugins()
        checks["plugin_discovery_before_moa_threads"] = True
        sys.addaudithook(fence)
        db = SessionDB(home / "state.db")
        sid = f"synthetic-{case}"
        db.create_session(sid, "cli")
        db.append_messages_batch(sid, synthetic_history(20))
        agent = AIAgent(base_url=server.base_url, api_key="no-key-required", provider="moa",
                        api_mode="chat_completions", model=MOA_PRESET,
                        enabled_toolsets=["wc_test"] if case in ("moa_auto_tool_loop", *COMPATIBILITY_CASES) else [],
                        disabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
                        skip_background_review=True, capabilities={"vision": False}, session_id=sid,
                        session_db=db, max_tokens=1024, platform="cli", cwd=str(scenario), max_iterations=5,
                        stream_delta_callback=lambda *_args, **_kwargs: None)
        engine = agent.context_compressor
        checks["plugin_selected"] = getattr(engine, "name", None) == "warm_compaction"
        checks["plugin_origin_is_temporary_install"] = Path(
            sys.modules[type(engine).__module__].__file__).resolve().is_relative_to(installed)
        history = db.get_messages_as_conversation(sid)
        run_moa(agent, engine, db, history, server, result, checks, case,
                _compress_session_history, finalize_context_engine_compression_notification)
        checks["physical_destinations_match_models"] = all(
            r["path"] == f"/{r['slot']}/v1/chat/completions" and r["authorization_ok"]
            for r in server.requests if r["slot"] in MOA_SLOTS)
        checks["plugin_made_no_blocked_call"] = fence.plugin_denied == 0
        checks["no_private_profile_read"] = "private_profile_read_blocked" not in fence.denied
        result["fence_denied"] = fence.denied
        result["settings_metadata"] = getattr(server, "settings_metadata", {})
        result["requests"] = [{"seq": r["seq"], "kind": r["kind"], "slot": r["slot"], "status": r["status"],
                               "stream": bool(r["body"].get("stream")),
                               "message_count": len(r["body"].get("messages") or [])} for r in server.requests]
        result["status"] = "passed" if checks and all(checks.values()) else "failed"
        return result
    finally:
        if agent is not None:
            agent.close()
        if db is not None:
            db.close()
        server.stop()


def run(source, cases, paths, plugin_source):
    """Discard synthetic histories and captured worker output after each case."""
    from check_hermes_compatibility import worker_environment

    report = {"check": "MOA full Hermes conversation loops", "python": sys.version.split()[0],
              "hermes": source_state(source), "synthetic": True, "live_provider": False,
              "cache_reuse_proved": False, "privacy": "metadata only", "cases": [], "error": None,
              "plugin_sha256": plugin_hashes(plugin_source), "helper_sha256": helper_hashes(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="wc-moa-loop-") as folder:
            scenario = Path(folder)
            home = scenario / "home"
            home.mkdir()
            shutil.copytree(plugin_source / "warm_compaction", home / "plugins" / "warm_compaction",
                            ignore=shutil.ignore_patterns("__pycache__"))
            tool = home / "plugins" / "wc_test_tools"
            tool.mkdir()
            for name, content in TOOL_PLUGIN.items():
                (tool / name).write_text(content, encoding="utf-8")
            target = scenario / "result.json"
            command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--hermes-source", str(source),
                       "--worker-case", case, "--report", str(target)]
            for path in paths:
                command += ["--dependency-path", str(path)]
            env = worker_environment(home)
            env.update({slot_key_env(name): f"fake-{name}-key" for name in MOA_SLOTS})
            completed = subprocess.run(command, cwd=scenario, env=env, capture_output=True, timeout=240)
            item = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {
                "case": case, "status": "failed", "error": "worker_no_report", "checks": {}}
            item["exit_ok"] = completed.returncode == 0
            report["cases"].append(item)
            print(f"{case}: {item['status']}", flush=True)
    report["checks"] = {
        "hermes_source_unchanged": source_state(source) == report["hermes"],
        "plugin_source_unchanged": plugin_hashes(plugin_source) == report["plugin_sha256"],
        "helpers_unchanged": helper_hashes() == report["helper_sha256"],
        "script_unchanged": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == report["script_sha256"],
    }
    report["passed"] = all(report["checks"].values()) and all(
        item["status"] == "passed" and item["exit_ok"] for item in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--dependency-path", type=Path, action="append", default=[])
    parser.add_argument("--plugin-source", type=Path, default=ROOT,
                        help="Use a separate plugin source for a counterfactual check.")
    parser.add_argument("--case", choices=(*CASES, *NATIVE_CASES, *COMPATIBILITY_CASES), action="append")
    parser.add_argument("--native", action="store_true", help="Also require the native two-role MOA checks.")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-case", choices=(*CASES, *NATIVE_CASES, *COMPATIBILITY_CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    source, paths = args.hermes_source.resolve(), [p.resolve() for p in args.dependency_path]
    try:
        selected = args.case or ((*CASES, *NATIVE_CASES) if args.native else CASES)
        report = worker(source, args.worker_case, paths) if args.worker_case else run(
            source, selected, paths, args.plugin_source.resolve())
    except Exception as error:
        report = {"passed": False, "status": "failed", "error": "worker_runtime_failed" if args.worker_case
                  else "setup_failed", "error_type": type(error).__name__, "checks": {},
                  "error_locations": [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                                      for f in traceback.extract_tb(error.__traceback__)[-3:]]}
    passed = report.get("passed", report.get("status") == "passed")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "error": report.get("error")}))
    return 0 if passed else 1


def check_moa_session_isolation(agent, db, server, result, checks, compress, finalize):
    """A second agent must not replay the first agent's capture without its own completed turn."""
    from run_agent import AIAgent

    sid = "wc-moa-isolation-no-capture"
    db.create_session(sid, "cli")
    history = synthetic_history(20)
    history[0]["content"] = history[0]["content"].replace("BLUE-7", "GREEN-9")
    db.append_messages_batch(sid, history)
    other = AIAgent(base_url=server.base_url, api_key="no-key-required", provider="moa",
                    api_mode="chat_completions", model=MOA_PRESET, enabled_toolsets=[], disabled_toolsets=[],
                    quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
                    capabilities={"vision": False}, session_id=sid, session_db=db, max_tokens=1024,
                    platform="cli", cwd=agent.session_cwd,
                    stream_delta_callback=lambda *_args, **_kwargs: None)
    session = {"agent": other, "history": db.get_messages_as_conversation(sid),
               "history_lock": threading.Lock(), "history_version": 1, "session_key": sid}
    start = len(server.requests)
    removed, _usage = compress(session, "")
    finalize(other, committed=True)
    status = dict(other.context_compressor.warm_last or {})
    result["isolation_path"] = status.get("path")
    result["isolation_reason"] = status.get("reason")
    checks["isolated_session_no_capture"] = (status.get("path"), status.get("reason")) == ("fallback", "no_capture")
    checks["isolated_session_sends_no_warm_request"] = not any(r["kind"] == "warm" for r in server.requests[start:])
    checks["isolated_session_fallback_once"] = sum(r["kind"] == "fallback" for r in server.requests[start:]) == 1
    checks["isolated_session_saved"] = removed > 0 and [row.get("content") for row in session["history"]] == [
        row.get("content") for row in db.get_messages_as_conversation(other.session_id)]


if __name__ == "__main__":
    raise SystemExit(main())
