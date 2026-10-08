"""Check native provider requests with real Hermes transports and a loopback fake server.

Run with a Hermes dependency interpreter and a clean Hermes source checkout. Each case has a fresh
process and home. The plugin loads from a copied plugin folder through normal Hermes discovery.
Only invented history, fake keys, and fake token counters are used. The report contains metadata only.
This check does not prove a live provider, cache reuse, summary quality, or a speed gain.
"""

from __future__ import annotations

import argparse
import copy
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("responses_text", "responses_tool", "responses_incomplete",
         "responses_codex", "responses_codex_reasoning", "anthropic_text", "anthropic_tool",
         "anthropic_refusal", "anthropic_signed")
MODEL = "synthetic-model"
KEY = "synthetic-api-key"
SYSTEM = "You are a synthetic test agent. Keep the code word BLUE-7."
MAIN_TEXT = "Synthetic main answer."
TOOL_NAME = "wc_note"
TOOL_ID = "call_new"
HANDOFF = (
    "## Goal\nFinish the synthetic task.\n\n"
    "## User instructions\n- Keep the code word BLUE-7.\n\n"
    "## Current state\n- [OPEN] Report the code word.\n\n"
    "## Key facts\n- The code word is BLUE-7.\n\n"
    "## Next step\nReport the code word BLUE-7."
)
TOOLS = [{"type": "function", "function": {"name": TOOL_NAME, "description": "Return a synthetic note.",
          "parameters": {"type": "object", "properties": {}, "required": []}}}]
WARM_MARK = "Stop the current task now. This request comes from the host program"
FALLBACK_MARK = "Write a handoff summary of the conversation transcript"


def mode_for(case):
    return "codex_responses" if case.startswith("responses_") else "anthropic_messages"


def model_for(case):
    return "claude-sonnet-4-6" if case == "anthropic_signed" else MODEL


def logical_url(case, server):
    """Official routes select the real renderer. Their test sends are redirected to loopback."""
    if "codex" in case:
        return "https://chatgpt.com/backend-api/codex"
    return "https://api.anthropic.com" if case == "anthropic_signed" else server.base_url


def plugin_hashes():
    """Hash only the public plugin files that each worker receives."""
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "warm_compaction").glob("*")) if p.is_file()}


def history_for(case):
    """Include a complete earlier tool pair in the tool cases."""
    rows = []
    for index in range(6):
        rows += [{"role": "user", "content": f"Synthetic question {index}. " + ("Detail text. " * 40).rstrip()},
                 {"role": "assistant", "content": f"Synthetic answer {index}. " + ("Answer text. " * 40).rstrip()}]
    if case.endswith("_tool"):
        rows += [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_old", "type": "function",
                  "function": {"name": TOOL_NAME, "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "call_old", "content": "Synthetic earlier result."},
                 {"role": "assistant", "content": "Synthetic tool round complete."}]
    rows.append({"role": "user", "content": "Continue the synthetic task."})
    if case == "responses_codex_reasoning":
        rows[1]["codex_reasoning_items"] = [{
            "type": "reasoning", "id": "rs_synthetic_old", "encrypted_content": "synthetic-sealed-bytes",
            "summary": [{"type": "summary_text", "text": "Synthetic reasoning."}],
            "_issuer_kind": "codex_backend", "_issuer_model": MODEL}]
        rows[1]["codex_message_items"] = [{
            "type": "message", "id": "msg_synthetic_old", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": rows[1]["content"]}], "phase": "final_answer"}]
    elif case == "anthropic_signed":
        rows[1]["reasoning_details"] = [{"type": "thinking", "thinking": "Synthetic earlier thought.",
                                          "signature": "synthetic-old-signature"}]
    return rows


class FakeServer:
    """Serve native protocols. Keep raw synthetic bodies in memory only."""

    def __init__(self, case):
        self.case, self.requests = case, []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                text = json.dumps(body)
                kind = "warm" if WARM_MARK in text else "fallback" if FALLBACK_MARK in text else "main"
                owner.requests.append({"kind": kind, "path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                payload = owner.payload(kind)
                if self.path.endswith("/chat/completions"):
                    payload = {"id": "synthetic", "object": "chat.completion", "created": 0, "model": MODEL,
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant",
                                                        "content": HANDOFF + "\n[END OF SUMMARY]"}}],
                               "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}}
                if body.get("stream") and mode_for(owner.case) == "codex_responses":
                    event = "response.incomplete" if payload["status"] == "incomplete" else "response.completed"
                    parts = []
                    if owner.case == "responses_codex" and kind == "warm":
                        # The consumer route can send completed items before an empty terminal envelope.
                        for index, item in enumerate(payload["output"]):
                            done = {"type": "response.output_item.done", "output_index": index, "item": item}
                            parts.append("event: response.output_item.done\ndata: " + json.dumps(done) + "\n\n")
                        payload = {**payload, "output": None}
                    parts.append(f"event: {event}\ndata: "
                                 + json.dumps({"type": event, "response": payload}) + "\n\n")
                    data = "".join(parts).encode()
                    content_type = "text/event-stream"
                elif body.get("stream") and mode_for(owner.case) == "anthropic_messages":
                    start = {**payload, "content": [], "stop_reason": None, "stop_sequence": None}
                    events = [("message_start", {"type": "message_start", "message": start})]
                    for index, block in enumerate(payload["content"]):
                        events.extend([
                            ("content_block_start", {"type": "content_block_start", "index": index,
                                                     "content_block": {"type": "text", "text": ""}}),
                            ("content_block_delta", {"type": "content_block_delta", "index": index,
                                                     "delta": {"type": "text_delta", "text": block["text"]}}),
                            ("content_block_stop", {"type": "content_block_stop", "index": index})])
                    events.extend([
                        ("message_delta", {"type": "message_delta", "delta": {
                            "stop_reason": payload["stop_reason"], "stop_sequence": None},
                            "usage": {"output_tokens": 40}}),
                        ("message_stop", {"type": "message_stop"})])
                    data = "".join(f"event: {event}\ndata: {json.dumps(value)}\n\n"
                                   for event, value in events).encode()
                    content_type = "text/event-stream"
                else:
                    data, content_type = json.dumps(payload).encode(), "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self):
        root = f"http://127.0.0.1:{self.httpd.server_port}"
        return root + "/v1" if mode_for(self.case) == "codex_responses" else root

    def payload(self, kind):
        is_tool = kind == "main" and self.case.endswith("_tool")
        text = MAIN_TEXT if kind == "main" else HANDOFF
        if kind == "fallback":
            text += "\n[END OF SUMMARY]"
        if mode_for(self.case) == "codex_responses":
            output = [{"type": "message", "id": "msg_synthetic", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]}]
            if is_tool:
                output = [{"type": "function_call", "id": "fc_new", "call_id": TOOL_ID,
                           "status": "completed", "name": TOOL_NAME, "arguments": "{}"}]
            if kind == "main" and self.case == "responses_codex_reasoning":
                output.insert(0, {"type": "reasoning", "id": "rs_synthetic_new",
                                  "encrypted_content": "synthetic-new-sealed-bytes",
                                  "summary": [{"type": "summary_text", "text": "Synthetic new reasoning."}]})
            incomplete = kind == "warm" and self.case.endswith("_incomplete")
            return {"id": "resp_synthetic", "object": "response", "created_at": 0, "model": MODEL,
                    "status": "incomplete" if incomplete else "completed", "output": output,
                    "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
                    "usage": {"input_tokens": 200, "output_tokens": 40, "total_tokens": 240,
                              "input_tokens_details": {"cached_tokens": 120}}}
        refusal = kind == "warm" and self.case.endswith("_refusal")
        content = ([{"type": "tool_use", "id": TOOL_ID, "name": TOOL_NAME, "input": {}}] if is_tool
                   else [{"type": "text", "text": text}])
        if kind == "main" and self.case == "anthropic_signed":
            content.insert(0, {"type": "thinking", "thinking": "Synthetic new thought.",
                               "signature": "synthetic-new-signature"})
        return {"id": "msg_synthetic", "type": "message", "role": "assistant", "model": model_for(self.case),
                "content": content, "stop_reason": "refusal" if refusal else "tool_use" if is_tool else "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 80, "output_tokens": 40,
                                                   "cache_read_input_tokens": 120, "cache_creation_input_tokens": 0}}


def worker(source, case, dependency_paths):
    """Build and send a main request through real Hermes hooks and middleware."""
    sys.path[:0] = [str(source), *map(str, dependency_paths), str(ROOT / "scripts")]
    from check_plugin_hermes import Fence, merge_config
    home = Path(os.environ["HERMES_HOME"])
    installed = home / "plugins" / "warm_compaction"
    server = FakeServer(case)
    server.thread.start()
    fence = Fence(home.parent, server.httpd.server_port, source, installed)

    def early_network_fence(event, args):
        # Block external network access during import and host capability discovery too.
        if event in {"socket.getaddrinfo", "socket.connect", "socket.connect_ex"}:
            fence(event, args)

    sys.addaudithook(early_network_fence)
    checks, result = {}, {"case": case, "checks": {}, "error": None}
    try:
        mode = mode_for(case)
        model, route_url = model_for(case), logical_url(case, server)
        # The native packet includes the exact replay carrier. Give that packet enough tail room.
        tail_tokens = 512 if case in {"responses_codex_reasoning", "anthropic_signed"} else 128
        merge_config(home / "config.yaml", {
            "model": {"provider": "custom", "model": model, "base_url": server.base_url,
                      "api_key": KEY, "api_mode": mode},
            "context": {"engine": "warm_compaction"},
            "compression": {"progress_notices": False},
            "auxiliary": {"warm_compaction": {"provider": "auto", "model": MODEL}},
            "plugins": {"enabled": ["warm_compaction"], "hook_callback_timeout": 0,
                        "entries": {"warm_compaction": {"settings": {"tail_tokens": tail_tokens}}}},
        })
        platform.uname()
        platform.platform()
        platform.architecture()
        from hermes_cli import plugins
        from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        from agent.transports.anthropic import AnthropicTransport
        from agent.transports.codex import ResponsesApiTransport
        manager = plugins.get_plugin_manager()
        manager.discover_and_load()
        engine = plugins.get_plugin_context_engine()
        # Host capability discovery can call system tools. Finish it before the request fence.
        checks["plugin_discovered"] = getattr(engine, "name", None) == "warm_compaction"
        checks["plugin_origin_is_temporary_install"] = engine is not None and Path(
            sys.modules[type(engine).__module__].__file__).resolve().is_relative_to(installed)
        checks["transport_origin_is_pinned_source"] = all(
            Path(sys.modules[transport.__module__].__file__).resolve().is_relative_to(source)
            for transport in (AnthropicTransport, ResponsesApiTransport))
        redirected = []
        if route_url != server.base_url:
            def loopback_post(url, data, headers, timeout_s, context=None):
                from urllib.parse import urlsplit
                from importlib import import_module
                expected_path = "/backend-api/codex/responses" if "codex" in case else "/v1/messages"
                if urlsplit(url).path != expected_path or not url.startswith(route_url):
                    raise ValueError("unexpected_logical_endpoint")
                redirected.append(True)
                warm = import_module(type(engine).__module__.rsplit(".", 1)[0] + ".warm")
                path = "/responses" if mode == "codex_responses" else "/v1/messages"
                return warm.urllib_post(server.base_url + path, data, headers, timeout_s)
            engine = type(engine)(store=engine._store, llm=engine._llm,
                                  settings={"tail_tokens": tail_tokens}, post=loopback_post)
        engine.on_session_start("synthetic-session", platform="cli")
        engine.update_model(model=model, context_length=200_000, base_url=route_url, api_key=KEY,
                            provider="custom", api_mode=mode)
        transport = ResponsesApiTransport() if mode == "codex_responses" else AnthropicTransport()
        history = history_for(case)
        tools = copy.deepcopy(TOOLS) if case.endswith("_tool") else []
        kwargs = transport.build_kwargs(
            model=model, messages=[{"role": "system", "content": SYSTEM}, *history], tools=tools,
            base_url=route_url, provider="custom", max_tokens=4096, is_codex_backend="codex" in case,
            session_id="synthetic-session", context_length=200_000,
            reasoning_config={"enabled": case.endswith(("_reasoning", "_signed")), "effort": "low"})
        if mode == "codex_responses":
            kwargs = transport.preflight_kwargs(kwargs, allow_stream=True, sanitize_harmony_tokens="codex" in case)
        headers = {"x-opencode-session": "synthetic-session-header"}
        if mode == "anthropic_messages":
            headers.update({"anthropic-beta": "synthetic-beta", "anthropic-version": "2023-06-01"})
        kwargs["extra_headers"] = {**kwargs.get("extra_headers", {}), **headers}
        runtime = {"api_request_id": "synthetic-main-request", "session_id": "synthetic-session",
                   "model": model, "base_url": route_url, "api_mode": mode}
        if mode == "codex_responses":
            from openai import OpenAI
            client = OpenAI(base_url=server.base_url, api_key=KEY, max_retries=0)
        else:
            from anthropic import Anthropic
            client = Anthropic(base_url=server.base_url, api_key=KEY, max_retries=0)
        _ = client.default_headers
        sys.addaudithook(fence)
        def terminal(body):
            return client.responses.create(**body) if mode == "codex_responses" else client.messages.create(**body)
        applied = apply_llm_request_middleware(kwargs, **runtime)
        plugins.invoke_hook("pre_api_request", conversation_history=history, **runtime)
        response = run_llm_execution_middleware(applied.payload, terminal,
                                               original_request=applied.original_payload, **runtime)
        normalized = transport.normalize_response(response)
        plugins.invoke_hook("post_api_request", finish_reason=normalized.finish_reason,
                            assistant_message=normalized, usage={"prompt_tokens": 200}, **runtime)
        capture = engine._store.latest("synthetic-session") or {}
        main = next(row for row in server.requests if row["kind"] == "main")
        checks["main_capture_exact_wire_body"] = capture.get("body") == main["body"]
        checks["capture_has_no_refusal"] = capture.get("refusal") is None
        endpoint = "/v1/responses" if mode == "codex_responses" else "/v1/messages"
        checks["main_native_endpoint"] = main["path"] == endpoint
        checks["main_protocol_auth"] = (main["headers"].get("authorization") == f"Bearer {KEY}"
                                        if mode == "codex_responses" else main["headers"].get("x-api-key") == KEY)
        checks["main_request_headers_sent"] = all(main["headers"].get(k) == v for k, v in headers.items())
        if mode == "codex_responses":
            checks["main_instructions_exact"] = main["body"].get("instructions") == SYSTEM
        else:
            checks["main_system_exact"] = main["body"].get("system") == SYSTEM
        row = {"role": "assistant", "content": normalized.content or ""}
        row.update(normalized.provider_data or {})
        if normalized.tool_calls:
            row["tool_calls"] = [{"id": tc.id, "type": "function", "function": {
                "name": tc.name, "arguments": tc.arguments}} for tc in normalized.tool_calls]
        history.append(row)
        if normalized.tool_calls:
            history.append({"role": "tool", "tool_call_id": TOOL_ID, "content": "Synthetic new result."})
        else:
            history.append({"role": "user", "content": "Keep the synthetic next step."})
        compacted = engine.compress(history)
        outcome = dict(engine.warm_last or {})
        result["outcome"] = {k: outcome.get(k) for k in ("path", "reason", "prompt_tokens", "cached_tokens")}
        failure = case.endswith(("_incomplete", "_refusal"))
        checks["warm_or_fallback_outcome"] = outcome.get("path") == ("fallback" if failure else "warm")
        checks["history_compacted"] = compacted is not history and len(compacted) < len(history)
        checks["summary_has_synthetic_handoff"] = any(HANDOFF in str(r.get("content")) for r in compacted)
        if not failure:
            checks["synthetic_token_counters"] = (
                outcome.get("prompt_tokens") == 200 and outcome.get("cached_tokens") == 120)
        carrier = "codex_reasoning_items" if case == "responses_codex_reasoning" else "reasoning_details"
        if case in {"responses_codex_reasoning", "anthropic_signed"}:
            checks["latest_native_carrier_retained_exact"] = bool(row.get(carrier)) and any(
                candidate.get(carrier) == row[carrier] for candidate in compacted)
        warm_rows = [r for r in server.requests if r["kind"] == "warm"]
        checks["warm_sent_once"] = len(warm_rows) == 1
        if warm_rows:
            sent = warm_rows[0]
            body, earlier = sent["body"], main["body"]
            field = "input" if mode == "codex_responses" else "messages"
            checks["warm_native_endpoint"] = sent["path"] == main["path"]
            checks["warm_prefix_exact"] = body[field][:len(earlier[field])] == earlier[field]
            checks["warm_prompt_and_tools_exact"] = all(
                body.get(k) == earlier.get(k) for k in ("instructions", "system", "tools", "prompt_cache_key"))
            checks["warm_request_headers_kept"] = all(sent["headers"].get(k) == v for k, v in headers.items())
            checks["warm_protocol_auth"] = (sent["headers"].get("authorization") == f"Bearer {KEY}"
                                            if mode == "codex_responses" else sent["headers"].get("x-api-key") == KEY)
            checks["request_headers_not_json"] = "extra_headers" not in body
            if "codex" in case:
                checks["consumer_codex_typed_parts"] = all(
                    isinstance(item["content"], list) for item in body["input"] if item.get("type") == "message")
                checks["consumer_codex_stream_required"] = body.get("stream") is True
                checks["consumer_codex_session_header"] = sent["headers"].get("session_id") == "synthetic-session"
            if case == "responses_codex_reasoning":
                items = [item for item in body["input"] if item.get("type") == "reasoning"]
                checks["old_and_new_encrypted_reasoning_exact"] = [item.get("encrypted_content") for item in items] == [
                    "synthetic-sealed-bytes", "synthetic-new-sealed-bytes"]
                checks["reasoning_local_metadata_not_sent"] = all(not any(
                    key.startswith("_") or key == "id" for key in item) for item in items)
            if case == "anthropic_signed":
                thinking = [block for msg in body["messages"] if isinstance(msg["content"], list)
                            for block in msg["content"] if block.get("type") == "thinking"]
                checks["old_and_new_thinking_signatures_exact"] = [block.get("signature") for block in thinking] == [
                    "synthetic-old-signature", "synthetic-new-signature"]
            if case.endswith("_tool"):
                if mode == "codex_responses":
                    ids = [(x.get("type"), x.get("call_id")) for x in body[field]]
                    checks["old_and_new_tool_pairs"] = all(ids.count((kind, call_id)) == 1
                        for kind in ("function_call", "function_call_output") for call_id in ("call_old", TOOL_ID))
                    calls = [item for item in body[field] if item.get("type") == "function_call"
                             and item.get("call_id") == TOOL_ID]
                    outputs = [item for item in body[field] if item.get("type") == "function_call_output"
                               and item.get("call_id") == TOOL_ID]
                    checks["new_tool_payloads_exact"] = (
                        len(calls) == len(outputs) == 1 and calls[0].get("arguments") == "{}"
                        and outputs[0].get("output") == "Synthetic new result.")
                else:
                    blocks = [block for msg in body[field] if isinstance(msg["content"], list)
                              for block in msg["content"]]
                    checks["old_and_new_tool_pairs"] = all(
                        sum(block.get("type") == kind and block.get(key) == call_id for block in blocks) == 1
                        for kind, key in (("tool_use", "id"), ("tool_result", "tool_use_id"))
                        for call_id in ("call_old", TOOL_ID))
                    calls = [block for block in blocks if block.get("type") == "tool_use"
                             and block.get("id") == TOOL_ID]
                    outputs = [block for block in blocks if block.get("type") == "tool_result"
                               and block.get("tool_use_id") == TOOL_ID]
                    checks["new_tool_payloads_exact"] = (
                        len(calls) == len(outputs) == 1 and calls[0].get("input") == {}
                        and outputs[0].get("content") == "Synthetic new result.")
        checks["failure_reason_expected"] = outcome.get("reason") == (
            "incomplete_response" if case.endswith("_incomplete")
            else "gate:finish_not_stop" if failure else "accepted")
        checks["fallback_only_on_failure"] = bool([r for r in server.requests if r["kind"] == "fallback"]) == failure
        checks["plugin_made_no_blocked_call"] = fence.plugin_denied == 0
        checks["no_external_network_attempt"] = not any(reason.startswith("network_") for reason in fence.denied)
        result["fence_denied"] = fence.denied
        result["fence_callers"] = sorted(fence.callers)
        result["fence_programs"] = sorted(fence.programs)
        if route_url != server.base_url:
            checks["official_logical_route_redirected_once"] = len(redirected) == 1
        result["request_counts"] = {kind: sum(r["kind"] == kind for r in server.requests)
                                    for kind in ("main", "warm", "fallback")}
        client.close()
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()
    result["checks"] = checks
    result["passed"] = bool(checks) and all(checks.values())
    return result


def run(source, cases, dependency_paths):
    """Run cases in clean processes. Discard raw output and temporary synthetic data."""
    from check_hermes_compatibility import source_state, worker_environment
    report = {"check": "native provider transport and warm loopback check", "python": sys.version.split()[0],
              "hermes": source_state(source), "synthetic": True, "live_provider": False,
              "cache_reuse_proved": False, "cases": [], "error": None,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "plugin_sha256": plugin_hashes()}
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="wc-provider-") as folder:
            scenario = Path(folder)
            home = scenario / "home"
            shutil.copytree(ROOT / "warm_compaction", home / "plugins" / "warm_compaction",
                            ignore=shutil.ignore_patterns("__pycache__"))
            result_path = scenario / "result.json"
            command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--hermes-source", str(source),
                       "--worker-case", case, "--report", str(result_path)]
            for path in dependency_paths:
                command += ["--dependency-path", str(path)]
            completed = subprocess.run(command, cwd=scenario, env=worker_environment(home),
                                       capture_output=True, timeout=180)
            result = json.loads(result_path.read_text()) if result_path.exists() else {
                "case": case, "passed": False, "error": "worker_no_report", "checks": {}}
            result["exit_ok"] = completed.returncode == 0
            report["cases"].append(result)
            print(f"{case}: {'passed' if result['passed'] else 'failed'}", flush=True)
    report["hermes_source_unchanged"] = source_state(source) == report["hermes"]
    report["plugin_source_unchanged"] = plugin_hashes() == report["plugin_sha256"]
    report["script_unchanged"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == report["script_sha256"]
    report["passed"] = all(report[key] for key in (
        "hermes_source_unchanged", "plugin_source_unchanged", "script_unchanged")) and all(
        r["passed"] and r["exit_ok"] for r in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--dependency-path", type=Path, action="append", default=[])
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = args.hermes_source.resolve()
    paths = [p.resolve() for p in args.dependency_path]
    try:
        report = worker(source, args.worker_case, paths) if args.worker_case else run(source, args.case or CASES, paths)
    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)[-3:]
        report = {"passed": False, "error": "worker_runtime_failed" if args.worker_case else "setup_failed",
                  "error_type": type(error).__name__, "checks": {},
                  "error_locations": [{"file": Path(frame.filename).name, "function": frame.name,
                                       "line": frame.lineno} for frame in frames]}
        # A fixed exception class is safe. Do not put exception text or raw provider data in the report.
        if args.worker_case:
            report["case"] = args.worker_case
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
