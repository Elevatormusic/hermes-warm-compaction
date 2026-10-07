"""Integration check: the standalone warm_compaction plugin on unpatched Hermes Agent.

Run this script with the Hermes dependency interpreter. The parent makes a temporary Git repository that
contains the plugin and a small test tool plugin. Each scenario has its own HERMES_HOME and two worker
processes. The install process installs the plugin with the Hermes install command. The run process uses
the Python that a Hermes launch selects, starts a loopback fake OpenAI-compatible server, selects the
engine, and runs real Hermes conversation and compaction code on synthetic rows only. The rollback scenario
has a third process: it selects the built-in compressor and continues the saved session that the plugin
compacted. An audit-hook fence in the run processes blocks network access other than the fake server, child
processes, and writes outside the scenario folder. The report has metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "warm_compaction"
MODEL = "fake-model"
CONTEXT_LENGTH = 200_000
WARM_MARK = "Stop the current task now. This request comes from the host program"
# The reply limit of the warm request (warm_compaction.warm). The check does not import the plugin.
HANDOFF_MIN_TOKENS, HANDOFF_MAX_TOKENS = 2048, 8192
FALLBACK_MARK = "Write a handoff summary of the conversation transcript"
SYSTEM = "You are a test agent. Answer in one short sentence."
SEED_QUESTION = "What is the code word?"
NEXT_QUESTION = "What is the next step?"
TOOL_QUESTION = "Call the note tool, then answer."
TOOL_AGAIN_QUESTION = "Call the note tool again, then answer."
QUESTIONS = (SEED_QUESTION, NEXT_QUESTION, TOOL_QUESTION, TOOL_AGAIN_QUESTION)
HANDOFF = (
    "## Goal\nFinish the synthetic test task.\n\n"
    "## User instructions\n- \"Answer in one short sentence.\"\n\n"
    "## Current state\n- [OPEN] Report the code word.\n\n"
    "## Key facts\n- The code word is BLUE-7.\n\n"
    "## Next step\nReport the code word BLUE-7."
)
HANDOFF_GOAL = "Finish the synthetic test task."
FALLBACK_SUMMARY = HANDOFF.replace("Finish the synthetic test task.", "Finish the synthetic test task (fallback).") + (
    "\n[END OF SUMMARY]")
TOOL_PLUGIN = {
    "plugin.yaml": (
        "manifest_version: 2\nname: wc_test_tools\nversion: 0.0.1\n"
        "description: \"Test tool for the warm_compaction integration check.\"\nkind: standalone\n"
    ),
    "__init__.py": (
        '"""Test tool plugin for the warm_compaction integration check."""\n\nimport json\n\n\n'
        "def _note(args, **_kwargs):\n"
        '    return json.dumps({"note": "The test note says BLUE-7."})\n\n\n'
        "def register(ctx):\n"
        '    ctx.register_tool(name="wc_note", toolset="wc_test", handler=_note, schema={\n'
        '        "name": "wc_note", "description": "Return a fixed test note.",\n'
        '        "parameters": {"type": "object", "properties": {}, "required": []}})\n'
    ),
}
SCENARIOS = ("enable_manual_warm", "manual_fallback", "manual_fixed", "auto_tool_loop", "auto_warm_failures",
             "rollback")
# The scenarios with automatic compaction in a tool loop.
AUTO_SCENARIOS = ("auto_tool_loop", "auto_warm_failures")
NOTICE_MARK = "Warm compaction unavailable"
# The rollback scenario has a third process: a new Hermes process with the built-in compressor.
PHASES = {"rollback": ("install", "run", "rollback")}
DEFAULT_PHASES = ("install", "run")
WORKER_TIMEOUT = 900
LOOPBACK = "127.0.0.1"
PLACEHOLDER_URL = "http://127.0.0.1:9/v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def inside(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------------------
# Fake OpenAI-compatible server
# ---------------------------------------------------------------------------------------------------------


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
    users = [row for row in messages if row.get("role") == "user"]
    if users and str(users[-1].get("content") or "") in QUESTIONS:
        return "main"
    return "other"


def completion(content=None, tool=None, *, prompt_tokens, cached_tokens=None):
    """Return (message, finish_reason, usage) for one fake reply."""
    message = {"role": "assistant", "content": content}
    finish = "stop"
    if tool:
        message["tool_calls"] = [{"id": tool["id"], "type": "function",
                                  "function": {"name": tool["name"], "arguments": "{}"}}]
        finish = "tool_calls"
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 20, "total_tokens": prompt_tokens + 20}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return message, finish, usage


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
                    {"id": MODEL, "object": "model", "context_length": CONTEXT_LENGTH}]}).encode("utf-8")
                self.reply(200, "application/json", data)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = {}
                status, payload = server.answer(body)
                if status != 200:
                    self.reply(status, "application/json", json.dumps({"error": {"message": "fake error"}}).encode())
                elif body.get("stream"):
                    self.reply(200, "text/event-stream", server.sse(*payload))
                else:
                    self.reply(200, "application/json", server.json_reply(*payload))

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

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def answer(self, body):
        kind = classify(body)
        messages = body.get("messages") or []
        size = len(json.dumps(messages)) // 4
        with self.lock:
            entry = {"seq": len(self.requests), "kind": kind, "body": body, "status": 200}
            self.requests.append(entry)
            if kind == "warm":
                # The first warm_fail_first warm requests fail; the later ones use warm_status.
                warm_seen = sum(1 for request in self.requests if request["kind"] == "warm")
                failing = warm_seen <= self.plan.get("warm_fail_first", 0)
                entry["status"] = 500 if failing else self.plan.get("warm_status", 200)
                payload = completion(HANDOFF, prompt_tokens=size, cached_tokens=self.plan.get("cached", size // 2))
            elif kind == "fallback":
                entry["status"] = self.plan.get("fallback_status", 200)
                payload = completion(FALLBACK_SUMMARY, prompt_tokens=size)
            elif kind == "main":
                step = self.main_replies.pop(0) if self.main_replies else {"content": "OK."}
                payload = completion(step.get("content"), step.get("tool"),
                                     prompt_tokens=step.get("prompt_tokens", size))
            else:
                payload = completion("Synthetic title", prompt_tokens=size)
        return entry["status"], payload

    @staticmethod
    def json_reply(message, finish, usage) -> bytes:
        return json.dumps({"id": "fake", "object": "chat.completion", "created": 0, "model": MODEL,
                           "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                           "usage": usage}).encode("utf-8")

    @staticmethod
    def sse(message, finish, usage) -> bytes:
        def event(choices, extra=None):
            data = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": MODEL,
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


# ---------------------------------------------------------------------------------------------------------
# Worker fence
# ---------------------------------------------------------------------------------------------------------


def socketpair_frame() -> bool:
    """True when the standard library makes a loopback socket pair. Windows asyncio does this for each loop."""
    frame = sys._getframe(1)
    for _ in range(6):
        if frame is None:
            return False
        code = frame.f_code
        if code.co_name in {"socketpair", "_fallback_socketpair"} and Path(code.co_filename).name == "socket.py":
            return True
        frame = frame.f_back
    return False


def program_name(event, args) -> str:
    """The base name of the program in a child process event. The report keeps only this name."""
    try:
        if event == "subprocess.Popen":
            executable, argv = args[0], args[1]
            if executable:
                text = os.fsdecode(executable)
            elif isinstance(argv, (str, bytes)):
                text = os.fsdecode(argv).split()[0]
            else:
                text = os.fsdecode(argv[0])
        elif event == "os.system":
            text = os.fsdecode(args[0]).split()[0]
        else:
            text = os.fsdecode(args[0])
        return Path(text.strip('"')).name.lower() or "unknown"
    except Exception:
        return "unknown"


class Fence:
    """After the install: network only to the fake server, no child process, no write outside the run."""

    def __init__(self, output, port, source, plugin):
        self.output = Path(output).resolve()
        self.port = int(port)
        self.source = Path(source).resolve()
        self.plugin = Path(plugin).resolve()
        self.denied = {}
        self.plugin_denied = 0
        self.programs = set()
        self.hosts = set()
        self.callers = set()
        self.socketpairs = 0

    def stack(self):
        """Hermes callers as path:function (first three, without the start-up module) and plugin presence."""
        frame, callers, plugin = sys._getframe(2), [], False
        while frame is not None:
            path = Path(frame.f_code.co_filename)
            plugin = plugin or inside(path, self.plugin)
            if inside(path, self.source) and path.name != "hermes_bootstrap.py" and len(callers) < 3:
                callers.append(f"{path.resolve().relative_to(self.source).as_posix()}:{frame.f_code.co_name}")
            frame = frame.f_back
        return callers, plugin

    def refuse(self, reason):
        self.denied[reason] = self.denied.get(reason, 0) + 1
        callers, plugin = self.stack()
        self.plugin_denied += int(plugin)
        self.callers.add(f"{reason} from {' < '.join(callers) or 'outside Hermes'}"
                         + (" (plugin code on the stack)" if plugin else ""))
        raise PermissionError(reason)

    def __call__(self, event, args):
        if event == "socket.getaddrinfo":
            host = args[0].decode() if isinstance(args[0], bytes) else str(args[0])
            if host not in (LOOPBACK, "localhost"):
                self.hosts.add(host)
                self.refuse("network_host_blocked")
        elif event in {"socket.connect", "socket.connect_ex"}:
            address = args[1]
            loopback = isinstance(address, tuple) and len(address) >= 2 and address[0] in (LOOPBACK, "::1")
            if loopback and address[1] == self.port and address[0] == LOOPBACK:
                return
            if loopback and socketpair_frame():
                self.socketpairs += 1
                return
            self.refuse("network_address_blocked")
        elif event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.startfile"}:
            self.programs.add(program_name(event, args))
            self.refuse("child_process_blocked")
        elif event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.fsdecode(args[0]) == os.devnull:
                return
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1], args[2]
            writing = isinstance(mode, str) and any(mark in mode for mark in "wax+")
            writing = writing or bool(isinstance(flags, int) and flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not inside(path, self.output):
                self.refuse("write_outside_run_blocked")
            if not inside(path, self.output) and (
                    path.name.startswith(".env") or any(part.lower() == ".hermes" for part in path.parts)):
                self.refuse("private_profile_read_blocked")


# ---------------------------------------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------------------------------------


def synthetic_history(pairs=30):
    rows = []
    for index in range(pairs):
        rows.append({"role": "user", "content": f"Synthetic question {index}. " + "Detail text. " * 160})
        rows.append({"role": "assistant", "content": f"Synthetic answer {index}. " + "Answer text. " * 160})
    rows[0]["content"] = "Remember the code word BLUE-7. " + rows[0]["content"]
    return rows


def plan_for(scenario):
    if scenario in ("enable_manual_warm", "rollback"):
        return {"main": [{"content": "The code word is BLUE-7."}, {"content": "Next: report BLUE-7."}]}
    if scenario == "manual_fallback":
        return {"warm_status": 500, "main": [{"content": "The code word is BLUE-7."}, {"content": "Next."}]}
    if scenario == "manual_fixed":
        return {"warm_status": 500, "fallback_status": 500,
                "main": [{"content": "The code word is BLUE-7."}, {"content": "Next."}]}
    if scenario == "auto_warm_failures":
        # Four automatic compactions; the server refuses the first three warm requests.
        main = []
        for index in range(4):
            main += [{"tool": {"id": f"call_{index + 1}", "name": "wc_note"}, "prompt_tokens": 150_000},
                     {"content": f"Done {index + 1}.", "prompt_tokens": 20_000}]
        return {"warm_fail_first": 3, "main": main}
    if scenario == "auto_tool_loop":
        return {"main": [
            {"tool": {"id": "call_1", "name": "wc_note"}, "prompt_tokens": 150_000},
            {"content": "Done.", "prompt_tokens": 20_000},
            {"tool": {"id": "call_2", "name": "wc_note"}, "prompt_tokens": 150_000},
            {"content": "Done again.", "prompt_tokens": 20_000},
        ]}
    raise ValueError(scenario)


def base_config(scenario):
    """The run phase replaces the placeholder URL with the URL of its fake server."""
    settings = {"threshold": 0.4} if scenario in AUTO_SCENARIOS else {}
    return {
        "model": {"default": MODEL, "provider": "custom", "base_url": PLACEHOLDER_URL,
                  "context_length": CONTEXT_LENGTH},
        "context": {"engine": "warm_compaction"},
        "auxiliary": {"transient_retries": 0, "title_generation": {"enabled": False}},
        "plugins": {"entries": {"warm_compaction": {"settings": settings}}},
    }


def yaml_load(text):
    """Hermes on Python 3.14 ships ruamel.yaml and not PyYAML. Use the package that is present."""
    try:
        import yaml
    except ImportError:
        from ruamel.yaml import YAML
        return YAML(typ="safe", pure=True).load(text)
    return yaml.safe_load(text)


def yaml_dump(data) -> str:
    try:
        import yaml
    except ImportError:
        from io import StringIO
        from ruamel.yaml import YAML
        dumper = YAML(typ="safe", pure=True)
        dumper.default_flow_style = False
        stream = StringIO()
        dumper.dump(data, stream)
        return stream.getvalue()
    return yaml.safe_dump(data, sort_keys=False)


def merge_config(path, extra):
    current = yaml_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    current = current or {}

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(current, extra)
    path.write_text(yaml_dump(current), encoding="utf-8")


def install_phase(spec, session_dir):
    """Install the plugins in one process, as the `hermes plugins install` command does.

    The first scenario enables the plugin with the documented install command. Hermes then runs its plugin
    admission, which can build a dependency runtime for a source checkout. The other scenarios install
    without enable and write plugins.enabled, to keep the check short.
    """
    scenario, home = spec["scenario"], session_dir / "home"
    sys.path.insert(0, spec["hermes_source"])
    result = {"phase": "install", "python": sys.version.split()[0], "checks": {}, "error": None}
    checks = result["checks"]
    try:
        merge_config(home / "config.yaml", base_config(scenario))
        from hermes_cli.plugins_cmd_install import cmd_install
        documented = scenario == "enable_manual_warm"
        cmd_install(spec["plugin_url"], enable=documented)
        names = ["warm_compaction"]
        if scenario in AUTO_SCENARIOS:
            cmd_install(spec["tool_plugin_url"], enable=False)
            names.append("wc_test_tools")
        if not documented:
            merge_config(home / "config.yaml", {"plugins": {"enabled": names}})
        config = yaml_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        result["enabled_keys"] = list((config.get("plugins") or {}).get("enabled") or [])
        result["enable_path"] = "install command with enable" if documented else "plugins.enabled in config"
        checks["enabled_in_config"] = all(name in result["enabled_keys"] for name in names)
        installed = home / "plugins" / "warm_compaction"
        checks["installed"] = (installed / "plugin.yaml").is_file()
        from hermes_cli.plugin_validate import validate_plugin_dir
        report = validate_plugin_dir(installed)
        result["validate"] = [[name, bool(ok)] for name, ok, _detail in report.checks]
        checks["validate_ok"] = bool(report.ok)
        passed = {name for name, ok, _detail in report.checks if ok}
        checks["security_scan_passed"] = "security scan" in passed
        checks["no_core_override_passed"] = "no core override" in passed
        # A Hermes launch uses the runtime Python when a dependency environment is committed. Without one,
        # that Python refuses to start Hermes, and the launch Python keeps its own packages.
        from hermes_cli._launchers import resolve_store_python
        from pm.environments import committed_venv
        source = Path(spec["hermes_source"])
        store_python = resolve_store_python(source) if committed_venv(source) else None
        result["run_python"] = str(store_python) if store_python else None
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"[:500]
    write_json(session_dir / "install.json", result)
    return 0 if result["error"] is None and checks and all(checks.values()) else 1


def run_phase(spec, session_dir):
    """Run real Hermes conversation and compaction code in a fresh process behind the fence."""
    scenario, home = spec["scenario"], session_dir / "home"
    sys.path.insert(0, spec["hermes_source"])
    result = {"scenario": scenario, "python": sys.version.split()[0], "status": "started", "checks": {},
              "requests": [], "error": None}
    checks = result["checks"]
    server = FakeServer(plan_for(scenario))
    server.start()

    def finish(status):
        result["status"] = status
        result["requests"] = [{"seq": r["seq"], "kind": r["kind"], "status": r["status"],
                               "stream": bool(r["body"].get("stream")),
                               "message_count": len(r["body"].get("messages") or [])} for r in server.requests]
        write_json(session_dir / "result.json", result)

    try:
        # A Hermes process selects its dependency set at start, before any third-party import.
        import hermes_bootstrap  # noqa: F401
        merge_config(home / "config.yaml", {"model": {"base_url": server.base_url},
                                          "compression": {"progress_notices": False}})
        import platform
        # Fill the platform cache before the fence: some Windows Pythons start "cmd /c ver" one time.
        platform.uname()
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from hermes_cli.plugins_cmd_toggle import _discover_context_engines
        from hermes_state import SessionDB
        from run_agent import AIAgent
        from tui_gateway.server import _compress_session_history
        checks["listed_in_engine_picker"] = any(name == "warm_compaction" for name, *_ in _discover_context_engines())
        fence = Fence(session_dir, server.httpd.server_port, spec["hermes_source"],
                      home / "plugins" / "warm_compaction")
        sys.addaudithook(fence)

        db = SessionDB(home / "state.db")
        # The status lines that Hermes shows the user (CLI print and the gateway status callback).
        statuses = []
        sid = f"wc-{scenario}"
        db.create_session(sid, "cli")
        db.append_messages_batch(sid, synthetic_history())
        agent = AIAgent(base_url=server.base_url, api_key="no-key-required", provider="custom",
                        api_mode="chat_completions", model=MODEL,
                        enabled_toolsets=["wc_test"] if scenario in AUTO_SCENARIOS else [],
                        disabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
                        skip_background_review=True, capabilities={"vision": False}, session_id=sid,
                        session_db=db, max_tokens=1024, platform="cli", cwd=str(session_dir),
                        status_callback=lambda kind, message: statuses.append([kind, str(message)]))
        engine = agent.context_compressor
        checks["engine_selected"] = getattr(engine, "name", None) == "warm_compaction"
        result["engine"] = {"name": getattr(engine, "name", None), "threshold_tokens": engine.threshold_tokens,
                            "context_length": engine.context_length}
        history = db.get_messages_as_conversation(sid)
        if scenario == "auto_tool_loop":
            run_auto(agent, engine, history, server, result, checks)
        elif scenario == "auto_warm_failures":
            run_auto_failures(agent, engine, history, server, result, checks, statuses, home)
        else:
            run_manual(agent, engine, db, history, server, result, checks, scenario,
                       _compress_session_history, finalize_context_engine_compression_notification)
        result["fence_denied"] = fence.denied
        result["fence_blocked_programs"] = sorted(fence.programs)
        result["fence_blocked_hosts"] = sorted(fence.hosts)
        result["fence_blocked_callers"] = sorted(fence.callers)
        result["fence_socketpairs"] = fence.socketpairs
        # The fence blocks host calls too. Those are information. The check is that plugin code made none.
        checks["plugin_made_no_blocked_call"] = fence.plugin_denied == 0
        db.close()
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"[:500]
        finish("failed")
        server.stop()
        raise
    finish("passed" if checks and all(checks.values()) else "failed")
    server.stop()
    return 0


def row_shape(row):
    """Metadata of one row for the report: role, summary flag, and size. No text."""
    return {"role": row.get("role"), "summary_flag": row.get("_compressed_summary"),
            "chars": len(str(row.get("content") or ""))}


def contains_goal(request):
    """True when a message of the request carries the goal line of the plugin handoff."""
    return any(HANDOFF_GOAL in str(message.get("content") or "") for message in request["body"].get("messages") or [])


def rollback_phase(spec, session_dir):
    """Start a new Hermes process with the built-in compressor on the history that the plugin compacted.

    The plugin stays enabled, as after the first rollback step in the plugin guide. The phase continues the saved
    session with more synthetic rows, then runs the built-in manual compaction.
    """
    home = session_dir / "home"
    sys.path.insert(0, spec["hermes_source"])
    result = {"phase": "rollback", "python": sys.version.split()[0], "status": "started", "checks": {},
              "requests": [], "error": None}
    checks = result["checks"]
    server = FakeServer({"main": [{"content": "Next: report BLUE-7."}]})
    server.start()

    def finish(status):
        result["status"] = status
        result["requests"] = [{"seq": r["seq"], "kind": r["kind"], "status": r["status"],
                               "stream": bool(r["body"].get("stream")),
                               "message_count": len(r["body"].get("messages") or [])} for r in server.requests]
        write_json(session_dir / "rollback.json", result)

    try:
        import hermes_bootstrap  # noqa: F401
        merge_config(home / "config.yaml", {"model": {"base_url": server.base_url},
                                            "context": {"engine": "compressor"}})
        import platform
        platform.uname()
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from hermes_state import SessionDB
        from run_agent import AIAgent
        from tui_gateway.server import _compress_session_history
        fence = Fence(session_dir, server.httpd.server_port, spec["hermes_source"],
                      home / "plugins" / "warm_compaction")
        sys.addaudithook(fence)

        sid = read_json(session_dir / "result.json", {}).get("final_session_id")
        db = SessionDB(home / "state.db")
        saved = db.get_messages_as_conversation(sid)
        result["reloaded_first_rows"] = [row_shape(row) for row in saved[:3]]
        checks["saved_history_has_plugin_summary"] = (
            len(saved) >= 3 and [row.get("role") for row in saved[:2]] == ["user", "assistant"]
            and str(saved[1].get("content") or "").startswith("[CONTEXT SUMMARY]:")
            and HANDOFF_GOAL in str(saved[1].get("content") or ""))
        agent = AIAgent(base_url=server.base_url, api_key="no-key-required", provider="custom",
                        api_mode="chat_completions", model=MODEL, enabled_toolsets=[], disabled_toolsets=[],
                        quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
                        capabilities={"vision": False}, session_id=sid, session_db=db, max_tokens=1024,
                        platform="cli", cwd=str(session_dir))
        engine = agent.context_compressor
        result["engine"] = getattr(engine, "name", None)
        checks["builtin_engine_selected"] = result["engine"] == "compressor"
        # The user continues the session after the rollback: more synthetic turns, then one question.
        history = list(saved) + synthetic_history(20)
        cont = agent.run_conversation(NEXT_QUESTION, system_message=SYSTEM, conversation_history=history)
        checks["continuation_reply"] = bool(str(cont.get("final_response") or "").strip())
        main = by_kind(server, "main")
        checks["continuation_sends_plugin_summary"] = bool(main) and contains_goal(main[-1])
        session = {"agent": agent, "history": list(cont.get("messages") or []), "history_lock": threading.Lock(),
                   "history_version": 1, "session_key": agent.session_id}
        before = len(server.requests)
        removed, _usage = _compress_session_history(session, "")
        finalize_context_engine_compression_notification(agent, committed=True)
        result["builtin_removed_rows"] = removed
        checks["builtin_compress_removed_rows"] = removed > 0
        summary_requests = [r for r in server.requests[before:] if r["kind"] != "main"]
        after = session["history"]
        result["builtin_summary_requests"] = len(summary_requests)
        result["plugin_summary_in_builtin_request"] = any(contains_goal(r) for r in summary_requests)
        result["plugin_summary_kept_verbatim"] = any(HANDOFF_GOAL in str(row.get("content") or "") for row in after)
        # The built-in compressor either summarizes the plugin summary or keeps it in its protected head.
        checks["plugin_summary_carried"] = (result["plugin_summary_in_builtin_request"]
                                            or result["plugin_summary_kept_verbatim"])
        stored = db.get_messages_as_conversation(agent.session_id)
        # Hermes strips the outer whitespace of a row that it stores for the first time. The added synthetic
        # rows end with a space, so compare the stripped text.
        texts = [[str(row.get("content") or "").strip() for row in rows] for rows in (after, stored)]
        result["history_rows"] = {"in_memory": len(after), "stored": len(stored)}
        mismatch = next((index for index, (a, b) in enumerate(zip(*texts)) if a != b), None)
        if mismatch is not None:
            result["first_mismatch"] = {"index": mismatch, "in_memory": row_shape(after[mismatch]),
                                        "stored": row_shape(stored[mismatch])}
        checks["history_saved"] = texts[0] == texts[1]
        result["fence_denied"] = fence.denied
        result["fence_blocked_programs"] = sorted(fence.programs)
        result["fence_blocked_hosts"] = sorted(fence.hosts)
        checks["plugin_made_no_blocked_call"] = fence.plugin_denied == 0
        db.close()
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"[:500]
        finish("failed")
        server.stop()
        raise
    finish("passed" if checks and all(checks.values()) else "failed")
    server.stop()
    return 0


def worker(session_dir, phase):
    session_dir = Path(session_dir).resolve()
    spec = json.loads((session_dir / "spec.json").read_text(encoding="utf-8"))
    if phase == "install":
        return install_phase(spec, session_dir)
    return rollback_phase(spec, session_dir) if phase == "rollback" else run_phase(spec, session_dir)


def by_kind(server, kind):
    return [r for r in server.requests if r["kind"] == kind]


def check_warm_request(server, checks, label, tool_rows=0):
    """The latest main request before the warm request must be an exact prefix of the warm request."""
    warm = by_kind(server, "warm")[-1]
    main = [r for r in server.requests if r["kind"] == "main" and r["seq"] < warm["seq"]][-1]
    sent, extended = main["body"]["messages"], warm["body"]["messages"]
    checks[f"{label}_prefix_exact"] = json.dumps(extended[: len(sent)], sort_keys=True) == json.dumps(
        sent, sort_keys=True)
    checks[f"{label}_new_rows"] = len(extended) == len(sent) + 2 + tool_rows
    checks[f"{label}_instruction_last"] = str(extended[-1].get("content") or "").startswith(WARM_MARK)
    checks[f"{label}_not_streamed"] = warm["body"].get("stream") is False and "stream_options" not in warm["body"]
    limits = {"max_tokens", "max_completion_tokens"}
    same = {key: value for key, value in main["body"].items()
            if key not in {"messages", "stream", "stream_options", *limits}}
    checks[f"{label}_settings_kept"] = all(warm["body"].get(key) == value for key, value in same.items())
    # The handoff has its own reply limit: the main limit kept between HANDOFF_MIN_TOKENS and HANDOFF_MAX_TOKENS.
    # Without a main limit, the handoff gets HANDOFF_MAX_TOKENS in one field (the field that Hermes uses).
    if any(type(main["body"].get(key)) is int and main["body"].get(key) > 0 for key in limits):
        checks[f"{label}_reply_limit"] = all(
            warm["body"].get(key) == (min(max(value, HANDOFF_MIN_TOKENS), HANDOFF_MAX_TOKENS)
                                      if type(value) is int and value > 0 else value)
            for key, value in ((key, main["body"].get(key)) for key in limits))
    else:
        checks[f"{label}_reply_limit"] = sorted(
            key for key in limits if warm["body"].get(key) == HANDOFF_MAX_TOKENS) in (
            ["max_tokens"], ["max_completion_tokens"])
    if tool_rows:
        roles = [row.get("role") for row in extended[len(sent):]]
        checks[f"{label}_tool_rows_sent"] = roles == ["assistant", *["tool"] * tool_rows, "user"]


def summary_rows_ok(history):
    return (len(history) >= 3 and history[0].get("role") == "user" and history[1].get("role") == "assistant"
            and history[0].get("_compressed_summary") is True and history[1].get("_compressed_summary") is True
            and str(history[1].get("content") or "").startswith("[CONTEXT SUMMARY]:"))


def run_manual(agent, engine, db, history, server, result, checks, scenario, compress, finalize):
    seed = agent.run_conversation(SEED_QUESTION, system_message=SYSTEM, conversation_history=history)
    messages = seed.get("messages") or []
    checks["seed_turn"] = len(messages) == len(history) + 2
    session = {"agent": agent, "history": list(messages), "history_lock": threading.Lock(),
               "history_version": 1, "session_key": agent.session_id}
    # The capture keeps the prompt count that the server reported, through the real post_api_request hook.
    capture = engine._store.latest(engine._wc_session_id) or {}
    seed_request = by_kind(server, "main")[-1]
    checks["capture_keeps_reported_prompt_tokens"] = capture.get("prompt_tokens") == len(
        json.dumps(seed_request["body"]["messages"])) // 4
    begin = time.perf_counter()
    failures_before = engine._warm_failures
    removed, _usage = compress(session, "")
    checks["failure_streak_waits_for_manual_commit"] = engine._warm_failures == failures_before
    finalize(agent, committed=True)
    expected_failures = failures_before + (scenario in ("manual_fallback", "manual_fixed"))
    checks["failure_streak_after_manual_commit"] = engine._warm_failures == expected_failures
    checks["manual_commit_notification_one_time"] = finalize(agent, committed=True) is False
    result["compress_seconds"] = round(time.perf_counter() - begin, 3)
    warm_last = dict(engine.warm_last or {})
    result["warm_last"] = warm_last
    expected = {"enable_manual_warm": ("warm", "accepted"), "manual_fallback": ("fallback", "provider_error"),
                "manual_fixed": ("fixed", "provider_error"), "rollback": ("warm", "accepted")}[scenario]
    checks["path_and_reason"] = (warm_last.get("path"), warm_last.get("reason")) == expected
    checks["removed_rows"] = removed > 0
    after = session["history"]
    checks["summary_rows"] = summary_rows_ok(after)
    checks["tail_starts_with_user"] = after[2].get("role") == "user"
    if scenario == "enable_manual_warm":
        check_warm_request(server, checks, "warm")
        checks["handoff_in_summary"] = HANDOFF in str(after[1].get("content"))
        checks["no_fallback_request"] = not by_kind(server, "fallback")
    if scenario == "manual_fallback":
        checks["one_fallback_request"] = len(by_kind(server, "fallback")) == 1
        checks["fallback_in_summary"] = "(fallback)" in str(after[1].get("content"))
    if scenario == "manual_fixed":
        checks["fixed_in_summary"] = "Summary unavailable." in str(after[1].get("content"))
    saved = db.get_messages_as_conversation(agent.session_id)
    checks["history_saved"] = [row.get("content") for row in saved] == [row.get("content") for row in after]
    result["session_rotated"] = agent.session_id != f"wc-{scenario}"
    result["final_session_id"] = agent.session_id
    checks["engine_session_follows"] = engine._wc_session_id == agent.session_id
    count = len(by_kind(server, "main"))
    cont = agent.run_conversation(NEXT_QUESTION, system_message=SYSTEM, conversation_history=list(after))
    checks["continuation_reply"] = bool(str(cont.get("final_response") or "").strip())
    later = by_kind(server, "main")[count:]
    checks["continuation_sends_summary"] = bool(later) and any(
        str(row.get("content") or "").startswith("[CONTEXT SUMMARY]:") for row in later[0]["body"]["messages"])
    # Information only: Hermes rebuilds the system prompt during /compress. This is host behavior.
    result["continuation_keeps_custom_system_text"] = bool(later) and SYSTEM in str(
        later[0]["body"]["messages"][0].get("content") or "")


def run_auto(agent, engine, history, server, result, checks):
    checks["threshold_from_settings"] = engine.threshold_tokens == int(CONTEXT_LENGTH * 0.4)
    first = agent.run_conversation(TOOL_QUESTION, system_message=SYSTEM, conversation_history=history)
    checks["first_turn_final"] = first.get("final_response") == "Done."
    checks["first_warm_request"] = len(by_kind(server, "warm")) == 1
    result["warm_last_1"] = dict(engine.warm_last or {})
    checks["first_path_warm"] = result["warm_last_1"].get("path") == "warm"
    if by_kind(server, "warm"):
        check_warm_request(server, checks, "auto1", tool_rows=1)
    after_first = first.get("messages") or []
    checks["first_summary_rows"] = summary_rows_ok(after_first)
    checks["wait_flag_cleared"] = engine.awaiting_real_usage_after_compression is False
    second = agent.run_conversation(TOOL_AGAIN_QUESTION, system_message=SYSTEM,
                                    conversation_history=list(after_first))
    checks["second_turn_final"] = second.get("final_response") == "Done again."
    checks["second_warm_request"] = len(by_kind(server, "warm")) == 2
    result["warm_last_2"] = dict(engine.warm_last or {})
    checks["second_path_warm"] = result["warm_last_2"].get("path") == "warm"
    if len(by_kind(server, "warm")) == 2:
        check_warm_request(server, checks, "auto2", tool_rows=1)
    checks["no_fallback_request"] = not by_kind(server, "fallback")
    result["compression_count"] = engine.compression_count


def run_auto_failures(agent, engine, history, server, result, checks, statuses, home):
    """Repeated warm refusals: each one is a WARNING in agent.log and errors.log, and after the third one the
    next automatic compaction status shows one notice to the user. A warm compaction ends the streak."""
    from agent.context_engine import automatic_compaction_status_message
    from gateway.run import _gateway_compression_progress_notices_enabled, _prepare_gateway_status_message

    checks["gateway_routine_progress_disabled"] = not _gateway_compression_progress_notices_enabled()
    questions = (TOOL_QUESTION, TOOL_AGAIN_QUESTION)
    outcomes, notice_turn = [], None
    messages = history
    for turn in range(4):
        before = len(statuses)
        reply = agent.run_conversation(questions[turn % 2], system_message=SYSTEM, conversation_history=messages)
        messages = list(reply.get("messages") or [])
        checks[f"turn{turn + 1}_final"] = reply.get("final_response") == f"Done {turn + 1}."
        last = dict(engine.warm_last or {})
        outcomes.append([last.get("path"), last.get("reason")])
        if turn == 2:
            # The host guard runs before the engine formatter. It must keep the notice for a later status.
            pending = engine._warm_notice
            engine.emit_automatic_compaction_status = False
            try:
                hidden = automatic_compaction_status_message(
                    engine, phase="compress", default_message="Compacting context")
                checks["disabled_host_status_is_silent"] = hidden is None
                checks["disabled_host_status_keeps_notice"] = bool(pending) and engine._warm_notice == pending
            finally:
                engine.emit_automatic_compaction_status = True
        if any(NOTICE_MARK in message for _kind, message in statuses[before:]):
            notice_turn = turn + 1
    result["outcomes"] = outcomes
    notices = [message for _kind, message in statuses if NOTICE_MARK in message]
    # Metadata only: the notice has no conversation text, so its text goes into the report.
    result["notices"] = notices
    result["status_count"] = len(statuses)
    checks["three_fallbacks_then_warm"] = outcomes == [["fallback", "provider_error"]] * 3 + [["warm", "accepted"]]
    checks["one_notice"] = len(notices) == 1
    delivered = [_prepare_gateway_status_message("telegram", kind, message)
                 for kind, message in statuses if NOTICE_MARK in message]
    result["gateway_notices"] = [message for message in delivered if message]
    checks["one_notice_survives_gateway_filter"] = len(result["gateway_notices"]) == 1
    checks["gateway_keeps_the_notice_text"] = bool(notices) and result["gateway_notices"] == notices
    checks["notice_has_no_routine_progress"] = bool(notices) and notices[0].startswith("\u26a0 Warm compaction")
    checks["notice_before_the_fourth_compaction"] = notice_turn == 4
    checks["notice_names_the_reason"] = bool(notices) and (
        "provider_error: the server refused the warm request" in notices[0])
    checks["notice_describes_the_summary_result"] = bool(notices) and "did not use the warm summary" in notices[0]
    checks["notice_has_no_cache_or_retained_detail_claim"] = bool(notices) and not any(
        claim in notices[0] for claim in ("could not reuse", "prompt cache", "slower", "no messages were dropped"))
    logs = home / "logs"
    def read_log(name):
        path = logs / name
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    agent_log, errors_log = read_log("agent.log"), read_log("errors.log")
    refusal = "Warm compaction skipped (provider_error)"
    result["log_counts"] = {"agent_refusals": agent_log.count(refusal), "errors_refusals": errors_log.count(refusal),
                            "errors_notice": errors_log.count("Warm compaction failed 3 times in a row")}
    checks["each_refusal_in_agent_log"] = agent_log.count(refusal) == 3
    checks["each_refusal_in_errors_log"] = errors_log.count(refusal) == 3
    checks["notice_in_errors_log"] = errors_log.count("Warm compaction failed 3 times in a row") == 1
    check_host_commit_boundaries(engine, checks)


def check_host_commit_boundaries(engine, checks):
    """Run staged, discarded, and committed synthetic summaries through the real host boundary helpers."""
    from types import SimpleNamespace
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification, _restore_compressor_attempt_state,
        _snapshot_compressor_attempt_state, finalize_context_engine_compression_notification)

    llm = SimpleNamespace(complete=lambda *_args, **_kwargs: SimpleNamespace(
        text=FALLBACK_SUMMARY, usage=SimpleNamespace(input_tokens=50)))
    probe = type(engine)(llm=llm)
    probe.on_session_start("wc-boundary-0", platform="cli")
    probe.update_model(model=MODEL, context_length=CONTEXT_LENGTH, base_url=PLACEHOLDER_URL,
                       api_key="no-key-required", provider="custom", api_mode="chat_completions")
    host = SimpleNamespace(context_compressor=probe, platform="cli")
    history = synthetic_history()

    def queue(next_session):
        _queue_context_engine_compression_notification(
            host, new_session_id=next_session, old_session_id=probe._wc_session_id)

    snapshot = _snapshot_compressor_attempt_state(probe)
    checks["fresh_engine_counter_in_host_snapshot"] = snapshot.get("compression_count") == 0
    staged = probe.compress(history)
    checks["boundary_probe_has_real_fallback_candidate"] = (
        staged is not history and probe.warm_last["path"] == "fallback")
    checks["first_attempt_advances_compression_count"] = probe.compression_count == 1
    checks["staged_failure_does_not_count"] = probe._warm_failures == 0
    queue("wc-boundary-discarded")
    checks["discarded_host_boundary_is_silent"] = not finalize_context_engine_compression_notification(
        host, committed=False)
    checks["discarded_failure_does_not_count"] = probe._warm_failures == 0
    checks["discarded_host_boundary_cannot_be_replayed"] = not finalize_context_engine_compression_notification(
        host, committed=True)
    _restore_compressor_attempt_state(probe, snapshot, durable_cooldown_authoritative=False)
    checks["host_rollback_restores_compression_count"] = probe.compression_count == snapshot["compression_count"]
    queue("wc-boundary-stale")
    finalize_context_engine_compression_notification(host, committed=True)
    checks["restored_candidate_cannot_count_at_stale_boundary"] = probe._warm_failures == 0

    probe.compress(history)
    checks["later_failure_waits_for_commit"] = probe._warm_failures == 0
    queue("wc-boundary-committed")
    finalize_context_engine_compression_notification(host, committed=True)
    checks["later_committed_failure_counts_once"] = probe._warm_failures == 1
    checks["committed_host_boundary_cannot_be_replayed"] = not finalize_context_engine_compression_notification(
        host, committed=True)
    checks["replayed_host_boundary_does_not_count"] = probe._warm_failures == 1

    # A rollback after an earlier commit must keep that committed streak and reject the new candidate.
    snapshot = _snapshot_compressor_attempt_state(probe)
    probe.compress(history)
    queue("wc-boundary-rollback-after-commit")
    finalize_context_engine_compression_notification(host, committed=False)
    _restore_compressor_attempt_state(probe, snapshot, durable_cooldown_authoritative=False)
    queue("wc-boundary-stale-after-commit")
    finalize_context_engine_compression_notification(host, committed=True)
    checks["rollback_keeps_prior_committed_streak"] = probe._warm_failures == 1
    checks["rollback_after_commit_restores_count"] = probe.compression_count == snapshot["compression_count"]
    probe.compress(history)
    queue("wc-boundary-retry")
    finalize_context_engine_compression_notification(host, committed=True)
    checks["retry_after_rollback_counts_once"] = probe._warm_failures == 2


# ---------------------------------------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------------------------------------


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=wc-check", "-c", "user.email=wc-check@example.invalid",
                    "-c", "core.autocrlf=false", *args], cwd=cwd, check=True, capture_output=True)


def plugin_repo(run_dir):
    repo = run_dir / "plugin-repo"
    shutil.copytree(PLUGIN, repo / "warm_compaction", ignore=shutil.ignore_patterns("__pycache__"))
    tools = repo / "wc_test_tools"
    tools.mkdir(parents=True)
    for name, text in TOOL_PLUGIN.items():
        (tools / name).write_text(text, encoding="utf-8")
    git("init", "-q", cwd=repo)
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "plugin under test", cwd=repo)
    return repo


def worker_environment(session_dir):
    environment = {key: value for key, value in os.environ.items()
                   if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "COMSPEC"}}
    environment.update({
        "HERMES_HOME": str(session_dir / "home"), "HERMES_RUNTIME_DIR": str(session_dir / "runtime"),
        "USERPROFILE": str(session_dir / "os-home"), "HOME": str(session_dir / "os-home"),
        "APPDATA": str(session_dir / "os-home" / "appdata"),
        "LOCALAPPDATA": str(session_dir / "os-home" / "localappdata"),
        "TEMP": str(session_dir / "tmp"), "TMP": str(session_dir / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1", "HERMES_DUMP_REQUESTS": "false", "OPENAI_API_KEY": "no-key-required",
        "PYTHONIOENCODING": "utf-8",
    })
    return environment


def plugin_files_sha256():
    return {path.relative_to(ROOT).as_posix(): sha(path.read_bytes())
            for path in sorted(PLUGIN.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def source_state(source):
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], capture_output=True, text=True,
                            check=True)
    return {"commit": head.stdout.strip(), "clean": status.stdout.strip() == ""}


def run(source, work, scenarios, report_path):
    source = Path(source).resolve()
    run_dir = Path(work).resolve() / time.strftime("run-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    repo = plugin_repo(run_dir)
    url = repo.resolve().as_uri()
    report = {
        "check": "standalone warm_compaction plugin on unpatched Hermes Agent",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hermes": source_state(source),
        "python": sys.version.split()[0],
        "script_sha256": sha(Path(__file__).read_bytes()),
        "plugin_files_sha256": plugin_files_sha256(),
        "privacy": "metadata only: names, counts, codes, booleans, and times; no message text",
        "scenarios": [],
    }
    for scenario in scenarios:
        session_dir = run_dir / scenario
        for folder in ("home", "runtime", "os-home", "tmp"):
            (session_dir / folder).mkdir(parents=True)
        write_json(session_dir / "spec.json", {
            "scenario": scenario, "hermes_source": str(source), "plugin_url": f"{url}#warm_compaction",
            "tool_plugin_url": f"{url}#wc_test_tools"})
        begin = time.perf_counter()
        python, exits = sys.executable, []
        phases = PHASES.get(scenario, DEFAULT_PHASES)
        for phase in phases:
            completed = subprocess.run(
                [python, "-I", "-B", str(Path(__file__).resolve()), "--worker", "--phase", phase,
                 "--session-dir", str(session_dir)],
                env=worker_environment(session_dir), cwd=str(session_dir), timeout=WORKER_TIMEOUT,
                capture_output=True)
            exits.append(completed.returncode)
            (session_dir / f"worker-{phase}.private.txt").write_bytes(
                completed.stdout[-200_000:] + b"\n---\n" + completed.stderr[-200_000:])
            if completed.returncode:
                break
            python = read_json(session_dir / "install.json", {}).get("run_python") or sys.executable
        install = read_json(session_dir / "install.json", {"checks": {}, "error": "no_install_result"})
        result = read_json(session_dir / "result.json", {
            "scenario": scenario, "status": "failed", "checks": {}, "error": "no_run_result"})
        result["install"] = {key: install.get(key) for key in (
            "python", "enable_path", "enabled_keys", "validate", "checks", "error")}
        result["run_interpreter"] = "pm_runtime_python" if install.get("run_python") else "launch_python"
        if install.get("error") or not install.get("checks") or not all(install["checks"].values()):
            result["status"] = "failed"
        if "rollback" in phases:
            rollback = read_json(session_dir / "rollback.json", {"status": "failed", "checks": {},
                                                                  "error": "no_rollback_result"})
            result["rollback"] = rollback
            if rollback.get("status") != "passed":
                result["status"] = "failed"
        result["exit"] = exits
        result["phases"] = list(phases)
        result["process_seconds"] = round(time.perf_counter() - begin, 3)
        report["scenarios"].append(result)
        print(f"{scenario}: {result['status']} ({result['process_seconds']} s)", flush=True)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["passed"] = all(item["status"] == "passed" and item["exit"] == [0] * len(item["phases"])
                           for item in report["scenarios"])
    report["run_folder"] = run_dir.relative_to(ROOT).as_posix() if inside(run_dir, ROOT) else "outside repository"
    if report_path:
        write_json(Path(report_path), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path)
    parser.add_argument("--work", type=Path, default=ROOT / ".work" / "plugin-integration")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("install", "run", "rollback"), help=argparse.SUPPRESS)
    parser.add_argument("--session-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.session_dir, args.phase)
    if args.hermes_source is None:
        parser.error("--hermes-source is required")
    report = run(args.hermes_source, args.work, args.scenario or list(SCENARIOS), args.report)
    print(json.dumps({"passed": report["passed"], "hermes_commit": report["hermes"]["commit"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
