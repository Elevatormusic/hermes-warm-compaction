"""Run one benchmark job with the compaction of another agent CLI: Codex CLI or Claude Code.

The parent process (run.py) starts this file with its own Python. The worker writes the synthetic fixture into the
session file format of the CLI, resumes that session with the same resume turn as the Hermes methods, forces one
compaction in headless mode, reads the context after compaction (summary and kept messages) back from the session
file, and asks the same questions with the same QA request as the other methods (worker.ask).

Isolation: each job has a new temporary CODEX_HOME or CLAUDE_CONFIG_DIR, a new empty working folder, and an
environment that has only runtime paths. The CLI configuration points only at the local endpoint, with a fake local
token. The worker never signs in. If a CLI asks for a login, the job stops with the error "login_required".

Limits: the session file formats are internal to the pinned CLI versions. Tool calls keep their Hermes tool names
and argument text, so the facts are the same in all methods.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import JobFailed, ask, estimate_tokens, route_headers, wire_rows  # noqa: E402

LOGIN_PATTERNS = re.compile(r"(?i)(please run /login|not logged in|invalid api key|login required|sign in to|"
                            r"codex login|authentication required)")
RUNTIME_NAMES = ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC", "SYSTEMDRIVE", "NUMBER_OF_PROCESSORS",
                 "PROCESSOR_ARCHITECTURE", "OS")


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")


def cli_environment(home: Path, extra: dict[str, str]) -> dict[str, str]:
    """Runtime paths only. Profile folders point at the job home, so no user configuration or credential is read."""
    env = {key: value for key, value in os.environ.items() if key.upper() in RUNTIME_NAMES}
    env.update({key: str(home) for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME",
                                           "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "TEMP", "TMP")})
    env.update(extra)
    return env


def is_link(path: Path) -> bool:
    """True for a symbolic link or a Windows junction."""
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def group_options() -> dict:
    """Popen options that start a command in its own process group, so kill_tree can stop its descendants."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(process: subprocess.Popen) -> None:
    """Stop the process and every descendant. A CLI can start a child (node starts the Codex binary), and a
    descendant that runs on would keep using the one-slot server during the next job."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def run_tree(command: list[str], timeout: float, **options) -> subprocess.CompletedProcess:
    """subprocess.run with captured output, where a timeout stops the whole process tree (kill_tree), not only the
    first process. On a timeout it raises subprocess.TimeoutExpired with the output read so far."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **group_options(), **options)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            stdout = stderr = None
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run_cli(command: list[str], cwd: Path, env: dict[str, str], timeout_s: float) -> dict:
    started = time.monotonic()
    try:
        done = run_tree(command, timeout_s, cwd=cwd, env=env, stdin=subprocess.DEVNULL, encoding="utf-8",
                        errors="replace")
        text = (done.stdout or "") + "\n" + (done.stderr or "")
        result = {"exit_code": done.returncode, "timeout": False, "text": text}
    except subprocess.TimeoutExpired as error:
        text = str(error.stdout or "") + "\n" + str(error.stderr or "")
        result = {"exit_code": None, "timeout": True, "text": text}
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    result["login_prompt"] = bool(LOGIN_PATTERNS.search(result["text"]))
    return result


# Codex CLI.

def codex_rollout(fixture: dict, session_id: str, cwd: Path, version: str, provider: str, base: datetime) -> list:
    """The fixture as a Codex rollout file: session metadata, then one response item for each row."""
    clock = [base]

    def stamp() -> str:
        clock[0] += timedelta(milliseconds=5)
        return iso(clock[0])

    lines = [{"timestamp": stamp(), "type": "session_meta", "payload": {
        "session_id": session_id, "id": session_id, "timestamp": iso(base), "cwd": str(cwd),
        "originator": "codex_exec", "cli_version": version, "source": "exec", "thread_source": "user",
        "model_provider": provider}}]
    for row in fixture["rows"]:
        role = row["role"]
        if role == "user":
            lines.append({"timestamp": stamp(), "type": "response_item", "payload": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": row["content"]}]}})
            lines.append({"timestamp": stamp(), "type": "event_msg", "payload": {
                "type": "user_message", "message": row["content"], "images": [], "local_images": [],
                "text_elements": []}})
        elif role == "assistant":
            if row.get("content"):
                lines.append({"timestamp": stamp(), "type": "response_item", "payload": {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": row["content"]}]}})
            for call in row.get("tool_calls") or []:
                lines.append({"timestamp": stamp(), "type": "response_item", "payload": {
                    "type": "function_call", "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"], "call_id": call["id"]}})
        elif role == "tool":
            lines.append({"timestamp": stamp(), "type": "response_item", "payload": {
                "type": "function_call_output", "call_id": row["tool_call_id"], "output": row["content"]}})
    return lines


def codex_message_text(item: dict) -> str:
    return "".join(part.get("text", "") for part in item.get("content") or []
                   if part.get("type") in ("input_text", "output_text", "text"))


def codex_text(value) -> str:
    """A tool output or input as text: a string as it is, else its JSON form."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def codex_rows(items: list[dict]) -> list[dict]:
    """Chat rows for the items of a Codex replacement history. Every kind that Codex keeps becomes a row: messages,
    tool calls (function, custom, local shell, web search) as assistant tool calls, tool outputs as tool rows, and
    the text of reasoning items. An unknown kind fails the job, so that no kept item is lost without a trace."""
    rows: list[dict] = []

    def call(call_id, name, arguments) -> None:
        tool_call = {"id": str(call_id or f"call_{len(rows)}"), "type": "function",
                     "function": {"name": str(name or "tool"), "arguments": codex_text(arguments)}}
        last = rows[-1] if rows else None
        if last and last["role"] == "assistant":
            last.setdefault("tool_calls", []).append(tool_call)
        else:
            rows.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})

    for item in items:
        kind = item.get("type")
        if kind == "message":
            role = item.get("role")
            text = codex_message_text(item)
            if role in ("user", "assistant"):
                rows.append({"role": role, "content": text})
            elif role in ("developer", "system"):
                # A mid-history instruction: a user row with its role named (a chat template allows one system row).
                rows.append({"role": "user", "content": f"[{role}] {text}"})
            else:
                raise JobFailed(f"codex_context_item:message_{role}")
        elif kind == "function_call":
            call(item.get("call_id"), item.get("name"), item.get("arguments"))
        elif kind == "custom_tool_call":
            call(item.get("call_id"), item.get("name"), item.get("input"))
        elif kind == "local_shell_call":
            call(item.get("call_id"), "local_shell", item.get("action"))
        elif kind == "web_search_call":
            call(item.get("id"), "web_search", item.get("action"))
        elif kind in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output")
            if isinstance(output, dict) and "content" in output:
                output = output["content"]
            rows.append({"role": "tool", "tool_call_id": str(item.get("call_id")), "content": codex_text(output)})
        elif kind == "reasoning":
            text = "\n".join(part.get("text", "") for part in (item.get("summary") or []) + (item.get("content") or [])
                             if isinstance(part, dict) and part.get("text"))
            if text:
                rows.append({"role": "assistant", "content": text})
        else:
            raise JobFailed(f"codex_context_item:{kind}")
    return rows


def codex_context(lines: list[dict]) -> tuple[list[dict] | None, dict]:
    """The context after the last compaction: the replacement history of the last compacted item, as chat rows."""
    compacted = [index for index, line in enumerate(lines) if line.get("type") == "compacted"]
    if not compacted:
        return None, {}
    index = compacted[-1]
    payload = lines[index]["payload"]
    rows = codex_rows(payload.get("replacement_history") or [])
    if not payload.get("replacement_history") and payload.get("message"):
        rows.append({"role": "user", "content": payload["message"]})
    # The compaction time: from the start of the turn that compacted to the compacted item.
    started = None
    for line in reversed(lines[:index]):
        if line.get("type") == "event_msg" and (line.get("payload") or {}).get("type") == "task_started":
            started = line["timestamp"]
            break
    meta = {"summary_chars": len(payload.get("message") or "")}
    if started:
        meta["compress_model_s"] = round((parse_iso(lines[index]["timestamp"]) - parse_iso(started))
                                         .total_seconds(), 3)
    return rows, meta


def codex_endpoint(endpoint: str) -> tuple[str, dict[str, str]]:
    """The Codex base_url and query_params of an endpoint. Codex adds /responses to base_url, so the query (for
    example an api-version) goes into query_params, which Codex puts after the path. A fragment is not sent over
    HTTP and is left out. A repeated query key stops the job (query_params has one value per key)."""
    parts = urllib.parse.urlsplit(endpoint)
    pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    if len({key for key, _value in pairs}) != len(pairs):
        raise ValueError("endpoint_query_repeated_key")
    return urllib.parse.urlunsplit(parts._replace(query="", fragment="")), dict(pairs)


def run_codex(job: dict, fixture: dict, work: Path, out: dict) -> None:
    home, cwd = work / "codex-home", work / "cwd"
    home.mkdir()
    cwd.mkdir()
    provider = "benchlocal"
    base_url, query = codex_endpoint(job["endpoint"])
    (home / "config.toml").write_text("\n".join([
        f'model = {json.dumps(job["model"])}',
        f'model_provider = "{provider}"',
        f'model_context_window = {int(job["context_length"])}',
        'approval_policy = "never"',
        'sandbox_mode = "read-only"',
        "",
        f"[model_providers.{provider}]",
        'name = "Local benchmark endpoint"',
        f'base_url = {json.dumps(base_url)}',
        'wire_api = "responses"',
        *(["query_params = {" + ", ".join(f"{json.dumps(key)} = {json.dumps(value)}" for key, value in query.items())
           + "}"] if query else []),
        # The same route as the Hermes jobs and the questions: the key and the allowed extra headers.
        'env_key = "WC_BENCH_API_KEY"',
        "http_headers = {" + ", ".join(f"{json.dumps(key)} = {json.dumps(value)}"
                                        for key, value in route_headers(job).items()) + "}",
        ""]), encoding="utf-8")
    env = cli_environment(home, {"CODEX_HOME": str(home), "WC_BENCH_API_KEY": job["api_key"]})
    session_id = str(uuid.uuid4())
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    day = home / "sessions" / base.strftime("%Y") / base.strftime("%m") / base.strftime("%d")
    day.mkdir(parents=True)
    rollout = day / f"rollout-{base.strftime('%Y-%m-%dT%H-%M-%S')}-{session_id}.jsonl"
    lines = codex_rollout(fixture, session_id, cwd, job["cli_version"], provider, base)
    seeded = len(lines)
    rollout.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    common = [*job["cli_command"], "exec", "--skip-git-repo-check",
              "-c", f"developer_instructions={json.dumps(fixture['system'])}"]
    timeout_s = float(job.get("cli_timeout_s", 1800))
    item = {"cycle": 1, "rows_before_main": len(fixture["rows"]),
            "est_tokens_before_main": estimate_tokens(text_rows_all(fixture["rows"]))}
    out["cycles"].append(item)
    # The main turn: no compaction (a limit above any request size).
    main = run_cli(common + ["-c", "model_auto_compact_token_limit=100000000", "resume", session_id,
                             fixture["resume"]], cwd, env, timeout_s)
    item.update(main_s=main["elapsed_s"], main_exit=main["exit_code"], main_timeout=main["timeout"])
    lines = [json.loads(line) for line in rollout.read_text(encoding="utf-8").splitlines() if line.strip()]
    # The reply of the main turn: only the rollout lines that the main turn added.
    reply = codex_main_reply(lines[seeded:])
    check_main(item, main, reply=reply)
    if any(line.get("type") == "compacted" for line in lines):
        # Only the planned compaction step may compact.
        raise JobFailed("main_turn_compacted")
    compaction_input(item, fixture, codex_added_rows(lines[seeded:], fixture["resume"]))
    # The manual compaction: the app server request thread/compact/start (what /compact sends). It starts a
    # compaction turn only; no model turn follows.
    compact = codex_app_server_compact([*job["cli_command"], "app-server", "-c",
                                        f"developer_instructions={json.dumps(fixture['system'])}"],
                                       session_id, cwd, env, timeout_s)
    item["compact_status"] = compact["status"]
    if compact["login_prompt"]:
        raise LoginRequired()
    lines = [json.loads(line) for line in rollout.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, meta = codex_context(lines)
    finish(item, job, fixture, rows, compact, meta, path="codex")


def codex_app_server_compact(command: list[str], thread_id: str, cwd: Path, env: dict[str, str],
                             timeout_s: float) -> dict:
    """Resume the thread in a Codex app server (JSON lines on stdio) and send thread/compact/start. Wait for the
    turn/completed notification. One deadline (timeout_s from the start) covers initialize, resume, and the
    compaction turn. Return the time, the turn status, and safe error fields; no message text."""
    started = time.monotonic()
    deadline = started + timeout_s
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, encoding="utf-8", errors="replace", **group_options())
    messages: list[dict] = []
    stderr: list[str] = []
    ready = threading.Condition()

    def read_stdout():
        for line in process.stdout:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            with ready:
                messages.append(item)
                ready.notify_all()

    def read_stderr():
        for line in process.stderr:
            stderr.append(line)

    for target in (read_stdout, read_stderr):
        threading.Thread(target=target, daemon=True).start()

    def send(item: dict) -> None:
        process.stdin.write(json.dumps(item) + "\n")
        process.stdin.flush()

    def wait(predicate) -> dict | None:
        """The first message that matches, or None at the deadline or when the app server stopped."""
        with ready:
            while True:
                found = next((item for item in messages if predicate(item)), None)
                left = deadline - time.monotonic()
                if found is not None or left <= 0 or process.poll() is not None:
                    return found
                ready.wait(min(left, 1.0))

    def stalled() -> bool:
        """True when the app server still runs at the deadline (a timeout, not a stop)."""
        return process.poll() is None and time.monotonic() >= deadline

    result = {"status": None, "timeout": False, "exit_code": None, "error": None}
    try:
        send({"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "wc-bench", "version": "1"}}})
        if wait(lambda item: item.get("id") == 1) is None:
            result["timeout"] = stalled()
            result["error"] = "initialize_failed"
        else:
            send({"method": "initialized"})
            send({"id": 2, "method": "thread/resume", "params": {"threadId": thread_id, "cwd": str(cwd)}})
            resumed = wait(lambda item: item.get("id") == 2)
            if resumed is None or "error" in resumed:
                result["timeout"] = resumed is None and stalled()
                result["error"] = "resume_failed"
            else:
                send({"id": 3, "method": "thread/compact/start", "params": {"threadId": thread_id}})
                done = wait(lambda item: item.get("method") == "turn/completed")
                if done is None:
                    result["timeout"] = stalled()
                    result["error"] = "no_turn_completed"
                else:
                    turn = (done.get("params") or {}).get("turn") or {}
                    result["status"] = turn.get("status")
                    error = turn.get("error")
                    if error:
                        result["error"] = str((error or {}).get("codexErrorInfo") or "turn_error")[:200]
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            kill_tree(process)
            process.wait(timeout=15)
        result["exit_code"] = process.returncode
    errors = " ".join(str((item.get("params") or {}).get("message") or "") for item in messages
                      if item.get("method") in ("error", "warning"))
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    result["login_prompt"] = bool(LOGIN_PATTERNS.search("".join(stderr) + " " + errors))
    return result


# Claude Code.

def claude_project_dir(config: Path, cwd: Path) -> Path:
    return config / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def claude_transcript(fixture: dict, session_id: str, cwd: Path, version: str, model: str,
                      base: datetime) -> list[dict]:
    """The fixture as a Claude Code transcript: one entry for each message, linked by parentUuid."""
    clock = [base]
    entries: list[dict] = []
    parent = [None]

    def add(kind: str, message: dict, **extra) -> None:
        clock[0] += timedelta(milliseconds=5)
        entry_id = str(uuid.uuid4())
        entries.append({"parentUuid": parent[0], "isSidechain": False, "type": kind, "message": message,
                        "uuid": entry_id, "timestamp": iso(clock[0]), "userType": "external",
                        "entrypoint": "sdk-cli", "cwd": str(cwd), "sessionId": session_id, "version": version,
                        "gitBranch": "", **extra})
        parent[0] = entry_id

    for number, row in enumerate(fixture["rows"]):
        role = row["role"]
        if role == "user":
            add("user", {"role": "user", "content": row["content"]})
        elif role == "assistant":
            blocks = [{"type": "text", "text": row["content"]}] if row.get("content") else []
            for call in row.get("tool_calls") or []:
                try:
                    arguments = json.loads(call["function"]["arguments"])
                except ValueError:
                    arguments = {"arguments": call["function"]["arguments"]}
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                               "input": arguments})
            add("assistant", {"id": f"msg_bench_{number:04d}", "type": "message", "role": "assistant",
                              "model": model, "content": blocks,
                              "stop_reason": "tool_use" if row.get("tool_calls") else "end_turn",
                              "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}},
                requestId=f"req_bench_{number:04d}")
        elif role == "tool":
            add("user", {"role": "user", "content": [{"type": "tool_result", "tool_use_id": row["tool_call_id"],
                                                      "content": row["content"]}]},
                toolUseResult=row["content"])
    return entries


def claude_text(content, thinking: bool = False) -> str:
    """The text of a Claude Code message: text, tool calls, and tool results; with thinking=True also the thinking
    text (a context keeps it, a reply text does not). An unknown block kind fails the job."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            if thinking:
                parts.append(block.get("thinking", ""))
        elif block.get("type") in ("redacted_thinking", "image", "document"):
            # No text that a fact check or the questions can use.
            continue
        elif block.get("type") == "tool_use":
            parts.append(f"[tool call {block.get('name')}: {json.dumps(block.get('input'), ensure_ascii=False)}]")
        elif block.get("type") == "tool_result":
            inner = block.get("content")
            parts.append(inner if isinstance(inner, str) else claude_text(inner, thinking))
        else:
            raise JobFailed(f"claude_context_block:{block.get('type')}")
    return "\n".join(part for part in parts if part)


def claude_context(entries: list[dict]) -> tuple[list[dict] | None, dict]:
    """The context after the last compact boundary: the summary and any kept messages, without command rows."""
    boundaries = [index for index, entry in enumerate(entries)
                  if entry.get("type") == "system" and entry.get("subtype") == "compact_boundary"]
    if not boundaries:
        return None, {}
    index = boundaries[-1]
    rows = []
    summary_chars = 0
    for entry in entries[index + 1:]:
        if entry.get("type") not in ("user", "assistant") or entry.get("isMeta"):
            continue
        message = entry.get("message") or {}
        text = claude_text(message.get("content"), thinking=True)
        if "<command-name>" in text or "<local-command-stdout>" in text:
            continue
        if entry.get("isCompactSummary"):
            summary_chars += len(text)
        rows.append({"role": message.get("role") or entry["type"], "content": text})
    meta = {"summary_chars": summary_chars,
            "compact_metadata": {key: value for key, value in (entries[index].get("compactMetadata") or {}).items()
                                 if isinstance(value, (str, int, float, bool, type(None)))}}
    # The compaction time: the duration that Claude Code records, else from the queued /compact to the boundary.
    duration = meta["compact_metadata"].get("durationMs")
    if isinstance(duration, (int, float)):
        meta["compress_model_s"] = round(duration / 1000, 3)
    else:
        for entry in reversed(entries[:index]):
            if entry.get("type") == "queue-operation" and entry.get("content") == "/compact":
                meta["compress_model_s"] = round((parse_iso(entries[index]["timestamp"])
                                                  - parse_iso(entry["timestamp"])).total_seconds(), 3)
                break
    return rows, meta


def run_claude(job: dict, fixture: dict, work: Path, out: dict) -> None:
    home, cwd = work / "claude-home", work / "cwd"
    home.mkdir()
    cwd.mkdir()
    config = home / ".claude"
    env = cli_environment(home, {
        "CLAUDE_CONFIG_DIR": str(config), "ANTHROPIC_BASE_URL": job["anthropic_base_url"],
        # The same route as the Hermes jobs and the questions: the key and the allowed extra headers.
        "ANTHROPIC_API_KEY": job["api_key"], "ANTHROPIC_MODEL": job["model"],
        "ANTHROPIC_CUSTOM_HEADERS": "\n".join(f"{key}: {value}" for key, value in route_headers(job).items()),
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": job["model"], "ANTHROPIC_DEFAULT_SONNET_MODEL": job["model"],
        "ANTHROPIC_DEFAULT_OPUS_MODEL": job["model"], "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "MAX_THINKING_TOKENS": "0",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"})
    session_id = str(uuid.uuid4())
    project = claude_project_dir(config, cwd)
    project.mkdir(parents=True)
    transcript = project / f"{session_id}.jsonl"
    entries = claude_transcript(fixture, session_id, cwd, job["cli_version"], job["model"],
                                datetime.now(timezone.utc) - timedelta(hours=1))
    seeded = len(entries)
    transcript.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
                          encoding="utf-8")
    # No tools, as in the Hermes jobs. No setting sources, so no user or project settings are read.
    common = [*job["cli_command"], "-p", "--setting-sources", "", "--tools", "", "--output-format", "json",
              "--append-system-prompt", fixture["system"], "--resume", session_id]
    timeout_s = float(job.get("cli_timeout_s", 1800))
    item = {"cycle": 1, "rows_before_main": len(fixture["rows"]),
            "est_tokens_before_main": estimate_tokens(text_rows_all(fixture["rows"]))}
    out["cycles"].append(item)
    main = run_cli(common + [fixture["resume"]], cwd, env, timeout_s)
    item.update(main_s=main["elapsed_s"], main_exit=main["exit_code"], main_timeout=main["timeout"],
                main_result=cli_result(main["text"]))
    entries = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
    # The reply of the main turn: only the transcript entries that the main turn added.
    reply = claude_main_reply(entries[seeded:])
    check_main(item, main, item["main_result"], reply=reply)
    if any(entry.get("type") == "system" and entry.get("subtype") == "compact_boundary" for entry in entries):
        # Only the planned compaction step may compact (Claude Code can compact on its own before a turn).
        raise JobFailed("main_turn_compacted")
    compaction_input(item, fixture, claude_added_rows(entries[seeded:], fixture["resume"]))
    compact = run_cli(common + ["/compact"], cwd, env, timeout_s)
    if compact["login_prompt"]:
        raise LoginRequired()
    item["compact_result"] = cli_result(compact["text"])
    lines = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, meta = claude_context(lines)
    finish(item, job, fixture, rows, compact, meta, path="claude_code")


def cli_result(text: str) -> dict:
    """Safe fields of the JSON result line of claude -p: no reply text."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"type"' in line:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            return {key: data.get(key) for key in ("subtype", "is_error", "api_error_status", "num_turns",
                                                   "stop_reason", "terminal_reason", "duration_ms")}
    return {}


# Shared.

class LoginRequired(Exception):
    """The CLI asked for a login. The benchmark never signs in."""


class MainTurnFailed(Exception):
    """The main turn of the CLI timed out, exited with an error, or reported an error result. The job stops before
    the compaction: a compaction of an incomplete session is not comparable."""


def compaction_input(item: dict, fixture: dict, added: list[dict]) -> None:
    """The size of the context that the compaction gets: the session rows, the main user row, and every row that
    the main turn added to the session (codex_added_rows, claude_added_rows): the reply, and also reasoning,
    thinking, tool calls, and tool results. The size before the main turn stays in est_tokens_before_main."""
    rows = text_rows_all(fixture["rows"]) + [{"role": "user", "content": fixture["resume"]}] + added
    item.update(rows_before=len(rows), est_tokens_before=estimate_tokens(rows))


# The kinds of Codex session items that codex_rows reads.
CODEX_ITEM_KINDS = ("message", "function_call", "custom_tool_call", "local_shell_call", "web_search_call",
                    "function_call_output", "custom_tool_call_output", "reasoning")


def codex_added_rows(lines: list[dict], resume: str) -> list[dict]:
    """The rows of the session items that the Codex main turn added (rollout response items), without the main user
    message (compaction_input adds it). An item of another kind or role is counted as its JSON text, so that the
    size does not leave it out. Encrypted reasoning has no text and adds only its summary text."""
    items = []
    for line in lines:
        item = line.get("payload") if line.get("type") == "response_item" else None
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "user" and codex_message_text(item) == resume:
            continue
        if item.get("type") not in CODEX_ITEM_KINDS or (
                item.get("type") == "message" and item.get("role") not in ("user", "assistant", "developer", "system")):
            item = {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": f"[{item.get('type')}] {codex_text(item)}"}]}
        items.append(item)
    return codex_rows(items)


def claude_added_rows(entries: list[dict], resume: str) -> list[dict]:
    """The rows of the transcript entries that the Claude Code main turn added, with thinking, tool calls, and tool
    results, without the main user message (compaction_input adds it) and without command rows."""
    rows = []
    for entry in entries:
        if entry.get("type") not in ("user", "assistant") or entry.get("isMeta"):
            continue
        message = entry.get("message") or {}
        role = message.get("role") or entry["type"]
        text = claude_text(message.get("content"), thinking=True)
        if (role == "user" and text == resume) or "<command-name>" in text or "<local-command-stdout>" in text:
            continue
        rows.append({"role": role, "content": text})
    return rows


def codex_main_reply(lines: list[dict]) -> str:
    for line in reversed(lines):
        payload = line.get("payload") or {}
        if line.get("type") == "response_item" and payload.get("type") == "message" and payload.get("role") == \
                "assistant":
            return codex_message_text(payload)
    return ""


def claude_main_reply(entries: list[dict]) -> str:
    for entry in reversed(entries):
        if entry.get("type") == "assistant" and not entry.get("isMeta"):
            return claude_text((entry.get("message") or {}).get("content"))
    return ""


def check_main(item: dict, main: dict, result: dict | None = None, reply: str = "") -> None:
    """Fail the job when the main turn timed out, exited with an error, reported an error result, or gave no
    assistant reply text."""
    if main["login_prompt"]:
        raise LoginRequired()
    reason = ("timeout" if main["timeout"] else f"exit_{main['exit_code']}" if main["exit_code"] != 0
              else "error_result" if (result or {}).get("is_error")
              else "empty_reply" if not str(reply or "").strip() else None)
    if reason:
        item["main_failed"] = True
        raise MainTurnFailed(reason)


def text_rows_all(rows: list[dict]) -> list[dict]:
    from worker import wire_rows
    return wire_rows(rows)


def compaction_failure(item: dict, compact: dict, path: str) -> str | None:
    """Why the compaction command itself did not complete successfully (None when it did). A CLI can write a
    compacted session record and still time out, exit with an error, or report a failed turn: such rows are not a
    successful compaction. Codex: the app server turn must have the status "completed" and no error. Claude Code:
    the command must exit with 0 and report no error result."""
    if compact["timeout"]:
        return "cli_timeout"
    if path == "codex":
        if compact.get("error"):
            return str(compact["error"])[:200]
        if compact.get("status") != "completed":
            return f"turn_{compact.get('status') or 'missing'}"
        return None
    if compact["exit_code"] != 0:
        return f"exit_{compact['exit_code']}"
    if (item.get("compact_result") or {}).get("is_error"):
        return "error_result"
    return None


def finish(item: dict, job: dict, fixture: dict, rows: list[dict] | None, compact: dict, meta: dict,
           path: str) -> None:
    item.update(compress_wall_s=compact["elapsed_s"], compress_exit=compact["exit_code"],
                compress_timeout=compact["timeout"], compress_s=meta.get("compress_model_s"))
    failure = compaction_failure(item, compact, path)
    if failure:
        # Rows that a failed command wrote are not the result of a compaction: no questions, no sizes.
        rows = None
    item["compacted"] = bool(rows)
    item["compress_error"] = None if rows else (failure or "no_compaction_record")
    item["outcome"] = {"path": path if rows else "aborted", "summary_chars": meta.get("summary_chars"),
                       **(meta.get("compact_metadata") or {})}
    # The chat fields of every row (also an assistant row with tool calls only, and the tool rows): the questions
    # and the size after compaction use the same rows.
    rows = wire_rows(rows or [])
    item["removed"] = max(item["rows_before"] - len(rows), 0) if item["compacted"] else 0
    item["rows_after"] = len(rows)
    item["est_tokens_after"] = estimate_tokens(rows)
    item["history"] = rows
    item["qa"] = ask(job, fixture["system"], rows) if item["compacted"] else {
        "batches": [], "calls": [], "error": "skipped_not_compacted"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    fixture = json.loads(Path(job["fixture_path"]).read_text(encoding="utf-8"))
    out = {"status": "failed", "error": None, "cycles": [], "python": sys.version.split()[0],
           "engine_name": job["engine"]}
    work = Path(args.out).resolve().parent
    try:
        if int(job.get("cycles", 1)) != 1:
            raise ValueError("CLI methods run one compaction cycle")
        (run_codex if job["engine"] == "codex_cli" else run_claude)(job, fixture, work, out)
        out["status"] = "ok"
    except LoginRequired:
        out["error"] = "login_required"
    except MainTurnFailed as error:
        out["error"] = f"main_turn_failed:{error}"
    except JobFailed as error:
        out["error"] = str(error)
    except Exception as error:
        out["error"] = type(error).__name__
        out["trace"] = "".join(traceback.format_exception(type(error), None, error.__traceback__))[-4000:]
    # The report replaces the file in one step: a stop during the write leaves no partial report.
    temporary = Path(args.out).with_name(Path(args.out).name + ".tmp")
    temporary.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
