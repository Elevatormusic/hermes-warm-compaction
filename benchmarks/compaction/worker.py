"""Run one benchmark job in a fresh process: the real Hermes agent loop, one compaction method, and the answers.

The parent process (run.py) starts this file with the Python of a Hermes virtual environment. HERMES_HOME points to
a new temporary home. The worker writes the Hermes configuration, loads the plugin from that home with the Hermes
plugin manager, sends the main request with AIAgent.run_conversation, compacts with the Hermes manual compress
entry, and asks the questions. It does not patch, subclass, or wrap Hermes code.

The output file has the answer text and the compacted history text, because the parent scores them. The parent
keeps only scores and metadata in the result records. All conversations are synthetic.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import missing_answers, prompt_sha256, qa_prompt  # noqa: E402

WIRE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")
# The only Hermes settings that hermes_config can set (an allow-list): extra request headers of the main model
# route, for a local gateway. Every other key (a route, an engine, an auxiliary model, a plugin) is refused.
HERMES_CONFIG_ALLOWED = ("model.headers",)
# Header names that the harness, the HTTP client, or a CLI sets for the route (lower case; names are compared
# without case). model.headers cannot set them: a second source with another value would replace one silently.
RESERVED_HEADERS = ("authorization", "content-type", "content-length", "host", "x-api-key", "anthropic-version")
# The only QA request fields that qa_extra_body can set: sampler and chat template options. The model, the messages,
# the output limit, the temperature, and streaming stay fixed.
QA_EXTRA_ALLOWED = ("chat_template_kwargs", "seed", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty",
                    "frequency_penalty")


class JobFailed(Exception):
    """A sub-step of the job failed (not the compaction under test). The job fails and is not scored."""


def estimate_tokens(rows) -> int:
    """A fixed size estimate for all methods: characters of the JSON form, divided by 4."""
    return len(json.dumps(rows, ensure_ascii=False)) // 4


def wire_rows(rows: list) -> list:
    """The chat fields of stored rows. Hermes keeps private keys (with a leading underscore) in stored rows."""
    out = []
    for row in rows:
        item = {key: copy.deepcopy(row[key]) for key in WIRE_KEYS if key in row and row[key] is not None}
        if item.get("role") == "assistant" and not item.get("tool_calls"):
            item.pop("tool_calls", None)
        if not isinstance(item.get("content"), (str, list)):
            item["content"] = "" if item.get("content") is None else str(item.get("content"))
        out.append(item)
    return out


# An HTTP field name is a token (RFC 9110): ASCII letters, digits, and these signs only. Other names (a space, a
# colon, a non-ASCII letter) are refused or read differently by HTTP clients.
HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


def header_conflicts(headers) -> list[str]:
    """The problems of model.headers: not a mapping, a name that is not an HTTP token (HEADER_NAME), a value that
    is not a string or has a character other than visible ASCII, a space, or a tab, a reserved name
    (RESERVED_HEADERS), or two names that differ only in case. Names are compared without case. A non-ASCII value
    is refused because the clients of the run encode it differently (urllib, which sends the questions, encodes
    header values as Latin-1 and fails on other characters after the main turn and the compaction)."""
    if not isinstance(headers, dict):
        return ["model.headers"]
    found, seen = [], set()
    for key, value in headers.items():
        name = str(key)
        if not isinstance(key, str) or not HEADER_NAME.fullmatch(key) or not isinstance(value, str) or any(
                not (32 <= ord(char) <= 126 or char == "\t") for char in value):
            found.append(f"model.headers.{name}")
        elif name.strip().lower() in RESERVED_HEADERS or name.strip().lower() in seen:
            found.append(f"model.headers.{name}")
        seen.add(name.strip().lower())
    return found


def route_headers(job: dict) -> dict[str, str]:
    """The allowed extra request headers of the model route (hermes_config.model.headers). The main turn, the
    compaction, the questions, and the CLI methods send them all. A reserved, duplicate, or malformed name fails
    the job (the run refuses it before the first job)."""
    headers = ((job.get("hermes_config") or {}).get("model") or {}).get("headers") or {}
    conflicts = header_conflicts(headers)
    if conflicts:
        # The error keeps a count only: the header names come from the user configuration.
        raise JobFailed(f"route_headers_refused:{len(conflicts)}")
    return dict(headers)


def post_chat(endpoint: str, api_key: str, body: dict, timeout_s: float, headers: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=data, headers={
        **(headers or {}), "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(job: dict, system: str, rows: list) -> dict:
    """Ask the questions in the batches of the gold plan (job "qa_batches"). Each batch is one request with the
    compacted history and one user row."""
    questions = {item["n"]: item for item in job["questions"]}
    if sorted(n for batch in job["qa_batches"] for n in batch) != sorted(questions):
        raise JobFailed("qa_plan_mismatch")
    # The extra headers of the route, checked before the first request (a refused name fails the job).
    headers = route_headers(job)
    result = {"batches": [], "calls": [], "error": None}
    for numbers in job["qa_batches"]:
        part = [questions[n] for n in numbers]
        text = qa_prompt([(item["n"], item["text"]) for item in part])
        messages = [{"role": "system", "content": system}, *wire_rows(rows), {"role": "user", "content": text}]
        conflicts = qa_body_conflicts(job.get("qa_extra_body") or {})
        if conflicts:
            raise JobFailed(f"qa_extra_body_protected:{','.join(conflicts)}")
        body = {**(job.get("qa_extra_body") or {}), "model": job["model"], "messages": messages, "temperature": 0,
                "max_tokens": int(job.get("qa_max_tokens", 2048))}
        started = time.monotonic()
        try:
            reply = post_chat(job["endpoint"], job["api_key"], body, float(job.get("qa_timeout_s", 600)),
                              headers)
        except Exception as error:
            # A failed request (timeout, HTTP error) gives no answers: the job fails, it is not scored as wrong.
            result["error"] = f"qa_request:{type(error).__name__}"
            raise JobFailed(result["error"]) from None
        # Only a full answer is accepted: a reply of another shape fails the job, it is not scored as wrong.
        if not isinstance(reply, dict) or not isinstance(reply.get("choices"), list) or not reply["choices"]:
            raise JobFailed("qa_request:no_choices")
        choice = reply["choices"][0] if isinstance(reply["choices"][0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise JobFailed("qa_request:no_content")
        if choice.get("finish_reason") != "stop":
            # "length" cuts the answers; "tool_calls" and "content_filter" are not answers.
            raise JobFailed(f"qa_request:finish_{str(choice.get('finish_reason'))[:24]}")
        if missing_answers(content, [item["n"] for item in part]):
            raise JobFailed("qa_request:incomplete_answers")
        usage = reply.get("usage") or {}
        result["calls"].append({
            "elapsed_s": round(time.monotonic() - started, 3), "ok": True,
            "finish_reason": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")})
        result["batches"].append({"numbers": [item["n"] for item in part], "text": content,
                                  "prompt_sha256": prompt_sha256(text)})
    return result


def qa_body_conflicts(extra: dict) -> list[str]:
    """The qa_extra_body keys that are not on the allow-list."""
    return sorted(str(key) for key in extra if key not in QA_EXTRA_ALLOWED)


def override_conflicts(override: dict, prefix: str = "") -> list[str]:
    """The leaf key paths of hermes_config that are not on the allow-list (HERMES_CONFIG_ALLOWED), and the
    model.headers names that header_conflicts refuses."""
    found = []
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if path == "model.headers":
            found += header_conflicts(value)
            continue
        if any(path == allowed or path.startswith(allowed + ".") for allowed in HERMES_CONFIG_ALLOWED):
            continue
        if isinstance(value, dict) and value and any(allowed.startswith(path + ".")
                                                     for allowed in HERMES_CONFIG_ALLOWED):
            found += override_conflicts(value, path)
        else:
            found.append(path)
    return found


def merge(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value


def base_config(job: dict) -> dict:
    """The Hermes configuration that the benchmark sets: the main model route, the engine, and the plugin settings."""
    return {
        "model": {"default": job["model"], "model": job["model"], "provider": "custom",
                  "base_url": job["endpoint"], "api_key": job["api_key"], "api_mode": "chat_completions",
                  "context_length": int(job["context_length"]), "streaming": False},
        "context": {"engine": "warm_compaction" if job["engine"] == "warm_compaction" else "compressor"},
        "tools": {"tool_search": {"enabled": "off"}},
        # The control (engine "none") turns automatic compaction off: no step of it may compact.
        "compression": {"enabled": job["engine"] != "none", "progress_notices": False,
                        "threshold": float(job["threshold"])},
        "auxiliary": {"transient_retries": 0, "warm_compaction": {"provider": "auto"},
                      "title_generation": {"enabled": False, "model_upgrade_enabled": False}},
        "plugins": {"enabled": ["warm_compaction"] if job["engine"] == "warm_compaction" else [],
                    "hook_callback_timeout": 0,
                    "entries": {"warm_compaction": {"settings": dict(job.get("settings") or {})}}},
    }


def write_config(home: Path, job: dict) -> None:
    """Write base_config with the allowed hermes_config additions. An override of a benchmark setting fails before
    any other step."""
    config = base_config(job)
    conflicts = override_conflicts(job.get("hermes_config") or {})
    if conflicts:
        # The error keeps a count only: the key names come from the user configuration.
        raise JobFailed(f"hermes_config_protected:{len(conflicts)}")
    try:
        import yaml

        def dump(data):
            return yaml.safe_dump(data, sort_keys=False)
    except ImportError:  # Some Hermes builds ship ruamel.yaml only.
        from io import StringIO

        from ruamel.yaml import YAML

        def dump(data):
            writer = YAML(typ="safe", pure=True)
            writer.default_flow_style = False
            stream = StringIO()
            writer.dump(data, stream)
            return stream.getvalue()
    merge(config, job.get("hermes_config") or {})
    (home / "config.yaml").write_text(dump(config), encoding="utf-8")
    # A local model catalogue: host discovery makes no request for it.
    (home / "models_dev_cache.json").write_text(json.dumps({"synthetic": {"models": {}}}), encoding="utf-8")


def engine_outcome(engine) -> dict:
    """Metadata of the last compaction: the plugin status, or the scalar status fields of the built-in engine."""
    last = getattr(engine, "warm_last", None)
    if isinstance(last, dict):
        return {key: value for key, value in last.items() if isinstance(value, (str, int, float, bool, type(None)))}
    status = {}
    try:
        status = engine.get_status() or {}
    except Exception:
        status = {}
    return {key: value for key, value in status.items()
            if isinstance(value, (str, int, float, bool, type(None))) and "path" not in key.lower()}


class CompressionLockSkipped(Exception):
    """Another compaction held the compression lock, so the manual compaction did not run."""


def manual_compress(agent, messages: list) -> tuple[int, list]:
    """One manual /compress of the history, through the core that every Hermes surface (CLI, gateway, TUI, ACP)
    uses: agent/conversation_compression_manual.py::compress_now (documented in the agent/AGENTS.md of Hermes).
    No focus topic, no partial compaction. Return (the removed row count, the history after). A history shorter
    than MIN_MESSAGES, or a compaction that changes nothing, returns (0, the same rows). The caller finalizes the
    context-engine notification, as the Hermes surfaces do after their history commit."""
    from agent.conversation_compression_manual import MIN_MESSAGES, compress_now, parse_compress_args
    if len(messages) < MIN_MESSAGES:
        return 0, list(messages)
    result = compress_now(agent, list(messages), parse_compress_args(""), task_id=agent.session_id or "default")
    if result.status == "lock_skipped":
        raise CompressionLockSkipped()
    if result.status != "compressed":
        return 0, list(messages)
    return result.removed, list(result.after_messages)


def run_cycles(job: dict, fixture: dict, home: Path, out: dict) -> None:
    sys.path[:0] = [job["hermes_source"]]
    write_config(home, job)
    from hermes_cli import plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    from agent.conversation_compression import finalize_context_engine_compression_notification
    manager = plugins.get_plugin_manager()
    manager.discover_and_load()
    # The public plugin list of the manager (Hermes uses it in its own commands), not its private registry.
    out["plugins_loaded"] = sorted(item["name"] for item in manager.list_plugins() if item.get("enabled"))
    db = SessionDB(home / "state.db")
    session_id = "synthetic-bench-session"
    db.create_session(session_id, "cli")
    agent = AIAgent(base_url=job["endpoint"], api_key=job["api_key"], provider="custom", api_mode="chat_completions",
                    model=job["model"], enabled_toolsets=[], disabled_toolsets=[], quiet_mode=True,
                    skip_context_files=True, skip_memory=True, skip_background_review=True,
                    capabilities={"vision": False}, session_id=session_id, session_db=db,
                    max_tokens=int(job.get("main_max_tokens", 4096)), reasoning_config={"enabled": False},
                    platform="cli", cwd=str(home.parent), max_iterations=2)
    try:
        engine = agent.context_compressor
        out["engine_name"] = getattr(engine, "name", None)
        expected = "warm_compaction" if job["engine"] == "warm_compaction" else "compressor"
        if job["engine"] == "none" and getattr(agent, "compression_enabled", None) is not False:
            raise JobFailed("control_auto_compaction_on")
        if out["engine_name"] != expected:
            # For example, the plugin did not load and Hermes used its built-in compressor.
            raise JobFailed(f"engine_mismatch:{out['engine_name']}")
        history = copy.deepcopy(fixture["rows"])
        for cycle in range(int(job.get("cycles", 1))):
            item = {"cycle": cycle + 1}
            out["cycles"].append(item)
            if cycle > 0:
                history = list(history) + copy.deepcopy(fixture["continuations"][cycle - 1])
            # Before the main turn. The compaction input (rows_before, est_tokens_before) is set below.
            item["rows_before_main"] = len(history)
            item["est_tokens_before_main"] = estimate_tokens(wire_rows(history))
            started = time.monotonic()
            reply = agent.run_conversation(fixture["resume"], system_message=fixture["system"],
                                           conversation_history=history)
            item["main_s"] = round(time.monotonic() - started, 3)
            item["main_failed"] = bool(reply.get("failed"))
            if item["main_failed"]:
                # A failed main turn gives a different history from the other methods: the job fails here.
                raise MainTurnFailed()
            if not str(reply.get("final_response") or "").strip():
                # A main turn without reply text is not a complete turn: the job fails here too.
                item["main_failed"] = True
                raise MainTurnFailed("empty_reply")
            messages = list(reply.get("messages") or history)
            item["main_prompt_tokens"] = getattr(engine, "last_prompt_tokens", None)
            item["rows_after_main"] = len(messages)
            if job["engine"] == "none":
                # The control: the same main turn, no compaction. The answers use the complete history.
                if len(messages) < len(history):
                    raise JobFailed("control_history_shrank")
                item.update(rows_before=len(messages), est_tokens_before=estimate_tokens(wire_rows(messages)))
                item.update(compress_error=None, compress_s=None, removed=0, outcome={"path": "none"},
                            rows_after=len(messages), est_tokens_after=estimate_tokens(wire_rows(messages)),
                            compacted=False, history=wire_rows(messages))
                item["qa"] = ask(job, fixture["system"], messages)
                history = messages
                continue
            if len(messages) < len(history):
                # The engine compacted during the main turn (its preflight check before the main request). That is
                # the compaction of this cycle: the main request ran on the compacted history. No manual compress.
                # Its input: the history and the new user row of the main turn.
                source = list(history) + [{"role": "user", "content": fixture["resume"]}]
                item.update(rows_before=len(source), est_tokens_before=estimate_tokens(wire_rows(source)))
                # Its output: the compacted rows up to the new user row. The rows after it are the main turn
                # (the reply), which the input does not have; the size after compaction leaves them out.
                compacted = compaction_output(messages, fixture["resume"])
                item.update(compress_error=None, compress_s=None, removed=len(source) - len(compacted),
                            outcome={**engine_outcome(engine), "trigger": "preflight"},
                            rows_after=len(compacted), est_tokens_after=estimate_tokens(wire_rows(compacted)),
                            compacted=True, preflight=True, history=wire_rows(messages))
                item["compress_s"] = item["outcome"].get("elapsed_s")
                item["qa"] = ask(job, fixture["system"], messages)
                history = messages
                continue
            item.update(rows_before=len(messages), est_tokens_before=estimate_tokens(wire_rows(messages)))
            after = list(messages)
            started = time.monotonic()
            try:
                removed, after = manual_compress(agent, messages)
                finalize_context_engine_compression_notification(agent, committed=removed > 0)
                item["compress_error"] = None
            except Exception as error:
                removed = 0
                item["compress_error"] = type(error).__name__
                item["compress_trace"] = "".join(traceback.format_exception(type(error), None,
                                                                              error.__traceback__))[-3000:]
                try:
                    finalize_context_engine_compression_notification(agent, committed=False)
                except Exception:
                    pass
            item["compress_s"] = round(time.monotonic() - started, 3)
            item["removed"] = removed
            item["outcome"] = engine_outcome(engine)
            history = list(after)
            item["rows_after"] = len(history)
            item["est_tokens_after"] = estimate_tokens(wire_rows(history))
            item["compacted"] = removed > 0 and item["compress_error"] is None
            item["history"] = wire_rows(history)
            # An aborted or empty compaction keeps the input history: its answers would measure the full history.
            item["qa"] = ask(job, fixture["system"], history) if item["compacted"] else {
                "batches": [], "calls": [], "error": "skipped_not_compacted"}
    finally:
        agent.close()
        db.close()


class MainTurnFailed(Exception):
    """The main turn (run_conversation) reported a failure or gave no reply text. The job stops before the
    compaction."""


def compaction_output(messages: list, resume: str) -> list:
    """The rows of a preflight compaction: the history after the main turn up to its user row (the last user row
    with the main request text). Fail the job when that row is not there."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user" and messages[index].get("content") == resume:
            return messages[:index + 1]
    raise JobFailed("preflight_user_row_missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    fixture = json.loads(Path(job["fixture_path"]).read_text(encoding="utf-8"))
    out = {"status": "failed", "error": None, "cycles": [], "python": sys.version.split()[0]}
    try:
        run_cycles(job, fixture, Path(os.environ["HERMES_HOME"]), out)
        out["status"] = "ok"
    except MainTurnFailed as error:
        out["error"] = f"main_turn_failed:{error}" if str(error) else "main_turn_failed"
    except JobFailed as error:
        out["error"] = str(error)
    except Exception as error:
        out["error"] = type(error).__name__
        # The trace has code locations only when the message is left out; it stays in the local work folder.
        out["trace"] = "".join(traceback.format_exception(type(error), None, error.__traceback__))[-4000:]
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
