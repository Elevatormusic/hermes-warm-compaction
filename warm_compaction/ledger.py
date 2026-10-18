"""Work ledger: commands, their results, and file changes, extracted by code from the rows before the tail.

No model writes the ledger. A model summary keeps the goal and the state; the ledger keeps the hard facts that a
summary often drops: which command failed, with which error, and which files changed. The ledger is reference
data for the next model, not a list of tasks.
"""

from __future__ import annotations

import json
import re
from typing import Any
from collections.abc import Iterable

from .layout import COPY_HEADING, is_summary
from .rows import attr, estimate_tokens, plain_text, tool_calls_of

HEADING = "## Work ledger"
NOTE = "Recorded by the plugin from tool calls and results before this point. Reference only; not new tasks."
COMMANDS_HEADING = "### Commands"
FILES_HEADING = "### Files changed"
MAX_TOKENS = 1_000
COMMAND_CHARS = 160
DETAIL_CHARS = 200
PATH_CHARS = 200
# Argument keys that hold a shell command, and keys that hold a file path.
COMMAND_KEYS = ("command", "cmd")
PATH_KEYS = ("path", "file_path", "filepath", "filename", "target_file")
# The file action of a tool, from words in its name. The first match wins.
FILE_ACTIONS = (("delete", "deleted"), ("remove", "deleted"), ("move", "moved"), ("rename", "moved"),
                ("write", "written"), ("create", "written"), ("patch", "changed"), ("edit", "changed"),
                ("replace", "changed"), ("apply", "changed"))
# File headers of a V4A patch (the Hermes patch tool in patch mode).
PATCH_HEADER = re.compile(r"^\*\*\* (Update|Add|Delete|Move) File: (.+?)\s*$", re.MULTILINE)
PATCH_ACTIONS = {"Update": "changed", "Add": "written", "Delete": "deleted", "Move": "moved"}
# A line that reports a failure in tool output.
ERROR_LINE = re.compile(r"\b(error|errors|failed|failure|exception|traceback|fatal|denied|not found)\b", re.I)
# The exit code in plain text output ("exit code 2", "Exit status: 1").
EXIT_TEXT = re.compile(r"\bexit(?:ed)?(?: with)?(?: code| status)?[:= ]+(-?\d+)\b", re.I)


def _one_line(text: str, limit: int) -> str:
    """Collapse whitespace and keep at most limit characters (with an ellipsis when cut)."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _first_error(text: str) -> str:
    for line in text.splitlines():
        if line.strip() and ERROR_LINE.search(line):
            return _one_line(line, DETAIL_CHARS)
    return ""


def _outcome(result: str | None) -> tuple[str, bool]:
    """Return (outcome text, failed) for one tool result. The outcome is only what the result reports."""
    if result is None:
        return "no result recorded", False
    data: Any = None
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        pass
    if isinstance(data, dict):
        error = data.get("error")
        code = data.get("exit_code")
        output = data.get("output")
        output = output if isinstance(output, str) else ""
        if isinstance(code, int) and not isinstance(code, bool):
            detail = _one_line(str(error), DETAIL_CHARS) if error else (_first_error(output) if code else "")
            return f"exit {code}" + (f"; {detail}" if detail else ""), code != 0
        if error:
            return "error; " + _one_line(str(error), DETAIL_CHARS), True
        if data.get("success") is False:
            return "failed", True
        detail = _first_error(output)
        return ("exit unknown; " + detail if detail else "exit unknown"), False
    match = EXIT_TEXT.search(result)
    detail = _first_error(result)
    if match:
        code = int(match.group(1))
        return f"exit {code}" + (f"; {detail}" if code and detail else ""), code != 0
    if result.lstrip().lower().startswith("error"):
        return "error; " + _one_line(result.lstrip(), DETAIL_CHARS), True
    return ("exit unknown; " + detail if detail else "exit unknown"), False


def _file_action(name: str) -> str | None:
    lowered = name.lower()
    for word, action in FILE_ACTIONS:
        if word in lowered:
            return action
    return None


def _prior(messages: list, prefixes: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """The ledger lines of earlier summary rows, so that a later compaction keeps them."""
    commands: list[str] = []
    files: dict[str, str] = {}
    for row in messages:
        if not is_summary(row, prefixes):
            continue
        lines = plain_text(attr(row, "content")).splitlines()
        if HEADING not in lines:
            continue
        section = None
        for line in lines[lines.index(HEADING) + 1:]:
            if line in (COMMANDS_HEADING, FILES_HEADING):
                section = line
            elif line.startswith("## ") or line == COPY_HEADING:
                break
            elif line.startswith("- ") and section == COMMANDS_HEADING:
                commands.append(line)
            elif line.startswith("- ") and section == FILES_HEADING and ": " in line:
                path, _, action = line[2:].rpartition(": ")
                files.pop(path, None)
                files[path] = action
    return commands, files


def entries(messages: Iterable[Any], prefixes: Iterable[str] = ()) -> tuple[list[str], dict[str, str]]:
    """Return (command lines, {path: action}) in the order of the rows, oldest first."""
    messages = list(messages)
    commands, files = _prior(messages, tuple(prefixes))
    # Each result goes to the oldest earlier call with its id that has no result yet: a provider can use the same
    # call id again in a later turn, and each turn keeps its own result.
    calls: list[tuple[str, Any, list[str | None]]] = []
    waiting: dict[str, list[list[str | None]]] = {}
    for row in messages:
        if attr(row, "role") == "assistant":
            for call_id, name, raw_arguments in tool_calls_of(row):
                slot: list[str | None] = [None]
                calls.append((name, raw_arguments, slot))
                if call_id:
                    waiting.setdefault(call_id, []).append(slot)
        elif attr(row, "role") == "tool":
            pending = waiting.get(str(attr(row, "tool_call_id") or ""))
            if pending:
                pending.pop(0)[0] = plain_text(attr(row, "content"))
    for name, raw_arguments, slot in calls:
        arguments = _arguments(raw_arguments)
        outcome, failed = _outcome(slot[0])
        command = next((arguments[key] for key in COMMAND_KEYS if isinstance(arguments.get(key), str)), None)
        if command is not None and command.strip():
            commands.append(f"- `{_one_line(command, COMMAND_CHARS).replace('`', chr(39))}`: {outcome}")
            continue
        action = _file_action(name)
        paths = [arguments[key] for key in PATH_KEYS if isinstance(arguments.get(key), str)]
        touched = [(path, action) for path in paths if action] if paths else []
        for value in arguments.values():
            if isinstance(value, str):
                touched.extend((path, PATCH_ACTIONS[kind]) for kind, path in PATCH_HEADER.findall(value))
        if failed:
            # A failed edit did not change the file: report the failure as a tool line instead.
            target = f" {_one_line(paths[0], PATH_CHARS)}" if paths else ""
            commands.append(f"- tool {_one_line(name, 60)}{target}: {outcome}")
            continue
        for path, kind in touched:
            path = _one_line(path, PATH_CHARS)
            if path:
                files.pop(path, None)
                files[path] = kind
    return commands, files


def build(messages: Iterable[Any], prefixes: Iterable[str] = (), max_tokens: int = MAX_TOKENS) -> str:
    """Return the ledger block, or "" when it is empty or does not fit in max_tokens.
    The oldest command lines go first when the block is too large, then the oldest files."""
    commands, files = entries(messages, prefixes)
    paths = list(files.items())

    def render() -> str:
        lines = [HEADING, NOTE]
        if commands:
            lines += [COMMANDS_HEADING, *commands]
        if paths:
            lines += [FILES_HEADING, *(f"- {path}: {action}" for path, action in paths)]
        return "\n".join(lines)

    if not commands and not paths:
        return ""
    text = render()
    while estimate_tokens(text) > max_tokens and (commands or paths):
        if commands:
            commands.pop(0)
        else:
            paths.pop(0)
        text = render() if commands or paths else ""
    return text
