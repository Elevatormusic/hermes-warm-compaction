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
IDENTIFIERS_HEADING = "### Identifiers"
REASONS_HEADING = "### Reasons"
MAX_TOKENS = 1_500
# Exact identifiers from the rows: no model can paraphrase them. Each kind keeps its most frequent values
# (the latest first for a tie), at most its cap.
IDENTIFIER_KINDS = (
    ("names", re.compile(r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]*[a-z0-9]|[A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9]"
                         r"|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"), 40),
    ("addresses", re.compile(r"https?://[^\s<>\"'`)\]]{4,120}|\b(?:localhost|\d{1,3}(?:\.\d{1,3}){3}|[\w-]+(?:\.[\w-]+)+)"
                             r":\d{2,5}\b"), 20),
    ("references", re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b|(?<![\w&])#\d{2,6}\b|\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)"
                              r"[0-9a-f]{7,40}\b|\bv\d+\.\d+(?:\.\d+)?\b"), 30),
    ("errors", re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception)\b"), 15),
    # Environment variables that code reads: a setting of the project, also when only a file read shows it. A tool
    # result can be JSON text, so a quote can have a backslash before it.
    ("environment", re.compile(r"(?:\benviron(?:\.get)?\(\s*|\benviron\[\s*|\bgetenv\(\s*)\\?[\"']"
                               r"([A-Z][A-Z0-9_]{2,63})\\?[\"']|\bprocess\.env\.([A-Z][A-Z0-9_]{2,63})\b"
                               r"|\$\{([A-Z][A-Z0-9_]{2,63})\}"), 20),
)
IDENTIFIER_CHARS = 2_400
# A sentence of the assistant that gives a reason: the exact words keep a decision and why it was made.
REASON_WORDS = re.compile(r"\b(?:because|since|so that|instead of|rather than|in order to|due to)\b", re.I)
REASON_CHARS = 240
MAX_REASONS = 20
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


def _prior_lines(messages: list, prefixes: tuple[str, ...]) -> dict[str, list[str]]:
    """The bullet lines of each ledger section of earlier summary rows, so that a later compaction keeps them."""
    found: dict[str, list[str]] = {COMMANDS_HEADING: [], FILES_HEADING: [], IDENTIFIERS_HEADING: [],
                                   REASONS_HEADING: []}
    for row in messages:
        if not is_summary(row, prefixes):
            continue
        lines = plain_text(attr(row, "content")).splitlines()
        if HEADING not in lines:
            continue
        section = None
        for line in lines[lines.index(HEADING) + 1:]:
            if line in found:
                section = line
            elif line.startswith("## ") or line == COPY_HEADING:
                break
            elif line.startswith("- ") and section is not None:
                found[section].append(line)
    return found


def _prior(messages: list, prefixes: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """The command lines and the file changes of earlier summary rows."""
    found = _prior_lines(messages, prefixes)
    files: dict[str, str] = {}
    for line in found[FILES_HEADING]:
        if ": " in line:
            path, _, action = line[2:].rpartition(": ")
            files.pop(path, None)
            files[path] = action
    return found[COMMANDS_HEADING], files


def _values(value: Any) -> list[str]:
    """The string values of tool call arguments, without their keys (a key such as old_string is not a name of
    the work)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _values(item)]
    if isinstance(value, list):
        return [text for item in value for text in _values(item)]
    return []


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def notes(messages: Iterable[Any], prefixes: Iterable[str] = ()) -> tuple[dict[str, list[str]], list[str]]:
    """Return ({kind: identifiers}, reason lines) of the rows and of earlier summary rows.
    Identifier names come from user and assistant text and tool call arguments, not from tool results: a file read
    has many names that the work did not use. The other kinds come from all rows. A reason line is one sentence of
    the assistant that gives a reason, in its exact words."""
    messages = list(messages)
    prefixes = tuple(prefixes)
    found = _prior_lines(messages, prefixes)
    counts: dict[str, dict[str, int]] = {kind: {} for kind, _pattern, _cap in IDENTIFIER_KINDS}
    seen: dict[str, dict[str, int]] = {kind: {} for kind, _pattern, _cap in IDENTIFIER_KINDS}
    order = 0
    # Earlier identifiers count one time each, older than every row.
    for line in found[IDENTIFIERS_HEADING]:
        kind, _, values = line[2:].partition(": ")
        if kind in counts:
            for value in values.split(", "):
                if value:
                    counts[kind][value] = counts[kind].get(value, 0) + 1
                    seen[kind].setdefault(value, -1)
    reasons = [line[2:] for line in found[REASONS_HEADING]]
    for row in messages:
        if is_summary(row, prefixes):
            continue
        role = attr(row, "role")
        text = plain_text(attr(row, "content"))
        arguments = "\n".join(text for _call_id, _name, raw in tool_calls_of(row)
                              for text in _values(_arguments(raw))) if role == "assistant" else ""
        for kind, pattern, _cap in IDENTIFIER_KINDS:
            source = text if role == "tool" and kind != "names" else ""
            if role != "tool":
                source = text + "\n" + arguments
            for match in pattern.finditer(source):
                # A kind with a group (environment) keeps the name only.
                value = next((group for group in match.groups() if group), match.group(0)).rstrip(".,;:")
                order += 1
                counts[kind][value] = counts[kind].get(value, 0) + 1
                seen[kind][value] = order
        if role == "assistant":
            for sentence in _sentences(text):
                if REASON_WORDS.search(sentence):
                    line = _one_line(sentence, REASON_CHARS)
                    if line not in reasons:
                        reasons.append(line)
    identifiers: dict[str, list[str]] = {}
    for kind, _pattern, cap in IDENTIFIER_KINDS:
        ranked = sorted(counts[kind], key=lambda value: (-counts[kind][value], -seen[kind][value]))[:cap]
        if ranked:
            identifiers[kind] = ranked
    return identifiers, reasons[-MAX_REASONS:]


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
        if failed or (touched and slot[0] is None):
            # A failed edit did not change the file, and an edit without a result is not confirmed: report a tool
            # line instead of a changed file.
            first = paths[0] if paths else touched[0][0] if touched else ""
            target = f" {_one_line(first, PATH_CHARS)}" if first else ""
            commands.append(f"- tool {_one_line(name, 60)}{target}: {outcome}")
            continue
        for path, kind in touched:
            path = _one_line(path, PATH_CHARS)
            if path:
                files.pop(path, None)
                files[path] = kind
    return commands, files


def build(messages: Iterable[Any], prefixes: Iterable[str] = (), max_tokens: int = MAX_TOKENS,
          require_work: bool = False) -> str:
    """Return the ledger block, or "" when it is empty or does not fit in max_tokens. With require_work, a block
    without commands or file changes is "" too.
    When the block is too large, the oldest command lines go first, then the oldest reasons, then the oldest
    files, then the identifier kinds from the last."""
    messages = list(messages)
    prefixes = tuple(prefixes)
    commands, files = entries(messages, prefixes)
    paths = list(files.items())
    identifiers, reasons = notes(messages, prefixes)
    kinds: list[str] = []
    used = 0
    for kind, values in identifiers.items():
        line = f"- {kind}: " + ", ".join(values)
        while len(line) > IDENTIFIER_CHARS - used and values:
            values = values[:-1]
            line = f"- {kind}: " + ", ".join(values)
        if values:
            kinds.append(line)
            used += len(line)

    def render() -> str:
        lines = [HEADING, NOTE]
        if commands:
            lines += [COMMANDS_HEADING, *commands]
        if paths:
            lines += [FILES_HEADING, *(f"- {path}: {action}" for path, action in paths)]
        if kinds:
            lines += [IDENTIFIERS_HEADING, *kinds]
        if reasons:
            lines += [REASONS_HEADING, *(f"- {reason}" for reason in reasons)]
        return "\n".join(lines)

    if require_work and not commands and not paths:
        return ""
    if not (commands or paths or kinds or reasons):
        return ""
    text = render()
    while estimate_tokens(text) > max_tokens:
        if commands:
            commands.pop(0)
        elif reasons:
            reasons.pop(0)
        elif paths:
            paths.pop(0)
        elif kinds:
            kinds.pop()
        else:
            return ""
        if not (commands or paths or kinds or reasons):
            return ""
        text = render()
    return text
