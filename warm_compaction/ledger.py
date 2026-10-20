"""Work ledger: commands, their results, and file changes, extracted by code from the rows before the tail.

No model writes the ledger. A model summary keeps the goal and the state; the ledger keeps the hard facts that a
summary often drops: which command failed, with which error, and which files changed. The ledger is reference
data for the next model, not a list of tasks.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any
from collections.abc import Iterable

from .layout import COPY_HEADING, is_summary
from .rows import attr, estimate_tokens, plain_text, tool_calls_of

HEADING = "## Work ledger"
NOTE = ("Recorded by the plugin from tool calls and results before this point. Reference only; not new tasks. "
        "An indented line is an exact copy of an output line or a changed line.")
COMMANDS_HEADING = "### Commands"
FILES_HEADING = "### Files changed"
TASKS_HEADING = "### Task status"
IDENTIFIERS_HEADING = "### Identifiers"
REASONS_HEADING = "### Reasons"
SECTIONS = (COMMANDS_HEADING, FILES_HEADING, TASKS_HEADING, IDENTIFIERS_HEADING, REASONS_HEADING)
MAX_TOKENS = 1_500
# An exact line under an entry: an output line of a command or a changed line of a file. It is a separate line, so
# that a label of the ledger ("exit 1;") and the text of the line do not mix.
SUB_LINE = "  - "
# Exact identifiers from the rows: no model can paraphrase them. Each kind keeps its most frequent values
# (the latest first for a tie), at most its cap.
IDENTIFIER_KINDS = (
    ("names", re.compile(r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]*[a-z0-9]|[A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9]"
                         r"|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"), 40),
    # A host and port, not a file and line: no path before it ("src/mod.rs:198") and no column after it.
    ("addresses", re.compile(r"https?://[^\s<>\"'`)\]\\]{4,120}|(?<![\w./\\-])(?:localhost|\d{1,3}(?:\.\d{1,3}){3}"
                             r"|[\w-]+(?:\.[\w-]+)+):\d{2,5}\b(?!:\d)"), 20),
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
ADDRESS = dict((kind, pattern) for kind, pattern, _cap in IDENTIFIER_KINDS)["addresses"]
# A sentence of the assistant that gives a reason: the exact words keep a decision and why it was made.
REASON_WORDS = re.compile(r"\b(?:because|since|so that|instead of|rather than|in order to|due to|to avoid|to prevent)\b"
                          r"|,\s+so\b", re.I)
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
# A line that reports a failure in tool output, also with an exception name ("AssertionError:",
# "botocore.exceptions.EndpointConnectionError:").
ERROR_LINE = re.compile(r"\b(error|errors|fail|failed|failure|exceptions?|traceback|fatal|panic|denied|not found)\b"
                        r"|(?-i:[a-z0-9](?:Error|Exception))\b", re.I)
# A compiler, linter, or test diagnostic: a file name and a line number ("store_test.go:57:", "a.py:3:1:").
FILE_LINE = re.compile(r"[\w./\\-]+\.\w{1,8}:\d+(?::\d+)?:")
# The error lines that a failed command keeps: a test failure can need its name and its message.
MAX_ERROR_LINES = 3
ERROR_MARK = "error lines:"
# A successful command keeps a short note of its output (the built-in compressor keeps short outputs as they are).
# The ledger cuts the oldest notes first when it is too large.
OUTPUT_MARK = "output:"
NOTE_CHARS = 200
NOTE_LINES = 3
# A changed file keeps its first added lines: a version, a setting, or a fix can be only in the edit. It also keeps
# the step of its latest edits: the first sentence of the assistant text with the edit call.
MAX_CHANGE_LINES = 2
CHANGE_CHARS = 200
STEP_MARK = "; step: "
STEP_CHARS = 120
MAX_STEPS = 2
# A task and its status in a progress list of the assistant ("- add the lock-order test: open (not started)", or
# "- [x] update the docs"), or in a call of the Hermes todo tool. The latest status of each task is kept.
STATUS_WORDS = (r"done|complete|completed|finished|fixed|open|pending|todo|to do|not started|in progress|in-progress"
                r"|ongoing|blocked|skipped|cancelled|canceled|failed|failing")
STATUS_LINE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX>~]\]\s+)?(?P<task>[^:\n]{3,120}?)\s*:\s*"
                         rf"(?P<status>{STATUS_WORDS})\b(?P<rest>[^\n]*)$", re.I)
CHECKBOX_LINE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+\[(?P<box>[ xX])\]\s+(?P<task>[^\n]{3,160})$")
TODO_STATUS = {"pending": "open", "in_progress": "in progress", "completed": "done", "cancelled": "cancelled"}
TASK_CHARS = 200
MAX_TASKS = 30
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


def _first_error(text: str, count: int = 1) -> list[str]:
    """The first count distinct lines that report an error or a diagnostic. A line of four characters or less
    ("FAIL") is a summary, not a diagnostic."""
    found: list[str] = []
    for line in text.splitlines():
        if len(line.strip()) > 4 and (ERROR_LINE.search(line) or FILE_LINE.search(line)):
            line = _one_line(line, DETAIL_CHARS)
            if line not in found:
                found.append(line)
            if len(found) >= count:
                break
    return found


def _output_note(text: str) -> list[str]:
    """A short note of a successful output: all of it when it has at most NOTE_LINES lines in NOTE_CHARS, else the
    first line with an address ("listening on 127.0.0.1:8743") and the last line (often the result count). Lines
    without a letter or a digit ("..", "---") are not counted."""
    lines = [" ".join(line.split()) for line in text.splitlines() if any(char.isalnum() for char in line)]
    if not lines:
        return []
    if len(lines) <= NOTE_LINES and sum(len(line) for line in lines) <= NOTE_CHARS:
        return lines
    address = next((line for line in lines[:-1] if ADDRESS.search(line)), None)
    return [_one_line(line, NOTE_CHARS) for line in (address, lines[-1]) if line]


def _with_lines(head: str, mark: str, lines: list[str]) -> str:
    """The head of an entry and its exact lines, each on its own indented line after the mark."""
    return head + (f"; {mark}" + "".join(f"\n{SUB_LINE}{line}" for line in lines) if lines else "")


def _outcome(result: str | None) -> tuple[str, bool]:
    """Return (outcome text, failed) for one tool result. The outcome is only what the result reports: the exit
    code, up to MAX_ERROR_LINES error lines of a failure, or a short output note of a success."""
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
            head = f"exit {code}"
            if error:
                return f"{head}; {_one_line(str(error), DETAIL_CHARS)}", code != 0
            if code:
                return _with_lines(head, ERROR_MARK, _first_error(output, MAX_ERROR_LINES)), True
            return _with_lines(head, OUTPUT_MARK, _output_note(output)), False
        if error:
            return "error; " + _one_line(str(error), DETAIL_CHARS), True
        if data.get("success") is False:
            return "failed", True
        errors = _first_error(output)
        return (_with_lines("exit unknown", ERROR_MARK, errors) if errors
                else _with_lines("exit unknown", OUTPUT_MARK, _output_note(output))), False
    match = EXIT_TEXT.search(result)
    if match:
        code = int(match.group(1))
        if code:
            return _with_lines(f"exit {code}", ERROR_MARK, _first_error(result, MAX_ERROR_LINES)), True
        rest = "\n".join(line for line in result.splitlines() if not EXIT_TEXT.search(line))
        return _with_lines(f"exit {code}", OUTPUT_MARK, _output_note(rest)), False
    if result.lstrip().lower().startswith("error"):
        return "error; " + _one_line(result.lstrip(), DETAIL_CHARS), True
    errors = _first_error(result)
    return (_with_lines("exit unknown", ERROR_MARK, errors) if errors
            else _with_lines("exit unknown", OUTPUT_MARK, _output_note(result))), False


def _added(text: str, old: str = "") -> list[str]:
    """The first MAX_CHANGE_LINES lines of text that an ordered diff against old shows as new (an unchanged line
    is not; a line that moves is), and that have a letter or a digit."""
    before = [" ".join(line.split()) for line in old.splitlines()]
    after = [" ".join(line.split()) for line in text.splitlines()]
    found: list[str] = []
    for tag, _start, _end, first, last in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag not in ("insert", "replace"):
            continue
        for line in after[first:last]:
            if line and any(char.isalnum() for char in line) and line not in found:
                found.append(_one_line(line, CHANGE_CHARS))
                if len(found) >= MAX_CHANGE_LINES:
                    return found
    return found


def _changes(arguments: dict[str, Any], paths: list[str]) -> dict[str, list[str]]:
    """{path: first added lines} of an edit: the new "+" lines of each file in a V4A patch against its "-" lines,
    or the new lines of a replacement (new_string against old_string)."""
    changes: dict[str, list[str]] = {}
    for value in arguments.values():
        if not isinstance(value, str) or not PATCH_HEADER.search(value):
            continue
        added: dict[str, list[str]] = {}
        removed: dict[str, list[str]] = {}
        current = None
        for line in value.splitlines():
            header = PATCH_HEADER.match(line)
            if header:
                current = header.group(2)
                added.setdefault(current, [])
                removed.setdefault(current, [])
            elif current is not None and line.startswith("+") and not line.startswith("+++"):
                added[current].append(line[1:])
            elif current is not None and line.startswith("-") and not line.startswith("---"):
                removed[current].append(line[1:])
        for path, lines in added.items():
            changes[path] = _added("\n".join(lines), "\n".join(removed[path]))
    new = next((arguments[key] for key in ("new_string", "new_str") if isinstance(arguments.get(key), str)), None)
    if new is not None and paths:
        old = next((arguments[key] for key in ("old_string", "old_str") if isinstance(arguments.get(key), str)), "")
        changes[paths[0]] = _added(new, old)
    return changes


def _file_action(name: str) -> str | None:
    lowered = name.lower()
    for word, action in FILE_ACTIONS:
        if word in lowered:
            return action
    return None


def _prior_lines(messages: list, prefixes: tuple[str, ...]) -> dict[str, list[str]]:
    """The entries of each ledger section of earlier summary rows, so that a later compaction keeps them. An entry
    is its bullet line and its indented exact lines."""
    found: dict[str, list[str]] = {heading: [] for heading in SECTIONS}
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
            elif line.startswith(SUB_LINE) and section is not None and found[section]:
                found[section][-1] += "\n" + line
    return found


FILE_ENTRY = re.compile(r"^- (.+?): ((?:changed|written|deleted|moved)(?:; step: .*)?)$")


def _prior(messages: list, prefixes: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """The command entries and the file changes of earlier summary rows."""
    found = _prior_lines(messages, prefixes)
    files: dict[str, str] = {}
    for entry in found[FILES_HEADING]:
        head, _, lines = entry.partition("\n")
        match = FILE_ENTRY.match(head)
        if match:
            path, action = match.groups()
        elif ": " in head:
            path, _, action = head[2:].rpartition(": ")
        else:
            continue
        files.pop(path, None)
        files[path] = action + (f"\n{lines}" if lines else "")
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
    calls: list[tuple[str, Any, list[str | None], str]] = []
    waiting: dict[str, list[list[str | None]]] = {}
    for row in messages:
        if attr(row, "role") == "assistant" and not is_summary(row, tuple(prefixes)):
            # The first sentence of the assistant text names the step of its calls ("Now add the label strings.").
            sentence = next(iter(_sentences(plain_text(attr(row, "content")))), "")
            step = _one_line(sentence, STEP_CHARS).replace(" | ", " / ")
            for call_id, name, raw_arguments in tool_calls_of(row):
                slot: list[str | None] = [None]
                calls.append((name, raw_arguments, slot, step))
                if call_id:
                    waiting.setdefault(call_id, []).append(slot)
        elif attr(row, "role") == "tool":
            pending = waiting.get(str(attr(row, "tool_call_id") or ""))
            if pending:
                pending.pop(0)[0] = plain_text(attr(row, "content"))
    for name, raw_arguments, slot, step in calls:
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
        changes = _changes(arguments, paths)
        for path, kind in touched:
            lines = changes.get(path, []) if kind in ("changed", "written") else []
            path = _one_line(path, PATH_CHARS)
            if path:
                head = files.pop(path, "").split("\n", 1)[0]
                steps = [item for item in head.partition(STEP_MARK)[2].split(" | ") if item]
                if step and step not in steps:
                    steps.append(step)
                steps = steps[-MAX_STEPS:]
                files[path] = (kind + (STEP_MARK + " | ".join(steps) if steps else "")
                               + "".join(f"\n{SUB_LINE}{line}" for line in lines))
    return commands, files


def tasks(messages: Iterable[Any], prefixes: Iterable[str] = ()) -> list[str]:
    """Return the latest status of each task, as "task: status" lines, the most recently changed last. The tasks
    come from progress lists in the assistant text, calls of the todo tool, and earlier summary rows."""
    messages = list(messages)
    prefixes = tuple(prefixes)
    status: dict[str, str] = {}

    def put(task: str, state: str) -> None:
        # ": " separates the task from its status, so a task does not keep one.
        task = _one_line(task.replace(": ", " - "), 120).strip(" -*`")
        if len(task) >= 3:
            status.pop(task.casefold(), None)
            status[task.casefold()] = _one_line(f"{task}: {state.strip()}", TASK_CHARS)

    for entry in _prior_lines(messages, prefixes)[TASKS_HEADING]:
        task, _, state = entry.partition("\n")[0][2:].partition(": ")
        if state:
            put(task, state)
    for row in messages:
        if is_summary(row, prefixes) or attr(row, "role") != "assistant":
            continue
        for line in plain_text(attr(row, "content")).splitlines():
            match = STATUS_LINE.match(line)
            if match:
                put(match.group("task"), match.group("status").lower() + match.group("rest"))
                continue
            match = CHECKBOX_LINE.match(line)
            if match:
                put(match.group("task"), "open" if match.group("box") == " " else "done")
        for _call_id, name, raw in tool_calls_of(row):
            if "todo" not in str(name).lower():
                continue
            items = _arguments(raw).get("todos")
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    state = str(item.get("status") or "pending").strip().lower()
                    put(item["content"], TODO_STATUS.get(state, state))
    return list(status.values())[-MAX_TASKS:]


def build(messages: Iterable[Any], prefixes: Iterable[str] = (), max_tokens: int = MAX_TOKENS,
          require_work: bool = False) -> str:
    """Return the ledger block, or "" when it is empty or does not fit in max_tokens. With require_work, a block
    without commands or file changes is "" too.
    When the block is too large, the output notes of the oldest commands go first, then the changed lines of the
    oldest files, then the oldest command entries, then the oldest reasons, then the oldest files, then the oldest
    tasks, then the identifier kinds from the last."""
    messages = list(messages)
    prefixes = tuple(prefixes)
    commands, files = entries(messages, prefixes)
    paths = list(files.items())
    states = tasks(messages, prefixes)
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
        if states:
            lines += [TASKS_HEADING, *(f"- {state}" for state in states)]
        if kinds:
            lines += [IDENTIFIERS_HEADING, *kinds]
        if reasons:
            lines += [REASONS_HEADING, *(f"- {reason}" for reason in reasons)]
        return "\n".join(lines)

    if require_work and not commands and not paths:
        return ""
    if not (commands or paths or states or kinds or reasons):
        return ""
    text = render()
    while estimate_tokens(text) > max_tokens:
        noted = next((index for index, line in enumerate(commands) if "; " + OUTPUT_MARK in line), None)
        changed = next((index for index, (_path, action) in enumerate(paths) if "\n" in action), None)
        if noted is not None:
            commands[noted] = commands[noted].split("; " + OUTPUT_MARK, 1)[0]
        elif changed is not None:
            path, action = paths[changed]
            paths[changed] = (path, action.split("\n", 1)[0])
        elif commands:
            commands.pop(0)
        elif reasons:
            reasons.pop(0)
        elif paths:
            paths.pop(0)
        elif states:
            states.pop(0)
        elif kinds:
            kinds.pop()
        else:
            return ""
        if not (commands or paths or states or kinds or reasons):
            return ""
        text = render()
    return text
