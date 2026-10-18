"""Build the SYNTHETIC benchmark sessions and their gold answers from scenarios.py.

The build is deterministic: the same scenario data and seed give the same rows. Each fixture has two files:
- <id>.history.json: the session rows in the Hermes message format, the continuation chunks for repeated
  compaction, and the resume message. The build writes it to a work folder; it is not committed.
- <id>.gold.json: the questions, the gold answers, the scoring rules, and the SHA-256 of the history file.
  It is committed under benchmarks/compaction/fixtures/.

Tool rows follow the Hermes tools: terminal results are JSON with output, exit_code, and error; read_file,
write_file, and patch (V4A patch mode and replace mode) take a path. All names, paths, and values are invented.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path

from .scenarios import SCENARIOS

LABEL = "SYNTHETIC benchmark fixture: invented project, people, paths, commands, and values."
RESUME = "Give a one-sentence status update, then wait for my next message."
WORKSPACE = "/workspace"
# The questioned facts stay out of the last part of the session, which a verbatim tail can keep. The recent
# facts go into that last part.
TAIL_GUARD_TOKENS = 14_000
ZONE_SHARE = 0.6
CONTINUATION_SHARE = 0.5
STATUS_WORDS = {"done": ["done", "completed", "complete", "finished"],
                "open": ["open", "pending", "not started", "in progress", "todo", "to do", "not done"],
                "blocked": ["blocked", "waiting"]}

WORDS = (
    "alpha amber anchor apex arrow aspen atlas basin beacon birch bolt border bramble breeze bridge brook cable "
    "canvas canyon carbon cedar cinder circuit cobalt comet copper coral crest delta dune ember falcon fern fjord "
    "flint forge garnet glacier granite grove harbor hazel helix heron horizon indigo iris ivory jasper juniper "
    "kelp lagoon lantern larch lattice ledger lichen linden lotus lumen maple marble meadow mesa meteor mint "
    "nectar nickel nimbus north oasis onyx opal orbit osprey pebble pepper pine pixel plume prairie prism quartz "
    "quill radar raven reef ridge river rocket saffron sage sapphire shale signal silver sparrow spruce summit "
    "tango thistle thunder timber topaz tundra umber valley vapor velvet vertex violet walnut willow zephyr"
).split()
VERBS = "apply build check collect compute convert copy count emit fetch find format load map merge parse read " \
        "render resolve scan select sort split sum trim update validate walk wrap write".split()


def estimate_tokens(rows) -> int:
    """Characters of the JSON form divided by 4. The worker uses the same estimate."""
    return len(json.dumps(rows, ensure_ascii=False)) // 4


def stop_words() -> set[str]:
    """Words of the keyword rules. Filler text does not use them, so it cannot look like a gold fact."""
    words: set[str] = set()
    for spec in SCENARIOS:
        groups = [*spec["goal"]["objective_kw"], *spec["goal"]["criterion_kw"], *spec["goal"]["out_of_scope_kw"]]
        groups += [group for decision in spec["decisions"] for group in decision[2]]
        groups += [group for key, value in spec["next"].items() if key.endswith("_kw") for group in value]
        for group in groups:
            for alternative in group:
                words.update(re.findall(r"[a-z]{3,}", alternative.lower()))
    return words


class Session:
    """Rows of one synthetic session, with the row index of each fact."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.rng = random.Random(spec["seed"])
        self.rows: list[dict] = []
        self.calls = 0
        stop = stop_words()
        self.words = [word for word in WORDS if not any(s in word or word in s for s in stop)]
        self.verbs = [verb for verb in VERBS if verb not in stop]

    # Rows.
    def user(self, text: str) -> int:
        self.rows.append({"role": "user", "content": text})
        return len(self.rows) - 1

    def _open_text(self) -> bool:
        """True when the last row is assistant text without a tool call: the next assistant output joins it.
        An OpenAI-format agent loop does not send two assistant messages in a row."""
        return bool(self.rows) and self.rows[-1]["role"] == "assistant" and "tool_calls" not in self.rows[-1]

    def say(self, text: str) -> int:
        if self._open_text():
            self.rows[-1]["content"] = (self.rows[-1]["content"] + "\n\n" + text).strip()
        else:
            self.rows.append({"role": "assistant", "content": text})
        return len(self.rows) - 1

    def call(self, name: str, arguments: dict, result: str, note: str = "") -> int:
        """Append one assistant tool call and its result. Return the index of the result row."""
        self.calls += 1
        call_id = f"call_{self.spec['id'][:2]}_{self.calls:04d}"
        calls = [{"id": call_id, "type": "function",
                  "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]
        if self._open_text():
            self.rows[-1]["content"] = (self.rows[-1]["content"] + "\n\n" + note).strip()
            self.rows[-1]["tool_calls"] = calls
        else:
            self.rows.append({"role": "assistant", "content": note, "tool_calls": calls})
        self.rows.append({"role": "tool", "tool_call_id": call_id, "content": result})
        return len(self.rows) - 1

    def terminal(self, command: str, code: int, lines: list[str], note: str = "") -> int:
        result = {"output": "\n".join(lines), "exit_code": code, "error": None}
        return self.call("terminal", {"command": command, "timeout": 300}, json.dumps(result, ensure_ascii=False),
                         note)

    def read(self, path: str, text: str, note: str = "") -> int:
        lines = text.split("\n")
        numbered = "\n".join(f"{number:6d}|{line}" for number, line in enumerate(lines, 1))
        result = {"content": numbered, "total_lines": len(lines), "file_size": len(text.encode("utf-8")),
                  "truncated": False}
        return self.call("read_file", {"path": path, "offset": 1, "limit": 500}, json.dumps(result), note)

    def modified(self, path: str) -> list[str]:
        return [f"{WORKSPACE}/{self.spec['project']}/{path}"]

    def tokens(self) -> int:
        return estimate_tokens(self.rows)

    # Filler: work that holds no questioned fact.
    def name(self) -> str:
        return f"{self.rng.choice(self.words)}_{self.rng.choice(self.words)}"

    def code(self, lines: int) -> str:
        """Synthetic source text in the style of the project language."""
        lang = self.spec["lang"]
        out: list[str] = []
        while len(out) < lines:
            fn, arg, other = self.name(), self.rng.choice(self.words), self.name()
            verb = self.rng.choice(self.verbs)
            if lang == "python":
                out += [f"def {verb}_{fn}({arg}, limit=64):", f"    \"\"\"{verb.title()} the {arg} items.\"\"\"",
                        f"    items = [{other}(x) for x in {arg} if x]", "    if len(items) > limit:",
                        "        items = items[:limit]", "    return items", ""]
            elif lang == "go":
                out += [f"func {verb.title()}{fn.title().replace('_', '')}({arg} []string) []string {{",
                        "\tout := make([]string, 0, len(" + arg + "))", f"\tfor _, v := range {arg} {{",
                        f"\t\tout = append(out, {other}(v))", "\t}", "\treturn out", "}", ""]
            elif lang == "rust":
                out += [f"pub fn {verb}_{fn}({arg}: &[String]) -> Vec<String> {{",
                        f"    {arg}.iter().filter(|v| !v.is_empty()).map(|v| {other}(v)).collect()", "}", ""]
            else:
                out += [f"export function {verb}{fn.title().replace('_', '')}({arg}: string[]): string[] {{",
                        f"  return {arg}.filter(Boolean).map((v) => {other}(v));", "}", ""]
        return "\n".join(out[:lines])

    def filler(self) -> None:
        vocab = self.spec["vocab"]
        kind = self.rng.choice(("read", "read", "read", "search", "test", "git", "list"))
        directory = self.rng.choice(vocab["dirs"])
        if kind == "read":
            path = f"{directory}/{self.name()}.{vocab['ext']}"
            self.read(path, self.code(self.rng.randint(60, 220)),
                      self.rng.choice(("Read this file for context.", "Check how this module works.", "")))
        elif kind == "search":
            word = self.rng.choice(self.words)
            matches = [{"path": f"{directory}/{self.name()}.{vocab['ext']}", "line": self.rng.randint(1, 400),
                        "content": f"{self.rng.choice(self.verbs)}_{word}_{self.name()}()"}
                       for _ in range(self.rng.randint(12, 40))]
            self.call("search_files", {"pattern": word, "path": directory},
                      json.dumps({"matches": matches, "total_count": len(matches)}), "Search for related uses.")
        elif kind == "test":
            names = [f"test_{self.rng.choice(self.verbs)}_{self.name()}" for _ in range(self.rng.randint(20, 70))]
            lines = [f"{directory}::{name} PASSED" for name in names] + [f"{len(names)} passed in "
                                                                         f"{self.rng.uniform(0.4, 9):.2f}s"]
            self.terminal(vocab["test"].format(dir=directory), 0, lines, "Run the nearby tests as a baseline.")
        elif kind == "git":
            lines = [f"{self.rng.getrandbits(28):07x} {self.rng.choice(self.verbs).title()} the "
                     f"{self.rng.choice(self.words)} {self.rng.choice(self.words)} code"
                     for _ in range(self.rng.randint(15, 35))]
            self.terminal(f"git log --oneline -n {len(lines)} -- {directory}", 0, lines, "")
        else:
            lines = [f"-rw-r--r-- 1 dev dev {self.rng.randint(300, 40000):6d} Sep {self.rng.randint(1, 28):2d} "
                     f"{self.name()}.{vocab['ext']}" for _ in range(self.rng.randint(10, 30))]
            self.terminal(f"ls -la {directory}", 0, lines, "")

    def fill_to(self, tokens: int) -> None:
        while self.tokens() < tokens:
            self.filler()


def opening(spec: dict) -> str:
    goal = spec["goal"]
    plan = "\n".join(f"{index}. {name}" for index, (name, _status, _note) in enumerate(spec["subtasks"], 1))
    rules = "\n".join(f"- {rule}" for _topic, rule in spec["instructions"][:3])
    return (f"{LABEL}\n\nPlease {goal['objective']}. The main change is in {goal['module']}. The work counts as "
            f"finished when {goal['criterion']}. This is for release {goal['release']}. "
            f"{goal['out_of_scope'][0].upper()}{goal['out_of_scope'][1:]} is out of scope.\n\n"
            f"Plan:\n{plan}\n\nRules:\n{rules}")


def v4a(path: str, old: str, new: str) -> str:
    minus = "\n".join("-" + line for line in old.split("\n"))
    plus = "\n".join("+" + line for line in new.split("\n"))
    return f"*** Begin Patch\n*** Update File: {path}\n@@\n{minus}\n{plus}\n*** End Patch"


def unified(path: str, old: str, new: str) -> str:
    minus = "\n".join("-" + line for line in old.split("\n"))
    plus = "\n".join("+" + line for line in new.split("\n"))
    return f"--- a/{path}\n+++ b/{path}\n@@\n{minus}\n{plus}"


def fact_value(session: Session, via: str, text) -> int:
    """Write a fact through a user row, an assistant row, a file read, or a terminal command. Return its row."""
    if via == "user":
        return session.user(text)
    if via == "assistant":
        return session.say(text)
    if via == "read":
        path, line = text
        if "\n" in line:
            return session.read(path, line, "Open this file.")
        body = session.code(session.rng.randint(30, 90)).split("\n")
        body.insert(session.rng.randint(3, len(body) - 3), line)
        return session.read(path, "\n".join(body), "Look at this file.")
    command, lines = text
    return session.terminal(command, 0, lines, "Check this.")


def build(spec: dict) -> tuple[dict, dict]:
    """Return (history, gold) for one scenario."""
    session = Session(spec)
    evidence: dict[str, list[int]] = {}
    target = int(spec["target_tokens"])
    zone = int(min(ZONE_SHARE * target, target - TAIL_GUARD_TOKENS))
    goal = spec["goal"]

    def mark(key: str, *rows: int) -> None:
        evidence.setdefault(key, []).extend(rows)

    def ev_opening():
        row = session.user(opening(spec))
        for key in ("goal", "instruction:0", "instruction:1", "instruction:2"):
            mark(key, row)
        session.say(f"Understood. I will start with {goal['module']} and reproduce the problem.")
        session.read(goal["module"], session.code(session.rng.randint(80, 160)), "Read the main file first.")

    def ev_command(index):
        def run():
            command, code, lines, _error = spec["commands"][index]
            notes = ("Reproduce the failure first.", "Run this check.", "Try the next check.", "Run it again.",
                     "Check the result.", "Run the last check of this step.")
            mark(f"command:{index}", session.terminal(command, code, lines, notes[index]))
        return run

    def ev_file(index):
        def run():
            tool, path, purpose, content = spec["files"][index]
            note = f"Now {purpose}."
            if tool == "write_file":
                result = {"bytes_written": len(content.encode("utf-8")), "dirs_created": False,
                          "files_modified": session.modified(path)}
                row = session.call("write_file", {"path": path, "content": content}, json.dumps(result), note)
            elif tool == "patch":
                old, new = content
                result = {"success": True, "diff": unified(path, old, new), "files_modified": session.modified(path)}
                row = session.call("patch", {"mode": "patch", "patch": v4a(path, old, new)}, json.dumps(result),
                                   note)
            else:
                old, new = content
                result = {"success": True, "diff": unified(path, old, new), "files_modified": session.modified(path)}
                row = session.call("patch", {"mode": "replace", "path": path, "old_string": old,
                                             "new_string": new}, json.dumps(result), note)
            mark(f"file:{index}", row - 1, row)
        return run

    def ev_identifier(index):
        def run():
            _question, _answer, via, text = spec["identifiers"][index]
            mark(f"identifier:{index}", fact_value(session, via, text))
        return run

    def ev_decision(index):
        def run():
            mark(f"decision:{index}", session.say(spec["decisions"][index][1]))
        return run

    def ev_progress(indexes):
        def run():
            lines = [f"- {spec['subtasks'][i][0]}: {spec['subtasks'][i][1]} ({spec['subtasks'][i][2]})"
                     for i in indexes]
            row = session.say("Progress:\n" + "\n".join(lines))
            for i in indexes:
                mark(f"subtask:{i}", row)
        return run

    def ev_blocked():
        index = next(i for i, item in enumerate(spec["subtasks"]) if item[1] == "blocked")
        name, _status, note = spec["subtasks"][index]
        mark(f"subtask:{index}", session.user(f"FYI: \"{name}\" is blocked: {note}. Skip it for now."))
        session.say(f"Noted. I will leave \"{name}\" blocked and continue with the other steps.")

    def ev_user_rule(index, extra=""):
        def run():
            row = session.user(f"One more rule: {spec['instructions'][index][1]}{extra}")
            mark(f"instruction:{index}", row)
            if extra:
                mark("next", row)
            session.say("Understood.")
        return run

    nxt = spec["next"]
    next_text = (f"\n\nNext: once {nxt['trigger']}, {nxt['action']} in {nxt['target']}. After that, "
                 f"{nxt['after']}. Do not merge until {nxt['approver']} approves, and only after {nxt['condition']}.")
    blocked = next(i for i, item in enumerate(spec["subtasks"]) if item[1] == "blocked")
    progress = [i for i in range(5) if i != blocked]
    events = [
        ev_opening, ev_command(0), ev_identifier(0), ev_file(0), ev_file(1), ev_decision(0),
        ev_progress(progress[:2]), ev_command(1), ev_identifier(1), ev_user_rule(3), ev_decision(1), ev_file(2),
        ev_command(2), ev_identifier(2), ev_decision(2), ev_blocked, ev_command(3), ev_file(3), ev_identifier(3),
        ev_decision(3), ev_user_rule(4, next_text), ev_command(4), ev_file(4), ev_identifier(4), ev_decision(4),
        ev_command(5), ev_progress(progress[2:]),
    ]
    for index, event in enumerate(events):
        event()
        session.fill_to(zone * (index + 1) // len(events))
    # The late part: work without questioned facts, then the two recent facts.
    session.user("Continue. Check the other modules for similar code paths and tell me if anything looks wrong.")
    session.fill_to(target - 2_500)
    for index, (_question, _answer, _kind, text) in enumerate(spec["control"]):
        via = "terminal" if isinstance(text[1], list) else "read"
        mark(f"recent:{index}", fact_value(session, via, text))
        if index == 0:
            session.filler()
    session.say("I checked the other modules. Nothing else needs a change for now.")
    rows = session.rows
    continuations = []
    for number in range(2):
        start = len(session.rows)
        session.user(f"Continue with the open work. Look through the remaining code (part {number + 1}) for anything "
                     "related, and keep me updated.")
        session.fill_to(estimate_tokens(rows[:start]) + int(CONTINUATION_SHARE * target))
        session.say(f"Part {number + 1} is checked. I found nothing that needs a change yet.")
        continuations.append(copy.deepcopy(session.rows[start:]))
        del session.rows[start:]
    history = {"label": LABEL, "id": spec["id"], "system": spec["system"], "resume": RESUME,
               "rows": rows, "continuations": continuations}
    return history, gold_for(spec, evidence, history)


def gold_for(spec: dict, evidence: dict, history: dict) -> dict:
    goal, nxt = spec["goal"], spec["next"]
    questions: list[dict] = []

    def add(category, key, text, kind, gold, retention=None, **extra):
        questions.append({"id": f"{spec['id']}:{len(questions) + 1:02d}", "category": category, "question": text,
                          "kind": kind, "gold": gold, "evidence_rows": sorted(set(evidence[key])),
                          "retention": retention if retention is not None else retention_for(kind, gold), **extra})

    add("goal", "goal", "What is the main goal of this session? Answer in one sentence.", "keywords",
        goal["objective_kw"])
    add("goal", "goal", "Which source file is the main target of the change? Give the path.", "path", goal["module"])
    add("goal", "goal", "What acceptance criterion did the user set for the work to count as finished?", "keywords",
        goal["criterion_kw"])
    add("goal", "goal", "Which release is this work for?", "text", goal["release"])
    add("goal", "goal", "What did the user say is out of scope?", "keywords", goal["out_of_scope_kw"])
    for index, (topic, rule) in enumerate(spec["instructions"]):
        add("user_instruction", f"instruction:{index}", f"The user gave a rule about {topic}. Quote that rule "
            "exactly.", "verbatim", rule)
    for index, (command, code, _lines, _error) in enumerate(spec["commands"]):
        add("command_outcome", f"command:{index}", f"What exit code did the command `{command}` return? Answer with "
            "the number.", "number", str(code), [[command[:60]], exit_patterns(code)], subkind="exit_code")
    for index, (command, _code, _lines, error) in enumerate(spec["commands"]):
        if error:
            add("command_outcome", f"command:{index}", f"What error line did the command `{command}` print? Quote "
                "it.", "text", error, subkind="error_line")
    for index, (tool, path, purpose, _content) in enumerate(spec["files"]):
        verb = "create" if tool == "write_file" else "change"
        add("file_changed", f"file:{index}", f"Which file did the agent {verb} to {purpose}? Give the path.", "path",
            path)
    for index, (question, answer, _via, _text) in enumerate(spec["identifiers"]):
        add("identifier", f"identifier:{index}", question, "text", answer)
    for index, (question, statement, groups) in enumerate(spec["decisions"]):
        anchor = decision_anchor(question, statement)
        add("decision", f"decision:{index}", question, "keywords", groups,
            [*groups, [anchor]] if anchor else groups)
    for index, (name, status, _note) in enumerate(spec["subtasks"]):
        add("task_state", f"subtask:{index}", f"What is the status of the subtask \"{name}\"? Answer with one "
            "word: done, open, or blocked.", "choice", status, [[name], STATUS_WORDS[status]])
    add("next_step", "next", f"What did the user say to do once {nxt['trigger']}?", "keywords", nxt["action_kw"])
    add("next_step", "next", "Which file is the target of that next step? Give the path.", "path", nxt["target"])
    add("next_step", "next", "What did the user say comes after that step?", "keywords", nxt["after_kw"])
    add("next_step", "next", "Who must approve before the merge?", "keywords", nxt["approver_kw"])
    add("next_step", "next", "Besides the approval, which condition must hold before the merge?", "keywords",
        nxt["condition_kw"])
    for index, (question, answer, kind, _text) in enumerate(spec["control"]):
        add("recent", f"recent:{index}", question, kind, answer)
    text = json.dumps(history, ensure_ascii=False, sort_keys=True)
    return {"label": LABEL, "id": spec["id"], "target_tokens": spec["target_tokens"],
            "history_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "rows": len(history["rows"]), "est_tokens": estimate_tokens(history["rows"]),
            "continuation_est_tokens": [estimate_tokens(chunk) for chunk in history["continuations"]],
            "questions": questions}


def exit_patterns(code: int) -> list[str]:
    """Alternatives that state an exit code in a ledger line, a summary, or a raw terminal result."""
    return [f"exit {code}", f"exit_code: {code}", f"exit code {code}", f"exit code: {code}", f"exited {code}",
            f"exited with {code}", f"returned {code}", f"status {code}", f"code {code}"]


def retention_for(kind: str, gold) -> list[list[str]]:
    if kind == "keywords":
        return gold
    return [[gold]]


def decision_anchor(question: str, statement: str) -> str | None:
    """A distinctive word of the question that the decision statement also has (for the retention check)."""
    words = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{5,}", question)), key=lambda w: (-len(w), w))
    skip = {"instead", "choose", "change", "agent", "through"}
    for word in words:
        if word.lower() not in skip and word.lower() in statement.lower():
            return word
    return None


def write_all(history_dir: Path, gold_dir: Path) -> list[dict]:
    """Build every scenario. Write the history files to history_dir and the gold files to gold_dir."""
    history_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in SCENARIOS:
        history, gold = build(spec)
        (history_dir / f"{spec['id']}.history.json").write_text(
            json.dumps(history, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        (gold_dir / f"{spec['id']}.gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1) + "\n",
                                                           encoding="utf-8")
        summary.append({"id": spec["id"], "rows": gold["rows"], "est_tokens": gold["est_tokens"],
                        "questions": len(gold["questions"])})
    return summary


def load(history_path: Path, gold_path: Path) -> tuple[dict, dict]:
    """Load one fixture. Refuse a history file that is not the one the gold file names."""
    history_text = history_path.read_text(encoding="utf-8")
    history = json.loads(history_text)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(history, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if digest != gold["history_sha256"]:
        raise ValueError(f"history_mismatch:{gold['id']}")
    return history, gold
