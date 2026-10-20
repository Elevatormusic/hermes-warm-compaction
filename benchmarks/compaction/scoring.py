"""Rule-based scoring and statistics for the compaction benchmark. No model judges an answer.

Answer rules (the question "kind" in a gold file):
- number: the first number in the answer, after numbers that the question itself contains, equals the gold value.
- text: the normalized gold text occurs in the normalized answer, not as part of a longer word or number.
- path: as text, with slashes normalized.
- verbatim: as text, but punctuation and case are ignored.
- choice: the answer names exactly one status class (done, open, blocked), and it is the gold class.
- keywords: every keyword group has one alternative in the answer.
An answer "unknown" or an empty answer is wrong.

Retention (model-free): the compacted history text has the fact. Each retention group needs one alternative, and
one occurrence of each group must fit in one span of RETENTION_WINDOW characters (from the first to the last start
position). This check is approximate: it shows whether the text is
still there, not whether a model can use it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from typing import Any

RETENTION_WINDOW = 600
QA_INSTRUCTION = (
    "Answer the numbered questions about this session. Use only the conversation above. If the conversation "
    "does not give the answer, write \"unknown\". Copy names, paths, numbers, commands, and rules exactly. "
    "Reply with one JSON object that maps each question number (as a string) to a short answer string. "
    "Write nothing else.")
QA_BATCH_MAX = 12
STATUS_CLASSES = {"done": ("done", "completed", "complete", "finished"),
                  "open": ("open", "pending", "not started", "in progress", "todo", "to do", "not done", "incomplete",
                           "unfinished"),
                  "blocked": ("blocked", "waiting")}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# A negation on normalized text (normalize removes apostrophes: "isn't" becomes "isnt"), then at most three words.
# The words between are letters only (a comma or a full stop ends the phrase), and a conjunction ends it too ("not
# tested but completed" is done).
# A contrastive "not only", "not merely", "not just", or "not simply" is not a negation ("not only completed but
# verified" is done).
# A failure or inability before "to" is a negation too: "failed to complete" and "unable to finish" are open.
NEGATION = (r"(?<![a-z0-9])(?:not|never|no|cannot|cant|isnt|arent|wasnt|werent|hasnt|havent|hadnt|wont|dont|doesnt|"
            r"didnt|couldnt|(?:failed|fails|failing|unable) to)(?! (?:only|merely|just|simply)(?![a-z0-9]))"
            r"(?: (?!(?:and|but|or|so|then|now|though|although|because|while)(?![a-z0-9]))[a-z]+){0,3}? ")
# A negated completion phrase names the open class. Examples: "not completed", "not fully completed", "not actually
# done", "cannot be completed", "has not been finished yet".
NEGATED_DONE = re.compile(NEGATION + r"(?:done|completed|complete|finished|finish)(?![a-z0-9])")
# A negated open or blocked label names no class: "still open, not blocked" is open, and "not open" alone is no
# class. ("not started" and "not done" are open phrases: the label after the negation is not in this list.)
NEGATED_LABEL = re.compile(NEGATION + r"(?:blocked|waiting|open|pending|in progress|todo|to do)(?![a-z0-9])")
# An answer that states uncertainty about the status names no class: "not sure if completed" is not an open answer.
# The uncertainty phrase must have a status word (or "status") at most four words after it in the same clause, so
# "Blocked, but the exact cause is unclear" stays blocked.
STATUS_WORDS = "|".join(sorted((re.escape(word) for words in STATUS_CLASSES.values() for word in words), key=len,
                               reverse=True))
UNCERTAIN = re.compile(r"(?<![a-z0-9])(?:not sure|not clear|not certain|unsure|unclear|uncertain|cannot tell|cant "
                       r"tell|unknown whether|unknown if)(?: [a-z]+){0,4}? (?:" + STATUS_WORDS + r"|status)"
                       r"(?![a-z0-9])")


def normalize(text: Any) -> str:
    """Lower case, one space between words, no quote characters or backticks, forward slashes."""
    return normalize_text(str(text or ""))


@functools.lru_cache(maxsize=65536)
def normalize_text(text: str) -> str:
    text = text.replace("\\\\", "\\").replace('\\"', '"').replace("\\n", "\n").replace("\\t", " ")
    text = re.sub(r"[`\"'‘’“”]", "", text).replace("\\", "/")
    return " ".join(text.lower().split())


def loose(text: Any) -> str:
    """normalize, and only letters, digits, and single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalize(text)).split())


# A dotted version ("7.3.0", "3.0.0-beta.2", "1.2.0+build.5") is one token. It can have a "v" before it ("v7.3.0"),
# but not another letter, a digit, or a sign. Another version part, prerelease, or build suffix after it makes
# another version ("7.3.0.1", "7.3.0-beta", "3.0.0-beta.2.1"). A one-dot number keeps the number edge after it.
VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)+(?:-[0-9a-z]+(?:\.[0-9a-z]+)*)?(?:\+[0-9a-z]+(?:\.[0-9a-z]+)*)?")
VERSION_LEFT = r"(?<![a-uw-z0-9+-])(?<![a-z0-9]v)(?<![0-9][.,])"
VERSION_RIGHT = r"(?![0-9a-z])(?![.+-][0-9a-z])"
NUMBER_RIGHT = r"(?![0-9])(?![.,][0-9])"
NUMERIC = re.compile(r"[0-9.,:/ x%+-]+")


def number_pattern(needle: str) -> str | None:
    """The regular expression of a numeric needle as a complete number, or None for another needle. A dotted
    version uses the version edges (VERSION). Another number has no letter, digit, decimal part, or sign before it
    ("-5" is not "5", unless the needle has the sign) and no digit or decimal part after it ("5.5" is not "5"); a
    unit such as "5s" or "5ms" can follow."""
    if VERSION.fullmatch(needle):
        # A one-dot number without a suffix ("0.412", "91.6") is a decimal: a unit can follow ("0.412s").
        decimal = needle.count(".") == 1 and not re.search(r"[+-]", needle)
        return VERSION_LEFT + re.escape(needle) + (NUMBER_RIGHT if decimal else VERSION_RIGHT)
    if NUMERIC.fullmatch(needle) and any(char.isdigit() for char in needle):
        sign = "" if needle[0] in "+-" else r"(?<![+-])"
        return r"(?<![a-z0-9])(?<![0-9][.,])" + sign + re.escape(needle) + NUMBER_RIGHT
    return None


def contains(haystack: str, needle: str) -> bool:
    """needle occurs in haystack, and an edge that is a letter or digit is not part of a longer word. A numeric
    needle matches only as a complete number (number_pattern)."""
    if not needle:
        return False
    numeric = number_pattern(needle)
    if numeric:
        return re.search(numeric, haystack) is not None
    left = r"(?<![a-z0-9])" if needle[0].isalnum() else ""
    right = r"(?![a-z0-9])" if needle[-1].isalnum() else ""
    return re.search(left + re.escape(needle) + right, haystack) is not None


def keyword_in(text: str, keyword: str) -> bool:
    """A keyword is a word start ("round" matches "rounding"). A short or numeric keyword is a whole word."""
    if len(keyword) <= 2 or re.fullmatch(r"[\d.,%]+", keyword):
        return contains(text, keyword)
    return re.search(r"(?<![a-z0-9])" + re.escape(keyword), text) is not None


def status_class(answer: str) -> str | None:
    """The one status class that the answer names, or None for no class or more than one. A negated completion
    phrase (NEGATED_DONE) names the open class, and its completion word does not name the done class. A negated
    open or blocked label (NEGATED_LABEL) names no class. An answer that states uncertainty (UNCERTAIN) names no
    class."""
    text = normalize(answer)
    if UNCERTAIN.search(text):
        return None
    negated = NEGATED_DONE.search(text) is not None
    text = NEGATED_LABEL.sub(" ", NEGATED_DONE.sub(" ", text))
    found = {name for name, words in STATUS_CLASSES.items() if any(contains(text, word) for word in words)}
    if negated and found != {"blocked"}:
        # A negated completion infers the open class. An explicit blocked label wins over it: "Blocked; not
        # completed yet" is blocked.
        found.add("open")
    return found.pop() if len(found) == 1 else None


# An answer that starts by saying that the conversation does not give the answer is unknown, also with a guess
# after it ("unknown, maybe 7.3.0").
UNKNOWN_START = re.compile(r"(?:unknown|not stated|not given|not mentioned|not specified|not established|not recorded|"
                           r"not known|not sure|unsure|unclear|uncertain|i dont know|i do not know|cannot tell|"
                           r"cant tell|cannot determine|cant determine|no information|n/a)(?![a-z0-9])")


def is_unknown(answer: Any) -> bool:
    """True for an empty answer, an answer of "none", or an answer that starts with an unknown phrase
    (UNKNOWN_START)."""
    text = normalize(answer)
    return not text or text in ("unknown.", "none") or UNKNOWN_START.match(text) is not None


def score(question: dict, answer: Any) -> bool:
    """True when the answer passes the rule of the question."""
    if is_unknown(answer):
        return False
    kind, gold = question["kind"], question["gold"]
    text = normalize(answer)
    if kind == "number":
        skip = set(NUMBER.findall(question["question"].replace(",", "")))
        numbers = [value for value in NUMBER.findall(text.replace(",", "")) if value not in skip]
        return bool(numbers) and float(numbers[0]) == float(gold)
    if kind in ("text", "path"):
        return contains(text, normalize(gold))
    if kind == "verbatim":
        return contains(loose(answer), loose(gold))
    if kind == "choice":
        return status_class(str(answer)) == gold
    if kind == "keywords":
        return all(any(keyword_in(text, normalize(alternative)) for alternative in group) for group in gold)
    raise ValueError(f"unknown question kind {kind}")


def history_text(rows: list) -> str:
    """The text of all compacted rows: content, tool call names, and tool call arguments."""
    parts = []
    for row in rows:
        content = row.get("content")
        if isinstance(content, list):
            content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        parts.append(str(content or ""))
        for call in row.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(f"{function.get('name', '')} {function.get('arguments', '')}")
    return "\n".join(parts)


def short_text(normal: str) -> bool:
    """True for a normalized fact text that is short (under 6 characters) or only digits and number signs: it also
    matches inside unrelated text."""
    return len(normal) < 6 or bool(re.fullmatch(r"[0-9.,:/ x%+-]+", normal))


def fact_positions(needle: str, normal: str) -> list[int]:
    """The start positions of a normalized fact text in normalized text. A numeric fact matches only as a whole
    number: "5" does not match in "15", "50", or "5.5" (a unit such as "5s" or "5ms" can follow). Another short
    fact matches only at the start of a word, as a word stem ("doc" in "docs", not in "undocumented")."""
    if not needle:
        return []
    pattern = number_pattern(needle) or re.escape(needle)
    if pattern == re.escape(needle) and short_text(needle):
        pattern = r"(?<![a-z0-9])" + pattern
    return [match.start() for match in re.finditer(pattern, normal)]


def retained(question: dict, text: str) -> bool:
    """True when the fact is still in the normalized compacted text (approximate)."""
    if question["kind"] == "verbatim":
        return loose(question["gold"]) in loose(text)
    normal = normalize(text)
    positions = []
    for group in question["retention"]:
        found = []
        for alternative in group:
            found += fact_positions(normalize(alternative), normal)
        if not found:
            return False
        positions.append(sorted(found))
    if len(positions) == 1:
        return True
    # One span for all groups: the span starts at the first chosen occurrence, so try each occurrence of any group
    # as the start, and each group needs an occurrence from there to RETENTION_WINDOW characters after it.
    for start in sorted({position for group in positions for position in group}):
        if all(any(start <= position <= start + RETENTION_WINDOW for position in group) for group in positions):
            return True
    return False


def parse_answers(text: str, numbers: list[int]) -> tuple[dict[str, str], bool]:
    """Return ({question number: answer}, parsed). Accept a JSON object, also inside a code fence or after
    thinking text; else accept lines of the form "N. answer" or "N: answer"."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    # Decode a JSON object at each "{" (a malformed brace span before the answer does not stop the scan). Keep the
    # outer objects only, and use the last one that has a question number as a key (else the last object).
    objects, position, decoder = [], 0, json.JSONDecoder()
    while (start := text.find("{", position)) >= 0:
        try:
            data, end = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 1
            continue
        if isinstance(data, dict):
            objects.append(data)
        position = end
    keyed = [data for data in objects if {str(n) for n in numbers} & {str(key).strip().rstrip(".") for key in data}]
    if keyed or objects:
        data = (keyed or objects)[-1]
        return {str(key).strip().rstrip("."): ("" if value is None else
                                                 json.dumps(value) if isinstance(value, (dict, list))
                                                 else str(value)) for key, value in data.items()}, True
    answers = {}
    for line in text.splitlines():
        match = re.match(r"\s*\"?(\d+)\"?\s*[.:)\-]\s*(.+)", line)
        if match and int(match.group(1)) in numbers:
            answers[match.group(1)] = match.group(2).strip().strip('",')
    return answers, bool(answers)


def qa_prompt(items: list[tuple[int, str]]) -> str:
    """The user row of one QA request: the instruction and the numbered questions (number, question text)."""
    return QA_INSTRUCTION + "\n\n" + "\n".join(f"{number}. {text}" for number, text in items)


def prompt_sha256(text: str) -> str:
    """The SHA-256 of a QA user row: --rescore checks that a raw answer is the answer to this exact prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_leaks(items: list[tuple[int, dict]]) -> list[str]:
    """The ids of the questions in one batch (number, gold question) whose gold rule passes on text that the batch
    prompt gives: the full prompt, each line (also the own line: an echo), or each word of another line (the
    instruction, another question, or a question number)."""
    text = qa_prompt([(number, question["question"]) for number, question in items])
    lines = [line for line in text.splitlines() if line.strip()]
    found = []
    for number, question in items:
        own = f"{number}. {question['question']}"
        words = dict.fromkeys(word for line in lines if line != own for word in line.split())
        if any(score(question, candidate) for candidate in [text, *lines, *words]):
            found.append(question["id"])
    return found


def missing_answers(text: str, numbers: list[int]) -> list[int]:
    """The question numbers of a batch that a reply does not answer (no answer or an empty answer). A reply that
    does not parse misses all."""
    answers, parsed = parse_answers(text, numbers)
    return [n for n in numbers if not parsed or not answers.get(str(n), "").strip()]


def wilson(successes: int, total: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """The 95% Wilson score interval of a rate."""
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)



def incomplete_beta(x: float, a: float, b: float) -> float:
    """The regularized incomplete beta function I_x(a, b) (continued fraction, modified Lentz method)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - incomplete_beta(1.0 - x, b, a)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)) / a
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 1000):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return front * result


def t_cdf(t: float, df: int) -> float:
    """The CDF of Student's t distribution with df degrees of freedom."""
    tail = 0.5 * incomplete_beta(df / (df + t * t), df / 2, 0.5)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(df: int, p: float = 0.975) -> float:
    """The exact p quantile of Student's t distribution (bisection on t_cdf), for each df >= 1."""
    if df <= 0:
        return float("nan")
    low, high = 0.0, 1.0
    while t_cdf(high, df) < p:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if t_cdf(middle, df) < p else (low, middle)
    return high


def mean_interval(values: list[float]) -> dict[str, Any]:
    """Mean, sample standard deviation, and the 95% t interval of the mean."""
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "low": None, "high": None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "sd": None, "low": None, "high": None}
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    half = t_quantile(len(values) - 1) * sd / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "sd": sd, "low": mean - half, "high": mean + half}


def mcnemar(first_only: int, second_only: int) -> float | None:
    """Exact two-sided McNemar p-value from the discordant pairs (binomial test with p = 0.5)."""
    n = first_only + second_only
    if n == 0:
        return None
    k = min(first_only, second_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
