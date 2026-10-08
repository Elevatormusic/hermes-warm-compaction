"""Check the issue, open PR limit, AI agent, and test requirements for new PRs.

Use one complete plain line at column zero: Closes #123, Fixes #123, or Resolves #123.
The same keywords can precede an exact HTTPS URL for this repository's issue.
Keywords are not case-sensitive. Comments and code examples do not count.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


class PolicyError(Exception):
    """The check cannot make a safe decision."""


class ApiError(PolicyError):
    """An API request failed without a policy decision."""

    def __init__(self, status: int | None = None):
        self.status = status
        super().__init__(f"GitHub API request failed (status {status or 'unknown'}).")


class NoRedirect(HTTPRedirectHandler):
    """Keep the token on the fixed GitHub API host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHubApi:
    def __init__(self, token: str):
        self.token = token
        self.opener = build_opener(NoRedirect())

    def get(self, path: str) -> dict:
        result = self._read(path)
        if not isinstance(result, dict):
            raise ApiError()
        return result

    def get_list(self, path: str) -> list[dict]:
        result = self._read(path)
        if not isinstance(result, list) or len(result) > 100 or any(not isinstance(item, dict) for item in result):
            raise ApiError()
        return result

    def _read(self, path: str):
        if not isinstance(path, str) or not path.startswith("/repos/") or re.search(r"[\x00-\x20\x7f]", path):
            raise ApiError()
        request = Request(
            "https://api.github.com" + path,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "warm-compaction-issue-policy",
            },
        )
        try:
            with self.opener.open(request, timeout=30) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ApiError()
            return json.loads(raw)
        except HTTPError as error:
            raise ApiError(error.code) from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise ApiError() from None


def timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise PolicyError("A required UTC timestamp is missing or invalid.")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise PolicyError("A required UTC timestamp is invalid.") from None


def plain_lines(body: str) -> list[str]:
    """Read bounded column-zero lines outside comments, code, and nested text."""
    if len(body) > 128 * 1024:
        raise PolicyError("The pull request body exceeds the policy size limit.")
    lines = []
    fence = None
    comment = False
    inline_code = None
    html_blocks = []
    raw_html = None
    markup_end = None
    math_block = False
    tag_parts = []
    tag_quote = None
    nested = False
    comment_end = re.compile(r"--!?>")
    tag_start = re.compile(r"</?[A-Za-z]")
    declaration_start = re.compile(r"<![A-Z]")
    tick_run = re.compile(r"`+")
    hidden_tags = {
        "pre", "code", "script", "style", "textarea", "blockquote", "ul", "ol", "li", "dl", "dt", "dd", "details",
    }
    literal_tags = {"pre", "code", "script", "style", "textarea"}
    block_tags = (
        "address article aside base basefont blockquote body caption center col colgroup dd details dialog dir div "
        "dl dt fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe "
        "legend li link main menu menuitem nav noframes ol optgroup option p param search section summary table "
        "tbody td tfoot th thead title tr track ul"
    )
    raw_block_tag = re.compile(r" {0,3}</?(?:" + "|".join(block_tags.split()) + r")(?=[ \t/>]|$)", re.IGNORECASE)
    attribute = r"[ \t]+[A-Za-z_:][A-Za-z0-9_.:-]*(?:[ \t]*=[ \t]*(?:[^\s\"'=<>`]+|'[^']*'|\"[^\"]*\"))?"
    complete_tag = re.compile(
        r" {0,3}(?:<[A-Za-z][A-Za-z0-9-]*(?:" + attribute + r")*[ \t]*/?>|</[A-Za-z][A-Za-z0-9-]*[ \t]*>)[ \t]*$"
    )
    math_marker = re.compile(r"(?<!\\)(?:\\\\)*\$\$")

    def scan_markup(line: str, allow_inline: bool = True) -> bool:
        nonlocal comment, tag_quote, inline_code, markup_end
        hidden = comment or bool(html_blocks) or bool(tag_parts) or inline_code is not None or markup_end is not None
        index = 0
        while index < len(line):
            if comment:
                end = comment_end.search(line, index)
                if not end:
                    break
                comment = False
                index = end.end()
                continue
            if markup_end is not None:
                end = line.find(markup_end, index)
                if end < 0:
                    break
                index = end + len(markup_end)
                markup_end = None
                continue
            if tag_parts:
                char = line[index]
                tag_parts.append(char)
                if tag_quote:
                    if char == tag_quote:
                        tag_quote = None
                elif char in ("'", '"'):
                    tag_quote = char
                elif char == ">":
                    tag = re.match(r"<(/?)([A-Za-z][A-Za-z0-9:-]*)(?=[\s/>])", "".join(tag_parts))
                    if tag and tag[2].lower() in hidden_tags:
                        name = tag[2].lower()
                        if not tag[1]:
                            html_blocks.append(name)
                        elif html_blocks and html_blocks[-1] == name:
                            html_blocks.pop()
                    tag_parts.clear()
                index += 1
                continue
            if inline_code is not None:
                marker = tick_run.search(line, index)
                if not marker:
                    break
                index = marker.end()
                if len(marker[0]) == inline_code:
                    inline_code = None
                continue
            if line.startswith("<!--", index):
                comment = True
                hidden = True
                index += 4
            elif line.startswith("<?", index):
                markup_end = "?>"
                hidden = True
                index += 2
            elif line.startswith("<![CDATA[", index):
                markup_end = "]]>"
                hidden = True
                index += 9
            elif declaration_start.match(line, index):
                markup_end = ">"
                hidden = True
                index += 2
            elif tag_start.match(line, index):
                tag_parts.append("<")
                hidden = True
                index += 1
            elif allow_inline and not any(tag in literal_tags for tag in html_blocks) and line[index] == "`":
                start = index
                while start and line[start - 1] == "\\":
                    start -= 1
                marker = tick_run.match(line, index)
                if (index - start) % 2 == 0:
                    inline_code = len(marker[0])
                    hidden = True
                index = marker.end()
            else:
                index += 1
        if tag_parts:
            tag_parts.append("\n")
        return hidden

    for line in body.splitlines():
        if not line.strip():
            nested = False
            if raw_html == "blank":
                raw_html = None
        if raw_html:
            scan_markup(line, allow_inline=False)
            continue
        if math_block:
            if math_marker.search(line):
                math_block = False
            continue
        if inline_code is not None:
            scan_markup(line)
            continue
        if fence:
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}[ \t]*", line):
                fence = None
            continue
        mark = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        literal_html = any(tag in literal_tags for tag in html_blocks)
        if (
            html_blocks and not literal_html and not tag_parts and not markup_end and not comment
            and re.match(r"(?: {4}| {0,3}\t)", line)
        ):
            continue
        if mark and not comment and not literal_html and not tag_parts and not markup_end:
            fence = (mark[1][0], len(mark[1]))
            continue
        if html_blocks or tag_parts or markup_end:
            starts_raw_close = (
                not literal_html and not comment and not tag_parts and not markup_end
                and re.match(r" {0,3}</", line) is not None and raw_block_tag.match(line) is not None
            )
            scan_markup(line, allow_inline=markup_end is None and not literal_html and not starts_raw_close)
            if starts_raw_close:
                # A block HTML close keeps the following lines raw until a blank.
                raw_html = "blank"
            continue
        if not comment:
            # Raw HTML blocks use their Markdown end rule, not a closing tag.
            if re.match(r" {0,3}(?:<\?|<![A-Z]|<!\[CDATA\[)", line):
                scan_markup(line, allow_inline=False)
                continue
            if re.match(r" {0,3}<(?:script|pre|style|textarea)(?=[ \t>]|$)", line, re.IGNORECASE):
                scan_markup(line, allow_inline=False)
                continue
            if raw_block_tag.match(line) or complete_tag.fullmatch(line):
                raw_html = "blank"
                scan_markup(line, allow_inline=False)
                continue
            if re.match(r" {0,3}\$\$", line):
                opening = math_marker.search(line)
                math_block = math_marker.search(line, opening.end()) is None
                continue
        # A comment must not change a code fence or join parts of a plain line.
        if scan_markup(line):
            continue
        # Require column zero so list continuations cannot become policy lines.
        list_or_quote = re.match(r" {0,3}(?:[-+*][ \t]+|[0-9]+[.)][ \t]+|>)", line)
        was_nested = nested
        if list_or_quote:
            nested = True
        if line.startswith((" ", "\t")):
            continue
        if list_or_quote:
            if was_nested or not re.fullmatch(
                r"- \[[xX]\] I ran all tests listed in CONTRIBUTING\.md and all passed\.", line
            ):
                continue
        elif re.match(r"#{1,6}[ \t]", line):
            nested = False
        elif nested:
            continue
        lines.append(line.rstrip(" \t"))
    return lines


def issue_references(body: str, repository: str) -> list[int]:
    """Read explicit references outside comments and code blocks."""
    pattern = re.compile(
        r"(?:Closes|Fixes|Resolves)[ \t]+(?:#([1-9][0-9]*)|"
        + re.escape(f"https://github.com/{repository}/issues/")
        + r"([1-9][0-9]*))",
        re.IGNORECASE,
    )
    numbers = []
    for line in plain_lines(body):
        match = pattern.fullmatch(line)
        if match:
            number = int(match[1] or match[2])
            if number not in numbers:
                numbers.append(number)
    if len(numbers) > 20:
        raise PolicyError("Use at most 20 issue reference lines.")
    return numbers


def ai_agent_requirement(lines: list[str]) -> str | None:
    """Return a failure reason when the AI agent section is incomplete."""
    headings = [index for index, line in enumerate(lines) if line == "## AI agent use"]
    if len(headings) != 1:
        return "Add one visible section with the exact heading: ## AI agent use."
    section = []
    for line in lines[headings[0] + 1:]:
        if re.match(r"#{1,6}(?:[ \t]|$)", line):
            break
        section.append(line)
    records = [re.fullmatch(r"(Harness|Model|Work):[ \t]*(.*)", line) for line in section]
    records = [record for record in records if record]
    no_agent = section.count("No AI agent used")
    if no_agent:
        if no_agent == 1 and not records:
            return None
        return "Use either No AI agent used or complete agent records, not both."
    if not records:
        return "In ## AI agent use, write No AI agent used or plain Harness:, Model:, and Work: lines."
    if len(records) > 60:
        return "Use at most 20 AI agent records."
    expected = ("Harness", "Model", "Work")
    for index, record in enumerate(records):
        field, value = record.groups()
        if field != expected[index % 3]:
            return "Each AI agent record must have Harness:, Model:, and Work: lines, in that order."
        value = value.strip()
        normalized = value.strip("`*_ ").casefold()
        placeholders = {
            "...", "…", "todo", "tbd", "unknown", "none", "n/a", "not applicable", "not run",
            "harness", "harness name", "model", "model name", "exact model", "exact model id",
            "work", "task", "task description", "your harness", "your model", "your task",
            "describe work", "describe the work", "describe task", "describe the task",
            "describe what the agent did", "list every harness", "list every model",
        }
        if (
            not value or normalized in placeholders or re.search(r"<[^>]*>", value)
            or re.fullmatch(r"\[[^]]*\]", value)
            or re.fullmatch(r"(?:replace|enter|insert)[ \t]+(?:value|here|your (?:harness|model|task))", normalized)
        ):
            return f"Give a nonempty {field}: value. Template placeholders do not count."
    if len(records) % 3:
        return f"Each AI agent record needs a {expected[len(records) % 3]}: line."
    return None


def body_requirement(body: str) -> str | None:
    """Check the AI agent statement and the required test confirmation."""
    lines = plain_lines(body)
    reason = ai_agent_requirement(lines)
    if reason:
        return reason
    if not any(re.fullmatch(
        r"- \[[xX]\] I ran all tests listed in CONTRIBUTING\.md and all passed\.", line
    ) for line in lines):
        return "Add the checked line: - [x] I ran all tests listed in CONTRIBUTING.md and all passed."
    return None


OPEN_PR_LIMIT = 5
OPEN_PR_PAGE_LIMIT = 100


def author_open_pulls(api, repository: str, author: str, number: int) -> frozenset[int]:
    """Read all open PRs by this author, including drafts and older PRs."""
    seen = set()
    numbers = set()
    for page in range(1, OPEN_PR_PAGE_LIMIT + 1):
        query = urlencode({
            "state": "open", "creator": author, "sort": "created", "direction": "asc", "per_page": 100, "page": page,
        })
        items = api.get_list(f"/repos/{repository}/issues?{query}")
        if not isinstance(items, list) or len(items) > 100:
            raise PolicyError("The open PR list is incomplete or inconsistent.")
        for item in items:
            try:
                item_number = item["number"]
                login = item["user"]["login"]
                if (
                    not isinstance(item_number, int) or isinstance(item_number, bool) or item_number < 1
                    or item_number in seen or item["state"] not in ("open", "closed")
                    or not isinstance(login, str) or login.casefold() != author.casefold()
                ):
                    raise ValueError
                seen.add(item_number)
                is_pull = "pull_request" in item
                kind = "pull" if is_pull else "issues"
                if item["html_url"].lower() != f"https://github.com/{repository}/{kind}/{item_number}".lower():
                    raise ValueError
                if is_pull:
                    details = item["pull_request"]
                    if not isinstance(details, dict) or details["url"].lower() != (
                        f"https://api.github.com/repos/{repository}/pulls/{item_number}".lower()
                    ):
                        raise ValueError
                    merged_at = details.get("merged_at")
                    if merged_at is not None:
                        timestamp(merged_at)
                    if "draft" in item and not isinstance(item["draft"], bool):
                        raise ValueError
                    if item["state"] == "open" and merged_at is None:
                        numbers.add(item_number)
            except (KeyError, TypeError, AttributeError, ValueError):
                raise PolicyError("The open PR list is incomplete or inconsistent.") from None
        if len(items) < 100:
            if number not in numbers:
                raise PolicyError(
                    "The current pull request is missing from the author's open PR list. Run the check again."
                )
            return frozenset(numbers)
    raise PolicyError("The open PR list exceeds the page limit. The count is unknown.")


def pull_policy_fields(pull: dict) -> tuple:
    """Compare policy inputs without mutable repository and profile metadata."""
    try:
        values = [pull[field] for field in ("number", "body", "created_at", "state", "merged")]
        if (
            not isinstance(pull["number"], int) or isinstance(pull["number"], bool)
            or not isinstance(pull["merged"], bool)
        ):
            raise TypeError
        user = pull["user"]
        values.extend((user["login"].casefold(), user.get("id")))
        for name in ("head", "base"):
            branch = pull.get(name, {})
            if not isinstance(branch, dict):
                raise TypeError
            repository = branch.get("repo")
            if repository is None:
                repository = {}
            elif not isinstance(repository, dict):
                raise TypeError
            values.extend((branch.get("ref"), repository.get("full_name"), repository.get("id")))
            if name == "head":
                values.append(branch.get("sha"))
        return tuple(values)
    except (KeyError, TypeError, AttributeError):
        raise PolicyError("Pull request data is incomplete or inconsistent.") from None


def check_policy(api, repository: str, number: int, activated_at: str) -> tuple[bool, str]:
    """Check fresh API data. This script has no write operation."""
    activated = timestamp(activated_at)
    pull = api.get(f"/repos/{repository}/pulls/{number}")
    try:
        if (
            pull["number"] != number
            or pull["state"] not in ("open", "closed")
            or not isinstance(pull["merged"], bool)
            or pull["base"]["repo"]["full_name"].lower() != repository.lower()
        ):
            raise ValueError
        if pull["state"] != "open" or pull["merged"]:
            return True, "Skipped: the pull request is closed or merged."
        created_at = timestamp(pull["created_at"])
        if created_at < activated:
            return True, "Skipped: the pull request predates the policy."
        body = pull["body"]
        if body is not None and not isinstance(body, str):
            raise ValueError
        author = pull["user"]["login"]
        if not isinstance(author, str) or not author.strip():
            raise ValueError
    except (KeyError, TypeError, AttributeError, ValueError):
        raise PolicyError("Pull request data is incomplete or inconsistent.") from None

    def finish(passed: bool, message: str) -> tuple[bool, str]:
        open_pulls = None
        if passed:
            open_pulls = author_open_pulls(api, repository, author, number)
            if len(open_pulls) > OPEN_PR_LIMIT:
                passed = False
                message = (
                    f"Keep at most {OPEN_PR_LIMIT} open pull requests per author in this repository. "
                    f"This author has {len(open_pulls)}, including this PR, drafts, and older open PRs. "
                    "Close or merge an existing PR, then run the check again. This check does not close PRs."
                )
        # Check again before reporting a decision based on PR or issue data.
        fresh = api.get(f"/repos/{repository}/pulls/{number}")
        if pull_policy_fields(fresh) != pull_policy_fields(pull):
            raise PolicyError("The pull request changed during the check. Run the check again.")
        if passed and author_open_pulls(api, repository, author, number) != open_pulls:
            raise PolicyError("The author's open PR list changed during the check. Run the check again.")
        return passed, message

    reason = body_requirement(body or "")
    if reason:
        return finish(False, reason)
    if author.casefold() == "elevatormusic":
        return finish(True, "Passed: AI agent and test requirements met; the PR author is exempt from the issue rule.")

    for issue_number in issue_references(body or "", repository):
        try:
            issue = api.get(f"/repos/{repository}/issues/{issue_number}")
        except ApiError as error:
            if error.status == 404:
                continue
            raise
        if issue.get("number") != issue_number:
            raise PolicyError("Issue data is incomplete or inconsistent.")
        if "pull_request" in issue:
            continue
        if (
            not isinstance(issue.get("html_url"), str)
            or issue["html_url"].lower() != f"https://github.com/{repository}/issues/{issue_number}".lower()
        ):
            raise PolicyError("Issue data is incomplete or inconsistent.")
        if timestamp(issue.get("created_at")) < created_at:
            return finish(True, f"Passed: issue #{issue_number} was opened before this pull request.")

    return finish(False, (
        "Add a complete plain line: Closes #123, Fixes #123, or Resolves #123. "
        "Use a real issue in this repository that was opened before the pull request. "
        "The keyword can also precede its exact https://github.com/owner/repo/issues/123 URL. "
        "Comments and code examples do not count."
    ))


def main() -> int:
    try:
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise PolicyError("The repository name is missing or invalid.")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        event_name = os.environ.get("GITHUB_EVENT_NAME")
        if event_name == "pull_request_target" and event.get("action") in {
            "opened", "reopened", "edited", "synchronize", "ready_for_review"
        }:
            number = event["number"]
        elif event_name == "workflow_dispatch":
            raw_number = event["inputs"]["pr_number"]
            if not isinstance(raw_number, str) or not re.fullmatch(r"[1-9][0-9]*", raw_number):
                raise PolicyError("Enter a positive pull request number.")
            number = int(raw_number)
        else:
            raise PolicyError("This event cannot run the policy check.")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise PolicyError("The pull request number is invalid.")
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise PolicyError("The GitHub token is missing.")
        passed, message = check_policy(GitHubApi(token), repository, number, os.environ["ISSUE_POLICY_ACTIVATED_AT"])
        print(("" if passed else "::error::") + message)
        return 0 if passed else 1
    except PolicyError as error:
        # Do not print input text, API bodies, or the token.
        print("::error::" + str(error))
        return 1
    except (KeyError, OSError, ValueError, TypeError, AttributeError):
        print("::error::Policy input could not be read.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
