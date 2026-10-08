"""Check the issue, AI agent, and test requirements for new pull requests.

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
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ApiError()
            return result
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
    html_code = []
    nested = False
    markup = re.compile(
        r"<!--|--!?>|<(/?)(pre|code|script|style|textarea)(?=[ \t/>]|$)[^>]*(?:>|$)", re.IGNORECASE
    )

    def scan_markup(line: str) -> bool:
        nonlocal comment
        hidden = comment or bool(html_code)
        for marker in markup.finditer(line):
            if marker[0] == "<!--" and not comment:
                comment = True
                hidden = True
            elif marker[0] in ("-->", "--!>") and comment:
                comment = False
            elif marker[2] and not comment:
                hidden = True
                tag = marker[2].lower()
                if not marker[1]:
                    html_code.append(tag)
                elif html_code and html_code[-1] == tag and marker[0].endswith(">"):
                    html_code.pop()
        return hidden

    for line in body.splitlines():
        if not line.strip():
            nested = False
        if html_code:
            scan_markup(line)
            continue
        if inline_code:
            for marker in re.finditer(r"`+", line):
                if inline_code is None:
                    inline_code = len(marker[0])
                elif len(marker[0]) == inline_code:
                    inline_code = None
            continue
        if fence:
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}[ \t]*", line):
                fence = None
            continue
        mark = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if mark and not comment:
            fence = (mark[1][0], len(mark[1]))
            continue
        # A comment must not change a code fence or join parts of a plain line.
        if scan_markup(line):
            continue
        ticks = list(re.finditer(r"(?<!\\)`+", line))
        if ticks:
            for marker in ticks:
                if inline_code is None:
                    inline_code = len(marker[0])
                elif len(marker[0]) == inline_code:
                    inline_code = None
            continue
        # Require column zero so list continuations cannot become policy lines.
        list_or_quote = re.match(r" {0,3}(?:[-+*][ \t]+|[0-9]+[.)][ \t]+|>)", line)
        if list_or_quote:
            nested = True
        if line.startswith((" ", "\t")):
            continue
        if list_or_quote:
            if not re.fullmatch(r"- \[[xX]\] I ran all tests listed in CONTRIBUTING\.md and all passed\.", line):
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
        # Check again before reporting a decision based on PR or issue data.
        fresh = api.get(f"/repos/{repository}/pulls/{number}")
        fields = ("number", "body", "created_at", "updated_at", "state", "merged", "head", "base", "user")
        if any(fresh.get(field) != pull.get(field) for field in fields):
            raise PolicyError("The pull request changed during the check. Run the check again.")
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
