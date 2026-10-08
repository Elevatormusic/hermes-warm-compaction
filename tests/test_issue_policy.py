"""Check the issue policy with fake API replies. No network is used."""

from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from scripts import check_issue_policy as policy

REPOSITORY = "owner/repo"
ACTIVATED_AT = "2026-10-08T17:09:42Z"
PR_PATH = "/repos/owner/repo/pulls/23"
TEST_CONFIRMATION = "- [x] I ran all tests listed in CONTRIBUTING.md and all passed."
NO_AGENT = "## AI agent use\nNo AI agent used\n\n" + TEST_CONFIRMATION
AGENT_USE = "## AI agent use\nHarness: Codex\nModel: gpt-6.1-sol\nWork: Write policy tests.\n\n" + TEST_CONFIRMATION


def pull(**changes):
    data = {
        "number": 23,
        "body": "Closes #7\n\n" + NO_AGENT,
        "created_at": "2026-10-08T18:00:00Z",
        "updated_at": "2026-10-08T18:00:00Z",
        "state": "open",
        "merged": False,
        "user": {"login": "contributor"},
        "base": {"repo": {"full_name": REPOSITORY}, "ref": "main", "sha": "base-sha"},
        "head": {"repo": {"full_name": "contributor/fork"}, "ref": "change", "sha": "head-sha"},
    }
    data.update(changes)
    return data


def issue(number=7, **changes):
    data = {
        "number": number,
        "html_url": f"https://github.com/{REPOSITORY}/issues/{number}",
        "created_at": "2026-10-08T16:00:00Z",
        "state": "open",
        "user": {"login": "another-author"},
    }
    data.update(changes)
    return data


class FakeApi:
    def __init__(self, *responses, list_responses=None):
        self.responses = list(responses)
        self.paths = []
        self.list_responses = list(list_responses or [])
        self.explicit_list_responses = list_responses is not None
        self.list_paths = []
        self.default_open_pulls = [open_pull(author=responses[0]["user"]["login"])] if (
            responses and isinstance(responses[0], dict) and isinstance(responses[0].get("user", {}).get("login"), str)
        ) else []

    def get_list(self, path):
        self.list_paths.append(path)
        if not self.list_responses:
            if self.explicit_list_responses:
                raise AssertionError("An unexpected list API read was requested.")
            return copy.deepcopy(self.default_open_pulls)
        response = self.list_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)

    def get(self, path):
        self.paths.append(path)
        if not self.responses:
            raise AssertionError("An unexpected API read was requested.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


def open_pull(number=23, author="contributor", **changes):
    data = issue(number, user={"login": author}, html_url=f"https://github.com/{REPOSITORY}/pull/{number}")
    data["pull_request"] = {"url": f"https://api.github.com/repos/{REPOSITORY}/pulls/{number}", "merged_at": None}
    data.update(changes)
    return data


class OpenPullLimitTests(unittest.TestCase):
    def test_five_open_prs_pass_after_two_count_reads(self):
        items = [open_pull(number) for number in (23, 24, 25, 26, 27)]
        api = FakeApi(pull(), issue(), pull(), list_responses=[items, items])
        self.assertTrue(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])
        self.assertEqual(len(api.list_paths), 2)

    def test_six_open_prs_fail(self):
        items = [open_pull(number) for number in (23, 24, 25, 26, 27, 28)]
        api = FakeApi(pull(), issue(), pull(), list_responses=[items])
        passed, message = policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)
        self.assertFalse(passed)
        self.assertIn("5", message)

    def test_owner_has_no_open_pr_limit_exception(self):
        for author in ("Elevatormusic", "ELEVATORMUSIC"):
            with self.subTest(author=author):
                data = pull(user={"login": author}, body=NO_AGENT)
                items = [open_pull(number, author) for number in (23, 24, 25, 26, 27, 28)]
                api = FakeApi(data, data, list_responses=[items])
                passed, message = policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)
                self.assertFalse(passed)
                self.assertIn("6", message)
                self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_drafts_older_prs_and_all_base_branches_count(self):
        items = [open_pull(number) for number in (23, 24, 25, 26)]
        items += [
            open_pull(27, draft=True, created_at="2026-10-08T17:09:41Z"),
            open_pull(28, base={"ref": "another-branch"}),
        ]
        api = FakeApi(pull(), issue(), pull(), list_responses=[items])
        self.assertFalse(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])
        query = parse_qs(urlsplit(api.list_paths[0]).query)
        self.assertEqual(query, {
            "state": ["open"], "creator": ["contributor"], "sort": ["created"],
            "direction": ["asc"], "per_page": ["100"], "page": ["1"],
        })

    def test_closed_merged_prs_and_ordinary_issues_do_not_count(self):
        merged = open_pull(29)
        merged["pull_request"]["merged_at"] = "2026-10-08T18:01:00Z"
        items = [open_pull(number) for number in (23, 24, 25, 26, 27)] + [
            open_pull(28, state="closed"), merged, issue(30, user={"login": "contributor"}),
        ]
        api = FakeApi(pull(), issue(), pull(), list_responses=[items, items])
        self.assertTrue(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])

    def test_open_pr_without_optional_merged_timestamp_counts(self):
        items = [open_pull(number) for number in (23, 24, 25, 26, 27)]
        for item in items:
            del item["pull_request"]["merged_at"]
        api = FakeApi(pull(), issue(), pull(), list_responses=[items, items])
        self.assertTrue(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])
        items.append(open_pull(28))
        del items[-1]["pull_request"]["merged_at"]
        api = FakeApi(pull(), issue(), pull(), list_responses=[items])
        self.assertFalse(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])

    def test_case_insensitive_author(self):
        data = pull(user={"login": "ConTRibutor"})
        items = [open_pull(author="CONTRIBUTOR")]
        api = FakeApi(data, issue(), data, list_responses=[items, items])
        self.assertTrue(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])

    def test_query_values_are_encoded(self):
        author = "contributor&state=all#fragment"
        items = [open_pull(author=author)]
        api = FakeApi(list_responses=[items])
        self.assertEqual(policy.author_open_pulls(api, REPOSITORY, author, 23), frozenset({23}))
        parsed = urlsplit(api.list_paths[0])
        self.assertEqual(parsed.path, "/repos/owner/repo/issues")
        self.assertEqual(parsed.fragment, "")
        self.assertEqual(parse_qs(parsed.query)["creator"], [author])
        self.assertEqual(parse_qs(parsed.query)["state"], ["open"])

    def test_pagination_passes_only_after_all_issue_pages_are_read_twice(self):
        first = [issue(number, user={"login": "contributor"}) for number in range(100, 200)]
        second = [open_pull(number) for number in (23, 24, 25, 26, 27)]
        api = FakeApi(pull(), issue(), pull(), list_responses=[first, second, first, second])
        self.assertTrue(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])
        self.assertEqual([parse_qs(urlsplit(path).query)["page"] for path in api.list_paths],
                         [["1"], ["2"], ["1"], ["2"]])
        self.assertFalse(api.list_responses)

    def test_sixth_pr_on_a_later_page_refuses(self):
        first = [open_pull(number) for number in (23, 24, 25, 26, 27)]
        first += [issue(number, user={"login": "contributor"}) for number in range(100, 195)]
        api = FakeApi(pull(), issue(), pull(), list_responses=[first, [open_pull(28)]])
        self.assertFalse(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])
        self.assertEqual(len(api.list_paths), 2)

    def test_full_final_page_needs_an_empty_page(self):
        items = [open_pull()] + [issue(number, user={"login": "contributor"}) for number in range(100, 199)]
        api = FakeApi(list_responses=[items, []])
        self.assertEqual(policy.author_open_pulls(api, REPOSITORY, "contributor", 23), frozenset({23}))
        self.assertEqual(len(api.list_paths), 2)

    def test_page_limit_refuses_an_unknown_count(self):
        pages = [
            [open_pull()] + [issue(number, user={"login": "contributor"}) for number in range(100, 199)],
            [issue(number, user={"login": "contributor"}) for number in range(200, 300)],
        ]
        api = FakeApi(list_responses=pages)
        with patch.object(policy, "OPEN_PR_PAGE_LIMIT", 2):
            with self.assertRaisesRegex(policy.PolicyError, "page limit.*unknown"):
                policy.author_open_pulls(api, REPOSITORY, "contributor", 23)
        self.assertEqual(len(api.list_paths), 2)

    def test_current_pr_must_be_present_and_open(self):
        for items in ([], [open_pull(24)], [open_pull(state="closed")]):
            with self.subTest(items=items):
                api = FakeApi(pull(), issue(), pull(), list_responses=[items])
                with self.assertRaisesRegex(policy.PolicyError, "current pull request is missing"):
                    policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)

    def test_malformed_list_data_refuses(self):
        cases = [
            {}, None, [None], ["not-an-item"], [open_pull(number=True)], [open_pull(number=0)],
            [open_pull(state="unknown")], [open_pull(user={})], [open_pull(user={"login": "another-author"})],
            [open_pull(html_url="https://github.com/other/repo/pull/23")], [open_pull(html_url=None)],
            [open_pull(pull_request=None)], [open_pull(pull_request={})],
            [open_pull(pull_request={"url": "https://api.example.com/repos/owner/repo/pulls/23", "merged_at": None})],
            [open_pull(pull_request={"url": "https://api.github.com/repos/owner/repo/pulls/23", "merged_at": "bad"})],
            [open_pull(draft="true")], [open_pull()] * 101,
        ]
        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(policy.PolicyError):
                    policy.author_open_pulls(FakeApi(list_responses=[items]), REPOSITORY, "contributor", 23)

    def test_duplicate_items_on_same_or_later_pages_refuse(self):
        first = [open_pull()] + [issue(number, user={"login": "contributor"}) for number in range(100, 199)]
        for pages in ([[open_pull(), open_pull()]], [first, [issue(100, user={"login": "contributor"})]]):
            with self.subTest(pages=len(pages)):
                with self.assertRaisesRegex(policy.PolicyError, "incomplete or inconsistent"):
                    policy.author_open_pulls(FakeApi(list_responses=pages), REPOSITORY, "contributor", 23)

    def test_list_api_errors_refuse_on_both_reads_and_later_pages(self):
        first = [open_pull()] + [issue(number, user={"login": "contributor"}) for number in range(100, 199)]
        for status in (None, 403, 404, 429, 500):
            for pages in ([policy.ApiError(status)], [[open_pull()], policy.ApiError(status)],
                          [first, policy.ApiError(status)]):
                with self.subTest(status=status, pages=len(pages)):
                    api = FakeApi(pull(), issue(), pull(), list_responses=pages)
                    with self.assertRaises(policy.ApiError):
                        policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)

    def test_open_pr_set_changes_abort_a_pass(self):
        first = [open_pull(number) for number in (23, 24, 25, 26, 27)]
        for last in (first + [open_pull(28)], first[:-1], first[:-1] + [open_pull(28)]):
            with self.subTest(numbers=[item["number"] for item in last]):
                api = FakeApi(pull(), issue(), pull(), list_responses=[first, last])
                with self.assertRaisesRegex(policy.PolicyError, "open PR list changed"):
                    policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)

    def test_old_candidate_skips_count_even_above_limit(self):
        api = FakeApi(pull(created_at="2026-10-08T17:09:41Z"), list_responses=[
            [open_pull(number) for number in (23, 24, 25, 26, 27, 28)],
        ])
        self.assertTrue(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0])
        self.assertEqual(api.list_paths, [])


class ReviewRegressionTests(unittest.TestCase):
    def test_raw_html_block_cannot_supply_declarations(self):
        body = "<div>\nCloses #7\n" + NO_AGENT + "\n</div>"
        self.assertEqual(policy.issue_references(body, REPOSITORY), [])
        self.assertIsNotNone(policy.body_requirement(body))

    def test_markup_after_multiline_inline_code_is_still_scanned(self):
        body = "`\nexample`<!--\nCloses #7\n" + NO_AGENT
        self.assertEqual(policy.issue_references(body, REPOSITORY), [])
        self.assertIsNotNone(policy.body_requirement(body))

    def test_block_math_cannot_supply_declarations(self):
        body = "$$\nCloses #7\n" + NO_AGENT + "\n$$"
        self.assertEqual(policy.issue_references(body, REPOSITORY), [])
        self.assertIsNotNone(policy.body_requirement(body))

    def test_unrelated_repository_metadata_does_not_abort_a_pass(self):
        first = pull()
        last = copy.deepcopy(first)
        last["head"]["repo"]["stargazers_count"] = 1
        last["base"]["repo"]["updated_at"] = "2026-10-08T18:01:00Z"
        last["user"]["avatar_url"] = "https://example.com/avatar.png"
        self.assertTrue(policy.check_policy(FakeApi(first, issue(), last), REPOSITORY, 23, ACTIVATED_AT)[0])

    def test_checkbox_after_an_example_needs_a_blank_line(self):
        body = NO_AGENT.replace(TEST_CONFIRMATION, "- Example:\n" + TEST_CONFIRMATION)
        self.assertIsNotNone(policy.body_requirement(body))

    def test_raw_html_tag_blocks_need_a_blank_line_after_their_close(self):
        for opening, closing in (
            ("<div>", "</div>"), ("<TABLE>", "</TABLE>"), ("<details>", "</details>"),
            ("<section class='example'>", "</section>"), ("<div\nclass='example'>", "</div>"),
            ("<span>", "</span>"), ("<custom data-value=example>", "</custom>"),
            ("<img src=example.png />", ""), ("</custom>", ""), ("</div>", ""),
        ):
            with self.subTest(opening=opening):
                body = opening + "\nCloses #7\n## AI agent use\nNo AI agent used\n" + TEST_CONFIRMATION
                body += ("\n" + closing if closing else "") + "\nCloses #8"
                self.assertEqual(policy.issue_references(body, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(body))
                public = body + "\n\nCloses #9\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(public, REPOSITORY), [9])
                self.assertIsNone(policy.body_requirement(public))

    def test_raw_html_special_blocks_keep_their_own_end_rule(self):
        for opening, closing in (("<?example", "?>"), ("<!DOCTYPE example", ">"), ("<![CDATA[", "]]>")):
            with self.subTest(opening=opening):
                body = opening + "\n\nCloses #7\n" + NO_AGENT + "\n" + closing
                self.assertEqual(policy.issue_references(body, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(body))
                public = body + "\nCloses #8\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                self.assertIsNone(policy.body_requirement(public))
                one_line = opening + closing + "\n" + NO_AGENT
                self.assertIsNone(policy.body_requirement(one_line))

    def test_script_pre_style_and_textarea_end_at_the_matching_close(self):
        for tag in ("script", "pre", "style", "textarea"):
            with self.subTest(tag=tag):
                hidden = f"<{tag}>\n`\n\nCloses #7\n{NO_AGENT}\n</{tag}>"
                self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(hidden))
                public = hidden + "\nCloses #8\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                self.assertIsNone(policy.body_requirement(public))

    def test_raw_html_does_not_interpret_code_or_math_markers(self):
        for marker in ("```", "`", "$$"):
            with self.subTest(marker=marker):
                body = "<div>\n" + marker + "\nCloses #7\n</div>\n\nCloses #8\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(body, REPOSITORY), [8])
                self.assertIsNone(policy.body_requirement(body))

    def test_text_and_autolinks_do_not_start_a_raw_html_block(self):
        for line in ("<https://example.com>", "<author@example.com>", "2 < 3", "Example: <span>word</span>"):
            with self.subTest(line=line):
                body = line + "\nCloses #7\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(body, REPOSITORY), [7])
                self.assertIsNone(policy.body_requirement(body))

    def test_inline_code_suffix_comments_and_pre_blocks_hide_declarations(self):
        for marker in ("`", "``"):
            for opening, closing in (("<!--", "-->"), ("<!--", "--!>"), ("<pre>", "</pre>")):
                with self.subTest(marker=marker, opening=opening, closing=closing):
                    hidden = marker + "\nexample" + marker + opening + "\nCloses #7\n" + NO_AGENT + "\n" + closing
                    self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])
                    self.assertIsNotNone(policy.body_requirement(hidden))
                    public = hidden + "\n\nCloses #8\n\n" + NO_AGENT
                    self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                    self.assertIsNone(policy.body_requirement(public))
        hidden = "``\nexample`different``<!--\nCloses #7\n" + NO_AGENT
        self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])

    def test_inline_code_does_not_interpret_markup_until_it_closes(self):
        for text in ("<!--", "<pre>", "<div>"):
            with self.subTest(text=text):
                body = "``\n" + text + "``\nCloses #7\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(body, REPOSITORY), [7])
                self.assertIsNone(policy.body_requirement(body))
        body = "`\nexample`<span title='<!--'>\nCloses #7\n\n" + NO_AGENT
        self.assertEqual(policy.issue_references(body, REPOSITORY), [7])
        self.assertIsNone(policy.body_requirement(body))

    def test_math_opening_content_and_blank_lines_stay_hidden(self):
        for opening in ("$$", "$$x + y", "   $$x + y"):
            with self.subTest(opening=opening):
                hidden = opening + "\n\nCloses #7\n" + NO_AGENT + "\n$$"
                self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(hidden))
                public = hidden + "\nCloses #8\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                self.assertIsNone(policy.body_requirement(public))
        self.assertIsNotNone(policy.body_requirement("$$\n" + NO_AGENT))

    def test_same_line_and_escaped_math_delimiters_do_not_hide_public_lines(self):
        for line in ("$$x + y$$", r"\$$", "Price: $$", r"$$x + y\\$$"):
            with self.subTest(line=line):
                body = line + "\nCloses #7\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(body, REPOSITORY), [7])
                self.assertIsNone(policy.body_requirement(body))
        body = "$$\n" + r"\$$" + "\nCloses #7\n" + NO_AGENT + "\n$$\n\nCloses #8\n\n" + NO_AGENT
        self.assertEqual(policy.issue_references(body, REPOSITORY), [8])
        self.assertIsNone(policy.body_requirement(body))

    def test_metadata_and_base_commit_changes_do_not_change_the_decision(self):
        for valid in (False, True):
            first = pull() if valid else pull(body="")
            last = copy.deepcopy(first)
            last.update(title="New title", labels=[{"name": "policy"}], draft=True, updated_at="2026-10-08T18:01:00Z")
            last["base"]["sha"] = "unrelated-main-commit"
            last["base"]["repo"]["pushed_at"] = "2026-10-08T18:01:00Z"
            last["user"]["login"] = "CONTRIBUTOR"
            api = FakeApi(first, issue(), last) if valid else FakeApi(first, last)
            self.assertEqual(policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)[0], valid)

    def test_real_policy_field_changes_abort_the_decision(self):
        changes = [
            ("number", 24), ("created_at", "2026-10-08T18:00:01Z"), ("user.login", "other-author"), ("user.id", 42),
            ("head.sha", "new-head"), ("head.ref", "new-ref"), ("head.repo.full_name", "other/fork"),
            ("head.repo.id", 43), ("base.ref", "release"), ("base.repo.full_name", "other/repo"), ("base.repo.id", 44),
        ]
        for path, value in changes:
            for valid in (False, True):
                with self.subTest(path=path, valid=valid):
                    first = pull() if valid else pull(body="")
                    last = copy.deepcopy(first)
                    parts = path.split(".")
                    node = last
                    for part in parts[:-1]:
                        node = node[part]
                    node[parts[-1]] = value
                    api = FakeApi(first, issue(), last) if valid else FakeApi(first, last)
                    with self.assertRaisesRegex(policy.PolicyError, "changed during"):
                        policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)

    def test_malformed_recheck_data_fails_closed(self):
        for path, value in (
            ("user", None), ("user", {}), ("head", []), ("base", "bad"), ("head.repo", []),
            ("number", 23.0), ("merged", 0),
        ):
            with self.subTest(path=path):
                last = pull()
                parts = path.split(".")
                node = last
                for part in parts[:-1]:
                    node = node[part]
                node[parts[-1]] = value
                with self.assertRaises(policy.PolicyError):
                    policy.check_policy(FakeApi(pull(), issue(), last), REPOSITORY, 23, ACTIVATED_AT)

    def test_checkbox_after_a_list_or_quote_needs_its_own_block(self):
        for prefix in ("- Example:", "1. Example:", "> Example:", "  - Example:", "- [ ] Example task:"):
            with self.subTest(prefix=prefix):
                body = NO_AGENT.replace(TEST_CONFIRMATION, prefix + "\n" + TEST_CONFIRMATION)
                self.assertIsNotNone(policy.body_requirement(body))
                public = NO_AGENT.replace(TEST_CONFIRMATION, prefix + "\n\n" + TEST_CONFIRMATION)
                self.assertIsNone(policy.body_requirement(public))
        public = NO_AGENT.replace(TEST_CONFIRMATION, "- Example:\n## Tests\n" + TEST_CONFIRMATION)
        self.assertIsNone(policy.body_requirement(public))


class ParserTests(unittest.TestCase):
    def test_supported_lines_and_duplicates(self):
        body = "Closes #7\nfixes #7\nResolves https://github.com/OWNER/REPO/issues/8  "
        self.assertEqual(policy.issue_references(body, REPOSITORY), [7, 8])

    def test_only_complete_plain_lines_count(self):
        examples = [
            "Related to #7", "Closes #7 because it fails", "- Closes #7", "> Fixes #7", "`Closes #7`",
            "    Closes #7", "\tCloses #7", "Closes #0", "Closes #07", "Closes #7, #8",
            " Closes #7", "  Closes #7", "   Closes #7",
            "Closes other/repo#7", "Closes https://github.com/other/repo/issues/7",
            "Closes http://github.com/owner/repo/issues/7", "Closes https://github.com/owner/repo/pull/7",
            "Closes https://github.com/owner/repo/issues/7?x=1", "[Closes #7](https://example.com)",
        ]
        for body in examples:
            with self.subTest(body=body):
                self.assertEqual(policy.issue_references(body, REPOSITORY), [])

    def test_comments_and_code_do_not_count(self):
        examples = [
            "<!-- Closes #7 -->", "<!--\nCloses #7\n-->", "<!--\nCloses #7",
            "```text\nCloses #7\n```", "~~~\nCloses #7\n~~~", "   ```\nCloses #7\n   ```",
            "````\n```\nCloses #7\n```\n````", "Closes <!-- example -->#7",
        ]
        for body in examples:
            with self.subTest(body=body):
                self.assertEqual(policy.issue_references(body, REPOSITORY), [])
        self.assertEqual(policy.issue_references("```\nCloses #7\n```\nCloses #8", REPOSITORY), [8])

    def test_limit_bounds_api_reads(self):
        body = "\n".join(f"Closes #{number}" for number in range(1, 22))
        with self.assertRaises(policy.PolicyError):
            policy.issue_references(body, REPOSITORY)

    def test_body_size_is_bounded(self):
        with self.assertRaisesRegex(policy.PolicyError, "size limit"):
            policy.plain_lines("x" * (128 * 1024 + 1))

    def test_ai_records_and_no_agent_statement_pass(self):
        second_agent = "\n\nHarness: Hermes Agent\nModel: example-model-v1\nWork: Diagnose a synthetic failure.\n"
        for body in (NO_AGENT, AGENT_USE, AGENT_USE + second_agent):
            with self.subTest(body=body):
                self.assertIsNone(policy.body_requirement(body))
        concrete_work = AGENT_USE.replace("Write policy tests.", "Replace [x] parser logic.")
        self.assertIsNone(policy.body_requirement(concrete_work))

    def test_agent_record_count_is_bounded(self):
        records = "Harness: Codex\nModel: gpt-6.1-sol\nWork: Write policy tests.\n"
        body = "## AI agent use\n" + records * 21 + TEST_CONFIRMATION
        self.assertIn("at most 20", policy.body_requirement(body))

    def test_ai_section_is_required_and_has_a_boundary(self):
        examples = [
            "No AI agent used\n" + TEST_CONFIRMATION,
            "## AI agent use\n## Tests\nNo AI agent used\n" + TEST_CONFIRMATION,
            NO_AGENT + "\n## AI agent use\nNo AI agent used",
            "## AI Agent Use\nNo AI agent used\n" + TEST_CONFIRMATION,
        ]
        for body in examples:
            with self.subTest(body=body):
                self.assertIsNotNone(policy.body_requirement(body))

    def test_no_agent_statement_is_exact_and_cannot_mix_with_records(self):
        for statement in ("no AI agent used", "No AI agent used.", "- No AI agent used", "> No AI agent used"):
            with self.subTest(statement=statement):
                self.assertIsNotNone(policy.body_requirement(NO_AGENT.replace("No AI agent used", statement)))
        self.assertIsNotNone(policy.body_requirement(AGENT_USE + "\n\nNo AI agent used"))

    def test_each_record_requires_all_fields_and_real_values(self):
        for field, value in (("Harness", "Codex"), ("Model", "gpt-6.1-sol"), ("Work", "Write policy tests.")):
            for replacement in ("", "...", "TODO", "<value>", "[value]", "unknown", "enter value"):
                with self.subTest(field=field, replacement=replacement):
                    body = AGENT_USE.replace(f"{field}: {value}", f"{field}: {replacement}")
                    self.assertIsNotNone(policy.body_requirement(body))
            with self.subTest(missing=field):
                self.assertIsNotNone(policy.body_requirement(AGENT_USE.replace(f"{field}: {value}\n", "")))
        self.assertIsNotNone(policy.body_requirement(AGENT_USE + "\n\nHarness: Hermes Agent"))

    def test_disclosure_in_comments_or_code_does_not_count(self):
        section = "## AI agent use\nNo AI agent used"
        for hidden in (
            "<!--\n" + section + "\n-->", "```text\n" + section + "\n```",
            "~~~\n" + section + "\n~~~", "    ## AI agent use\n    No AI agent used",
            " \t## AI agent use\n \tNo AI agent used",
        ):
            with self.subTest(hidden=hidden):
                self.assertIsNotNone(policy.body_requirement(hidden + "\n" + TEST_CONFIRMATION))
        hidden_model = AGENT_USE.replace("Model: gpt-6.1-sol", "<!-- Model: gpt-6.1-sol -->")
        self.assertIsNotNone(policy.body_requirement(hidden_model))

    def test_comments_cannot_turn_code_text_into_assertions(self):
        for end in ("-->", "--!>"):
            with self.subTest(end=end):
                body = f"```text\n<!-- example {end}```\n" + NO_AGENT + "\n```"
                self.assertIsNotNone(policy.body_requirement(body))
                body = f"```text\n<!-- example {end}```\nCloses #7\n```"
                self.assertEqual(policy.issue_references(body, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(f"```text <!-- example {end}\n" + NO_AGENT + "\n```"))
                self.assertIsNone(policy.body_requirement(f"<!--\n```\n{end}\n" + NO_AGENT))

    def test_both_html_comment_end_tags_restore_public_lines(self):
        for end in ("-->", "--!>"):
            with self.subTest(end=end):
                body = f"Closes #6\n<!--\nCloses #7\n{end}\nCloses #8"
                self.assertEqual(policy.issue_references(body, REPOSITORY), [6, 8])
                hidden = "<!--\n" + NO_AGENT + f"\n{end}"
                self.assertIsNotNone(policy.body_requirement(hidden))
                self.assertIsNone(policy.body_requirement(NO_AGENT + "\n\n" + hidden))
                self.assertIsNone(policy.body_requirement(hidden + "\n\n" + NO_AGENT))
                hidden_ai = f"<!--\n## AI agent use\nNo AI agent used\n{end}\n" + TEST_CONFIRMATION
                self.assertIsNotNone(policy.body_requirement(hidden_ai))
                hidden_test = f"## AI agent use\nNo AI agent used\n<!--\n{TEST_CONFIRMATION}\n{end}"
                self.assertIsNotNone(policy.body_requirement(hidden_test))
                self.assertEqual(policy.issue_references(f"<!-- example {end}Closes #7", REPOSITORY), [])

    def test_multiline_inline_code_does_not_count(self):
        for marker in ("`", "``"):
            with self.subTest(marker=marker):
                self.assertEqual(policy.issue_references(marker + "\nCloses #7\n" + marker, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(marker + "\n" + NO_AGENT + "\n" + marker))
                visible = marker + "\nexample\n" + marker + "\n\n" + NO_AGENT
                self.assertIsNone(policy.body_requirement(visible))
        self.assertEqual(policy.issue_references("`\n` text `\nCloses #7\n`", REPOSITORY), [])

    def test_html_code_blocks_do_not_count(self):
        for tag in ("pre", "code", "script", "style", "textarea", "PRE"):
            with self.subTest(tag=tag):
                body = f'<{tag} class="example">\nCloses #7\n{NO_AGENT}\n</{tag}>'
                self.assertEqual(policy.issue_references(body, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(body))
                self.assertIsNone(policy.body_requirement(body + "\n\n" + NO_AGENT))
        self.assertIsNotNone(policy.body_requirement("<pre>\n</pre><pre>\n" + NO_AGENT + "\n</pre>"))
        self.assertIsNotNone(policy.body_requirement("<pre\n>\n" + NO_AGENT + "\n</pre>"))

    def test_nested_html_code_stays_hidden_until_the_outer_close(self):
        for outer, inner in (("code", "code"), ("code", "pre"), ("pre", "code")):
            with self.subTest(outer=outer, inner=inner):
                hidden = f"<{outer}><{inner}>\nexample\n</{inner}>\nCloses #7\n{NO_AGENT}\n</{outer}>"
                self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(hidden))
                public = hidden + "\n\nCloses #8\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                self.assertIsNone(policy.body_requirement(public))

    def test_comment_tags_cannot_close_or_hide_an_html_code_block(self):
        for end in ("-->", "--!>"):
            for opening in (f"<code>\n<!--\n</code>\n{end}", f"<code><!-- </code> {end}"):
                with self.subTest(end=end, opening=opening):
                    hidden = opening + "\nCloses #7\n" + NO_AGENT + "\n</code>"
                    self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])
                    self.assertIsNotNone(policy.body_requirement(hidden))
                    public = hidden + "\n\nCloses #8\n\n" + NO_AGENT
                    self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                    self.assertIsNone(policy.body_requirement(public))

    def test_quoted_html_attributes_cannot_end_code_or_start_comments(self):
        examples = (
            '<code><span title="</code>">', "<code><span title='</code>'>",
            '<code><span\ntitle="\n</code>\n">', "<code><span\ntitle='\n</code>\n'>",
            '<code><span title="</code><!--">',
        )
        for opening in examples:
            with self.subTest(opening=opening):
                hidden = opening + "\nCloses #7\n" + NO_AGENT + "\n</span></code>"
                self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(hidden))
                public = hidden + "\n\nCloses #8\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                self.assertIsNone(policy.body_requirement(public))

    def test_html_quote_and_list_containers_do_not_count(self):
        for tag in ("blockquote", "ul", "ol", "li", "dl"):
            with self.subTest(tag=tag):
                hidden = f"<{tag}>\nCloses #7\n{NO_AGENT}\n</{tag}>"
                self.assertEqual(policy.issue_references(hidden, REPOSITORY), [])
                self.assertIsNotNone(policy.body_requirement(hidden))
                public = hidden + "\n\nCloses #8\n\n" + NO_AGENT
                self.assertEqual(policy.issue_references(public, REPOSITORY), [8])
                self.assertIsNone(policy.body_requirement(public))

    def test_list_and_quote_continuations_do_not_count(self):
        for prefix in ("- Example:", "1. Example:", "> Example:", "  - Example:", "  > Example:"):
            for indent in ("", "  "):
                with self.subTest(prefix=prefix, indent=indent):
                    self.assertEqual(policy.issue_references(prefix + "\n" + indent + "Closes #7", REPOSITORY), [])
        self.assertEqual(policy.issue_references("- Example:\n\nCloses #7", REPOSITORY), [7])
        self.assertIsNotNone(policy.body_requirement("- Example:\n  " + NO_AGENT.replace("\n", "\n  ")))

    def test_test_confirmation_must_be_visible_exact_and_checked(self):
        examples = [
            "", TEST_CONFIRMATION.replace("[x]", "[ ]"), TEST_CONFIRMATION.replace("all passed.", "all passed"),
            TEST_CONFIRMATION + " except one.", "<!-- " + TEST_CONFIRMATION + " -->",
            "```\n" + TEST_CONFIRMATION + "\n```", "~~~\n" + TEST_CONFIRMATION + "\n~~~",
            "    " + TEST_CONFIRMATION, " \t" + TEST_CONFIRMATION, "> " + TEST_CONFIRMATION,
        ]
        for line in examples:
            with self.subTest(line=line):
                reason = policy.body_requirement(NO_AGENT.replace(TEST_CONFIRMATION, line))
                self.assertIn("checked line", reason)
        self.assertIsNone(policy.body_requirement(NO_AGENT.replace("[x]", "[X]")))


class DecisionTests(unittest.TestCase):
    def check(self, api):
        return policy.check_policy(api, REPOSITORY, 23, ACTIVATED_AT)

    def test_older_issue_passes_and_pr_is_fetched_again(self):
        api = FakeApi(pull(), issue(), pull())
        passed, message = self.check(api)
        self.assertTrue(passed)
        self.assertIn("#7", message)
        self.assertEqual(api.paths, [PR_PATH, "/repos/owner/repo/issues/7", PR_PATH])

    def test_closed_issue_and_different_author_pass(self):
        api = FakeApi(pull(), issue(state="closed"), pull())
        self.assertTrue(self.check(api)[0])

    def test_equal_or_later_issue_fails(self):
        for created_at in ("2026-10-08T18:00:00Z", "2026-10-08T18:00:01Z"):
            with self.subTest(created_at=created_at):
                api = FakeApi(pull(), issue(created_at=created_at), pull())
                self.assertFalse(self.check(api)[0])

    def test_pr_is_not_an_issue(self):
        api = FakeApi(pull(), issue(pull_request={}, html_url="https://github.com/owner/repo/pull/7"), pull())
        self.assertFalse(self.check(api)[0])

    def test_missing_issue_is_confirmed_invalid(self):
        api = FakeApi(pull(), policy.ApiError(404), pull())
        self.assertFalse(self.check(api)[0])

    def test_missing_and_external_references_fail_without_issue_reads(self):
        for body in (None, "", "Closes https://github.com/another/repo/issues/7", "<!-- Closes #7 -->"):
            with self.subTest(body=body):
                data = pull(body=(body or "") + "\n\n" + NO_AGENT)
                api = FakeApi(data, data)
                self.assertFalse(self.check(api)[0])
                self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_any_one_valid_issue_passes(self):
        data = pull(body="Closes #7\nFixes #8\n\n" + NO_AGENT)
        api = FakeApi(data, policy.ApiError(404), issue(8), data)
        self.assertTrue(self.check(api)[0])

    def test_api_errors_abort_the_decision(self):
        cases = [
            (policy.ApiError(403),),
            (pull(), policy.ApiError(403)),
            (pull(), policy.ApiError(429)),
            (pull(), policy.ApiError(500)),
            (pull(), policy.ApiError()),
            (pull(), issue(), policy.ApiError(500)),
        ]
        for replies in cases:
            with self.subTest(replies=len(replies)):
                with self.assertRaises(policy.ApiError):
                    self.check(FakeApi(*replies))

    def test_bad_issue_data_aborts_the_decision(self):
        cases = [issue(number=8), issue(created_at=None), issue(html_url="https://github.com/other/repo/issues/7")]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(policy.PolicyError):
                    self.check(FakeApi(pull(), data))

    def test_changed_pr_aborts_both_pass_and_fail(self):
        changes = [
            {"body": "Closes #8"}, {"state": "closed"}, {"merged": True},
            {"head": {"sha": "new-head"}}, {"base": {"ref": "release", "repo": {"full_name": REPOSITORY}}},
            {"user": {"login": "Elevatormusic"}},
        ]
        for change in changes:
            for valid in (False, True):
                with self.subTest(change=change, valid=valid):
                    first = pull() if valid else pull(body="")
                    last = copy.deepcopy(first)
                    last.update(change)
                    api = FakeApi(first, issue(), last) if valid else FakeApi(first, last)
                    with self.assertRaisesRegex(policy.PolicyError, "changed during"):
                        self.check(api)

    def test_mergeability_calculation_does_not_change_the_policy(self):
        api = FakeApi(pull(mergeable=None), issue(), pull(mergeable=True))
        self.assertTrue(self.check(api)[0])

    def test_closed_or_merged_pr_is_skipped(self):
        for change in ({"state": "closed"}, {"merged": True}):
            api = FakeApi(pull(**change))
            self.assertTrue(self.check(api)[0])
            self.assertEqual(api.paths, [PR_PATH])

    def test_old_pr_is_skipped_even_with_current_edits(self):
        data = pull(created_at="2026-10-08T17:09:41Z", body=None, updated_at="2026-10-09T00:00:00Z")
        api = FakeApi(data)
        self.assertTrue(self.check(api)[0])
        self.assertEqual(api.paths, [PR_PATH])

    def test_owner_is_exempt_only_from_the_issue_requirement(self):
        for login in ("Elevatormusic", "ELEVATORMUSIC", "eLeVaToRmUsIc"):
            with self.subTest(login=login):
                data = pull(user={"login": login}, body=NO_AGENT)
                api = FakeApi(data, data)
                self.assertTrue(self.check(api)[0])
                self.assertEqual(api.paths, [PR_PATH, PR_PATH])
        for body in ("", "## AI agent use\nNo AI agent used", TEST_CONFIRMATION):
            with self.subTest(body=body):
                data = pull(user={"login": "Elevatormusic"}, body=body)
                self.assertFalse(self.check(FakeApi(data, data))[0])

    def test_owner_association_and_fork_owner_cannot_exempt_a_contributor(self):
        data = pull(
            body=NO_AGENT, author_association="OWNER",
            head={"repo": {"full_name": "Elevatormusic/fork", "owner": {"login": "Elevatormusic"}}},
        )
        api = FakeApi(data, data)
        self.assertFalse(self.check(api)[0])
        self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_body_requirements_fail_before_issue_reads(self):
        for body in ("Closes #7", "Closes #7\n## AI agent use\nNo AI agent used"):
            with self.subTest(body=body):
                data = pull(body=body)
                api = FakeApi(data, data)
                passed, reason = self.check(api)
                self.assertFalse(passed)
                self.assertTrue(reason)
                self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_agent_disclosure_does_not_replace_a_contributor_issue(self):
        data = pull(body=AGENT_USE)
        api = FakeApi(data, data)
        self.assertFalse(self.check(api)[0])
        self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_author_race_aborts_owner_exemption(self):
        data = pull(user={"login": "Elevatormusic"}, body=NO_AGENT)
        changed = pull(user={"login": "contributor"}, body=NO_AGENT)
        with self.assertRaisesRegex(policy.PolicyError, "changed during"):
            self.check(FakeApi(data, changed))

    def test_pr_at_activation_is_checked(self):
        data = pull(created_at=ACTIVATED_AT, body="")
        api = FakeApi(data, data)
        self.assertFalse(self.check(api)[0])

    def test_missing_or_invalid_activation_fails_before_api_reads(self):
        for activated_at in (None, "", "invalid", "2026-10-08T17:09:42", "2026-02-30T17:09:42Z"):
            with self.subTest(activated_at=activated_at):
                api = FakeApi()
                with self.assertRaisesRegex(policy.PolicyError, "timestamp"):
                    policy.check_policy(api, REPOSITORY, 23, activated_at)
                self.assertEqual(api.paths, [])

    def test_malformed_pr_and_timestamp_fail(self):
        for data in (
            pull(number=24), pull(merged="false"), pull(created_at="invalid"), pull(base={}),
            pull(user={}), pull(user={"login": ""}), pull(user={"login": None}),
        ):
            with self.subTest(data=data):
                with self.assertRaises(policy.PolicyError):
                    self.check(FakeApi(data))


class EntryTests(unittest.TestCase):
    def run_main(self, event_name, event, api):
        with tempfile.TemporaryDirectory() as folder:
            event_path = Path(folder) / "event.json"
            event_path.write_text(json.dumps(event), encoding="utf-8")
            env = {
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_EVENT_PATH": str(event_path),
                "GITHUB_EVENT_NAME": event_name,
                "GITHUB_TOKEN": "fake-test-token",
                "ISSUE_POLICY_ACTIVATED_AT": ACTIVATED_AT,
            }
            output = io.StringIO()
            with patch.dict(os.environ, env, clear=True), patch.object(policy, "GitHubApi", return_value=api):
                with redirect_stdout(output):
                    result = policy.main()
            self.assertNotIn("fake-test-token", output.getvalue())
            return result, output.getvalue()

    def test_old_pr_remains_exempt_for_every_event(self):
        for action in ("opened", "reopened", "edited", "synchronize", "ready_for_review"):
            with self.subTest(action=action):
                api = FakeApi(pull(created_at="2026-10-08T17:09:41Z", body=""))
                result, _ = self.run_main("pull_request_target", {"action": action, "number": 23}, api)
                self.assertEqual(result, 0)
                self.assertEqual(api.paths, [PR_PATH])

    def test_dispatch_reads_fresh_pr_and_never_writes(self):
        data = pull(body="")
        api = FakeApi(data, data)
        result, output = self.run_main("workflow_dispatch", {"inputs": {"pr_number": "23"}}, api)
        self.assertEqual(result, 1)
        self.assertTrue(output.startswith("::error::"))
        self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_event_body_is_never_used_as_current_data(self):
        api = FakeApi(pull(), issue(), pull())
        result, _ = self.run_main(
            "pull_request_target", {"action": "opened", "number": 23, "pull_request": {"body": "bad"}}, api
        )
        self.assertEqual(result, 0)

    def test_event_actor_and_event_author_cannot_exempt_a_contributor(self):
        data = pull(body=NO_AGENT)
        api = FakeApi(data, data)
        result, _ = self.run_main(
            "pull_request_target",
            {"action": "opened", "number": 23, "sender": {"login": "Elevatormusic"},
             "pull_request": {"user": {"login": "Elevatormusic"}, "body": "Closes #7"}},
            api,
        )
        self.assertEqual(result, 1)
        self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_fresh_api_owner_controls_exemption(self):
        data = pull(user={"login": "Elevatormusic"}, body=AGENT_USE)
        api = FakeApi(data, data)
        result, _ = self.run_main(
            "pull_request_target",
            {"action": "opened", "number": 23, "sender": {"login": "contributor"},
             "pull_request": {"user": {"login": "contributor"}}},
            api,
        )
        self.assertEqual(result, 0)
        self.assertEqual(api.paths, [PR_PATH, PR_PATH])

    def test_bad_dispatch_input_and_unsupported_events_fail_before_api(self):
        cases = [
            ("workflow_dispatch", {"inputs": {"pr_number": "23; echo unsafe"}}),
            ("workflow_dispatch", {"inputs": {"pr_number": "-1"}}),
            ("workflow_dispatch", {"inputs": {"pr_number": True}}),
            ("pull_request", {"action": "opened", "number": 23}),
            ("pull_request_target", {"action": "closed", "number": 23}),
        ]
        for event_name, event in cases:
            with self.subTest(event_name=event_name, event=event):
                api = FakeApi()
                self.assertEqual(self.run_main(event_name, event, api)[0], 1)
                self.assertEqual(api.paths, [])


class TransportTests(unittest.TestCase):
    def reply(self, raw):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self, limit):
                return raw[:limit]

        return Response()

    def test_request_is_a_get_to_the_fixed_host(self):
        api = policy.GitHubApi("fake-test-token")
        with patch.object(api.opener, "open", return_value=self.reply(b'{"number": 7}')) as opened:
            self.assertEqual(api.get("/repos/owner/repo/issues/7"), {"number": 7})
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.github.com/repos/owner/repo/issues/7")
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        self.assertEqual(opened.call_args.kwargs, {"timeout": 30})

    def test_http_and_network_errors_are_safe_and_do_not_log_response_data(self):
        errors = [
            HTTPError("https://api.github.com", 403, "private-body", {}, None),
            URLError("private-url-or-token"), TimeoutError("private-url-or-token"),
        ]
        for error in errors:
            with self.subTest(kind=type(error).__name__):
                api = policy.GitHubApi("fake-test-token")
                with patch.object(api.opener, "open", side_effect=error):
                    with self.assertRaises(policy.ApiError) as raised:
                        api.get(PR_PATH)
                self.assertNotIn("private", str(raised.exception))
                self.assertNotIn("fake-test-token", str(raised.exception))

    def test_invalid_json_and_large_response_fail(self):
        for raw in (b"not-json", b"[]", b"x" * (1024 * 1024 + 1)):
            with self.subTest(size=len(raw)):
                api = policy.GitHubApi("fake-test-token")
                with patch.object(api.opener, "open", return_value=self.reply(raw)):
                    with self.assertRaises(policy.ApiError):
                        api.get(PR_PATH)

    def test_list_request_keeps_the_fixed_host_and_does_not_weaken_dict_checks(self):
        api = policy.GitHubApi("fake-test-token")
        path = "/repos/owner/repo/issues?state=open&creator=contributor&per_page=100&page=1"
        with patch.object(api.opener, "open", return_value=self.reply(b'[{"number": 23}]')) as opened:
            self.assertEqual(api.get_list(path), [{"number": 23}])
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.github.com" + path)
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        self.assertEqual(opened.call_args.kwargs, {"timeout": 30})
        with patch.object(api.opener, "open", return_value=self.reply(b'[{"number": 23}]')):
            with self.assertRaises(policy.ApiError):
                api.get(PR_PATH)

    def test_list_invalid_json_type_items_size_and_api_errors_fail(self):
        for raw in (b"not-json", b"{}", b"null", b"[null]", b"[true]", b'["item"]',
                    json.dumps([{}] * 101).encode(), b"x" * (1024 * 1024 + 1)):
            with self.subTest(size=len(raw)):
                api = policy.GitHubApi("fake-test-token")
                with patch.object(api.opener, "open", return_value=self.reply(raw)):
                    with self.assertRaises(policy.ApiError):
                        api.get_list("/repos/owner/repo/issues")
        for error in (HTTPError("https://api.github.com", 500, "private-body", {}, None),
                      URLError("private-url-or-token"), TimeoutError("private-url-or-token")):
            api = policy.GitHubApi("fake-test-token")
            with patch.object(api.opener, "open", side_effect=error):
                with self.assertRaises(policy.ApiError) as raised:
                    api.get_list("/repos/owner/repo/issues")
            self.assertNotIn("private", str(raised.exception))
            self.assertNotIn("fake-test-token", str(raised.exception))

    def test_invalid_api_paths_fail_before_a_request(self):
        for path in ("@example.com", "//example.com/path", "https://example.com", "/repos/owner/repo\r\nunsafe"):
            for method in ("get", "get_list"):
                with self.subTest(path=path, method=method):
                    api = policy.GitHubApi("fake-test-token")
                    with patch.object(api.opener, "open") as opened:
                        with self.assertRaises(policy.ApiError):
                            getattr(api, method)(path)
                        opened.assert_not_called()

    def test_redirects_are_not_followed(self):
        self.assertIsNone(policy.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com"))


if __name__ == "__main__":
    unittest.main()
