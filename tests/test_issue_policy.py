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
    def __init__(self, *responses):
        self.responses = list(responses)
        self.paths = []

    def get(self, path):
        self.paths.append(path)
        if not self.responses:
            raise AssertionError("An unexpected API read was requested.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


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
            {"head": {"sha": "new-head"}}, {"base": {"sha": "new-base"}},
            {"updated_at": "2026-10-08T18:01:00Z"},
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

    def test_redirects_are_not_followed(self):
        self.assertIsNone(policy.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com"))


if __name__ == "__main__":
    unittest.main()
