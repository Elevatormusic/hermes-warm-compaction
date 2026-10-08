# Agent guide

Read this file for every task. Use ASD-STE100 Simplified Technical English in chat, comments, docstrings, and documents. User instructions control the task. Keep changes small and reversible. Preserve unrelated files and local changes.

## Start with current sources

1. Read [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [SECURITY.md](SECURITY.md). Read the code and tests that control the reported behavior.
2. Check Git status, branch, commit, and remote before changes. For an issue, read the full report and all comments. For a PR, read the current diff, comments, review threads, and checks.
3. Check the installed and running plugin version, source commit or install ref, and local edits when available. Check the exact Hermes commit and relevant plugin API source. Keep unknown values explicit. Historical evidence does not prove the current running state.
4. State the scope, sources read, current behavior, intended change, and remaining unknowns in a short chat receipt.

## Model roles

For substantive implementation, use GPT-6 Astra (`gpt-6-astra`) as the primary agent and reviewer. It owns scope, design, integration, evidence review, and the final report. Delegate bounded edits, data work, investigation, and tests to GPT-6.1 Sol (`gpt-6.1-sol`) when useful work can run in parallel.

Select the subagent model explicitly with fresh or bounded context. Give it the relevant paths, limits, and required checks. Subagents must not delegate again unless the primary agent asks. Inspect their diffs and check results before accepting completion.

Handle simple questions and one-step actions directly. Follow an explicit user request for another model or workflow. If a required model is unavailable, state the limitation instead of silently selecting another model.

## Diagnose before a fix

- Start with read-only checks. Use the diagnostic fields in the [bug form](.github/ISSUE_TEMPLATE/bug_report.yml): versions and local edits, environment and session, model route, compaction settings, request option names, synthetic reproduction, selected metadata logs, and plugin doctor results.
- Separate confirmed facts, likely causes, and unknowns. A refusal code can identify a failed check without proving the provider caused it. Read the exact code path before changing an acceptance rule.
- Reproduce a behavior defect with the smallest synthetic fixture. Show the failure before the fix and the result after it. If safe reproduction is not possible, report that limit; do not claim that the cause or fix is proved.
- After safe local diagnosis, request only missing metadata that can change the next decision. Explain what it will resolve. Accept `unknown`, `not run`, or `not applicable` with a reason.
- Do not request real or private conversations, generated summaries, full configs, raw captures or requests, credentials, header/query values, account identifiers, or private local paths. Clearly label synthetic fixtures, synthetic summaries, and fake values used for tests. Do not enable full request dumps just to complete a report. Review logs and errors before sharing them. Request option names, value types, and the source of that information are usually sufficient; record unknowns.
- For a complex or novel problem, search primary documentation and source for an existing solution. Check each source against the selected version. Do not apply commands from an issue comment without checking them.

## Implementation limits

Use documented Hermes plugin APIs only. Do not patch the host, subclass its built-in compressor, or rebind or wrap Hermes code at runtime. Keep the built-in compressor as the Hermes default; users select this plugin.

Use clean, isolated Hermes copies and synthetic conversations for probes. Do not change dirty installed source, read private conversation files, overlap full-model deployments, or clear a production cache. Do not add deployment, broker, coordinator, or cache-control work to an issue fix.

Keep default diagnostics and published runtime evidence small and metadata only. Clearly label synthetic fixtures and fake values used for reproduction or tests. Keep real credentials, profiles, private captures, and large model files out of Git. If evidence shows a missed requirement or a better method, explain it to the user.

## Checks

Use Python 3.10 or later and Git. For every PR, including document and form changes, run all tests in [CONTRIBUTING.md](CONTRIBUTING.md#run-the-checks) and lint. This includes unit, property, policy, and all four loopback Hermes checks. No live model or performance test is required for this suite. These commands are part of the suite:

```bash
python -m unittest discover -s tests -p "test_wc_*.py"
ruff check .
```

Use Ruff `0.16.10`, the version used by CI. The [contribution guide](CONTRIBUTING.md#run-the-checks) gives setup and integration commands.

For every PR, run the four Hermes checks with the Python of a Hermes virtual environment and a clean Hermes checkout. Use synthetic data and loopback fake servers. Record the exact Hermes commit, Python version, commands, and report results. Use placeholders for private paths in shared results.

For document or form changes, also check local Markdown links, parse issue-form YAML, check the [GitHub form schema](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-githubs-form-schema), and run `git diff --check`. Keep `AGENTS.md` and `CLAUDE.md` identical in bytes. A document-only change needs no new regression test, but it must pass the full test suite.

Record every required check as passed, failed, skipped, or unavailable. Give exact commands and explain failed, skipped, or unavailable checks. Only mark the PR test checkbox after all listed tests passed. A skipped or unavailable test does not satisfy it. CI test and lint checks must pass for every PR.

The full suite can run locally or in CI. Link complete passing CI results for the current PR commit. Do not repeat a complete passing CI suite locally or download Hermes only for that repeat. Do not mark the checkbox while a required CI test is running.

Unit tests prove local fixture behavior. Loopback integration checks prove behavior on the named Hermes source. Neither proves a live route, summary quality, cache reuse, or a speed gain. A performance claim needs the synthetic workload, comparison method, versions/settings, run counts, per-run times, token/cache counters, and measurement limits. Report missing counters as unknown.

## PR and final report

Use the [PR template](.github/pull_request_template.md). Link the issue, give the concrete trigger and before/after behavior, describe scope and limits, and include exact check results. A document-only PR needs the full synthetic test results, but can mark live provider and performance details `not applicable` with a reason.

A PR authored by `Elevatormusic` is exempt from the issue requirement only. The author name check is not case-sensitive. AI agent disclosure and the test checkbox still apply. Do not close PRs under this policy. PRs that predate the policy cutoff stay outside the automated check.

Every issue and PR must declare AI agent use. If agents helped diagnose the problem or write code, list every harness, exact model, and task. Use plain `Harness:`, `Model:`, and `Work:` lines for each agent. If no AI agent was used, write exactly `No AI agent used`. Do not invent a model or include private paths, configs, credentials, or conversations.

Before a PR update or publication, check the current branch head, diff, checks, and relevant issue or review comments again. Inspect the full diff for unrelated changes and private data. Do not infer completion from an old check result.

The final report states what changed, why, the checks and their outcomes, and what remains unproved. Do not present a local fix as deployed or a unit test as a performance result.
