## Problem and change

<!-- An issue is required for every new PR, including drafts, except for a PR authored by Elevatormusic (name is not case-sensitive). Only that author can omit the issue line. AI agent disclosure and the test checkbox are still required. Open the issue first, or use an existing issue for the same change. Replace the line below with Closes #123, Fixes #123, or Resolves #123. The issue must be in this repository and older than this PR. Keep the line outside comments and code blocks. A PR without a required valid earlier issue fails the policy check. Give the concrete trigger and before/after behavior. For a bug fix without a complete linked report, include applicable diagnostic fields from CONTRIBUTING.md. -->

Closes #

Trigger and before/after:

## Scope

<!-- State the files/components changed, why they are needed, and any limits or risks. Explain any README Limits or CHANGELOG update that is needed for a behavior change. Use documented Hermes plugin APIs only: no host patch, compressor subclass, or runtime wrapping. -->

## AI agent use

<!-- If an AI agent helped diagnose the problem or write code, list every harness, exact model, and task. A harness is the app or tool that ran the agent. Use plain Harness:, Model:, and Work: lines for each agent, outside comments and code blocks. Start each required line at the left margin, with no leading spaces. Put a blank line after any preceding list or quote. If no AI agent was used, write exactly No AI agent used. Do not guess a model. Do not include private paths, configs, credentials, or conversations. -->

## Check results

<!-- Every PR, including document and form changes and PRs authored by Elevatormusic, must pass all tests listed in CONTRIBUTING.md and lint. Run them locally or in CI. Give exact commands, versions, and results, or links to all passing CI test jobs for this commit. Do not mark the checkbox while any required test is running, failed, skipped, or unavailable. Do not repeat the full suite locally after it passed in CI. A document-only change needs no new regression test or live model/server details. Check links and form schema when applicable. No live model or performance test is required for the full suite. -->

| Check | Exact command | Outcome and evidence |
| --- | --- | --- |
| Regression test for a behavior fix (fails before, passes after) | | |
| Unit tests | | |
| Property tests | | |
| Issue policy tests | | |
| Lint | | |
| Hermes middleware compatibility | | |
| Hermes plugin integration | | |
| Native provider API checks | | |
| Native conversation checks | | |
| Document links / issue form YAML and schema, when applicable | | |
| `git diff --check` | `git diff --check` | |

- [ ] I ran all tests listed in CONTRIBUTING.md and all passed.

<!-- For integration, give the exact clean Hermes commit, Python version, and report result. Commands and check limits are in CONTRIBUTING.md. Use placeholders for private paths. -->

Integration versions and reports:

## Evidence and limits

<!-- Link small redacted metadata evidence and synthetic fixtures when needed. For a performance claim, give workload, comparison method, versions/settings, number of runs, per-run times, token/cache counts, and concurrent load when known. Missing counters are unknown. State what was not measured. For documents only, write not applicable with a reason. -->

Evidence:
Limits:

- [ ] I reviewed the diff and attachments. They contain no real or private conversation, summary, full config, raw request, credential, header/query value, account identifier, or private local path. Synthetic fixtures and fake values are clearly labeled.
