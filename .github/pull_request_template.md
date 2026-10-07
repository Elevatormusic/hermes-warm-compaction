## Problem and change

<!-- Link the issue, or explain why there is none. Give the concrete trigger and before/after behavior. For a bug fix without a complete linked report, include applicable diagnostic fields from CONTRIBUTING.md. -->

Issue:
Trigger and before/after:

## Scope

<!-- State the files/components changed, why they are needed, and any limits or risks. Explain any README Limits or CHANGELOG update that is needed for a behavior change. Use documented Hermes plugin APIs only: no host patch, compressor subclass, or runtime wrapping. -->

## Check results

<!-- Give the exact command and outcome for each applicable check: passed, failed, skipped, or unavailable. Explain every skipped or unavailable check. A document-only change needs link, form schema (when applicable), and diff checks; it needs no code regression test or model/server details. CI unit and lint checks must pass for every PR. -->

| Check | Exact command | Outcome and evidence |
| --- | --- | --- |
| Regression test for a behavior fix (fails before, passes after) | | |
| Unit tests | | |
| Lint | | |
| Hermes integration, for changes to Hermes interaction | | |
| Document links / issue form YAML and schema, when applicable | | |
| `git diff --check` | `git diff --check` | |

<!-- For integration, give the exact clean Hermes commit, Python version, and report result. Commands and check limits are in CONTRIBUTING.md. Use placeholders for private paths. -->

Integration versions and report (or not applicable with reason):

## Evidence and limits

<!-- Link small redacted metadata evidence and synthetic fixtures when needed. For a performance claim, give workload, comparison method, versions/settings, number of runs, per-run times, token/cache counts, and concurrent load when known. Missing counters are unknown. State what was not measured. For documents only, write not applicable with a reason. -->

Evidence:
Limits:

- [ ] I reviewed the diff and attachments. They contain no real or private conversation, summary, full config, raw request, credential, header/query value, account identifier, or private local path. Synthetic fixtures and fake values are clearly labeled.
