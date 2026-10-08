# Contributing

Thank you for your help. This guide tells you how to report a problem, propose a change, and run the checks.

## Report a problem

Use the [Bug report form](.github/ISSUE_TEMPLATE/bug_report.yml) for a failure. Use the [Feature request form](.github/ISSUE_TEMPLATE/feature_request.yml) for a proposed capability. For a security problem, use the private route in [SECURITY.md](SECURITY.md).

Complete each bug report section. Give exact versions when you have them. Write `unknown`, `not run`, or `not applicable` with a short reason when you cannot give a value. Do not guess. These fields reduce follow-up requests, but a new or version-specific problem can still need more checks.

Every issue and PR must declare AI agent use. If an AI agent helped diagnose the problem or write code, list every harness, exact model, and task. A harness is the app or tool that ran the agent. Use plain `Harness:`, `Model:`, and `Work:` lines for each agent. Start these lines at the left margin, with no leading spaces. If no AI agent was used, write exactly `No AI agent used`. Do not guess a model or use a default disclosure. Keep private paths, configs, credentials, and conversations out of this statement.

Include:

- The installed plugin version, source commit or install ref, install method, and local edits. Give the running Hermes version and commit, and any local Hermes edits.
- OS version, CPU architecture, Python version used by Hermes, and run mode (interactive CLI, one-shot CLI, Desktop, gateway, container, or WSL). State whether the session is new or resumed, whether a main-model reply completed in this process before compaction, and whether Hermes restarted after the plugin or settings changed.
- Main provider, exact model ID, API mode, server name and version when known, and whether a proxy or gateway is used. Include a hostname only if needed to explain routing. Do not give full endpoint URLs.
- Compaction trigger and frequency, selected context engine, plugin settings, context window, reply limit, and the auxiliary `warm_compaction` route. State any recent route or settings change and other context engines, plugins, or middleware that can affect the request.
- Names of extra headers, query options, and `extra_body` options, their value types when known, and whether tool, structured-response, audio, or multiple-choice request options are set. State where this information came from, or why it is unknown. Give no real header, query, or `extra_body` values. State whether TLS checks are enabled, whether a custom CA is used, and whether a gateway needs cookies. Give no certificate or cookie data.
- Minimal steps with a synthetic conversation, expected behavior, actual behavior, and how often the failure occurs. State whether the built-in compressor has the same problem, if already checked. A baseline check is optional.
- Redacted diagnostic output, as described below. For a performance claim, also give the comparison method, per-run times, token counts, cache counters, versions, and limits. A missing cache counter is `unknown`, not zero.

### Safe reproduction and diagnostics

Use invented messages that reproduce the failure. State the approximate message count and token count. If no safe reproduction is available, explain that limit. Do not attach real or private histories, generated summaries, full configurations, raw requests, credentials, header/query values, account identifiers, or private local paths. Clearly label synthetic fixtures, synthetic summaries, and fake credential or option values used for tests. Do not enable full request dumps just to complete a report.

Copy the relevant `warm_compaction:` lines from `logs/agent.log`, and any adjacent plugin warnings about capture, fallback, or fixed summaries. Keep path, reason, elapsed time, token counts, and error class when available. Report `no log line` with a reason when the plugin never loaded or no compaction ran. Review the selected lines before you paste them; do not attach the whole log.

Run this command in the same terminal environment where Hermes runs, then provide the redacted check results:

```bash
hermes plugins doctor warm_compaction
```

If the installed plugin name cannot be resolved, use `hermes plugins doctor <plugin-folder>` locally. Replace private paths in the shared command and output with `<plugin-folder>`. If the command is unavailable or you did not run it, give that status and reason. Doctor checks discovery and registration; it does not prove that a running session loaded the plugin, that a summary is correct, or that a server reused its cache.

Use a clean, isolated Hermes copy for further probes. Keep installed source and production sessions intact. A report does not require a production cache reset, a model deployment change, or a live benchmark.

## Propose a change

1. Open an issue before you open a pull request. If an issue already covers the same change, use that issue. This rule also applies to small fixes, documents, draft PRs, and dependency updates. A PR authored by `Elevatormusic` is exempt from the issue requirement only; the name check is not case-sensitive. For a large change, agree on the design in the issue before implementation. Report security problems through the private route in [SECURITY.md](SECURITY.md).
2. Fork the repository and make a branch from `main`.
3. For a behavior fix, add a test that fails without your change. Then make the change. A document-only change does not need a new regression test, but it must pass the full test suite.
4. Run all tests listed below and lint for every PR, including document and form changes. Run them locally, or let CI run after you open the PR. Record the exact commands, versions, and results, or link the complete passing CI results. Do not repeat the same suite locally if CI already ran it in full. All tests must pass before you mark the required test checkbox. Record a failed, skipped, or unavailable check with its reason; it does not satisfy this requirement.
5. Open a pull request. Complete the [PR template](.github/pull_request_template.md): link the earlier issue, declare AI agent use, state the concrete trigger and before/after behavior, explain the scope, and give check results and limits. Mark `I ran all tests listed in CONTRIBUTING.md and all passed.` only after you ran the full suite and it passed.

### Open pull request limit

Each author can have at most five open pull requests in this repository. The count includes the PR being checked, draft PRs, and open PRs on every base branch. Open PRs that predate policy activation also count. Closed or merged PRs and ordinary issues do not count. The limit applies to every author, including `Elevatormusic`.

The policy check fails a new PR when its author has more than five open PRs. Wait until another PR is merged or close a PR you no longer need. Then ask a maintainer to rerun the failed policy job, or edit the remaining PR description to trigger a new check. GitHub can still create the sixth PR; the required check prevents it from passing the merge policy. The workflow does not close PRs.

PRs opened before policy activation keep their existing exemption from the automated check. They still use open PR slots when a later PR is checked. An API failure or an incomplete PR count fails the check. The check reads the open PR set again before it reports a pass.

### Required issue link

Unless the PR author is `Elevatormusic`, put one of these lines in the PR description. Replace `123` with the issue number:

```text
Closes #123
Fixes #123
Resolves #123
```

Use one complete line at the left margin, with no leading spaces. Keep it outside a comment, code block, raw HTML block, math block, list, or quote. Put a blank line after any preceding list or quote. These format rules also apply to the AI declaration. Put the test checkbox in its own task list, with a blank line before it. You can also use the full issue URL, for example `Closes https://github.com/Elevatormusic/hermes-warm-compaction/issues/123`.

The linked item must be an issue in this repository, and it must have been opened before the PR. An issue from another repository or a link to another PR does not count. An existing issue can have a different author or be closed.

GitHub cannot enforce this rule before it creates a PR. The [issue policy workflow](.github/workflows/require-issue.yml) checks each new PR and fails if it has no valid earlier issue, except for PRs authored by `Elevatormusic`. It checks draft PRs too. The author exception applies only to the issue link: the five-open-PR limit, AI agent disclosure, and the test checkbox still apply. If the link is missing, add a valid earlier issue link. If you opened the issue after the PR, open a new PR for that issue. API failures fail the check. The workflow preserves PRs that predate the policy cutoff and does not close any PR.

The workflow reads PR and issue metadata. It checks out the trusted workflow commit and does not run code from the PR. Manual dispatch checks one PR without changing it.

To activate the policy, a maintainer sets the repository variable `ISSUE_POLICY_ACTIVATED_AT` once to the actual activation time in UTC, in `YYYY-MM-DDTHH:MM:SSZ` format. Do not advance this value for later policy edits. A missing or invalid value fails the check. Then require the `issue-first` check in the `main` branch protection settings and allow `pull_request_target` for this workflow in the Actions event policy.

Run its local tests with:

```bash
python -B -m unittest discover -s tests -p "test_issue_policy.py" -v
```

Keep each pull request to one subject. The plugin uses documented Hermes plugin APIs only: no patch of Hermes, no subclass of the built-in compressor, and no runtime wrapping of Hermes code. A change that needs one of those cannot be merged.

For a bug fix, link a complete bug report or include its applicable diagnostic fields in the PR. For a performance change, include the comparison method and small metadata evidence. Clearly label synthetic fixtures and redact runtime evidence before publication. A document-only PR needs no model, server, or private runtime data; write `not applicable` with a reason for those sections.

Agents must read [AGENTS.md](AGENTS.md) before work. [CLAUDE.md](CLAUDE.md) has the same instructions.

## Run the checks

You need Python 3.10 or later and Git. The plugin has no third-party dependency.

Every PR must pass the full suite: unit tests, property tests, issue policy tests, and all four loopback Hermes checks below. Run lint too. This rule includes document and form changes. These checks use synthetic data. They do not prove deployed compatibility, summary quality, or cache reuse. No live model or performance test is required for this suite.

The full suite can run in CI. Link all passing test jobs for the current PR commit, including the Hermes checks, before marking the test checkbox. A running, failed, skipped, or unavailable job is not a passing result. The local commands below provide the same checks when you run them outside CI. You do not need to download a Hermes runtime only to repeat a complete passing CI suite.

```bash
python -m unittest discover -s tests -p "test_wc_*.py"
```

Install the test tools in a separate test environment, then run the property tests and branch coverage:

```bash
python -m pip install -r requirements-test.txt
python -m coverage run -m unittest discover -s tests -p "test_wc_*.py"
python -m coverage run --append -m unittest discover -s tests/property -p "test_*.py"
python -B -m unittest discover -s tests -p "test_issue_policy.py" -v
python -m coverage report -m
```

The [property tests](tests/property/test_native_properties.py) use Hypothesis to generate synthetic native records and request options. The [coverage settings](.coveragerc) measure branches in the plugin. Coverage reports show which paths need more checks; there is no minimum percentage gate. These tools are test dependencies only. The plugin still has no third-party dependency.

Lint with [ruff](https://docs.astral.sh/ruff/) (the version that CI uses):

```bash
pip install ruff==0.16.10
ruff check .
```

For every PR, run the middleware and integration checks. They run real Hermes code against loopback fake servers with synthetic data. Run them with the Python of a Hermes virtual environment and a **clean** Hermes checkout, not your installed Hermes. Record the exact Hermes commit, Python version, commands, and report results. Use placeholders for private paths in shared results:

```bash
<hermes-venv-python> -B scripts/check_hermes_compatibility.py --hermes-source <clean-hermes-checkout> --report .work/hermes-compatibility-report.json
<hermes-venv-python> -B scripts/check_plugin_hermes.py --hermes-source <clean-hermes-checkout> --report .work/plugin-integration-report.json
```

The integration check covers install, request, compaction, and history behavior on that Hermes version. It does not call a real model or prove live cache reuse or a performance gain. Keep evidence small and metadata only.

For every PR, also run both native checks. The provider check uses real Hermes transports and synthetic loopback replies. The conversation check runs the real Hermes agent loop, manual and automatic compaction, native tools, and saved-session reload. Each case uses a fresh process and temporary home. The isolated interpreter needs the Hermes dependencies for both API formats. Use `--dependency-path <dependency-folder>` if those dependencies are in a separate local folder.

```bash
<hermes-venv-python> -B scripts/check_provider_apis.py --hermes-source <clean-hermes-checkout> --report .work/provider-api-report.json
<hermes-venv-python> -B scripts/check_native_conversations.py --hermes-source <clean-hermes-checkout> --report .work/native-conversations-report.json
```

The [Hermes compatibility workflow](.github/workflows/hermes-compatibility.yml) uses the minimum Hermes commit `45871e100feceb89769536c88e5e6e265226a409` and current upstream `main`. It prepares a separate runtime from each checkout's `uv.lock`, with the `anthropic` extra. It does not change the plugin's runtime dependencies. Native check failures fail the job, including upstream changes; they are not treated as unsupported-version skips. The report records the exact Hermes commit and source hashes. The jobs have read-only repository access, no stored checkout credentials, no secrets, and no Actions cache access. Only metadata reports are saved.

See [provider support](docs/provider-support.md) for the supported request and authentication limits. A passing local check does not qualify a live provider.

Run `git diff --check` for every PR. For document or form changes, also check local Markdown links. For issue forms, parse the YAML and check its fields against the [GitHub form schema](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-githubs-form-schema). These checks are in addition to the full test suite.

CI runs unit and property tests on Python 3.10 to 3.13 (Linux) and on Python 3.12 (Windows and macOS). It saves branch coverage reports for each job and runs lint. A pull request must pass the test and lint checks. The Hermes compatibility workflow also runs on pull requests, pushes to `main`, each day, and by manual dispatch.

## Style

- Match the code around your change: names, comment density, and line length (120).
- Write comments, docstrings, and documents in short, simple sentences.
- A measured result needs its record in `evidence/` with metadata only. Report a missing counter as unknown.

## License

By contributing, you agree that your contribution is licensed under the [Apache License 2.0](LICENSE). Keep the [NOTICE](NOTICE) file in copies and derivative works.
