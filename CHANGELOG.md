# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Add a work ledger after the summary: the commands that ran, with their exit codes and first error lines, and the files that tools changed. Code writes it from the tool calls and results; no model writes it. A later compaction keeps the earlier ledger lines. Setting `ledger` (default true).
- Add a ledger-only summary when both model summaries fail and the rows before the tail have tool work (`path=ledger`). Setting `ledger_on_failure` (default true).

### Changed

- The fallback summary uses the free handoff sections of the warm request by default (`fallback_format: relaxed`). The source-quote format stays available as `fallback_format: quoted`. A relaxed reply without its end marker passes when all headings are present and the reported output count is below the generation limit. Small local models could not write the quoted format: a local 4B model passed 0 of 16 quoted replies and 10 of 10 relaxed replies in one synthetic test.

- Give fallback generation an 8,192-token allowance while keeping the visible summary within 2,048 estimated tokens. Check a fresh end marker and the summary content instead of refusing a short reply because its reported output count can include thinking. Retry an invalid reply once from the original transcript, within a shared deadline and cancellation checks. The public API still cannot report the server stop reason or stop an in-flight call.
- Ask the fallback model to keep unreported progress unknown and keep rules separate from confirmed actions. Preserve approval conditions and do not add a goal from a prohibition. These prompt rules do not guarantee factual accuracy.
- Run CI checks automatically for pushes to `main` through the approval-free `ci-main` environment, which allows only the branch named `main`. Run CI checks automatically for pull requests from branches in this repository. Keep approval through `ci-approval` for pull requests from forks, scheduled runs, and manual dispatch. Local checks still need an explicit request.
- Keep separate Hermes compatibility concurrency groups for each event and ref. Scheduled and manual runs cannot cancel an automatic push run.
- Remove the fixed summary without a model. If the warm request and model-based fallback cannot produce a usable summary, stop the current compaction pass, preserve its input history, and report safe error codes. Hermes can retry an automatic failure; the plugin does not stop the entire host turn. This includes incomplete or oversized fallback replies and a retained-history layout that cannot fit safely.
- Keep the fallback on `auxiliary.warm_compaction`; `auto` follows the main chat route. This task stays separate from the built-in compressor's `auxiliary.compression` task.

### Limits

- Hermes controls automatic chat notices through `compression.progress_notices` (default `false`). Its current completion notice is generic. A notice that names the completed warm or model-based fallback path needs a public hook after a confirmed history commit, with no success notice for an error or cancellation. This host dependency is unmet; the plugin does not patch Hermes or use private callbacks.

### Fixed

- Require at least one verified source-content quote after fallback construction. Use the existing one retry for all-unknown or generated-label-only output, then preserve history if the second reply is also unusable. Keep actual text and argument provenance separate from generated labels. Give a recognized earlier summary core the existing share before its copied appendix uses any remainder; preserve the legacy whole-summary bound for unknown formats.
- Use the exact bounded focus and memory block sent to the fallback as a separate source block. Keep the existing request reserve and reject an encoded prior quote that also occurs inside an ordinary source line.
- Construct all five fallback sections from complete source lines. Restore the complete line for a unique exact quoted part. Decode one JSON layer from a recognized, bounded earlier summary without wrapping prior quotes again or decoding copied user messages as Next step. Keep Unicode separators inside valid JSON bullets. Drop ambiguous, malformed, cut, oversized, or changed evidence and report each field unknown. Deduplicate identical decoded quotes without reporting a loss. Each section has a 512-token bound; the complete body keeps its 2,048-token bound. Keep evidence in the order User instructions, Next step, Goal, Current state, Key facts. Repair that keeps source content needs no retry. Source support does not prove truth, applicability, evidence selection, or complete retention.
- Report safe fallback token counts, request limits, attempt counts, and end-marker status when a reply is refused. Keep summary text and random markers out of status and logs. Document a task configuration workaround for Hermes versions that omit the generation limit on custom Chat Completions routes; a general correction remains a Hermes dependency.
- Stop before both summary requests when a usable capture shows zero summary room below the compaction threshold or context limit. Report `summary_too_large` for both paths and preserve history. Keep the normal fallback when the request overhead is unknown.
- Match the blank-text rules of Hermes `6e7b83a` on Anthropic Messages routes: omit blank text blocks and use `(empty)` for an empty assistant turn. Keep nonblank text, signed thinking, tool data, and the captured prefix unchanged. Other unsupported transforms still use the fallback. This fixes a synthetic mismatch class; it does not establish the cause on a modified Hermes install.
- Accept unchanged stored `reasoning` text in captured Chat Completions assistant rows. Do not require reasoning padding on captured summary or notice rows that have no reasoning. Keep checks for changed text, reasoning details, and tool signatures, and keep the captured request prefix unchanged.
- Match captured content with the stored `api_content` text for context-injected rows. Keep role, tool, and native replay checks. Report each history refusal stage and use the fallback when the capture is ahead of the live history.
- Keep the original history when an indivisible native replay block exceeds the compaction budget. Check the threshold, reply reserve, and unknown request overhead before accepting the new history.
- Match Copilot hostnames at a domain boundary. A lookalike hostname no longer gets Copilot replay handling.
- Accept trailing whitespace removal by Hermes from complete text strings, including `api_content`, without a false `source_transform_unsupported` fallback. Keep the captured request prefix unchanged.
- Keep safe request refusal codes in status and logs, instead of the generic `settings_unsupported`.
- Keep the captured `x-opencode-session` header on warm requests, including the header that Hermes adds for OpenCode routes. Refuse unknown headers and header rewrites; send the session value only as an HTTP header.
- Report `missing_end_marker` or `body_token_limit` when a fallback reply fails the completion or size check. Incomplete replies stop the current compaction pass and preserve its input history.

### Added

- Warm request adapters for Responses and Anthropic Messages. Keep native input prefixes, cache controls, and verified encrypted or signed reasoning. Unsupported request shapes and authentication routes use the fallback. See [Provider support](docs/provider-support.md) for limits.
- Native reply completion checks, token counters, and a synthetic real-Hermes provider integration check. Native replay blocks count in tail estimates and cannot be cut apart.
- Native provider and conversation checks in CI against the minimum Hermes version and upstream `main`. The conversation checks cover manual and automatic compaction, tools, and saved-session reload.
- Hypothesis tests for native histories, stream events, and capture changes, plus branch coverage reports. These tools are test dependencies only.
- A WARNING for each saved compaction that Hermes confirms without the warm path, with its reason (Hermes copies it to `errors.log`).
- After 3 confirmed compactions in a row without the warm path, one notice with a hint for each distinct reason, in the logs and at the next automatic compaction when engine status is enabled. The notice has no routine progress text. Later confirmed failures update the pending notice until Hermes shows it. Aborted attempts do not count as confirmed compactions.

### Security

- Use PBKDF2-HMAC-SHA-256 with a random salt for each plugin load to check API key changes. Captures keep only the digest in memory.

## [0.2.0] - 2026-10-06

First public release. This section records historical behavior; the unreleased changes above remove the fixed summary.

### Added

- `warm_compaction` context engine for unpatched Hermes Agent (`45871e10` or later), with documented plugin APIs only.
- Warm request: at each compaction, the main model writes the handoff on the cached prefix of the last main-model request. Manual `/compress` and automatic compaction.
- Fallback summary through the Hermes auxiliary model route, and a fixed-format summary without a model request when that also fails.
- Checks before the warm request is sent: route, key, and session identity; middleware rewrites; capacity; the five-heading handoff gate.
- Settings: `threshold`, `tail_tokens`, `user_copy_chars`, `warm`.
- Integration check against real Hermes code and a loopback fake server, and live results on DGX and LM Studio.

[Unreleased]: https://github.com/Elevatormusic/hermes-warm-compaction/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Elevatormusic/hermes-warm-compaction/releases/tag/v0.2.0
