# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Run CI checks automatically for pushes to `main` through the approval-free `ci-main` environment, which allows only the branch named `main`. Keep approval through `ci-approval` for pull requests, scheduled runs, and manual dispatch. Local checks still need an explicit request.
- Keep separate Hermes compatibility concurrency groups for each event and ref. Scheduled and manual runs cannot cancel an automatic push run.
- Remove the fixed summary without a model. If the warm request and model-based fallback cannot produce a usable summary, stop the current compaction pass, preserve its input history, and report safe error codes. Hermes can retry an automatic failure; the plugin does not stop the entire host turn. This includes incomplete or oversized fallback replies and a retained-history layout that cannot fit safely.
- Keep the fallback on `auxiliary.warm_compaction`; `auto` follows the main chat route. This task stays separate from the built-in compressor's `auxiliary.compression` task.

### Limits

- Hermes controls automatic chat notices through `compression.progress_notices` (default `false`). Its current completion notice is generic. A notice that names the completed warm or model-based fallback path needs a public hook after a confirmed history commit, with no success notice for an error or cancellation. This host dependency is unmet; the plugin does not patch Hermes or use private callbacks.

### Fixed

- Match captured content with the stored `api_content` text for context-injected rows. Keep role, tool, and native replay checks. Report each history refusal stage and use the fallback when the capture is ahead of the live history.
- Keep the original history when an indivisible native replay block exceeds the compaction budget. Check the threshold, reply reserve, and unknown request overhead before accepting the new history.
- Match Copilot hostnames at a domain boundary. A lookalike hostname no longer gets Copilot replay handling.
- Accept trailing whitespace removal by Hermes from complete text strings, including `api_content`, without a false `source_transform_unsupported` fallback. Keep the captured request prefix unchanged.
- Keep safe request refusal codes in status and logs, instead of the generic `settings_unsupported`.
- Keep the captured `x-opencode-session` header on warm requests, including the header that Hermes adds for OpenCode routes. Refuse unknown headers and header rewrites; send the session value only as an HTTP header.
- Report `missing_end_marker` or `output_token_limit` when a fallback reply fails the completion check. Incomplete replies stop the current compaction pass and preserve its input history.

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
