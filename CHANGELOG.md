# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Accept trailing whitespace removal by Hermes from complete text strings, including `api_content`, without a false `source_transform_unsupported` fallback. Keep the captured request prefix unchanged.
- Keep safe request refusal codes in status and logs, instead of the generic `settings_unsupported`.
- Keep the captured `x-opencode-session` header on warm requests, including the header that Hermes adds for OpenCode routes. Refuse unknown headers and header rewrites; send the session value only as an HTTP header.
- Report `missing_end_marker` or `output_token_limit` when a fallback reply fails the completion check. Incomplete replies still use the fixed summary.

### Added

- Warm request adapters for Responses and Anthropic Messages. Keep native input prefixes, cache controls, and verified encrypted or signed reasoning. Unsupported request shapes and authentication routes use the fallback. See [Provider support](docs/provider-support.md) for limits.
- Native reply completion checks, token counters, and a synthetic real-Hermes provider integration check. Native replay blocks count in tail estimates and cannot be cut apart.
- A WARNING for each saved compaction that Hermes confirms without the warm path, with its reason (Hermes copies it to `errors.log`).
- After 3 confirmed compactions in a row without the warm path, one notice with a hint for each distinct reason, in the logs and at the next automatic compaction when engine status is enabled. The notice has no routine progress text. Later confirmed failures update the pending notice and its fixed-summary count until Hermes shows it.

### Security

- Use PBKDF2-HMAC-SHA-256 with a random salt for each plugin load to check API key changes. Captures keep only the digest in memory.

## [0.2.0] - 2026-10-06

First public release.

### Added

- `warm_compaction` context engine for unpatched Hermes Agent (`45871e10` or later), with documented plugin APIs only.
- Warm request: at each compaction, the main model writes the handoff on the cached prefix of the last main-model request. Manual `/compress` and automatic compaction.
- Fallback summary through the Hermes auxiliary model route, and a fixed-format summary without a model request when that also fails.
- Checks before the warm request is sent: route, key, and session identity; middleware rewrites; capacity; the five-heading handoff gate.
- Settings: `threshold`, `tail_tokens`, `user_copy_chars`, `warm`.
- Integration check against real Hermes code and a loopback fake server, and live results on DGX and LM Studio.

[Unreleased]: https://github.com/Elevatormusic/hermes-warm-compaction/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Elevatormusic/hermes-warm-compaction/releases/tag/v0.2.0
