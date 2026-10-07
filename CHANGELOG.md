# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-06

First public release.

### Added

- `warm_compaction` context engine for unpatched Hermes Agent (`45871e10` or later), with documented plugin APIs only.
- Warm request: at each compaction, the main model writes the handoff on the cached prefix of the last main-model request. Manual `/compress` and automatic compaction.
- Fallback summary through the Hermes auxiliary model route, and a fixed-format summary without a model request when that also fails.
- Checks before the warm request is sent: route, key, and session identity; middleware rewrites; capacity; the five-heading handoff gate.
- Settings: `threshold`, `tail_tokens`, `user_copy_chars`, `warm`.
- Integration check against real Hermes code and a loopback fake server, and live results on DGX and LM Studio.

[0.2.0]: https://github.com/Elevatormusic/hermes-warm-compaction/releases/tag/v0.2.0
