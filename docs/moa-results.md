# MOA integration results

The MOA warm path passed the local and loopback checks. A live Luna-reference
session passed. A live Luna-aggregator session accepted its warm summary, then
failed the first action check. These results do not qualify a release, a speed
gain, or general task retention. Native MOA requires the draft
[Hermes API PR #134922](https://github.com/NousResearch/hermes-agent/pull/134922).

## Sources and local checks

- Public plugin base: `c4103189ed8bfa3f3a2663ca21ea741e9b5c46c6`, the merged PR #15.
- Tested plugin code: `7c221dcb987494a335c73a39a24278eede751e3d`.
- Tested Hermes API: `3a088a310a73d8fd1aad2d9bd5ddec6ddff044f7`, based on
  `08165d58931841cee713468ae89032af7c57060a`.
- Plugin unit tests: 462 passed. Property tests: 12 passed. Branch coverage: 91%.
- Ruff 0.16.10, local document links, agent-file byte parity, and diff checks:
  passed.
- Real-Hermes loopback checks: five Chat MOA cases, four native MOA cases,
  seven existing plugin scenarios, nine native provider cases, four native
  conversation/reload cases, and four middleware cases passed.
- The separate Hermes change passed 124 targeted tests and all 11 source checks.
  The full Hermes test suite was not run.

The [loopback evidence](../evidence/moa-integration.json) records commands,
versions, source hashes, failed fixture attempts, and their corrections. An
old-host check confirms safe fallback when the complete tool observer is absent.
The PR #15 counterfactual used the ordinary fallback for MOA; it did not send
separate warm role requests. Loopback replies are synthetic and prove no live
cache behavior.

## Live method

Two fresh synthetic sessions ran on 2026-10-07 local time. Each had about 12K
seed tokens, three ordinary MOA turns, one manual compaction, and up to two
follow-up turns. The action check required an export of the approved target
`GREEN-9` without a new approval request. Each session used a separate profile.
Reference handoffs were enabled and ran before the aggregator handoff.

The routes were `gpt-6-luna` through native Codex Responses and a local DGX
OpenAI-compatible route with model alias `dgx-model`. Exact deployed DGX weights,
server build, initial shared cache state, and concurrent server load are unknown.
Python was 3.11.14 with OpenAI 2.24.0 and httpx 0.28.1. The source, fixture,
configuration, and helper hashes were fixed before inference and checked after
both attempts. Scoped authentication files were removed after each attempt.

The limit was 12 calls per session, 24 total, one worker at a time, and no retry.
A failed action check stopped the session. There were **22 calls**: 12 in the
Luna-reference case and 10 in the Luna-aggregator case. There was no cache reset,
model deployment, or installed-source change.

## Warm request results

Each row is one physical handoff request. Time runs from the observed transport
start to its completed response. A cache percentage uses the provider's reported
cached tokens divided by its reported input tokens.

| MOA arrangement | Handoff role | Cached / input tokens | Cache share | Request time |
| --- | --- | --- | --- | --- |
| Luna reference, DGX aggregator | Luna reference | 11,008 / 12,173 | 90.4% | 2.868 s |
| Luna reference, DGX aggregator | DGX aggregator | 12,224 / 13,139 | 93.0% | 8.383 s |
| DGX reference, Luna aggregator | DGX reference | 11,520 / 12,124 | 95.0% | 7.667 s |
| DGX reference, Luna aggregator | Luna aggregator | 0 / 13,386 | 0% | 3.041 s |

Both sessions kept the exact decoded warm input prefix and native settings.
Chat used the documented handoff controls. Both reference handoffs were
included, and both aggregator summaries passed the plugin's format gate.
Host compaction time was 11.462 s with the DGX aggregator and 11.246 s with the
Luna aggregator. There is no matched built-in-compactor timing control, so these
times do not establish a speed gain.

## Ordinary turns and retained action

Luna reported zero cached tokens on all three ordinary turns before compaction
in both arrangements. As a reference, it later reported 7,936 cached tokens out
of 8,783 on the second turn after compaction. Thus the observed ordinary-call
cache behavior was variable.

The Luna-reference session passed both follow-up action checks. The
Luna-aggregator session returned the wrong action on its first follow-up. The
session stopped; the second follow-up was not sent. A format-valid handoff does
not prove that a later answer will retain the task correctly. There was no
matched uncompacted action control, so this failure does not establish that
compaction caused the wrong answer.

The [live evidence](../evidence/moa-live.json) includes each request's counters,
time, prefix/settings checks, stop reason, and source hashes. Zero counters do
not prove expiry or a missing Hermes cache feature. The local runner and private
profiles are not published. M0 and M3 qualification remain open.
