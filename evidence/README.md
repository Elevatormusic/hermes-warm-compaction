# Evidence

These records cover research runs and issue checks. They have metadata only: counts, codes, booleans, times, and hashes. They contain no message text, request body, or key. Paths such as `.work/...` and short commit IDs in the research records refer to the research repository.

| File | Content |
| --- | --- |
| [issue24-validation.json](issue24-validation.json) | Issue 24: assembled-capture refusal before the fix, effective-content matching after it, safe capture-ahead refusal, and synthetic checks on clean minimum and current Hermes source |
| [issue24-live-dgx-20261009.json](issue24-live-dgx-20261009.json) | Issue 24: one manual and one automatic live DGX test with synthetic context; exact request prefixes, saved-history reloads, reported cached tokens, and measurement limits |
| [issue-12-whitespace.json](issue-12-whitespace.json) | Issue 12: refusal before the fix, three warm compactions after it, and seven integration scenarios on clean Hermes `e36a8180`; synthetic conversations and a loopback fake server |
| [plugin-integration.json](plugin-integration.json) | The integration check: install with the Hermes install command, manual and automatic compaction, fallback, tools, and rollback on unpatched Hermes `45871e10`, against a loopback fake server |
| [plugin-live.json](plugin-live.json) | Live runs, plugin against the built-in compressor: DGX and LM Studio, manual and automatic, ten synthetic cases of 100,000 or more tokens each |
| [lcm-bench.json](lcm-bench.json) | Live runs, plugin against hermes-lcm 0.21.0-rc2 and the built-in compressor: DGX, automatic compaction, ten synthetic cases of about 105,000 tokens |
