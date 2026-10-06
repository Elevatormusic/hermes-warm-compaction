# Evidence

These records come from the research repository of this plugin. They have metadata only: counts, codes, booleans, times, and hashes. They contain no message text, request body, or key. Paths such as `.work/...` and short commit IDs refer to that research repository.

| File | Content |
| --- | --- |
| [plugin-integration.json](plugin-integration.json) | The integration check: install with the Hermes install command, manual and automatic compaction, fallback, tools, and rollback on unpatched Hermes `45871e10`, against a loopback fake server |
| [plugin-live.json](plugin-live.json) | Live runs, plugin against the built-in compressor: DGX and LM Studio, manual and automatic, ten synthetic cases of 100,000 or more tokens each |
| [lcm-bench.json](lcm-bench.json) | Live runs, plugin against hermes-lcm 0.21.0-rc2 and the built-in compressor: DGX, automatic compaction, ten synthetic cases of about 105,000 tokens |
