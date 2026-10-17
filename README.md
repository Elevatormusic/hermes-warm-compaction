<p align="center">
  <img src="assets/banner.svg" alt="Warm Compaction: hot-cache context compaction for Hermes Agent" width="100%">
</p>

<p align="center">
  <a href="https://github.com/Elevatormusic/hermes-warm-compaction/actions/workflows/tests.yml"><img src="https://github.com/Elevatormusic/hermes-warm-compaction/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/Elevatormusic/hermes-warm-compaction/actions/workflows/lint.yml"><img src="https://github.com/Elevatormusic/hermes-warm-compaction/actions/workflows/lint.yml/badge.svg" alt="Lint"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-FF6B3D" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10 or later">
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Hermes%20Agent-45871e10%2B-5E6878" alt="Hermes Agent 45871e10 or later"></a>
  <img src="https://img.shields.io/badge/dependencies-none-2EA44F" alt="No dependencies">
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#settings">Settings</a> ·
  <a href="#limits">Limits</a>
</p>

`warm_compaction` is a context engine plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). At each compaction, the main model writes the handoff summary on the cached prefix of its last request. The server reads only the new rows, so the compaction is faster.

It works on unpatched Hermes Agent. It uses documented plugin APIs only: no host patch, no subclass of the built-in compressor, and no runtime wrapping of Hermes code.

<a id="warm_compaction-vs-hermes-lcm-vs-built-in"></a>

<p align="center">
  <img src="assets/bench-speed.svg" alt="Median compaction time: warm_compaction 16.8 s, hermes-lcm 43.5 s, built-in 78.5 s. Compaction plus next reply: 27.9 s, 66.1 s, 97.1 s." width="100%">
</p>

<p align="center">
  <img src="assets/bench-quality.svg" alt="Compacted facts kept: warm_compaction 59/60, hermes-lcm 43/60, built-in 50/60. Fact in the middle of a long message: 10/10, 0/10, 0/10. Standing rule kept: 10/10, 5/10, 6/10." width="100%">
</p>

<p align="center"><sub>Synthetic sessions of about 105K tokens on a DGX server, ten cases. Method, all measures, and limits: <a href="#results">Results</a>.</sub></p>

## Quick start

Run these commands in a terminal (PowerShell, Terminal, or any shell) where the `hermes` command works: the same place where you start Hermes. Do not type them in a Hermes chat.

1. Install and enable the plugin:

   ```bash
   hermes plugins install Elevatormusic/hermes-warm-compaction#warm_compaction --enable
   ```

2. Select the engine:

   ```bash
   hermes config set context.engine warm_compaction
   ```

   This writes the setting to your Hermes `config.yaml` (`hermes config path` shows where it is). You can also edit the file yourself (`hermes config edit`):

   ```yaml
   context:
     engine: warm_compaction
   ```

3. Start a new Hermes session. Each compaction writes one `warm_compaction:` line to `logs/agent.log` in the Hermes home folder.

## How it works

<p align="center">
  <img src="assets/how-it-works.svg" alt="The built-in compressor sends a separate summary request with a new prompt prefix. warm_compaction sends the last request again with the new rows and a handoff instruction, so the server reuses its prefix cache." width="100%">
</p>

At each compaction, the plugin sends the last main-model request of the session again. It adds the rows that came after that request and one handoff instruction at the end. This is the *warm request*. The server can reuse the cached prefix of the earlier request, so it reads only the new rows. The reply is a Markdown handoff with five headings: Goal, User instructions, Current state, Key facts, and Next step. The plugin replaces the older rows with the handoff and keeps a verbatim tail of recent rows.

If the warm request cannot run, or if its reply fails the gate, the same compaction attempt uses a model-based fallback summary through the Hermes auxiliary route. If neither path produces a usable summary, the plugin stops the current compaction pass, preserves its original input history, and reports an error with safe codes only. It never creates a summary without a model. Hermes can retry an automatic failure; the plugin cannot stop the entire host turn through its documented API. When Hermes cancels an attempt, the history stays unchanged.

The plugin does this for manual `/compress` and for automatic compaction.

**The speedup needs the same model and a prefix cache.** The warm request goes to the main model route. It runs on any OpenAI-compatible server, but it is faster only when that server reuses its prefix cache. The fallback summary can use a different model.

| Server | Prefix cache | Reports `cached_tokens` |
| --- | --- | --- |
| Hosted APIs with prompt caching | yes | yes |
| vLLM | yes (default) | only with `--enable-prompt-tokens-details` |
| SGLang | yes, radix cache (default) | only with `--enable-cache-report` |
| TensorRT-LLM `trtllm-serve` | yes, KV block reuse | yes |
| llama.cpp `llama-server` | yes, per slot | yes |
| MLX `mlx_lm.server` | yes | yes |
| Ollama (`/v1`) | yes | no |
| LM Studio | yes | no |

From the server documentation and issue trackers; tested here only on DGX and LM Studio. Where the server does not report the counter, the log shows `cached_tokens=None` (unknown). With llama.cpp `--parallel` above 1, the warm request can go to a different slot and miss the cache.

## Requirements

- Hermes Agent `45871e10` (2026-10-02) or a later version that keeps the plugin APIs that the plugin uses. At load time, the plugin checks for each API. If an API is missing, the plugin registers nothing, writes a warning to the log, and Hermes keeps its built-in compressor.
- For the warm path: `chat_completions`, `codex_responses`, or `anthropic_messages`, with a supported request and authentication route. See [Provider support](docs/provider-support.md) for the tested scope. Other formats use the fallback summary.
- Python 3.10 or later. No third-party package.

## Install

Ask your agent:

```text
Install this plugin and set it up: https://github.com/Elevatormusic/hermes-warm-compaction
```

To install it yourself, run this command in a terminal on the computer where Hermes runs, then [select the engine](#select-the-engine):

```bash
hermes plugins install Elevatormusic/hermes-warm-compaction#warm_compaction --enable
```

The `#warm_compaction` part selects the plugin folder in the repository. To install a specific commit, add `--ref <40-character commit SHA>`. To install without enable, use `--no-enable`, then `hermes plugins enable warm_compaction`.

The install runs the Hermes security scan and the plugin checks. The plugin passed them in the [integration check](evidence/plugin-integration.json).

## Select the engine

Hermes never selects a context engine automatically. Select it one time:

- `hermes plugins`, then Provider Plugins, then Context Engine, then `warm_compaction`; or
- set this value in `config.yaml`:

```yaml
context:
  engine: warm_compaction
```

Start a new Hermes session after the change.

## Native warm handoff

Some Hermes versions include warm handoff in the built-in compressor. This plugin checks the installed
constructor and default configuration for that feature. It does not use a release number or the merge state
of a pull request. Updating Hermes keeps `context.engine: warm_compaction` selected.

Update this plugin to get the notice. When the feature is available, the plugin supplies one notice through
the automatic compaction status API for each engine instance. A pending warm-request failure warning has
priority; the native notice stays pending for the next permitted status. Session reset and warm-path recovery
do not repeat the notice. A new engine instance, including a clone, gets its own notice.

Hermes controls how each surface shows automatic status. The plugin keeps the notice pending when the API
suppresses status, but it cannot detect suppression after it returns the text. This is not a startup banner
or a guaranteed notice on every surface. Manual `/compress` does not use this status API. The first actual
compaction also writes the notice once to `logs/agent.log` for each engine instance. There is no persistent
notice record, configuration change, or conversation message.

To use native warm handoff, set these values in your Hermes configuration and restart Hermes:

```yaml
context:
  engine: compressor
compression:
  warm_handoff: "on"
```

The native default is `"off"`. `"on"` tries the warm request when a usable captured request is available.
`"auto"` also requires the compression summary route to use the main model, reported cached prompt tokens
greater than zero, and a captured request no more than five minutes old. A server that does not report cached
tokens needs `"on"` to try the native warm path. Refused or failed warm requests use the normal auxiliary summary.
The native mode keeps the built-in history policy, so retained history can differ from this plugin.

## Settings

The settings are in `plugins.entries.warm_compaction.settings`. An invalid value uses the default and writes a warning to the log.

| Key | Type and range | Default | Use |
| --- | --- | --- | --- |
| `threshold` | float, 0.10 to 0.95 | 0.50 | Fraction of the context window at which automatic compaction starts |
| `tail_tokens` | int, 0 or more | 0: 2.5% of the context window, from 10,000 to 25,000 | Size of the verbatim tail, at most half of the compaction threshold |
| `user_copy_chars` | int, 0 or more | 24,000 | Total characters of user messages that the summary copies |
| `warm` | bool | true | Set false to use only the fallback summary, for comparison runs |
| `summary_words` | int, 1 or more | 600 | Word count the handoff instruction asks for. It applies to the warm request and the fallback summary. A larger value records more of a long session. The task, not this value, controls the final reply size |
| `warm_providers` | text | empty: every provider | Comma-separated provider names that may use the warm path. A provider outside the list sends no warm request, so a route that can miss its prefix cache does not pay for the whole prompt. Names are matched without case. An empty value keeps every provider |

```yaml
plugins:
  entries:
    warm_compaction:
      settings:
        threshold: 0.5
```

The threshold also limits the request after compaction. The fixed system prompt and tool schemas must fit below it, with room for the summary and retained rows. For example, `threshold: 0.15` on a 131,072-token window gives 19,660 tokens. A fixed prefix of about 41,000 tokens already exceeds that limit. No smaller summary can make that request fit below the threshold, even though the full context window has room. Start with the default `0.5`, or select a threshold above the fixed overhead with room for the summary and tail. The request must also fit in the context window with its reply reserve. Lowering the threshold can cause `summary_too_large`. The plugin estimates these sizes; provider token counters can use a different tokenizer.

If a usable capture shows zero room for a summary after the fixed overhead, reply reserve, and summary row structure, the plugin stops that compaction pass before either model summary request. It reports `path=aborted`, `reason=summary_too_large`, and `fallback_reason=summary_too_large`. History and the capture stay unchanged. An unknown overhead does not trigger this early check. Hermes controls any later retry.

`gate:finish_not_stop` means that the warm reply did not report `finish_reason: stop`. It does not by itself prove that the reply reached its output limit. Use the finish reason and completion-token count from a synthetic test to check that cause before changing `summary_words`. A host fallback timeout alone does not identify a provider cause or prove that its output limit was reached.

On Chat Completions routes, the warm request sends the whole earlier request again, plus the new rows and the reply reserve. The reply limit of the warm request is the `max_tokens` or `max_completion_tokens` value of the earlier request, kept between 2,048 and 8,192 tokens. If the earlier request had no limit, the warm request gets 8,192 tokens in the field that Hermes uses for the route (`max_completion_tokens` for OpenAI, Azure OpenAI, GitHub Copilot, and the newer OpenAI models; else `max_tokens`). The warm request sends the default headers of the Hermes client for the route: the provider headers, `model.default_headers`, and `providers.<name>.extra_headers`. It also keeps a captured `x-opencode-session` request header. Hermes can add this header automatically for OpenCode routes. The captured value stays in memory and overrides a default header with the same name, without regard to case. It goes through both middleware checks and is sent as an HTTP header, never as a JSON field. It uses the `ssl_ca_cert` of a custom provider route. A route with `ssl_verify: false` uses the fallback summary (refusal code `tls_unverified`): the plugin does not send without certificate checks. Thus it must fit in the context window. At the default threshold, about half of the window stays free for it. If the request does not fit, the plugin uses the fallback summary (refusal code `capacity`). The plugin uses the prompt token count that the server reported for the earlier request, and estimates only the new rows.

## Fallback model and keys

The fallback summary uses the auxiliary task `warm_compaction`. Set `auxiliary.warm_compaction.provider` and `auxiliary.warm_compaction.model` to use a different model. The default `auto` follows the main chat route. This task is separate from `auxiliary.compression`, which controls the built-in compressor.

The plugin asks for up to 8,192 generated tokens, so a thinking model has room to think. The visible summary must still fit within 2,048 estimated tokens and pass the five-heading, section, byte, and summary-marker checks. The reported output count is diagnostic data; it can include thinking and does not set the visible summary size.

The prompt asks the model to use only stated facts, keep unreported progress unknown, and keep user rules separate from confirmed actions. It also asks the model to preserve approval conditions. These instructions aim to reduce errors; the content checks cannot verify factual accuracy.

The plugin constructs all five fallback sections from source quotes. It keeps complete nonblank lines from the bounded transcript. If a JSON quote is an exact part of one distinct source line, it restores that complete line, including its negation and scope. Repeated source lines do not create ambiguity; repeated decoded quotes within a section are kept once. For a recognized earlier summary, it decodes one JSON layer from the five sections after the transcript bounds. The known core gets the existing 8,000-character / 2,000-token share first; Copied user messages uses only the remainder. A core that fits stays intact. Unstructured legacy summaries keep the existing whole-summary bound. The earlier Next step ends before the engine's separate Copied user messages section. This decoded text can differ from the encoded text in the transcript. It keeps the earlier summary encoded and does not decode arbitrary user or tool text. A collision with ordinary source text is ambiguous.

Allowed source also includes the exact bounded focus and memory block sent with the request. This block stays separate from transcript lines; it cannot supply memory text that the request bounds removed or create new line adjacency. If an encoded prior quote also occurs inside an ordinary source line, its decoded meaning is ambiguous and is dropped.

Each constructed section has at most 512 estimated tokens, including JSON encoding and space for its unknown line. The plugin keeps evidence in this priority order: User instructions, Next step, Goal, Current state, Key facts. The stored headings keep their normal order. It drops malformed, unsupported, ambiguous, cut, or oversized evidence, or evidence that JSON encoding cannot preserve exactly. When any evidence was dropped or none could be kept, it adds the section's unknown line: `- goal unknown`, `- instructions unknown`, `- progress unknown`, `- facts unknown`, or `- next step unknown`. It can lose useful facts and instructions; an unknown line records that loss.

After construction, at least one verified quote must contain retained source content. Any section can supply it; the other fields can stay unknown. An all-unknown summary or quotes of only generated labels fail with `no_useful_evidence` and use the existing one retry. A second unusable reply preserves the input history. Generated empty tool-call labels, attachment marks without retained references, and bare empty copied quotes do not count. Real text literals that look like these labels can count when their retained payload occurrence is verified. Known prior JSON quotes stay trusted as reported data; the plugin cannot recover or prove their original provenance from the old summary alone. This gate does not prove truth, relevance, or complete retention.

The complete body still has at most 2,048 estimated tokens. The original reply must pass the marker and raw byte/token bounds. Valid source JSON bullets escape Unicode line separators before the shared heading check. The constructed result must pass the content, byte, token, and source checks again. Every line after the final Next step heading is checked and constructed. A source quote records a reported claim; it does not prove current truth, current applicability, correct evidence selection, or retention of every needed detail. The warm summary keeps its existing content checks.

Each request gets a fresh random end marker, in the form `[END OF SUMMARY <nonce>]`. The reply must contain that exact marker once, as its final visible line. The plugin removes the marker before it stores the summary. A fixed marker, an old marker, text after the marker, or a missing marker fails the check. This checks the reply format. It does not prove that every fact is correct or tell the plugin why the server stopped; the public auxiliary API does not report a finish reason.

If a reply fails the checks, the plugin retries once; a missing or duplicated end marker earns one more retry. Each retry uses the same original transcript, a fresh marker, and a shorter summary instruction of at most 300 words. When the failure is a missing or duplicated end marker, the retry repeats the marker requirement explicitly. It does not try to complete a partial reply. All calls share a 120-second deadline. The plugin checks cancellation and route changes before each call and after each reply. It does not retry a request error. If no reply passes, it preserves the input history.

When the server applies the requested generation limit, the request is about 18,000 tokens or less: about 8,000 transcript tokens, the instruction, and up to 8,192 generated tokens. The focus topic and memory context take their space from the transcript budget. Use a fallback model with at least a 24,000-token context window. The deadline can prevent a retry or reject a late reply, but the public API cannot stop an in-flight call. Hermes controls its own retries, so two plugin calls can cause more than two server requests.

### Reply limits on custom Chat Completions routes

The plugin asks Hermes for an 8,192-token fallback generation limit. Hermes `d4d2ab7b` can omit that limit on a custom Chat Completions route. Then the server uses its own limit, and the call can take longer. The request estimate above assumes that the server applies the requested generation limit.

For a custom Chat Completions server that accepts `max_tokens`, set the limit in the auxiliary task body:

```yaml
auxiliary:
  warm_compaction:
    extra_body:
      max_tokens: 8192
```

Keep other task options that your server needs. This workaround applies only to Chat Completions routes that accept `max_tokens`; do not use it for Responses, Anthropic Messages, or a model that requires `max_completion_tokens`. Restart Hermes after the configuration change. A loopback check verifies the request field on the named Hermes source. It does not prove that a real server applies the limit. A general correction needs Hermes to forward the limit from the plugin API.

The generation limit includes any thinking tokens that the server counts as output. A model can still use that allowance before it finishes. Set thinking options only as your server documents them. The plugin does not change them automatically. A short summary at the reported token limit can pass if its fresh end marker and all content checks pass. A reply without that marker fails, even when it is short.

For a custom endpoint that needs a key, put the key name in the model settings:

```yaml
model:
  provider: custom
  base_url: https://example.invalid/v1
  key_env: MY_ENDPOINT_KEY
```

The warm request uses the key that Hermes gives to the engine. The fallback request goes through the Hermes auxiliary route, which reads the key from `model.key_env` or `model.api_key`. Without one of them, Hermes sends the placeholder key `no-key-required`, and a server that needs a key refuses the fallback request. A host that gives the key only in code, for example `AIAgent(api_key=...)`, cannot give it to the fallback request.

## Check the result

Hermes writes compaction metadata to `logs/agent.log` in the Hermes home folder:

```text
warm_compaction: path=warm reason=accepted elapsed_s=10.656 prompt_tokens=108021 cached_tokens=107968
```

- A successful `path` is `warm` or `fallback`. `path=aborted` means that the pass failed without replacing its input history. `reason` records the warm refusal; `fallback_reason` records the safe fallback error code, such as `unavailable`, `gate:<code>`, `summary_too_large`, `quote_capacity`, `capacity`, or `error:<exception class>`. No raw exception text is reported.
- `reason` is `accepted` or the refusal code of the warm request: `disabled`, `no_capture`, `provider_not_allowed`, `api_mode_unsupported`, `auth_unsupported`, `route_changed`, `settings_unsupported`, `request_not_mapping`, `request_options_unsupported`, `request_not_json`, `source_transform_unsupported`, `history_changed`, `capacity`, `cancelled`, `middleware_unavailable`, `middleware_refused`, `middleware_rewrite`, `middleware_repeated`, `middleware_after_capture`, `middleware_order_unknown`, `headers_unknown`, `tls_unknown`, `tls_unverified`, `middleware_changed_reply`, `provider_error`, `timeout`, `incomplete_response`, or `gate:<reason>`. The warm request goes through the Hermes `llm_request` and `llm_execution` middleware, as a main request does.
- `cached_tokens` is `None` when the server does not report it. Then the cache reuse is unknown.
- A refused fallback reply also reports `fallback_output_tokens`, `fallback_requested_max_tokens`, `fallback_body_tokens`, `fallback_attempts`, and `fallback_end_marker`. The output count is `None` when it is unavailable or invalid. The body count is an estimate. The marker code is `present`, `missing`, `not_last`, `duplicate`, or `empty`; it contains no reply text or random marker. The requested limit does not prove that Hermes sent it or that the server applied it.

Capture refusal codes identify the failed check: `request_not_mapping` means that the request is not a mapping; `request_options_unsupported` means that extra headers, query options, or `extra_body` cannot be kept; `request_not_json` means that a body value cannot be sent as JSON (including NaN and infinity). `settings_unsupported` still applies to unsupported model settings. These codes contain no request values.

History refusal codes identify the failed stage: `history_changed:capture_ahead` means that the capture has more rows than the live history; `history_changed:reply_missing` means that the captured reply is absent; `history_changed:digest` means that a captured row differs; `history_changed:reply` means that the next row does not match the reply; `history_changed:tool_extra` or `history_changed:tool_count` means that tool results do not match the captured calls; `history_changed:trailing` means that later rows include a role other than user. These codes contain no message text or tool IDs. A shorter live history uses the fallback; the plugin cannot reuse unmatched future rows.

The engine status (`get_status()`) has the same values in `warm_last`. After a new attempt starts, an older worker cannot replace this status or write a final metadata line. The log never contains message text, request bodies, or keys.

Manual `/compress` can report `No changes from compression` when all messages fit in the recent-history tail. Then the plugin has no older history to summarize. The request estimate also includes system instructions and tool definitions, which the plugin does not compress. Use `/context` to see the token counts for each part. This unchanged result does not count as a warm-path failure.

If both summary paths fail, the plugin raises a compaction error before it can replace the input history. In the inspected Hermes source (`05eecbcd`), manual `/compress` reports that safe error before history installation. The automatic post-tool handler can catch the error and retry the whole turn, up to eight times. Each failed plugin pass preserves its own input history; this does not promise that the host stops immediately or that the whole turn stays unchanged. The plugin reports only safe error codes, not provider response text, credentials, or conversation text. Hermes controls how each surface shows the error. This source review does not qualify a particular UI.

### Automatic completion notices

Set `compression.progress_notices: true` to show routine automatic compaction start and completion notices on chat platforms. The default is `false`. This Hermes setting applies to both this plugin and the built-in compressor. See the [Hermes configuration guide at the inspected source](https://github.com/NousResearch/hermes-agent/blob/05eecbcd972c8737ebc7722ea08aab47fb538043/website/docs/user-guide/configuration.md#L1032).

The current Hermes completion notice is generic: `Context compaction complete`. The documented plugin API cannot label a completed pass as warm compaction or model-based fallback. That label needs a public completion hook that runs only after a confirmed history commit and never reports success for an error or cancellation. This host dependency is unmet. The plugin does not use private callbacks or patch Hermes to add it.

### When the warm path keeps failing

When the warm summary cannot be used, the plugin tries the model-based fallback summary. If that summary cannot be used, the current compaction pass stops and preserves its input history. The plugin reports these results:

- Each saved compaction that Hermes confirms without the warm path writes a WARNING with its reason, for example `Warm compaction skipped (provider_error); used the fallback summary`. Hermes copies warnings to `logs/errors.log`.
- A host compatibility failure (`middleware_order_unknown` or `middleware_unavailable`) prepares a notice after the first confirmed failure. It identifies the failed Hermes check and the summary path used. It does not wait for three failures.
- After 3 confirmed compactions in a row without the warm path, the plugin writes one WARNING with a hint for each distinct reason. When automatic engine status is enabled, Hermes shows the notice at the next automatic compaction. The notice has no routine progress text, so a gateway filter for that text does not discard it. Later confirmed failures update the pending notice until Hermes shows it. For example:

  ```text
  ⚠ Warm compaction unavailable: the last 3 compactions did not use the warm summary (provider_error: the server refused the warm request (a provider error, or a gateway that needs a cookie)). Compaction continues with the fallback summary. Details: the warm_compaction lines in logs/agent.log.
  ```

- When the fallback summary cannot be used, the attempt reports an error. It does not count as a confirmed compaction in the notice streak.
- A warm summary can fail the handoff checks even when the server reports cached tokens. The notice describes the summary result. It does not prove a cache miss, a speed change, or that all details were kept.
- A confirmed warm compaction or a session reset ends the streak. The notice can then show again at the first host compatibility failure, or after 3 other failures. A cancelled or rejected attempt does not count or clear the streak, and `warm: false` is not a failure.
- When the engine's automatic status is disabled, warnings stay in the logs. The engine keeps a pending notice until status is enabled or a confirmed warm compaction or session reset clears it.
- The streak belongs to one engine instance. It is not saved across a restart or shared with separate gateway manual-command engines. A host that sends no successful compaction notification has attempt metadata only.
- Manual `/compress` has no status line from the engine: after manual compactions only, the notice is in the logs.

## Results

These are historical results from the versions and settings in the linked records. They do not test the current stop-on-failure behavior. All runs used synthetic conversations of 100,000 or more prompt tokens on unpatched Hermes `45871e10`, one session at a time. The records have metadata only. "DGX" is an OpenAI-compatible server on an NVIDIA DGX that reports cached tokens. "LM Studio 4B" is LM Studio with a Qwen3.5 4B model; it reports no cached-token counter.

**Plugin against the built-in compressor** ([record](evidence/plugin-live.json)), ten cases per engine and mode, median seconds:

| Engine | Mode | Warm path accepted | Compaction, plugin / built-in | Compaction + next reply, plugin / built-in |
| --- | --- | --- | --- | --- |
| DGX | manual | 10/10 | 10.2 / 32.7 | 12.5 / 36.9 |
| DGX | automatic | 10/10 | 10.1 / 39.9 | 11.5 / 42.9 |
| LM Studio 4B | manual | 9/10 | 20.9 / 34.4 | 25.2 / 64.5 |
| LM Studio 4B | automatic | 9/10 | 21.1 / 41.8 | 25.7 / 54.9 |

On DGX, each warm request read about 108,000 prompt tokens, and the server reported at least 99.9% of them as cached.

**Plugin, hermes-lcm 0.21.0-rc2, and the built-in compressor** ([record](evidence/lcm-bench.json)), DGX, automatic compaction, ten cases: see the [charts at the top](#warm_compaction-vs-hermes-lcm-vs-built-in).

<details>
<summary>Table with all measures</summary>

| Measure | Plugin | hermes-lcm | Built-in |
| --- | --- | --- | --- |
| Compaction, median (s) | 16.8 | 43.5 | 78.5 |
| Compaction + next reply, median (s) | 27.9 | 66.1 | 97.1 |
| Compacted facts correct | 59/60 | 43/60 | 50/60 |
| Fact in the middle of a long row | 10/10 | 0/10 | 0/10 |
| Standing rule kept (next reply) | 10/10 | 5/10 | 6/10 |
| Control fact (in the recent rows) | 10/10 | 10/10 | 10/10 |
| Next action, action and target | 9/10 | 9/10 | 10/10 |

The charts come from the record: `python assets/make_charts.py`.

</details>

The plugin was faster than both other engines in 10 of 10 cases. hermes-lcm and the built-in compressor cut long messages before they summarize them. The warm request lets the main model read the full rows.

### Limits of the results

- One synthetic task family, one session shape, and one model per engine.
- The cache reuse depends on the server. The plugin cannot force it. On LM Studio, the cache reuse is unknown, and part of the gain comes from the smaller new history.
- The results describe these versions and settings only.

## Limits

- The warm path needs a captured main-model request in the current Hermes process. After a restart or a resume, the first compaction uses the fallback summary if no main-model request completed before it (`no_capture`).
- The warm request uses its own HTTP connection. It sends the route URL (with its query, for example an Azure `api-version`), the route headers, and the TLS settings, but not the cookies of the Hermes client. A gateway that needs a cookie from an earlier response refuses the warm request, and the fallback summary runs (`provider_error`).
- A model or session switch, or a host timeout, during the attempt stops it, also while the warm request is on the network: the history stays unchanged (`route_changed` or `cancelled`).
- A provider or API key switch forgets the capture of the session: the next compaction uses the fallback summary (`no_capture`) until a main request on the new configuration completes. A key that Hermes gives as a function can change without a switch (a token that refreshes or rotates): the capture keeps a PBKDF2-HMAC-SHA-256 digest of the key that sent it, never the key, and the warm request is not sent with another key (reason `credential_changed`). The digest uses a random salt made when the plugin module loads. The salt and the captures stay in memory. A key change while a request is open keeps no capture.
- A captured row with a field that Hermes does not send for the stored row (other than the prompt caching marker `cache_control`) refuses the warm path (`source_transform_unsupported`).
- On Chat Completions routes, the source check accepts exact text or removal of trailing whitespace from a complete text string, including a stored `api_content` string. Leading or inner whitespace changes, added whitespace, and changes to text inside a content list still refuse the warm path (`source_transform_unsupported`). The captured request prefix stays unchanged.
- Responses and Anthropic Messages use their own source checks. Native signed and encrypted replay records stay whole and count in tail estimates. A record that leaves no room for a valid new history fails with `path=aborted` and a safe capacity code, without replacing the input history. See [Provider support](docs/provider-support.md).
- The handoff must start with `## Goal`: text before the first heading is refused (`gate:heading_preamble`).
- The verbatim tail keeps the newest rows that fit in `tail_tokens`, and in the free room after the system rows and tool schemas of a usable capture, the reply reserve, and a summary: a large system prompt can make a short history compact. It stops at the first older row that does not fit, and that row goes into the summary. When the newest unit alone is larger than the tail or the free room, its largest sent payloads keep only their start and end: message text, tool call arguments (kept as a JSON object), and reasoning text. A large image or other media part is replaced by its attachment mark: a web URL or a file name stays (at most 200 characters), a data URL does not. If the cuts are not enough, the payloads are dropped, largest first; only the row structure stays. Signed `reasoning_details` stay as they are. The fallback summary gets the cut middles. The fallback body is limited to 2,048 estimated tokens, including dense CJK text. The existing 4,096-token reserve also leaves room for bounded quotes. If the summary and quotes cannot fit safely, the current compaction pass stops and preserves its input history. The sizes come from the fields that Hermes sends: row metadata does not count; reasoning counts only on a route that needs it back (DeepSeek, Kimi, MiMo thinking; a capture of the route shows it, else a stored row with `reasoning_content`); `reasoning_details` counts only on a route that replays it, without the private native-assistant entries; the native-assistant entry type that the provider profile declares (`native_reasoning_details_type`) counts on every route of the provider, as Hermes replays it; a tool call counts only its id, type, name, and arguments, and a usable thought signature counts only for a Gemini or Gemma model. When the fallback summary is used and the latest user request in front of the tail must be cut, the cut middle goes after the summary as a bounded quote, in what the summary budget has left: the fallback transcript had only the start and end of a long row. When no quote fits there, the current compaction pass stops and preserves its input history. A fallback summary that does not fit in the free room also stops the current compaction pass. A model, session, window, or threshold change at any time during the compaction discards its result (reason `route_changed`). On a Gemini or Gemma route, a tool call with a thought signature stays whole: Gemini needs it back as it was. A change of the provider or the key forgets the capture, also from an empty provider and key. A warm summary that does not fit in the free room (a dense reply under a low threshold, for example) is not used: the fallback summary runs (reason `summary_too_large`). A retry middleware cannot send the warm request a second time, also after a failed send. A warm reply body above 1 MiB is not read (reason `response_too_large`). The fallback request does not start after a model, provider, or session switch. The plugin API has no request on a fixed route (an override needs a trust setting), so a switch in the short time after that check, before the host resolves the auxiliary route, can still send the transcript to the new route. Reasoning is cut only when a usable capture of the route shows that the route sends it: other reasoning (of an earlier route after a model switch, for example) does not go to the fallback model. The latest user request in front of the tail gets the room that stays after the tail is cut. Hermes notifications, task lists, and recovery nudges are not user requests only when the whole text matches the Hermes template. The copied user messages get the room that stays after the tail is cut. Without a usable capture (after a restart, after a model change, or when a middleware changed the captured rows), the request overhead is unknown: the summary has no copied user messages, the summary, the prepended row, and the tail take at most half of the free room, the check before a request compacts when the history estimate reaches half of the window less the reply reserve, and the reply reserve is a quarter of the window (at most 65,536 and at least 4,096 tokens). The same reserve applies when the captured request had no positive `max_tokens` or `max_completion_tokens`: the provider default is not known. The host token count is not used for the overhead: it does not use the plugin tokenizer.
- Responses warm requests are streamed; Chat Completions and Messages warm requests are not streamed. No warm request is retried. Its time limit is 120 seconds. The fallback has one shared 120-second deadline for at most two plugin calls (three when the failure is a missing or duplicated end marker); the host can keep an in-flight call running beyond that deadline. Native formats have additional limits in [Provider support](docs/provider-support.md).
- On Chat Completions routes, per-request `extra_headers` supports only one `x-opencode-session` header, with a non-empty plain ASCII string value, no control characters, and no leading or trailing space. Other headers, duplicate names with different case, non-empty `extra_query`, or an invalid `extra_body` use the fallback (`request_options_unsupported`). A JSON `extra_body.extra_headers` field cannot be combined with the session header. A request with `n` above 1, a forced `tool_choice`, `response_format`, or audio settings uses the fallback (`settings_unsupported`).
- After `/compress`, Hermes builds the system prompt again from its configuration. A system message that a host gave in code is not kept. This is Hermes behavior, and it is the same for the built-in compressor.
- Each new agent writes this Hermes warning to the log: `Context engine 'warm_compaction' loaded but no engine instance found`. Hermes first tries its context engine loader, which cannot make this engine. Then Hermes uses the engine of the enabled plugin. The warning has no effect.

## Rollback

Set the built-in compressor again and start a new Hermes session:

```yaml
context:
  engine: compressor
```

The new history uses summary rows that Hermes recognizes. The [integration check](evidence/plugin-integration.json) checked a rollback with the plugin still enabled: the built-in compressor loaded the saved history and compacted it again with its own summary. To remove the plugin, use `hermes plugins disable warm_compaction` or `hermes plugins remove warm_compaction`.

## Hermes update compatibility

A future Hermes update can change an API that this plugin uses. The plugin currently reads the private middleware registry to confirm that no execution middleware runs after its capture. A later middleware could send a different request, so removing this check would make the capture unsafe. If the order cannot be checked, the plugin refuses the warm path and uses a fallback. Fallback summaries can omit details; they are not a guarantee of complete retention.

The [compatibility workflow](.github/workflows/hermes-compatibility.yml) tests the real Hermes middleware, plugin install, native provider requests, and native conversation loop against the minimum supported revision and upstream `main`. It runs on pull requests, pushes to `main`, each day, and by manual dispatch. Each report records the exact Hermes commit and plugin file hashes. The checks use synthetic data and a loopback server. Native failures fail the job; an upstream change is not skipped. A failed workflow needs review; it does not update or change an installed Hermes. GitHub Actions notification settings control failure notifications.

The check uses synthetic requests and a fake provider callback. It checks capture ordering, request changes before and after capture, and refusal when ordering cannot be read. It does not call a model. Run it with a Hermes dependency interpreter and a clean, isolated source checkout:

```bash
<hermes-venv-python> -B scripts/check_hermes_compatibility.py --hermes-source <clean-hermes-checkout> --report .work/hermes-compatibility.json
```

The middleware check alone does not prove full conversation-loop compatibility or a cache benefit. Before adopting a Hermes update, also run the plugin and native checks in [Contributing](CONTRIBUTING.md#run-the-checks). Check `path`, `reason`, and available token counters in an isolated synthetic session. Keep the last working Hermes version available until these checks pass.

To remove the private ordering dependency, Hermes needs a documented execution-middleware contract that tells each callback whether another callback follows it. That information must come from the same chain snapshot that the request executes. It must not be inferred from a separate registry read or from request text. The current Hermes API does not provide it. This plugin does not patch Hermes or assume that this proposed API exists.

## Development

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests -p "test_wc_*.py"
```

Run the integration check on a clean Hermes checkout with the Python of the Hermes virtual environment:

```bash
<hermes-venv-python> -B scripts/check_plugin_hermes.py --hermes-source <clean-hermes-checkout> --report .work/plugin-integration-report.json
```

The integration check installs the plugin with the Hermes install command in a temporary Hermes home. It runs real Hermes conversation and compaction code against a loopback fake server, and it sends no request to a real model. An audit-hook fence blocks other network access, child processes, and writes outside the scenario folder. The report has metadata only.

For native provider edits, run both `scripts/check_provider_apis.py` and `scripts/check_native_conversations.py`. The second check covers the real agent loop, manual and automatic compaction, native tool rounds, and saved-session reload. See [Provider support](docs/provider-support.md#retained-history-and-checks) for commands and limits. The [contribution guide](CONTRIBUTING.md#run-the-checks) also gives the Hypothesis property tests and branch coverage commands. Test tools are separate from the plugin's runtime dependencies.

## Contributing

Bug reports, server results, and pull requests are welcome. Each author can have at most five open PRs, including drafts. Each new PR must link an earlier issue, such as `Closes #123`; PRs authored by `Elevatormusic` are exempt from the issue rule. Issues and PRs must declare the AI harness and model used, or state that no AI agent was used. Each new PR must also confirm that all tests passed. See [CONTRIBUTING.md](CONTRIBUTING.md) for the rules and checks, and [SECURITY.md](SECURITY.md) to report a vulnerability privately. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License and credit

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). A copy, a port, or a derivative work of this code must keep the NOTICE file. If you use this design in Hermes Agent or in a different project, please credit Elevatormusic and link to this repository.
