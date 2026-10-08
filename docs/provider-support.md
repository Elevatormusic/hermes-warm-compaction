# Provider support

The warm path supports three API formats. A model name alone does not identify its API format. Check the selected Hermes route and the `warm_compaction:` log line.

| API format | Warm path scope |
| --- | --- |
| `chat_completions` | Existing OpenAI-compatible chat endpoints. |
| `codex_responses` | Stateless Responses requests with text and function tools, including the consumer Codex endpoint. Verified encrypted reasoning and native assistant message phases are replayed. |
| `anthropic_messages` | API-key Messages routes with text and function tools. Native Claude signed and redacted thinking blocks retain their exact order and bytes. |

Requests that cannot pass the source, route, credential, middleware, or capacity checks use the normal auxiliary summary. A failed auxiliary summary uses the fixed summary. A live cache or speed benefit needs a separate provider test.

## Responses

The request keeps the captured `input` prefix, `instructions`, tools, reasoning settings, and cache controls. New rows and the host handoff instruction follow the prefix. The warm request uses `/responses` and streaming. Only a terminal, completed response can pass the handoff gate. Partial text, a failed or incomplete response, tool calls, and refusals cannot become a warm summary.

Supported session headers stay in memory and are sent as HTTP headers. The official consumer Codex route keeps the Hermes identity and account headers. A key change after capture still refuses the attempt.

Standard Responses routes use a bounded `max_output_tokens`. The consumer Codex route does not get an added output limit that its endpoint rejects. It keeps a conservative reserve of a quarter of the context window, from 4,096 to 65,536 tokens, when its output limit is unknown.

Native server compaction checkpoints, stateful `previous_response_id` or conversation requests, media, forced tools, built-in server tools, structured output, and ambiguous tool IDs use the fallback. Opaque reasoning from a different model or issuer refuses. Fields that Hermes removes after the capture also refuse when the plugin cannot prove the final request shape.

## Anthropic Messages

The request keeps the captured native `messages` prefix, system blocks, tools, cache markers, tool choice, and thinking settings. The warm request uses `/v1/messages`. Native replies use their stop reason and content blocks; the reader also accepts complete native SSE replies.

The adapter uses `x-api-key`, `anthropic-version`, the checked Hermes beta defaults, and custom provider headers. Claude Code OAuth or subscription authentication, bearer routes (MiniMax, Kimi, CommandCode, Palantir, and Nous), Azure/Entra authentication, and Bedrock remain on the fallback. Custom `Authorization` headers also refuse. Their authentication and final-client transformations need separate integration work.

Forced or server tools, media, structured output, unknown request controls, and unproved replay blocks use the fallback. Signed thinking is replayed only on the native Anthropic host. A manual thinking budget stays unchanged; the output limit leaves at least 2,048 tokens after that budget. A request that then exceeds the window uses the fallback.

## Retained history and checks

Native replay fields count in retained-tail estimates. A signed or encrypted assistant record stays whole, including the canonical text and tool data that its replay blocks describe. If an indivisible retained record exceeds the context budget, the attempt returns the original history with `path=unchanged reason=capacity`.

The provider transport check uses synthetic data and a loopback server. It calls real Hermes transports, hooks, and middleware explicitly. A separate native conversation check runs the real Hermes agent loop. It covers manual and automatic compaction, native tool rounds, and saved-session reload in a fresh process. Both checks require a clean source checkout and keep source and plugin hashes unchanged.

The [compatibility workflow](../.github/workflows/hermes-compatibility.yml) runs these checks against the minimum Hermes commit `45871e100feceb89769536c88e5e6e265226a409` and current upstream `main`. It installs the locked Hermes runtime with the `anthropic` extra. Native failures fail the job; upstream changes are not silently skipped. The jobs have read-only repository access, no stored checkout credentials, no secrets, and no Actions cache access. Their saved reports contain metadata only.

A passing report proves synthetic request and history behavior on the exact source commit in that report. The checks do not prove every deployed provider, summary quality, cached-token reuse, or a speed gain. The plugin has no documented final-wire capture or raw provider-dispatch API; future host transforms still need review.

Run the provider integration check with an isolated Hermes dependency interpreter:

```bash
<hermes-venv-python> -B scripts/check_provider_apis.py --hermes-source <clean-hermes-checkout> --report .work/provider-api-report.json
<hermes-venv-python> -B scripts/check_native_conversations.py --hermes-source <clean-hermes-checkout> --report .work/native-conversations-report.json
```

Use `--dependency-path <dependency-folder>` if the native API dependencies are in a separate folder. The [property tests](../tests/property/test_native_properties.py) generate synthetic native records and request options for local checks. Branch coverage reports show untested plugin paths without imposing a percentage gate.

The [contribution guide](../CONTRIBUTING.md#run-the-checks) gives the full regression checks.
