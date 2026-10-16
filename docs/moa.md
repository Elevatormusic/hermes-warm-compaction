# Mixture of Agents

MOA has one acting aggregator and one or more reference models. Each model can
have its own prefix cache. The plugin keeps each physical request separate. It
does not share a cache between models or control cache expiry.

At compaction, the aggregator writes the canonical handoff. With
`moa_references: true`, the plugin first asks eligible reference models for small
handoffs from their own earlier views. Those handoffs are advisory data. They
cannot replace user instructions, add completed work, or write session history.
Only the aggregator handoff can become the new history.

## Supported routes

| Physical API | Requirement |
| --- | --- |
| Chat Completions | Hermes auxiliary request hooks and an explicit plugin route. |
| Native Codex Responses | Hermes native auxiliary request hooks and `ctx.llm.complete_native`. |
| Anthropic Messages in MOA | Not supported by the MOA warm path. |

Chat requests with long or nested tool schemas also need the host's copied
`request_tools` observer field. Older observer previews truncate these schemas;
the plugin then uses a fallback with `moa_payload_incomplete`.

The native API is a separate Hermes extension proposed in
[Hermes PR #134922](https://github.com/NousResearch/hermes-agent/pull/134922).
That PR is a draft. The API is not part of the plugin or plugin PR #15.
Hermes versions without that extension use a fallback for
native MOA compaction. Solo native compaction and Chat MOA do not require the new
API. The plugin does not change or wrap Hermes code.

For native MOA, Hermes owns the login and sends the request. A signed in-memory
route record binds the captured model, endpoint, profile, session, credential,
and headers. Hermes checks them again before the warm request. The plugin does
not read the Codex login file or copy its access token.

## Settings

Configure the MOA preset in Hermes first. Then list the physical routes that the
plugin can replay. These are exact matches for provider, model, base URL, and API
format. The route names are short labels for metadata logs.

For native Codex, the plugin normalizes a Hermes model alias to the bare lower
case model ID that Hermes sends. Other physical route fields must still match.
The warm native request keeps the original output controls; it does not add an
output limit to the consumer Codex endpoint.

This example uses fake names and endpoints. Replace them with the selected
Hermes routes. Set the Chat route's key in the named environment variable. Do not
put a credential in the route settings.

```yaml
context:
  engine: warm_compaction
plugins:
  entries:
    warm_compaction:
      settings:
        moa_references: true
        moa_routes:
          - name: local
            provider: custom
            model: local-model
            base_url: https://model.example.invalid/v1
            api_mode: chat_completions
            api_key_env: LOCAL_MODEL_API_KEY
            context_length: 131072
          - name: luna
            provider: openai-codex
            model: gpt-6-luna
            base_url: https://chatgpt.com/backend-api/codex
            api_mode: codex_responses
            context_length: 272000
      llm:
        allow_provider_override: true
        allowed_providers: [openai-codex]
        allow_model_override: true
        allowed_models: [gpt-6-luna]
```

The `llm` allowlists permit the host-owned native request on those physical
routes. They are separate from plugin settings. Keep them narrow. Do not add an
`api_key_env` for a Codex route; Hermes uses its existing login.

`moa_routes` defaults to an empty list. `moa_references` defaults to `false`.
Reference handoffs are optional because they add requests and latency. Start
with the aggregator alone unless the reference views contain useful information.

## Checks and limits

- Captures stay in memory. They are bounded by request size, session count,
  route count, and open attempt count. A reset removes the session's captures.
- The plugin matches the acting history to the captured aggregator view. It
  rejects a changed route, stale attempt, incomplete request, or unsupported
  request option. A reference sees only its own captured view and reply.
- Each compaction has a 120-second warm budget. Each reference has at most
  20 seconds. The plugin reserves 30 seconds for the aggregator. Warm requests
  do not retry. A reference failure can be skipped; an aggregator failure uses
  the normal fallback path.
- Chat MOA rejects per-request headers. Its configured key is checked before
  and after replay, but the old auxiliary hooks do not prove which credential
  sent the original request. Native MOA uses the host's signed route check.
- If Hermes changes a native tool name through an alias, the raw reply can
  differ from the accepted history. That case uses a fallback. Legacy native
  call IDs are normalized before the tool call and result are matched.
- The aggregator's new history can change the next request prefix. A warm
  compaction cache hit does not promise cache reuse in the next chat turn.
- `warm_last.moa_slots` reports each role, route label, result, elapsed time,
  and reported token counters. Missing counters stay unknown. A zero counter
  does not by itself prove expiry, a disabled cache, or a Hermes defect.

Run the synthetic loopback check with a clean Hermes checkout that has the
native auxiliary API:

```bash
<hermes-venv-python> -B scripts/check_moa_hermes.py --hermes-source <clean-hermes-checkout> --native --report .work/moa-report.json
```

This command requires all five Chat cases and all four native cases to pass.
Omit `--native` for the Chat cases alone. Older Hermes versions can also lack
the full tool-schema observer field; that limit must stay explicit in results.

This check proves request and history behavior only on the named source. Live
cache reuse, summary quality, and performance need separate evidence. The
broader qualification gates remain open.

See the [integration and live results](moa-results.md) for exact tested source
revisions, cache counters, and the failed Luna-aggregator continuation.
