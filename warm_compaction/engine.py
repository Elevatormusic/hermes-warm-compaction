"""The warm_compaction context engine."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable

from agent.context_engine import ContextEngine

from . import fallback, handoff, layout, warm
from .capture import CaptureStore
from .rows import SendPolicy, api_content, attr, estimate_tokens, hermes_value, sent_rows, sent_tokens

logger = logging.getLogger(__name__)

NAME = "warm_compaction"
DEFAULTS: dict[str, Any] = {"threshold": 0.50, "tail_tokens": 0, "user_copy_chars": 24_000, "warm": True}
THRESHOLD_RANGE = (0.10, 0.95)
TAIL_SHARE = 0.025
TAIL_MIN = 10_000
TAIL_MAX = 25_000
# The summary row without the summary text and the copies: the header, the headings, the end marker, and the
# quote marks, with a margin for the estimate.
CARRIER_TOKENS = 500
# The summary size that the tail cap keeps free before the fallback summary is known: two times its reply
# limit, because the estimate and the server count do not use the same tokenizer.
# Rounds of the fixed path: cut the tail, quote the cut, and size the cap again.
FIXED_ROUNDS = 10
SUMMARY_RESERVE = 2 * fallback.MAX_TOKENS
# The largest reply reserve for an unknown reply limit: the Hermes output reserve of a native Gemini route.
UNKNOWN_RESERVE_MAX = 65_536
HERMES_END_MARKER = "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
HERMES_DB_MARKER = "_db_persisted"


def _valid(key: str, value: Any) -> bool:
    if key == "warm":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if key == "threshold":
        return isinstance(value, (int, float)) and THRESHOLD_RANGE[0] <= float(value) <= THRESHOLD_RANGE[1]
    return isinstance(value, int) and value >= 0


def read_settings(get_config: Callable[..., Any] | None) -> dict[str, Any]:
    """Read the plugin settings. An invalid value logs a warning and uses the default."""
    settings = dict(DEFAULTS)
    if get_config is None:
        return settings
    for key, default in DEFAULTS.items():
        try:
            value = get_config(key, default)
        except Exception:
            value = default
        if value is None:
            continue
        if _valid(key, value):
            settings[key] = float(value) if key == "threshold" else value
        else:
            logger.warning("warm_compaction: the setting %s is not valid; the default %r applies", key, default)
    return settings


def tail_budget(setting: int, context_length: int, threshold_tokens: int = 0) -> int:
    """Return the tail size: the setting, or 2.5% of the context window kept between 10,000 and 25,000.
    The size is also at most half of the compaction threshold, for a set value too. Otherwise the tail can hold
    the whole history when compaction starts, and nothing comes before the tail."""
    budget = setting if setting > 0 else max(TAIL_MIN, min(TAIL_MAX, int(context_length * TAIL_SHARE)))
    return min(budget, threshold_tokens // 2) if threshold_tokens > 0 else budget


def sanitize_memory(memory_context: Any) -> str:
    """Return memory provider text after the Hermes redaction. Return an empty text when that fails."""
    if not memory_context:
        return ""
    try:
        from agent.context_engine import sanitize_memory_context
        return str(sanitize_memory_context(str(memory_context)))
    except Exception:
        return ""


def request_overhead(capture: dict[str, Any] | None, messages: list | None = None) -> int | None:
    """Estimated tokens of the request parts that are not history rows: the system rows and the tool schemas of
    the captured request. None without a captured request body (for example, after a restart). The host's token
    count less the history estimate is not used: the two do not use the same tokenizer, and a compressible
    history can have an estimate far above the server count, so the difference can be much too small."""
    body = (capture or {}).get("body")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return None
    count = max(len(body["messages"]) - len(capture.get("digests") or ()), 0)
    # Every structured field comes again with the next request: tools, the legacy functions, response schemas.
    structured = {key: value for key, value in body.items() if key != "messages" and isinstance(value, (dict, list))}
    overhead = estimate_tokens({"messages": body["messages"][:count], **structured})
    if messages is not None:
        # Request-time text in the captured rows (context that Hermes or a middleware added) comes again with
        # the next request, so it is overhead too.
        for wire, row in zip(body["messages"][count:], messages):
            overhead += max(0, estimate_tokens(attr(wire, "content")) - estimate_tokens(api_content(row)))
    return overhead


def request_reserve(capture: dict[str, Any] | None, context_length: int = 0) -> int:
    """The reply reserve of the captured request (warm.reply_reserve). Without a captured body or a positive
    captured limit, the reply limit of the next request is unknown (Hermes does not give it to a context engine): a quarter of the window, at most
    UNKNOWN_RESERVE_MAX, and at least the default."""
    body = (capture or {}).get("body")
    if isinstance(body, dict) and isinstance(body.get("messages"), list) and any(
            type(body.get(key)) is int and body[key] > 0 for key in warm.LIMIT_KEYS):
        return warm.reply_reserve(body)
    # Without a positive captured limit, the provider default applies to the next request: it is unknown too.
    return max(warm.DEFAULT_RESERVE, min(context_length // 4, UNKNOWN_RESERVE_MAX))


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


class WarmCompactionEngine(ContextEngine):
    """Context engine that asks the main model for the handoff on the cached request prefix."""

    threshold_percent = DEFAULTS["threshold"]
    awaiting_real_usage_after_compression = False

    def __init__(self, *, store: CaptureStore | None = None, llm: Any = None,
                 settings: dict[str, Any] | None = None, task: str | None = fallback.TASK, post: Any = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._store = store if store is not None else CaptureStore()
        self._llm = llm
        self._settings = {**DEFAULTS, **(settings or {})}
        self._task = task
        self._post = post
        self._clock = clock
        self.threshold_percent = float(self._settings["threshold"])
        self.last_real_prompt_tokens = 0
        self.warm_last: dict[str, Any] | None = None
        self._wc_session_id = ""
        self._wc_route: tuple = (None, None, None)
        self._wc_api_key: Any = ""
        self._wc_provider: str = ""
        # True after the first update_model: an empty provider and key are an identity too.
        self._wc_identity_set = False

    @property
    def name(self) -> str:
        return NAME

    def clone_for_agent(self) -> "WarmCompactionEngine":
        """Return a new engine for one agent. The clone shares the capture store and the model access."""
        clone = WarmCompactionEngine(store=self._store, llm=self._llm, settings=self._settings, task=self._task,
                                     post=self._post, clock=self._clock)
        # The per-model thresholds that update_model resolves (a copy: the clone must not change with this engine).
        if getattr(self, "model_thresholds", None):
            clone.model_thresholds = copy.deepcopy(self.model_thresholds)
        return clone

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        if kwargs.get("boundary_reason") == "compression" and kwargs.get("old_session_id"):
            self._store.forget(session_id=kwargs["old_session_id"])
        self._wc_session_id = str(session_id or "")

    def update_model(self, model: str, context_length: int, base_url: str = "", api_key: Any = "",
                     provider: str = "", api_mode: str = "") -> None:
        super().update_model(model, context_length, base_url=base_url, api_key=api_key, provider=provider,
                             api_mode=api_mode)
        # A capture belongs to the provider and key that sent it: two configurations can share the model, the base
        # URL, and the API mode, and the old body must not go out with the new key and headers. The first call (no
        # identity yet) is not a switch; a change from an empty provider or key is.
        if self._wc_identity_set and (provider, api_key) != (self._wc_provider, self._wc_api_key):
            self._store.forget(session_id=self._wc_session_id)
        self._wc_identity_set = True
        self._wc_route = (model, base_url, api_mode)
        self._wc_api_key = api_key
        self._wc_provider = provider

    def update_from_response(self, usage: dict[str, Any]) -> None:
        usage = usage or {}
        self.last_prompt_tokens = _int(usage.get("prompt_tokens"))
        self.last_completion_tokens = _int(usage.get("completion_tokens"))
        self.last_total_tokens = (_int(usage.get("total_tokens"))
                                  or self.last_prompt_tokens + self.last_completion_tokens)
        if self.last_prompt_tokens > 0:
            self.last_real_prompt_tokens = self.last_prompt_tokens
        # Hermes sets this flag on the engine after each compaction. The next response clears it.
        self.awaiting_real_usage_after_compression = False

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = self.last_prompt_tokens if prompt_tokens is None else prompt_tokens
        return self.threshold_tokens > 0 and _int(tokens) >= self.threshold_tokens

    def should_compress_preflight(self, messages: list) -> bool:
        """The history estimate, with the system rows and tool schemas of a usable capture, reaches the threshold;
        or with the reply reserve it does not fit in the window. Without a usable capture (after a restart, for
        example) the overhead is unknown: as in compress, half of the room (the window less the reply reserve) is
        for the system rows and the tool schemas."""
        if self.threshold_tokens <= 0:
            return False
        # As Hermes sends the rows: the stored display text of a row with api_content is not in the request.
        tokens = estimate_tokens(sent_rows(messages, self._policy(messages)))
        window = int(self.context_length or 0)
        budget = self._budget_capture(self._store.latest(self._wc_session_id), messages)
        if budget is not None:
            tokens += request_overhead(budget, messages) or 0
            if window > 0 and tokens + request_reserve(budget, window) > window:
                return True
        elif window > 0 and tokens >= (window - request_reserve(None, window)) // 2:
            return True
        return tokens >= self.threshold_tokens

    def has_content_to_compress(self, messages: list) -> bool:
        start, _prepend = layout.tail_start(messages, self._start_tail(messages, self._store.latest(
            self._wc_session_id)), self._prefixes(), self._policy(messages))
        return start > 0

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.warm_last = None
        # The calibration of the old session: Hermes uses the real prompt count as a floor unless the latch is set.
        self.last_real_prompt_tokens = 0
        self.awaiting_real_usage_after_compression = False

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status["warm_last"] = dict(self.warm_last) if self.warm_last else None
        return status

    def compress(self, messages: list, current_tokens: int | None = None, focus_topic: str | None = None,
                 force: bool = False, memory_context: str = "") -> list:
        """Replace the rows before the tail with a summary. Keep the history when no row comes before the tail."""
        started = self._clock()
        # The route, key, and session of this attempt, before anything reads them. Hermes can switch them while
        # the attempt still runs (it runs on a pooled thread and can outlive a host timeout); the warm request goes
        # only to this route, and the result is used only when they did not change.
        attempt = self._attempt()
        prefixes = self._prefixes()
        policy = self._policy(messages)
        capture = self._store.latest(attempt[3])
        # An unknown overhead can be most of the window: then no copies. A capture of another route says nothing
        # about the system prompt, the tools, and the reply limit of this one.
        budget = self._budget_capture(capture, messages)
        overhead = request_overhead(budget, messages)
        reserve = request_reserve(budget, int(self.context_length or 0))
        start, prepend = layout.tail_start(messages, self._tail_cap(SUMMARY_RESERVE, overhead, reserve), prefixes,
                                           policy)
        if start <= 0:
            return messages
        memory = sanitize_memory(memory_context)
        record: dict[str, Any] = {"path": None, "reason": None, "elapsed_s": None, "prompt_tokens": None,
                                  "cached_tokens": None}
        summary = self._warm_summary(messages, capture, focus_topic, memory, prefixes, record, attempt)
        if summary is not None and not self._summary_fits(summary, overhead, reserve):
            # A dense handoff can pass the byte gate and still not fit below the threshold: the next request would
            # compact again at once. The fallback summary has a smaller limit.
            logger.warning("warm_compaction: the warm summary does not fit in the room; fallback used")
            record.update(path=None, reason="summary_too_large")
            summary = None
        # The middles that the tail cuts (see layout.bound_tail): the rows before the tail do not have them.
        removed: list = []
        if summary is None and not self._cancelled() and self._attempt() == attempt:
            layout.bound_tail(messages[start:], self._tail_cap(SUMMARY_RESERVE, overhead, reserve), removed, policy)
            # Only the rows before the tail: the tail stays as it is, and a transcript of the whole history
            # can spend its budget on the tail.
            summary, _tokens = fallback.llm_summary(self._llm, [*messages[:start], *removed], prefixes,
                                                    focus_topic=focus_topic, memory_context=memory, task=self._task)
            if summary is not None and (estimate_tokens(summary) > SUMMARY_RESERVE
                                        or not self._summary_fits(summary, overhead, reserve)):
                # A dense summary (CJK, for example) above the reserve: the tail would cut more than the
                # transcript had. Or above the room (a large system prompt and a low threshold): the next request
                # would compact again at once. The fixed summary quotes what the final tail cuts, in the room.
                logger.warning("warm_compaction: the fallback summary is above the reserve; fixed summary used")
                summary = None
            if summary is not None:
                record["path"] = "fallback"
        if self._cancelled():
            record.update(path="cancelled", reason=record["reason"] or "cancelled")
            self._finish(record, started)
            return messages
        if self._attempt() != attempt:
            # A model or session switch during the attempt (while a request was on the network, for example): the
            # summary is of the old route or session. Hermes keeps the history.
            record.update(path="cancelled", reason="route_changed")
            self._finish(record, started)
            return messages
        # The fixed summary takes at most the reserve, and at most the room.
        fixed_budget = self._fixed_budget(overhead, reserve)
        if summary is None:
            summary = fallback.fixed_summary([*messages[:start], *removed], prefixes, focus_topic, memory, fixed_budget)
            record["path"] = "fixed"
        # The prepended user row must be in the tail: when it does not fit in the room, keep its start and end.
        # With an unknown overhead, the room is unknown too: the row keeps only its minimum. The room is after the
        # tail as the cap cuts it: an uncut large tool result would leave no room.
        bounded = layout.bound_tail(messages[start:], self._tail_cap(estimate_tokens(summary), overhead, reserve),
                                    policy=policy)
        room = self._room(bounded, summary, overhead or 0, reserve, policy)
        cut: list = []
        if prepend is not None and room is not None:
            allowed = room if overhead is not None else 0
            if sent_tokens(prepend, policy) > allowed:
                if record["path"] == "warm":
                    # The warm request had the whole row.
                    prepend = layout.fit_user_row(prepend, allowed)
                elif record["path"] == "fallback":
                    # The fallback transcript had only the start and end of a long row: the middle that this cut
                    # removes goes after the summary as a quote. The room keeps space for that quote.
                    prepend = layout.fit_user_row(prepend, allowed - fallback.CUT_QUOTE_CHARS // 4, cut)
                    block = fallback.cut_quote(cut, fallback.CUT_QUOTE_CHARS // 4)
                    if block:
                        summary = summary.rstrip() + "\n\n" + block
                else:
                    # The row is not copied and the fixed summary does not have it: its cut middle goes into the
                    # summary as a quote. The room keeps space for that quote.
                    prepend = layout.fit_user_row(prepend, allowed - fallback.CUT_QUOTE_CHARS // 4, cut)
                    summary = fallback.fixed_summary([*messages[:start], *removed, *cut], prefixes, focus_topic, memory,
                                                    fixed_budget)
        prepend_tokens = sent_tokens(prepend, policy) if prepend is not None else 0
        tail_tokens = self._tail_cap(estimate_tokens(summary), overhead, reserve, prepend_tokens)
        if record["path"] == "fixed":
            # The fixed summary quotes what the tail cuts: cut and quote at the same cap. A larger summary makes a
            # smaller cap, and the cap only goes down. When the rounds stop before the cap is stable, the tail is
            # cut at the cap that the summary quotes: a little above the room, but no cut text is lost.
            for _round in range(FIXED_ROUNDS):
                final: list = []
                layout.bound_tail(messages[start:], tail_tokens, final, policy)
                if final != removed:
                    removed = final
                    summary = fallback.fixed_summary([*messages[:start], *removed, *cut], prefixes, focus_topic, memory,
                                                    fixed_budget)
                lower = self._tail_cap(estimate_tokens(summary), overhead, reserve, prepend_tokens)
                if lower >= tail_tokens or _round == FIXED_ROUNDS - 1:
                    break
                tail_tokens = lower
        # The copies get the room after the tail as build cuts it: the uncut tail can be much larger.
        copy_tokens = 0 if overhead is None else self._copy_tokens(
            layout.bound_tail(messages[start:], tail_tokens, policy=policy), summary, overhead, reserve, policy)
        new = layout.build(
            messages, summary, start=start, prepend=prepend, copy_chars=int(self._settings["user_copy_chars"]),
            header_prefix=hermes_value("agent.context_compressor", "SUMMARY_PREFIX", handoff.LEGACY_PREFIX),
            prefixes=prefixes,
            end_marker=hermes_value("agent.context_compressor", "_SUMMARY_END_MARKER", HERMES_END_MARKER),
            marker=hermes_value("agent.context_compressor", "_DB_PERSISTED_MARKER", HERMES_DB_MARKER),
            copy_tokens=copy_tokens, tail_tokens=tail_tokens, policy=policy)
        # Again before the result is used: a switch or a cancel while the new history was built also stops it.
        if self._cancelled() or self._attempt() != attempt:
            record.update(path="cancelled", reason="cancelled" if self._cancelled() else "route_changed")
            self._finish(record, started)
            return messages
        self.compression_count += 1
        self._finish(record, started)
        return new

    def _policy(self, messages: list) -> SendPolicy:
        """The route-dependent fields that warm.wire_row sends: reasoning_details on a route that replays them,
        the tool-call thought signature for a model that reads it, and reasoning_content on a route that needs it
        back. Hermes does not give that last rule to a context engine: a capture of this route shows it (Hermes
        sends the field on every assistant row, or on none); without one, a stored row that has the field."""
        # Only a usable capture (_budget_capture: this route, and the stored rows) shows what the route sends.
        capture = self._budget_capture(self._store.latest(self._wc_session_id), messages)
        body = capture.get("body") if capture else None
        sent = [row for row in (body.get("messages") or []) if isinstance(row, dict) and row.get("role") == "assistant"
                ] if isinstance(body, dict) else []
        if sent:
            echo = known = any("reasoning_content" in row for row in sent)
        elif capture:
            # A first request has no assistant row: the rows after it (its reply first) are of this route.
            echo = known = warm.needs_reasoning_echo({}, messages[len(capture["digests"]):])
        else:
            echo, known = warm.needs_reasoning_echo({}, messages), False
        module = "agent.transports.chat_completions"
        return SendPolicy(
            # Without a capture, stored reasoning can be of an earlier route: count it, but do not cut it.
            echo=echo, cut_reasoning=known,
            details=bool(hermes_value(module, "_route_replays_reasoning_details",
                                      warm._route_replays_reasoning_details)(self._wc_route[1])),
            signatures=bool(hermes_value(module, "_model_consumes_thought_signature",
                                         warm._model_consumes_thought_signature)(self._wc_route[0])))

    def _start_tail(self, messages: list, capture: dict[str, Any] | None) -> int:
        """The tail budget for the tail start: the tail setting, and at most the room after the overhead of a
        usable capture, the reply reserve, and a summary (_tail_cap). A large system prompt or tool schema can
        leave no room for a history that fits the tail setting."""
        budget = self._budget_capture(capture, messages)
        return self._tail_cap(SUMMARY_RESERVE, request_overhead(budget, messages),
                              request_reserve(budget, int(self.context_length or 0)))

    def _attempt(self) -> tuple:
        """(route, api key, provider, session id) now."""
        return tuple(self._wc_route), self._wc_api_key, self._wc_provider, self._wc_session_id

    def _warm_summary(self, messages: list, capture: dict[str, Any] | None, focus_topic: str | None, memory: str,
                      prefixes: tuple[str, ...], record: dict[str, Any], attempt: tuple) -> str | None:
        try:
            if not self._settings["warm"]:
                raise warm.WarmRefusal("disabled")
            if capture is None:
                raise warm.WarmRefusal("no_capture")
            instruction = handoff.build_instruction(focus_topic, memory)
            body = warm.build_request(capture, messages, attempt[0], self.context_length, instruction)
            if self._cancelled():
                raise warm.WarmRefusal("cancelled")
            reply = self._execute(body, instruction, len(capture["body"]["messages"]), attempt,
                                  capture.get("prompt_tokens"))
            record.update(prompt_tokens=reply["prompt_tokens"], cached_tokens=reply["cached_tokens"])
            text, reason = handoff.gate(reply, prefixes)
            if text is None:
                raise warm.WarmRefusal(f"gate:{reason}")
        except warm.WarmRefusal as refusal:
            record["reason"] = refusal.code
            return None
        except Exception as error:
            record["reason"] = f"error:{type(error).__name__}"
            return None
        record.update(path="warm", reason="accepted")
        return text

    def _execute(self, body: dict[str, Any], instruction: str, captured: int, attempt: tuple,
                 measured: int | None = None) -> dict[str, Any]:
        """Send the warm request through the Hermes llm_request and llm_execution middleware, as Hermes sends a
        main request. A request middleware can change the request, for example to redact the new rows. An
        execution middleware can audit, block, or replace the request. A block, a rewrite, or a replaced reply
        stops the warm request; the fallback summary then runs. The first captured messages of the body are
        the captured request, which already went through the request middleware."""
        try:
            from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        except Exception as error:
            raise warm.WarmRefusal("middleware_unavailable") from error
        # Before any middleware sees the request: a route without its headers does not send it.
        route, api_key, provider, session_id = attempt
        headers = warm.route_headers(api_key, route[1], provider)
        tls = warm.route_tls(route[1])
        context = {"purpose": NAME, "api_request_id": None, "session_id": session_id,
                   "model": route[0], "base_url": route[1], "api_mode": route[2]}
        try:
            # A copy: a host chain that passes the request itself to a middleware that changes it in place
            # would change the body that the checks below compare with (and the stored capture).
            changed = apply_llm_request_middleware(copy.deepcopy(body), **context).payload
        except Exception as error:
            raise warm.WarmRefusal("middleware_refused") from error
        if not isinstance(changed, dict) or not isinstance(changed.get("messages"), list):
            raise warm.WarmRefusal("middleware_refused")
        # The captured part went through the request middleware already. A middleware that changes it again (for
        # example, adds a system row) would apply twice and change the cached prefix. It can change the new rows.
        if ({k: v for k, v in changed.items() if k != "messages"} != {k: v for k, v in body.items() if k != "messages"}
                or changed["messages"][:captured] != body["messages"][:captured]
                # The host instruction must stay the last block of the last row: without it, the reply is not a
                # handoff. The user text in front of it (a trailing user row) can change, as the other new rows.
                or not changed["messages"] or not warm.ends_with_instruction(changed["messages"][-1], instruction)):
            raise warm.WarmRefusal("middleware_rewrite")
        body = changed
        # A request middleware can add text to the new rows. Check the size again before the request is sent.
        if not warm.fits(body, int(self.context_length or 0), measured, captured):
            raise warm.WarmRefusal("capacity")
        # A copy that no middleware can change in place.
        base = copy.deepcopy(body)
        sent: list[tuple[dict[str, Any], dict[str, Any]]] = []
        started: list[bool] = []
        repeated: list[bool] = []

        def terminal(request: Any) -> dict[str, Any]:
            # The capture keeps the body before the execution middleware. An execution middleware that rewrites
            # this request can also have rewritten the captured request, so the warm request is not sent.
            if request != base:
                raise warm.WarmRefusal("middleware_rewrite")
            if started:
                # One request only: a middleware that calls next_call again does not send it again (also after a
                # failed send: the provider can have the request), and the attempt stops even when the middleware
                # catches this refusal.
                repeated.append(True)
                raise warm.WarmRefusal("middleware_repeated")
            # The last check before the provider: an execution middleware can run after the host gave up on the
            # attempt, or after a model or session switch.
            if self._cancelled():
                raise warm.WarmRefusal("cancelled")
            if self._attempt() != attempt:
                raise warm.WarmRefusal("route_changed")
            started.append(True)
            result = warm.send(base, route[1], api_key, post=self._post, extra_headers=headers, ssl_context=tls)
            # The send blocks on the network: a switch can occur before it returns.
            if self._attempt() != attempt:
                raise warm.WarmRefusal("route_changed")
            sent.append((result, dict(result)))
            return result
        try:
            reply = run_llm_execution_middleware(body, terminal, original_request=base, **context)
        except warm.WarmRefusal:
            raise
        except Exception as error:
            raise warm.WarmRefusal("middleware_refused") from error
        if repeated:
            raise warm.WarmRefusal("middleware_repeated")
        # Only the reply of this request: a middleware that skipped the request or changed its reply stops it.
        if not sent or reply is not sent[0][0] or reply != sent[0][1]:
            raise warm.WarmRefusal("middleware_changed_reply")
        return reply

    def _prefixes(self) -> tuple[str, ...]:
        current = hermes_value("agent.context_compressor", "SUMMARY_PREFIX", handoff.LEGACY_PREFIX)
        historical = hermes_value("agent.context_compressor", "_HISTORICAL_SUMMARY_PREFIXES", ())
        values = (current, handoff.LEGACY_PREFIX, *historical)
        return tuple(dict.fromkeys(value for value in values if isinstance(value, str) and value))

    def _tail_tokens(self) -> int:
        return tail_budget(int(self._settings["tail_tokens"]), int(self.context_length or 0),
                           int(self.threshold_tokens or 0))

    def _room(self, tail_rows: list, summary: str, overhead: int = 0, reserve: int = 0,
              policy: SendPolicy = SendPolicy()) -> int | None:
        """Tokens that stay free below the compaction threshold, and below the context window less the reply
        reserve, after the request overhead (system rows and tool schemas), the tail rows, the summary, and the
        summary row headings. None without a threshold."""
        threshold = int(self.threshold_tokens or 0)
        if threshold <= 0:
            return None
        window = int(self.context_length or 0)
        limit = min(threshold, window - reserve) if window > 0 else threshold
        return limit - (overhead + estimate_tokens(sent_rows(tail_rows, policy)) + estimate_tokens(summary)
                        + CARRIER_TOKENS)

    def _fixed_budget(self, overhead: int | None, reserve: int) -> int:
        """The token budget of the fixed summary: the summary reserve, and at most the free room (half of it with
        an unknown overhead)."""
        free = self._room([], "", overhead or 0, reserve)
        if free is None:
            return SUMMARY_RESERVE
        if overhead is None:
            free //= 2
        # Not above a small room: the headings take less, and the quotes then have no share.
        return max(0, min(SUMMARY_RESERVE, free))

    def _summary_fits(self, summary: str, overhead: int | None, reserve: int) -> bool:
        """True when the summary row fits in the free room (as _tail_cap: half of it with an unknown overhead)."""
        free = self._room([], "", overhead or 0, reserve)
        if free is None:
            return True
        if overhead is None:
            free //= 2
        return estimate_tokens(summary) <= free

    def _tail_cap(self, summary_tokens: int, overhead: int | None, reserve: int, prepend_tokens: int = 0) -> int:
        """The tail budget, and at most the free room for the tail after the overhead, the summary, and the
        prepended row. With an unknown overhead, the tail takes at most half of that room: the other half is for
        the system rows and the tool schemas that the next request also sends."""
        tail = self._tail_tokens()
        free = self._room([], "", overhead or 0, reserve)
        if free is None:
            return tail
        free -= summary_tokens + prepend_tokens
        if overhead is None:
            free //= 2
        return max(0, min(tail, free))

    def _budget_capture(self, capture: dict[str, Any] | None, messages: list) -> dict[str, Any] | None:
        """The capture for the request budget, or None. Its body must be of this route, and it must be the system
        rows followed by the stored rows (the warm path checks): otherwise, the count of its system rows is not
        known."""
        if capture is None or tuple(capture.get("route") or ()) != tuple(self._wc_route):
            return None
        body = capture.get("body")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return None
        try:
            warm.split_history(capture, messages)
            warm.check_source(body, messages[: len(capture["digests"])], self._wc_route[1])
        except Exception:
            return None
        return capture

    def _copy_tokens(self, tail_rows: list, summary: str, overhead: int = 0, reserve: int = 0,
                     policy: SendPolicy = SendPolicy()) -> int:
        """Token allowance for the copied user messages and the prepended user row: at most the tail size, and
        the free room (_room)."""
        tail = self._tail_tokens()
        room = self._room(tail_rows, summary, overhead, reserve, policy)
        return tail if room is None else max(0, min(tail, room))

    def _cancelled(self) -> bool:
        check = getattr(self, "_compression_cancelled_check", None)
        if not callable(check):
            return False
        try:
            return bool(check())
        except Exception:
            return False

    def _finish(self, record: dict[str, Any], started: float) -> None:
        record["elapsed_s"] = round(self._clock() - started, 3)
        self.warm_last = record
        logger.info("warm_compaction: path=%s reason=%s elapsed_s=%s prompt_tokens=%s cached_tokens=%s",
                    record["path"], record["reason"], record["elapsed_s"], record["prompt_tokens"],
                    record["cached_tokens"])
