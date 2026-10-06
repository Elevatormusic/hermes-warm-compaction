"""The warm_compaction context engine."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable

from agent.context_engine import ContextEngine

from . import fallback, handoff, layout, warm
from .capture import CaptureStore
from .rows import estimate_tokens, hermes_value

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
    The automatic size is also at most half of the compaction threshold. Otherwise the tail can hold the
    whole history when compaction starts, and nothing comes before the tail."""
    if setting > 0:
        return setting
    budget = max(TAIL_MIN, min(TAIL_MAX, int(context_length * TAIL_SHARE)))
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


def request_overhead(capture: dict[str, Any] | None) -> int:
    """Estimated tokens of the request parts that are not history rows: the system rows and the tool schemas of
    the captured request. 0 without a captured request body."""
    body = (capture or {}).get("body")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return 0
    count = max(len(body["messages"]) - len(capture.get("digests") or ()), 0)
    return estimate_tokens({"messages": body["messages"][:count], "tools": body.get("tools")})


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

    @property
    def name(self) -> str:
        return NAME

    def clone_for_agent(self) -> "WarmCompactionEngine":
        """Return a new engine for one agent. The clone shares the capture store and the model access."""
        return WarmCompactionEngine(store=self._store, llm=self._llm, settings=self._settings, task=self._task,
                                    post=self._post, clock=self._clock)

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        if kwargs.get("boundary_reason") == "compression" and kwargs.get("old_session_id"):
            self._store.forget(session_id=kwargs["old_session_id"])
        self._wc_session_id = str(session_id or "")

    def update_model(self, model: str, context_length: int, base_url: str = "", api_key: Any = "",
                     provider: str = "", api_mode: str = "") -> None:
        super().update_model(model, context_length, base_url=base_url, api_key=api_key, provider=provider,
                             api_mode=api_mode)
        self._wc_route = (model, base_url, api_mode)
        self._wc_api_key = api_key

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
        return self.threshold_tokens > 0 and estimate_tokens(messages) >= self.threshold_tokens

    def has_content_to_compress(self, messages: list) -> bool:
        start, _prepend = layout.tail_start(messages, self._tail_tokens(), self._prefixes())
        return start > 0

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.warm_last = None

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status["warm_last"] = dict(self.warm_last) if self.warm_last else None
        return status

    def compress(self, messages: list, current_tokens: int | None = None, focus_topic: str | None = None,
                 force: bool = False, memory_context: str = "") -> list:
        """Replace the rows before the tail with a summary. Keep the history when no row comes before the tail."""
        started = self._clock()
        prefixes = self._prefixes()
        start, prepend = layout.tail_start(messages, self._tail_tokens(), prefixes)
        if start <= 0:
            return messages
        memory = sanitize_memory(memory_context)
        record: dict[str, Any] = {"path": None, "reason": None, "elapsed_s": None, "prompt_tokens": None,
                                  "cached_tokens": None}
        capture = self._store.latest(self._wc_session_id)
        summary = self._warm_summary(messages, capture, focus_topic, memory, prefixes, record)
        if summary is None and not self._cancelled():
            # Only the rows before the tail: the tail stays as it is, and a transcript of the whole history
            # can spend its budget on the tail.
            summary, _tokens = fallback.llm_summary(self._llm, messages[:start], prefixes, focus_topic=focus_topic,
                                                    memory_context=memory, task=self._task)
            if summary is not None:
                record["path"] = "fallback"
        if self._cancelled():
            record.update(path="cancelled", reason=record["reason"] or "cancelled")
            self._finish(record, started)
            return messages
        if summary is None:
            summary = fallback.fixed_summary(messages[:start])
            record["path"] = "fixed"
        new = layout.build(
            messages, summary, start=start, prepend=prepend, copy_chars=int(self._settings["user_copy_chars"]),
            header_prefix=hermes_value("agent.context_compressor", "SUMMARY_PREFIX", handoff.LEGACY_PREFIX),
            prefixes=prefixes,
            end_marker=hermes_value("agent.context_compressor", "_SUMMARY_END_MARKER", HERMES_END_MARKER),
            marker=hermes_value("agent.context_compressor", "_DB_PERSISTED_MARKER", HERMES_DB_MARKER),
            copy_tokens=self._copy_tokens(messages[start:], summary, request_overhead(capture)))
        self.compression_count += 1
        self._finish(record, started)
        return new

    def _warm_summary(self, messages: list, capture: dict[str, Any] | None, focus_topic: str | None, memory: str,
                      prefixes: tuple[str, ...], record: dict[str, Any]) -> str | None:
        try:
            if not self._settings["warm"]:
                raise warm.WarmRefusal("disabled")
            if capture is None:
                raise warm.WarmRefusal("no_capture")
            body = warm.build_request(capture, messages, self._wc_route, self.context_length,
                                      handoff.build_instruction(focus_topic, memory))
            if self._cancelled():
                raise warm.WarmRefusal("cancelled")
            reply = self._execute(body)
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

    def _execute(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send the warm request through the Hermes llm_request and llm_execution middleware, as Hermes sends a
        main request. A request middleware can change the request, for example to redact the new rows. An
        execution middleware can audit, block, or replace the request. A block, a rewrite, or a replaced reply
        stops the warm request; the fallback summary then runs."""
        try:
            from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
        except Exception as error:
            raise warm.WarmRefusal("middleware_unavailable") from error
        context = {"purpose": NAME, "api_request_id": None, "session_id": self._wc_session_id,
                   "model": self._wc_route[0], "base_url": self._wc_route[1], "api_mode": self._wc_route[2]}
        try:
            body = apply_llm_request_middleware(body, **context).payload
        except Exception as error:
            raise warm.WarmRefusal("middleware_refused") from error
        if not isinstance(body, dict):
            raise warm.WarmRefusal("middleware_refused")
        # A copy that no middleware can change in place.
        base = copy.deepcopy(body)

        def terminal(request: Any) -> dict[str, Any]:
            # The capture keeps the body before the execution middleware. An execution middleware that rewrites
            # this request can also have rewritten the captured request, so the warm request is not sent.
            if request != base:
                raise warm.WarmRefusal("middleware_rewrite")
            return warm.send(base, self._wc_route[1], self._wc_api_key, post=self._post)
        try:
            reply = run_llm_execution_middleware(body, terminal, original_request=base, **context)
        except warm.WarmRefusal:
            raise
        except Exception as error:
            raise warm.WarmRefusal("middleware_refused") from error
        if not isinstance(reply, dict) or not {"content", "finish_reason", "prompt_tokens"} <= reply.keys():
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

    def _copy_tokens(self, tail_rows: list, summary: str, overhead: int = 0) -> int:
        """Token allowance for the copied user messages: at most the tail size, and small enough that the request
        overhead (system rows and tool schemas), the tail rows, the summary, the summary row headings, and the
        copies fit below the compaction threshold."""
        tail = self._tail_tokens()
        threshold = int(self.threshold_tokens or 0)
        if threshold <= 0:
            return tail
        used = overhead + estimate_tokens(tail_rows) + estimate_tokens(summary) + CARRIER_TOKENS
        return max(0, min(tail, threshold - used))

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
