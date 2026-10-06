"""Stand-ins for the Hermes modules that the warm_compaction plugin imports. Not collected as tests."""

from __future__ import annotations

import abc
import copy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SUMMARY_PREFIX = "[HERMES PREFIX] Earlier turns were compacted."
END_MARKER = "--- END OF CONTEXT SUMMARY - stand-in marker ---"


class StubContextEngine(abc.ABC):
    """The ContextEngine rules of Hermes 45871e10 that the plugin uses."""

    last_prompt_tokens = 0
    last_completion_tokens = 0
    last_total_tokens = 0
    threshold_tokens = 0
    context_length = 0
    compression_count = 0
    threshold_percent = 0.75

    @property
    @abc.abstractmethod
    def name(self):
        """Engine name."""

    @abc.abstractmethod
    def update_from_response(self, usage):
        """Keep the token counts."""

    @abc.abstractmethod
    def should_compress(self, prompt_tokens=None):
        """Return the trigger state."""

    @abc.abstractmethod
    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        """Return the new history."""

    def should_compress_preflight(self, messages):
        return False

    def has_content_to_compress(self, messages):
        return True

    def on_session_start(self, session_id, **kwargs):
        return None

    def on_session_reset(self):
        self.last_prompt_tokens = self.last_completion_tokens = self.last_total_tokens = 0
        self.compression_count = 0

    def get_status(self):
        last = max(self.last_prompt_tokens, 0)
        return {"last_prompt_tokens": last, "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length, "usage_percent": 0,
                "compression_count": self.compression_count}

    def clone_for_agent(self):
        return copy.deepcopy(self)

    def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
        self.context_length = context_length
        from agent.context_compressor import resolve_model_threshold
        if not hasattr(self, "_config_threshold_percent"):
            self._config_threshold_percent = self.threshold_percent
        self.threshold_percent = resolve_model_threshold(
            model, getattr(self, "model_thresholds", {}), self._config_threshold_percent, provider)
        self.threshold_tokens = int(context_length * self.threshold_percent)


def resolve_model_threshold(model, model_thresholds, default, provider=""):
    """Longest matching key wins, else the default."""
    matches = [key for key in (model_thresholds or {}) if model and key in model]
    return float(model_thresholds[max(matches, key=len)]) if matches else default


def sanitize_memory_context(text):
    return text.strip()


def _module(name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    return module


AGENT = _module("agent")
CONTEXT_ENGINE = _module("agent.context_engine", ContextEngine=StubContextEngine,
                         sanitize_memory_context=sanitize_memory_context)
CONTEXT_COMPRESSOR = _module(
    "agent.context_compressor", SUMMARY_PREFIX=SUMMARY_PREFIX, _HISTORICAL_SUMMARY_PREFIXES=("[OLD PREFIX]",),
    _SUMMARY_END_MARKER=END_MARKER, _DB_PERSISTED_MARKER="_db_persisted",
    resolve_model_threshold=resolve_model_threshold)
AGENT.context_engine = CONTEXT_ENGINE
AGENT.context_compressor = CONTEXT_COMPRESSOR
# The sources of the default headers of the Hermes OpenAI client, in the order of agent.agent_init: a host
# factory or the provider profile, then model.default_headers, then providers.<name>.extra_headers.
HOST_HEADERS: dict = {}
PROFILE_HEADERS: dict = {}
# Other fields of the provider profile (native_reasoning_details_type, for example).
PROFILE_FIELDS: dict = {}
USER_HEADERS: dict = {}
CUSTOM_HEADERS: dict = {}
# The value that agent.ssl_verify.resolve_httpx_verify gives (True, False, or an SSL context); an exception
# in it is raised.
TLS_VERIFY: list = []


def resolve_httpx_verify(*, ca_bundle=None, ssl_verify=None, base_url=""):
    value = TLS_VERIFY[0] if TLS_VERIFY else True
    if isinstance(value, Exception):
        raise value
    return value


SSL_VERIFY = _module("agent.ssl_verify", resolve_httpx_verify=resolve_httpx_verify)
AGENT.ssl_verify = SSL_VERIFY
AGENT_INIT = _module("agent.agent_init", _host_default_headers_factory=lambda base_url: (
    (lambda api_key, base: dict(HOST_HEADERS)) if HOST_HEADERS else None))
AUXILIARY_CLIENT = _module("agent.auxiliary_client", _apply_user_default_headers=lambda headers: (
    {**(headers or {}), **USER_HEADERS} if USER_HEADERS else headers))
PROVIDERS = _module("providers", get_provider_profile=lambda name: (
    SimpleNamespace(default_headers=dict(PROFILE_HEADERS), **PROFILE_FIELDS)
    if PROFILE_HEADERS or PROFILE_FIELDS else None))
AGENT.agent_init = AGENT_INIT
AGENT.auxiliary_client = AUXILIARY_CLIENT
HERMES_CLI = _module("hermes_cli")
PLUGINS = _module("hermes_cli.plugins", VALID_HOOKS={
    "pre_api_request", "post_api_request", "on_session_start", "on_session_end", "on_session_finalize",
    "on_session_reset"})
EXECUTION_MIDDLEWARE = []
REQUEST_MIDDLEWARE = []
# The llm_execution callbacks in the order that the Hermes plugin manager keeps them. A test that runs a capture
# store adds its on_llm_execution here, as the plugin registration does.
CAPTURE_CHAIN = []
PLUGINS._delivery_manager = lambda: SimpleNamespace(_middleware={"llm_execution": list(CAPTURE_CHAIN)})


def apply_llm_request_middleware(request, **context):
    """The llm_request chain of Hermes 45871e10: a middleware can return {"request": {...}} to replace it."""
    current = copy.deepcopy(request)
    trace = []
    for middleware in REQUEST_MIDDLEWARE:
        result = middleware(request=copy.deepcopy(current), original_request=request, **context)
        if isinstance(result, dict) and isinstance(result.get("request"), dict):
            current = copy.deepcopy(result["request"])
            trace.append({"source": "plugin"})
    return SimpleNamespace(payload=current if trace else request, original_payload=request, changed=bool(trace),
                           trace=trace)


def run_llm_execution_middleware(request, next_call, **context):
    """The llm_execution chain of Hermes 45871e10: a callback that raises before next_call is skipped."""
    def call_at(index, payload):
        if index >= len(EXECUTION_MIDDLEWARE):
            return next_call(payload)
        called = []

        def downstream(new=None):
            called.append(True)
            return call_at(index + 1, payload if new is None else new)
        try:
            return EXECUTION_MIDDLEWARE[index](request=payload, next_call=downstream, **context)
        except Exception:
            if called:
                raise
            return call_at(index + 1, payload)
    return call_at(0, request)


MIDDLEWARE = _module("hermes_cli.middleware", VALID_MIDDLEWARE={
    "tool_request", "tool_execution", "llm_request", "llm_execution"},
    run_llm_execution_middleware=run_llm_execution_middleware,
    apply_llm_request_middleware=apply_llm_request_middleware)
CONFIG_PROVIDERS = _module("hermes_cli.config_providers",
                           get_custom_provider_extra_headers=lambda base_url, *args, **kwargs: dict(CUSTOM_HEADERS),
                           get_custom_provider_tls_settings=lambda base_url, *args, **kwargs: {})
HERMES_CLI.config_providers = CONFIG_PROVIDERS
HERMES_CLI.plugins = PLUGINS
HERMES_CLI.middleware = MIDDLEWARE
MODULES = {
    "agent": AGENT, "agent.context_engine": CONTEXT_ENGINE, "agent.context_compressor": CONTEXT_COMPRESSOR,
    "hermes_cli": HERMES_CLI, "hermes_cli.plugins": PLUGINS, "hermes_cli.middleware": MIDDLEWARE,
    "agent.agent_init": AGENT_INIT, "agent.auxiliary_client": AUXILIARY_CLIENT, "providers": PROVIDERS,
    "hermes_cli.config_providers": CONFIG_PROVIDERS, "agent.ssl_verify": SSL_VERIFY,
}


def install(test_case):
    """Install the stand-in modules for one test, and remove them after the test."""
    patcher = patch.dict(sys.modules, MODULES)
    patcher.start()
    test_case.addCleanup(patcher.stop)
    test_case.addCleanup(EXECUTION_MIDDLEWARE.clear)
    test_case.addCleanup(REQUEST_MIDDLEWARE.clear)
    test_case.addCleanup(CAPTURE_CHAIN.clear)
    for headers in (HOST_HEADERS, PROFILE_HEADERS, PROFILE_FIELDS, USER_HEADERS, CUSTOM_HEADERS, TLS_VERIFY):
        test_case.addCleanup(headers.clear)
