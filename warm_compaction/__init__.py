"""warm_compaction: a standalone context engine plugin for Hermes Agent."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

TASK = "warm_compaction"
HOOKS = ("pre_api_request", "post_api_request", "on_session_finalize", "on_session_reset")
MIDDLEWARE = "llm_execution"
CONTEXT_METHODS = (
    "register_context_engine", "register_hook", "register_middleware", "register_auxiliary_task", "get_config",
)
# The Hermes context engine loader gives a context without these APIs. Hermes tries that loader first and then
# uses the engine that the enabled plugin registered through the full plugin API.
PLUGIN_SYSTEM_ONLY = ("register_middleware", "register_auxiliary_task", "get_config", "llm.complete")
HOST_PREFIXES = ("hook:", "middleware:", "agent.")


def _read(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:
        return None


def missing_apis(ctx: Any) -> list[str]:
    """Return the names of the needed Hermes plugin APIs that are not available."""
    missing = [name for name in CONTEXT_METHODS if not callable(getattr(ctx, name, None))]
    try:
        llm = ctx.llm
    except Exception:
        llm = None
    if not callable(getattr(llm, "complete", None)):
        missing.append("llm.complete")
    hooks = _read("hermes_cli.plugins", "VALID_HOOKS") or ()
    missing.extend(f"hook:{hook}" for hook in HOOKS if hook not in hooks)
    if MIDDLEWARE not in (_read("hermes_cli.middleware", "VALID_MIDDLEWARE") or ()):
        missing.append(f"middleware:{MIDDLEWARE}")
    if _read("agent.context_engine", "ContextEngine") is None:
        missing.append("agent.context_engine.ContextEngine")
    return missing


def engine_loader_context(missing: list[str]) -> bool:
    """True when only the plugin-system APIs are missing, and this Hermes has the hooks and the middleware."""
    return (all(name in missing for name in PLUGIN_SYSTEM_ONLY)
            and not any(name.startswith(HOST_PREFIXES) for name in missing)
            and "register_context_engine" not in missing)


def register(ctx: Any) -> None:
    """Hermes entry point: register the fallback task, the engine, the hooks, and the middleware."""
    missing = missing_apis(ctx)
    if missing and engine_loader_context(missing):
        logger.debug("warm_compaction: this plugin context does not have these APIs: %s. The plugin registers "
                     "nothing in this context. The enabled plugin registers the engine through the full plugin "
                     "API.", ", ".join(missing))
        return
    if missing:
        logger.warning("warm_compaction: Hermes does not have these plugin APIs: %s. The plugin registers "
                       "nothing, and Hermes keeps its built-in compressor.", ", ".join(missing))
        return
    from .capture import CaptureStore
    from .engine import WarmCompactionEngine, read_settings

    task: str | None = TASK
    try:
        ctx.register_auxiliary_task(TASK, display_name="Warm compaction fallback",
                                    description="Model for the warm_compaction fallback summary.",
                                    defaults={"timeout": 120})
    except Exception as error:
        task = None
        logger.warning("warm_compaction: the auxiliary task was not registered (%s). The fallback summary uses "
                       "the main model route.", type(error).__name__)
    store = CaptureStore()
    engine = WarmCompactionEngine(store=store, llm=ctx.llm, settings=read_settings(ctx.get_config), task=task)
    if ctx.register_context_engine(engine) is None:
        logger.warning("warm_compaction: Hermes did not accept the context engine. The plugin registers no hooks.")
        return
    ctx.register_hook("pre_api_request", store.on_pre_api_request)
    ctx.register_middleware(MIDDLEWARE, store.on_llm_execution)
    ctx.register_hook("post_api_request", store.on_post_api_request)
    ctx.register_hook("on_session_finalize", store.forget)
    ctx.register_hook("on_session_reset", store.forget)


__all__ = ["engine_loader_context", "missing_apis", "register"]
