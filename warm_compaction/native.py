"""Check for native warm handoff without a version or merge-status check."""

from __future__ import annotations

import importlib
import inspect

NOTICE = (
    'Hermes includes native warm handoff. The warm_compaction plugin remains active. '
    'To switch, set context.engine: compressor and compression.warm_handoff: "on", then restart Hermes. '
    'The native mode keeps the built-in history policy; retained history can differ from the plugin. '
    'For auto mode and details, see https://github.com/Elevatormusic/hermes-warm-compaction#native-warm-handoff.'
)


def native_available() -> bool:
    """Return true only when the constructor and default config support warm handoff."""
    try:
        compressor = importlib.import_module("agent.context_compressor").ContextCompressor
        parameter = inspect.signature(compressor.__init__).parameters.get("warm_handoff")
        defaults = importlib.import_module("hermes_cli.config").DEFAULT_CONFIG
        value = defaults.get("compression", {}).get("warm_handoff")
        return (parameter is not None
                and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
                and value in ("off", "on", "auto"))
    except Exception:
        return False
