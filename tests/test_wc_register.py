"""Tests for plugin registration with a fake Hermes plugin context."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import wc_hermes_stub

ROOT = Path(__file__).resolve().parents[1]


class FakeContext:
    def __init__(self, config=None, aux_error=None, accept_engine=True):
        self.config = dict(config or {})
        self.aux_error = aux_error
        self.accept_engine = accept_engine
        self.llm = SimpleNamespace(complete=lambda *args, **kwargs: None)
        self.calls = []

    def register_auxiliary_task(self, key, *, display_name, description, defaults=None):
        if self.aux_error:
            raise self.aux_error
        self.calls.append(("task", key, defaults))

    def register_context_engine(self, engine):
        self.calls.append(("engine", engine))
        return "handle" if self.accept_engine else None

    def register_hook(self, name, callback):
        self.calls.append(("hook", name, callback))

    def register_middleware(self, kind, callback):
        self.calls.append(("middleware", kind, callback))

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class EngineLoaderContext:
    """Like the Hermes context engine loader context: register_context_engine and no-op registrations only."""

    def __init__(self):
        self.engine = None

    def _noop(self, *args, **kwargs):
        pass

    register_tool = register_hook = register_cli_command = register_memory_provider = register_command = _noop

    def register_context_engine(self, engine):
        self.engine = engine


def engine_of(ctx):
    return next(call[1] for call in ctx.calls if call[0] == "engine")


class RegisterTest(unittest.TestCase):
    def setUp(self):
        wc_hermes_stub.install(self)
        import warm_compaction
        self.plugin = warm_compaction

    def test_registers_in_order_with_one_store(self):
        ctx = FakeContext()
        self.plugin.register(ctx)
        names = [(call[0], call[1].name if call[0] == "engine" else call[1]) for call in ctx.calls]
        self.assertEqual(names, [("task", "warm_compaction"), ("engine", "warm_compaction"),
                                 ("hook", "pre_api_request"), ("middleware", "llm_execution"),
                                 ("hook", "post_api_request"), ("hook", "on_session_finalize"),
                                 ("hook", "on_session_reset")])
        engine = engine_of(ctx)
        self.assertTrue(all(call[2].__self__ is engine._store for call in ctx.calls[2:]))
        self.assertEqual((engine._task, engine._llm), ("warm_compaction", ctx.llm))
        self.assertEqual(ctx.calls[0][2], {"timeout": 120})

    def test_missing_method_registers_nothing(self):
        ctx = FakeContext()
        ctx.register_middleware = None
        with self.assertLogs("warm_compaction", level="WARNING") as logs:
            self.plugin.register(ctx)
        self.assertEqual(ctx.calls, [])
        self.assertIn("register_middleware", logs.output[0])

    def test_missing_hook_registers_nothing(self):
        ctx = FakeContext()
        with patch.object(wc_hermes_stub.PLUGINS, "VALID_HOOKS", {"pre_api_request"}):
            with self.assertLogs("warm_compaction", level="WARNING") as logs:
                self.plugin.register(ctx)
        self.assertEqual(ctx.calls, [])
        self.assertIn("hook:post_api_request", logs.output[0])

    def test_missing_llm_registers_nothing(self):
        ctx = FakeContext()
        ctx.llm = None
        with self.assertLogs("warm_compaction", level="WARNING"):
            self.plugin.register(ctx)
        self.assertEqual(ctx.calls, [])

    def test_engine_loader_context_registers_nothing_without_a_warning(self):
        # Hermes tries this loader before the plugin system. The enabled plugin then supplies the engine.
        ctx = EngineLoaderContext()
        with self.assertLogs("warm_compaction", level="DEBUG") as logs:
            self.plugin.register(ctx)
        self.assertIsNone(ctx.engine)
        self.assertEqual([record.levelname for record in logs.records], ["DEBUG"])
        self.assertIn("register_middleware", logs.output[0])

    def test_package_exports_no_engine_class(self):
        # The Hermes engine loader makes an instance of any ContextEngine subclass in the package. Such an
        # instance would have no capture store and no llm.
        self.plugin.register(FakeContext())
        base = wc_hermes_stub.StubContextEngine
        found = [name for name in dir(self.plugin)
                 if isinstance(getattr(self.plugin, name), type) and issubclass(getattr(self.plugin, name), base)]
        self.assertEqual(found, [])

    def test_task_error_uses_the_main_model_route(self):
        ctx = FakeContext(aux_error=ValueError("taken"))
        with self.assertLogs("warm_compaction", level="WARNING"):
            self.plugin.register(ctx)
        self.assertIsNone(engine_of(ctx)._task)

    def test_refused_engine_gets_no_hooks(self):
        ctx = FakeContext(accept_engine=False)
        with self.assertLogs("warm_compaction", level="WARNING"):
            self.plugin.register(ctx)
        self.assertEqual([call[0] for call in ctx.calls], ["task", "engine"])

    def test_settings_come_from_the_plugin_config(self):
        ctx = FakeContext(config={"threshold": 0.7})
        self.plugin.register(ctx)
        self.assertEqual(engine_of(ctx).threshold_percent, 0.7)

    def test_missing_auxiliary_hooks_keeps_the_base_engine(self):
        ctx = FakeContext(config={"moa_routes": [{"name": "synthetic"}]})
        with self.assertLogs("warm_compaction", level="WARNING"):
            self.plugin.register(ctx)
        self.assertIsNone(engine_of(ctx)._store._moa)
        self.assertNotIn("pre_auxiliary_call", [call[1] for call in ctx.calls if call[0] == "hook"])

    def test_moa_hooks_share_the_engine_store(self):
        route = {"name": "aggregator", "provider": "custom", "model": "fake-model",
                 "base_url": "http://127.0.0.1:9/v1", "context_length": 200000}
        ctx = FakeContext(config={"moa_routes": [route], "moa_references": True})
        with patch.object(wc_hermes_stub.PLUGINS, "VALID_HOOKS",
                          wc_hermes_stub.PLUGINS.VALID_HOOKS | set(self.plugin.MOA_HOOKS)):
            self.plugin.register(ctx)
        tracker = engine_of(ctx)._store._moa
        self.assertIsNotNone(tracker)
        for call in ctx.calls:
            if call[0] == "hook" and call[1] in self.plugin.MOA_HOOKS:
                self.assertIs(call[2].__self__, tracker)

    def test_invalid_moa_routes_keep_the_base_engine(self):
        ctx = FakeContext(config={"moa_routes": [{"name": "invalid"}]})
        with patch.object(wc_hermes_stub.PLUGINS, "VALID_HOOKS",
                          wc_hermes_stub.PLUGINS.VALID_HOOKS | set(self.plugin.MOA_HOOKS)):
            with self.assertLogs("warm_compaction", level="WARNING"):
                self.plugin.register(ctx)
        self.assertIsNone(engine_of(ctx)._store._moa)

    def test_manifest_declares_the_settings(self):
        text = (ROOT / "warm_compaction" / "plugin.yaml").read_text(encoding="utf-8")
        for line in ("manifest_version: 2", "name: warm_compaction", "config_schema:", "  threshold:",
                     "  tail_tokens:", "  user_copy_chars:", "  warm:", "  moa_routes:", "  moa_references:"):
            self.assertIn(line + "\n", text)

    def test_the_alpha_files_are_gone(self):
        for name in ("checkpoint.py", "policy.py", "checkpoint.schema.json", "LICENSE.codex"):
            self.assertFalse((ROOT / "warm_compaction" / name).exists(), name)
        self.assertFalse((ROOT / "patches" / "hermes" / "warm-compaction-alpha.patch").exists())


if __name__ == "__main__":
    unittest.main()
