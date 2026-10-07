"""Native capability and one-time notice tests with stand-in Hermes modules."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import wc_hermes_stub


class NativeCapabilityTest(unittest.TestCase):
    def probe(self, constructor, defaults):
        from warm_compaction.native import native_available
        modules = {"agent.context_compressor": SimpleNamespace(ContextCompressor=constructor),
                   "hermes_cli.config": SimpleNamespace(DEFAULT_CONFIG=defaults)}
        with patch("warm_compaction.native.importlib.import_module", side_effect=modules.__getitem__):
            return native_available()

    def test_explicit_native_setting_and_constructor_are_required(self):
        class Native:
            def __init__(self, *, warm_handoff="off"):
                pass

        class Old:
            def __init__(self, **kwargs):
                pass

        defaults = {"compression": {"warm_handoff": "off"}}
        self.assertTrue(self.probe(Native, defaults))
        self.assertFalse(self.probe(Old, defaults))
        self.assertFalse(self.probe(Native, {"compression": {}}))
        self.assertFalse(self.probe(Native, {"compression": {"warm_handoff": "unknown"}}))

    def test_import_failure_is_quiet(self):
        from warm_compaction.native import native_available
        with patch("warm_compaction.native.importlib.import_module", side_effect=ImportError):
            self.assertFalse(native_available())


class NativeNoticeTest(unittest.TestCase):
    def setUp(self):
        wc_hermes_stub.install(self)
        from warm_compaction.engine import WarmCompactionEngine
        self.engine_class = WarmCompactionEngine
        self.probe = patch("warm_compaction.engine.native.native_available", return_value=True)
        self.probe.start()
        self.addCleanup(self.probe.stop)
        self.engine = self.engine_class()

    def status(self, engine=None):
        return (engine or self.engine).get_automatic_compaction_status_message(
            phase="compress", default_message="Normal compaction status")

    def test_notice_once_and_not_reset_by_session_or_recovery(self):
        from warm_compaction.native import NOTICE
        self.assertEqual(self.status(), NOTICE)
        self.engine.on_session_reset()
        self.engine.on_session_start("new", boundary_reason="compression", old_session_id="old")
        self.engine._clear_warm_failures()
        self.assertEqual(self.status(), "Normal compaction status")

    def test_suppressed_status_keeps_the_notice(self):
        from warm_compaction.native import NOTICE
        self.engine.emit_automatic_compaction_status = False
        self.assertIsNone(self.status())
        self.engine.emit_automatic_compaction_status = True
        self.assertEqual(self.status(), NOTICE)

    def test_failure_notice_has_priority_and_native_notice_stays_pending(self):
        from warm_compaction.native import NOTICE
        self.engine._warm_notice = "Warm request failed; fallback summary used"
        self.engine.emit_automatic_compaction_status = False
        self.assertIsNone(self.status())
        self.engine.emit_automatic_compaction_status = True
        self.assertEqual(self.status(), "Warm request failed; fallback summary used")
        self.assertEqual(self.status(), NOTICE)
        self.assertEqual(self.status(), "Normal compaction status")

    def test_clone_has_its_own_notice_state(self):
        from warm_compaction.native import NOTICE
        self.assertEqual(self.status(), NOTICE)
        clone = self.engine.clone_for_agent()
        self.assertEqual(self.status(clone), NOTICE)
        self.assertEqual(self.status(), "Normal compaction status")

    def test_old_host_keeps_normal_status(self):
        with patch("warm_compaction.engine.native.native_available", return_value=False):
            engine = self.engine_class()
        self.assertEqual(self.status(engine), "Normal compaction status")

    def test_manual_compaction_logs_once_after_it_has_content(self):
        from warm_compaction.native import NOTICE
        from wc_fixtures import assistant, user
        rows = [user("synthetic old request " + "x" * 500), assistant("synthetic old answer " + "y" * 500),
                user("synthetic latest request")]
        self.engine.update_model("synthetic-model", 200_000)
        self.engine._settings["tail_tokens"] = 20
        with patch("warm_compaction.engine.logger.info") as log:
            self.engine.compress([])
            log.assert_not_called()
            self.engine.compress(rows)
            self.engine.compress(rows)
        self.assertEqual(sum(call.args == (NOTICE,) for call in log.call_args_list), 1)
        self.assertEqual(self.status(), NOTICE)
