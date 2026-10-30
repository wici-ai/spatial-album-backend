import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "execution_control", Path(__file__).resolve().parents[1] / "backend/execution_control.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class ExecutionControlTests(unittest.TestCase):
    def test_standalone_keeps_handler_unchanged(self):
        def handler():
            return "standalone"
        with patch.dict("os.environ", {}, clear=True):
            wrapped = control.controlled_http_task("orbit", {"/ingest"}, finish=lambda: None)(handler)
        self.assertIs(wrapped, handler)
        self.assertEqual(wrapped(), "standalone")

    def test_selected_adapter_receives_service_paths_and_completion(self):
        calls = []
        module = types.ModuleType("fixture_scheduler")
        def adapter(service, paths, *, finish):
            calls.append((service, paths, finish))
            return lambda handler: ("controlled", handler)
        module.controlled_http_task = adapter
        finish = lambda: None
        with patch.dict(sys.modules, {"fixture_scheduler": module}), patch.dict(
                "os.environ", {"WICI_REQUEST_CONTROL_MODULE": "fixture_scheduler"}):
            wrapped = control.controlled_http_task("orbit", {"/ingest"}, finish=finish)(str)
        self.assertEqual(calls, [("orbit", {"/ingest"}, finish)])
        self.assertEqual(wrapped, ("controlled", str))

    def test_selected_missing_adapter_fails_closed(self):
        with patch.dict("os.environ", {"WICI_REQUEST_CONTROL_MODULE": "missing_wici_adapter"}):
            with self.assertRaises(ModuleNotFoundError):
                control.controlled_http_task("orbit", {"/ingest"}, finish=lambda: None)

    def test_noncallable_adapter_fails_closed(self):
        module = types.ModuleType("fixture_scheduler")
        module.controlled_http_task = None
        with patch.dict(sys.modules, {"fixture_scheduler": module}), patch.dict(
                "os.environ", {"WICI_REQUEST_CONTROL_MODULE": "fixture_scheduler"}):
            with self.assertRaises(TypeError):
                control.controlled_http_task("orbit", {"/ingest"}, finish=lambda: None)


if __name__ == "__main__":
    unittest.main()
