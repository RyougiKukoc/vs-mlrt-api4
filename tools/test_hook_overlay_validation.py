from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("vsmlrt_hook", Path(__file__).parents[1] / "packaging/vsmlrt_build.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class HookOverlayTests(unittest.TestCase):
    def test_complete_variants_validate(self):
        hook = object.__new__(module.CustomBuildHook)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            variants = {
                "generic": ["vsncnn.dll", "vsov.dll"],
                "cu121": ["vsncnn.dll", "vsov.dll", "vstrt.dll"],
                "cu129": ["vsncnn.dll", "vsov.dll", "vstrt.dll", "vstrt_rtx.dll"],
            }
            for variant, plugins in variants.items():
                directory = root / variant
                directory.mkdir()
                (directory / "models").mkdir()
                for plugin in plugins:
                    (directory / plugin).write_bytes(b"plugin")
                if variant != "generic":
                    helper = directory / "vsmlrt-cuda"
                    helper.mkdir()
                    for name in ("trtexec.exe", "trtexec-build.json"):
                        (helper / name).write_bytes(b"helper")
                if variant == "cu129":
                    (directory / "vsmlrt-cuda" / "tensorrt_rtx.exe").write_bytes(b"helper")
                with patch.object(module.platform, "system", return_value="Windows"):
                    hook._validate_plugin_dir(directory, variant)

    def test_missing_builder_fails_cuda_variant(self):
        hook = object.__new__(module.CustomBuildHook)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "models").mkdir()
            for plugin in ("vsncnn.dll", "vsov.dll", "vstrt.dll"):
                (directory / plugin).write_bytes(b"plugin")
            with patch.object(module.platform, "system", return_value="Windows"):
                with self.assertRaisesRegex(RuntimeError, "builder helper"):
                    hook._validate_plugin_dir(directory, "cu121")


if __name__ == "__main__":
    unittest.main()
