"""Cross-platform path and child-process environment checks for vsmlrt.py."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vsmlrt.py"


class FakeVapourSynth(types.ModuleType):
    def __getattr__(self, name: str) -> object:
        return object


def load_vsmlrt() -> types.ModuleType:
    fake_vs = FakeVapourSynth("vapoursynth")
    fake_vs.core = types.SimpleNamespace()
    fake_dll_paths = types.ModuleType("vsmlrt_dll_paths")
    saved_modules = {name: sys.modules.get(name) for name in ("vapoursynth", "vsmlrt_dll_paths")}
    sys.modules["vapoursynth"] = fake_vs
    sys.modules["vsmlrt_dll_paths"] = fake_dll_paths
    try:
        spec = importlib.util.spec_from_file_location("vsmlrt_paths_test_module", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous


class VsmlrtPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vsmlrt = load_vsmlrt()

    def test_linux_plugin_root_accepts_elf_plugin_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package_root = root / "vapoursynth" / "plugins" / "vsmlrt"
            package_root.mkdir(parents=True)
            (package_root / "vstrt.so").touch()
            old_file = self.vsmlrt.__file__
            self.vsmlrt.__file__ = str(root / "vsmlrt.py")
            try:
                with patch.object(self.vsmlrt.platform, "system", return_value="Linux"):
                    self.assertEqual(self.vsmlrt.get_plugins_path(), str(package_root))
            finally:
                self.vsmlrt.__file__ = old_file

    def test_payload_tools_prefer_override_then_package_then_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            override = root / "override-trtexec"
            packaged = root / "plugins" / "vsmlrt-cuda" / "trtexec"
            override.touch()
            packaged.parent.mkdir(parents=True)
            packaged.touch()
            old_plugins_path = self.vsmlrt.plugins_path
            self.vsmlrt.plugins_path = str(root / "plugins")
            try:
                with patch.dict(os.environ, {"VSMLRT_TRTEXEC_PATH": str(override)}, clear=False):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Linux"):
                        self.assertEqual(
                            self.vsmlrt._get_payload_path(
                                (), "vsmlrt-cuda", "trtexec", environment_variable="VSMLRT_TRTEXEC_PATH"
                            ),
                            str(override),
                        )
                with patch.dict(os.environ, {}, clear=True):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Linux"):
                        self.assertEqual(
                            self.vsmlrt._get_payload_path(
                                (), "vsmlrt-cuda", "trtexec", environment_variable="VSMLRT_TRTEXEC_PATH"
                            ),
                            str(packaged),
                        )
                packaged.unlink()
                with patch.dict(os.environ, {}, clear=True):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Linux"):
                        with patch.object(self.vsmlrt.shutil, "which", return_value="/usr/bin/trtexec"):
                            self.assertEqual(
                                self.vsmlrt._get_payload_path(
                                    (), "vsmlrt-cuda", "trtexec", environment_variable="VSMLRT_TRTEXEC_PATH"
                                ),
                                "/usr/bin/trtexec",
                            )
            finally:
                self.vsmlrt.plugins_path = old_plugins_path

    def test_windows_prefers_exe_and_reports_exe_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "plugins" / "vsmlrt-cuda" / "trtexec.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()
            old_plugins_path = self.vsmlrt.plugins_path
            self.vsmlrt.plugins_path = str(root / "plugins")
            try:
                with patch.dict(os.environ, {}, clear=True):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Windows"):
                        self.assertEqual(
                            self.vsmlrt._get_payload_path(
                                (), "vsmlrt-cuda", "trtexec", environment_variable="VSMLRT_TRTEXEC_PATH"
                            ),
                            str(executable),
                        )
                executable.unlink()
                with patch.dict(os.environ, {}, clear=True):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Windows"):
                        with patch.object(self.vsmlrt.shutil, "which", return_value=None):
                            self.assertTrue(
                                self.vsmlrt._get_payload_path(
                                    (), "vsmlrt-cuda", "trtexec", environment_variable="VSMLRT_TRTEXEC_PATH"
                                ).endswith("trtexec.exe")
                            )
            finally:
                self.vsmlrt.plugins_path = old_plugins_path

    def test_every_external_tool_uses_path_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            old_plugins_path = self.vsmlrt.plugins_path
            self.vsmlrt.plugins_path = temporary
            try:
                with patch.dict(os.environ, {}, clear=True):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Linux"):
                        for executable, variable, payload_dir in (
                            ("trtexec", "VSMLRT_TRTEXEC_PATH", "vsmlrt-cuda"),
                            ("migraphx-driver", "VSMLRT_MIGRAPHX_DRIVER_PATH", "vsmlrt-hip"),
                            ("tensorrt_rtx", "VSMLRT_TENSORRT_RTX_PATH", "vsmlrt-cuda"),
                        ):
                            with patch.object(self.vsmlrt.shutil, "which", return_value=f"/usr/bin/{executable}"):
                                self.assertEqual(
                                    self.vsmlrt._get_payload_path(
                                        (), payload_dir, executable, environment_variable=variable
                                    ),
                                    f"/usr/bin/{executable}",
                                )
            finally:
                self.vsmlrt.plugins_path = old_plugins_path

    def test_child_environment_preserves_host_and_sets_tool_library_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_root = root / "plugins"
            executable = plugin_root / "vsmlrt-cuda" / "trtexec"
            executable.parent.mkdir(parents=True)
            executable.touch()
            old_plugins_path = self.vsmlrt.plugins_path
            self.vsmlrt.plugins_path = str(plugin_root)
            try:
                with patch.dict(os.environ, {"PRESERVED": "yes", "LD_LIBRARY_PATH": "/host/lib"}, clear=True):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Linux"):
                        env = self.vsmlrt._tool_environment(
                            str(executable), {"CUSTOM": "set", "CUDA_MODULE_LOADING": "EAGER"}, cuda=True
                        )
                self.assertEqual(env["PRESERVED"], "yes")
                self.assertEqual(env["CUSTOM"], "set")
                self.assertEqual(env["CUDA_MODULE_LOADING"], "EAGER")
                self.assertEqual(
                    env["LD_LIBRARY_PATH"].split(os.pathsep),
                    [str(executable.parent), str(plugin_root), "/host/lib"],
                )
            finally:
                self.vsmlrt.plugins_path = old_plugins_path

    def test_windows_child_environment_keeps_exe_and_package_dll_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_root = root / "plugins"
            executable = plugin_root / "vsmlrt-cuda" / "trtexec.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()
            old_plugins_path = self.vsmlrt.plugins_path
            self.vsmlrt.plugins_path = str(plugin_root)
            try:
                with patch.dict(os.environ, {"PATH": r"C:\Windows\System32", "CUDA_MODULE_LOADING": "EAGER"}, clear=True):
                    with patch.object(self.vsmlrt.platform, "system", return_value="Windows"):
                        with patch.object(os, "pathsep", ";"):
                            env = self.vsmlrt._tool_environment(str(executable), {}, cuda=True)
                self.assertEqual(env["CUDA_MODULE_LOADING"], "EAGER")
                self.assertEqual(
                    env["PATH"].split(";"),
                    [str(executable.parent), str(plugin_root), r"C:\Windows\System32"],
                )
            finally:
                self.vsmlrt.plugins_path = old_plugins_path

    def test_missing_tool_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "VSMLRT_TRTEXEC_PATH"):
            self.vsmlrt._require_tool("/missing/trtexec", "trtexec", "VSMLRT_TRTEXEC_PATH")


if __name__ == "__main__":
    unittest.main()
