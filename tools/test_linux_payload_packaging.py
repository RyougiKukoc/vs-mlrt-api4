from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile


class LinuxPayloadTests(unittest.TestCase):
    def test_builder_is_separate_from_runtime(self):
        script = Path(__file__).with_name("package_linux_payload.py")
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "stage"
            (stage / "vsmlrt-cuda").mkdir(parents=True)
            for name in ("vsncnn.so", "vsov.so", "vstrt.so", "libnvinfer.so.11", "libnvinfer_builder_resource.so.11"):
                (stage / name).write_bytes(name.encode())
            (stage / "vsmlrt-cuda" / "trtexec").write_bytes(b"trtexec")
            (stage / "vsmlrt-cuda" / "trtexec-build.json").write_bytes(b"{}")
            runtime = Path(temp) / "runtime.zip"
            builder = Path(temp) / "builder.zip"
            subprocess.run([sys.executable, str(script), "--stage-dir", str(stage), "--variant", "cu129", "--component", "tensorrt", "--output", str(runtime)], check=True)
            subprocess.run([sys.executable, str(script), "--stage-dir", str(stage), "--variant", "cu129", "--component", "builder", "--output", str(builder)], check=True)
            with zipfile.ZipFile(runtime) as archive:
                self.assertNotIn("vsmlrt/libnvinfer_builder_resource.so.11", archive.namelist())
            with zipfile.ZipFile(builder) as archive:
                self.assertIn("vsmlrt/libnvinfer_builder_resource.so.11", archive.namelist())
                self.assertIn("vsmlrt/vsmlrt-cuda/trtexec", archive.namelist())


if __name__ == "__main__":
    unittest.main()
