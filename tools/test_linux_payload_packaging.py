from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


_spec = importlib.util.spec_from_file_location(
    "payload_archive", Path(__file__).resolve().parents[1] / "packaging" / "payload_archive.py"
)
assert _spec and _spec.loader
_archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_archive)

SCRIPT = Path(__file__).with_name("package_linux_payload.py")


def build_stage(root: Path, variant: str = "cu129", *, windows_resource: bool = False) -> Path:
    stage = root / "stage"
    (stage / "vsmlrt-cuda").mkdir(parents=True)
    for name in ("vsncnn.so", "vsov.so", "vstrt.so", "libnvinfer.so.11", "libnvinfer_builder_resource_sm90.so.11"):
        (stage / name).write_bytes(name.encode())
    for name in ("trtexec", "trtexec-build.json"):
        (stage / "vsmlrt-cuda" / name).write_bytes(name.encode())
    if variant == "cu129":
        (stage / "vstrt_rtx.so").write_bytes(b"rtx")
        (stage / "vsmlrt-cuda" / "tensorrt_rtx").write_bytes(b"rtx")
    if windows_resource:
        (stage / "libnvinfer_builder_resource_win_sm90.so.11").write_bytes(b"windows")
    (stage / "manifest.vs").write_text("[VapourSynth Manifest V1]\nvsncnn\nvsov\nvstrt\n", encoding="ascii")
    return stage


def package(stage: Path, output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--stage-dir", str(stage), "--variant", "cu129", "--output", str(output), *extra],
        capture_output=True,
        text=True,
    )


class LinuxPayloadTests(unittest.TestCase):
    def test_payload_is_one_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = build_stage(root)
            output = root / "vs-mlrt-linux-x64-cu129.zip"
            package(stage, output)
            self.assertTrue(output.is_file())
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertIn("vsmlrt/vstrt.so", names)
                self.assertIn("vsmlrt/vsncnn.so", names)
                self.assertIn("vsmlrt/vsmlrt-cuda/trtexec", names)
                for info in archive.infolist():
                    self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED, info.filename)

    def test_oversized_payload_splits_into_volumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = build_stage(root)
            # Incompressible bytes, so the archive really exceeds the limit.
            (stage / "libnvinfer.so.11").write_bytes(os.urandom(200_000))
            output = root / "vs-mlrt-linux-x64-cu129.zip"
            package(stage, output, "--max-bytes", "90000")
            self.assertFalse(output.exists(), "the split archive should be replaced by its volumes")
            volumes = sorted(root.glob("vs-mlrt-linux-x64-cu129.zip.[0-9][0-9][0-9]"))
            self.assertGreater(len(volumes), 1)
            for volume in volumes:
                self.assertLess(volume.stat().st_size, 2 * 1024 ** 3)
            with _archive.open_payload(volumes) as archive:
                self.assertIn("vsmlrt/libnvinfer.so.11", archive.namelist())
                self.assertEqual(archive.read("vsmlrt/vsncnn.so"), b"vsncnn.so")

    def test_windows_builder_resource_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = build_stage(root, windows_resource=True)
            completed = package(stage, root / "payload.zip")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Windows-target builder resources", completed.stderr)

    def test_missing_required_file_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = build_stage(root, variant="generic")
            (stage / "vstrt.so").unlink()
            completed = package(stage, root / "payload.zip")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Missing required payload files", completed.stderr)


if __name__ == "__main__":
    unittest.main()
