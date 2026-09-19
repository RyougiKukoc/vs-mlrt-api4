from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile
import importlib.util

spec = importlib.util.spec_from_file_location("linux_payload_verify", Path(__file__).with_name("verify_linux_staged_payload.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
verify = module.verify


class LinuxStagedPayloadTests(unittest.TestCase):
    def write_zip(self, root: Path, name: str, members: list[str]) -> None:
        with zipfile.ZipFile(root / name, "w") as archive:
            for member in members:
                content = b"payload"
                if member.endswith("manifest.vs"):
                    content = b"[VapourSynth Manifest V1]\nvsncnn\nvsov\n" if "generic" in name else b"[VapourSynth Manifest V1]\nvstrt\n"
                archive.writestr(member, content)

    def test_generic_rejects_cuda_member(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_zip(root, "vs-mlrt-linux-x64-generic.zip", ["vsmlrt/vsncnn.so", "vsmlrt/vsov.so", "vsmlrt/manifest.vs", "vsmlrt/vsmlrt-cuda/cudart.so"])
            with self.assertRaisesRegex(RuntimeError, "CUDA files"):
                verify("generic", root)

    def test_cuda_requires_builder_and_rtx_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_zip(root, "vs-mlrt-linux-x64-generic.zip", ["vsmlrt/vsncnn.so", "vsmlrt/vsov.so", "vsmlrt/manifest.vs"])
            for name, members in {
                "vs-mlrt-linux-x64-tensorrt-cu129.zip": ["vsmlrt/vstrt.so", "vsmlrt/manifest.vs", "vsmlrt/libnvinfer.so", "vsmlrt/libnvinfer_plugin.so", "vsmlrt/libnvonnxparser.so"],
                "vs-mlrt-linux-x64-cuda-cu129.zip": ["vsmlrt/libcudart.so", "vsmlrt/libcublas.so"],
                "vs-mlrt-linux-x64-cudnn-cu129.zip": ["vsmlrt/libcudnn.so"],
                "vs-mlrt-linux-x64-tensorrt-builder-cu129.zip": ["vsmlrt/vsmlrt-cuda/trtexec", "vsmlrt/vsmlrt-cuda/trtexec-build.json", "vsmlrt/libnvinfer_builder_resource.so"],
            }.items():
                self.write_zip(root, name, members)
            with self.assertRaisesRegex(RuntimeError, "RTX"):
                verify("cu129", root)


if __name__ == "__main__":
    unittest.main()
