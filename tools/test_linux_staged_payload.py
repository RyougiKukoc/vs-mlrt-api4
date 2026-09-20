from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("verify_linux_staged_payload.py")
spec = importlib.util.spec_from_file_location("linux_payload_verify", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
verify = module.verify

GENERIC = [
    "vsmlrt/vsncnn.so",
    "vsmlrt/vsov.so",
    "vsmlrt/manifest.vs",
    "vsmlrt/libopenvino.so.2460",
    "vsmlrt/libtbb.so.12",
]
CUDA = [
    "vsmlrt/vstrt.so",
    "vsmlrt/libnvinfer.so.11",
    "vsmlrt/libnvinfer_plugin.so.11",
    "vsmlrt/libnvonnxparser.so.11",
    "vsmlrt/libcudart.so.12",
    "vsmlrt/libcublas.so.12",
    "vsmlrt/libcudnn.so.9",
    "vsmlrt/libnvinfer_builder_resource_sm90.so.11",
    "vsmlrt/vsmlrt-cuda/trtexec",
    "vsmlrt/vsmlrt-cuda/trtexec-build.json",
]
CU129 = CUDA + ["vsmlrt/vstrt_rtx.so", "vsmlrt/libtensorrt_rtx.so.1", "vsmlrt/vsmlrt-cuda/tensorrt_rtx"]

MANIFESTS = {
    "generic": "[VapourSynth Manifest V1]\nvsncnn\nvsov\n",
    "cu121": "[VapourSynth Manifest V1]\nvsncnn\nvsov\nvstrt\n",
    "cu129": "[VapourSynth Manifest V1]\nvsncnn\nvsov\nvstrt\nvstrt_rtx\n",
}


class LinuxStagedPayloadTests(unittest.TestCase):
    def write_payload(self, root: Path, variant: str, members: list[str], *, compression=zipfile.ZIP_DEFLATED) -> Path:
        archive = root / f"vs-mlrt-linux-x64-{variant}.zip"
        with zipfile.ZipFile(archive, "w", compression=compression) as out:
            for member in members:
                content = MANIFESTS[variant].encode() if member.endswith("manifest.vs") else b"payload"
                out.writestr(member, content)
        return archive

    def cu129_members(self) -> list[str]:
        return [*GENERIC, *CU129]

    def test_complete_payload_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_payload(root, "cu129", self.cu129_members())
            verify("cu129", root)

    def test_generic_rejects_cuda_member(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_payload(root, "generic", [*GENERIC, "vsmlrt/vsmlrt-cuda/cudart.so"])
            with self.assertRaisesRegex(RuntimeError, "CUDA files"):
                verify("generic", root)

    def test_generic_rejects_windows_builder_resource(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_payload(root, "generic", [*GENERIC, "vsmlrt/libnvinfer_builder_resource_win_sm90.so.11"])
            with self.assertRaisesRegex(RuntimeError, "Windows-target builder resource"):
                verify("generic", root)

    def test_payload_rejects_a_library_stored_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_payload(root, "generic", [*GENERIC, "vsmlrt/libopenvino.so.2024", "vsmlrt/libopenvino.so.2024.6.0"])
            with self.assertRaisesRegex(RuntimeError, "under two names"):
                verify("generic", root)

    def test_payload_accepts_an_abi_soname_with_two_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_payload(root, "generic", [*GENERIC, "vsmlrt/libnvrtc-builtins.so.12.9"])
            verify("generic", root)

    def test_payload_rejects_stored_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_payload(root, "generic", GENERIC, compression=zipfile.ZIP_STORED)
            with self.assertRaisesRegex(RuntimeError, "stored uncompressed"):
                verify("generic", root)

    def test_cu129_requires_the_rtx_plugin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_payload(root, "cu129", [*GENERIC, *CUDA])
            with self.assertRaisesRegex(RuntimeError, "vstrt_rtx"):
                verify("cu129", root)

    def test_cu129_requires_builder_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            members = [name for name in self.cu129_members() if "builder_resource" not in name]
            self.write_payload(root, "cu129", members)
            with self.assertRaisesRegex(RuntimeError, "builder resources"):
                verify("cu129", root)

    def test_split_volumes_verify_as_one_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.write_payload(root, "cu129", self.cu129_members())
            payload = archive.read_bytes()
            archive.unlink()
            for index, start in enumerate(range(0, len(payload), 64), start=1):
                (root / f"{archive.name}.{index:03d}").write_bytes(payload[start:start + 64])
            verify("cu129", root)


if __name__ == "__main__":
    unittest.main()
