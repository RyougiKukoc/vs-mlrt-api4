"""Regression checks for the staged Linux payload's size discipline."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "linux_native_build", Path(__file__).resolve().parents[1] / "packaging" / "linux_native_build.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def minimal_shared_object(soname: str | None, needed: tuple[str, ...] = ()) -> bytes:
    """Build enough of an ELF64 to exercise the DT_SONAME/DT_NEEDED readers."""
    strings = bytearray(b"\0")
    offsets: dict[str, int] = {}

    def add(name: str) -> int:
        if name not in offsets:
            offsets[name] = len(strings)
            strings.extend(name.encode() + b"\0")
        return offsets[name]

    soname_at = add(soname) if soname else None
    needed_at = [add(name) for name in needed]
    dynamic_entries = [(5, 0), (10, 0)]  # DT_STRTAB / DT_STRSZ, filled in below
    if soname_at is not None:
        dynamic_entries.append((14, soname_at))
    dynamic_entries.extend((1, offset) for offset in needed_at)
    dynamic_entries.append((0, 0))
    header_size, program_size = 64, 56
    dynamic_offset = header_size + program_size * 2
    dynamic_size = len(dynamic_entries) * 16
    strtab_offset = dynamic_offset + dynamic_size
    total = strtab_offset + len(strings)
    dynamic_entries[0] = (5, strtab_offset)
    dynamic_entries[1] = (10, len(strings))
    header = struct.pack(
        "<4sBBBBB7xHHIQQQIHHHHHH",
        b"\x7fELF", 2, 1, 1, 0, 0, 3, 0x3E, 1, 0, header_size, 0, 0, header_size, program_size, 2, 0, 0, 0,
    )
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, total, total, 0x1000)
    dynamic = struct.pack("<IIQQQQQQ", 2, 6, dynamic_offset, dynamic_offset, dynamic_offset, dynamic_size, dynamic_size, 8)
    body = b"".join(struct.pack("<QQ", tag, value) for tag, value in dynamic_entries) + bytes(strings)
    return header + load + dynamic + body


class LinuxStagingTests(unittest.TestCase):
    def test_versioned_library_is_renamed_to_soname(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            (stage / "libcublas.so.12.9.2.10").write_bytes(b"cublas bytes")
            module.write_elf_soname_aliases(stage)
            self.assertEqual([path.name for path in stage.iterdir()], ["libcublas.so.12"])
            self.assertEqual((stage / "libcublas.so.12").read_bytes(), b"cublas bytes")

    def test_duplicate_name_is_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            (stage / "libcudnn.so.9").write_bytes(b"cudnn bytes")
            (stage / "libcudnn.so.9.19.0").write_bytes(b"cudnn bytes")
            module.write_elf_soname_aliases(stage)
            self.assertEqual([path.name for path in stage.iterdir()], ["libcudnn.so.9"])

    def test_conflicting_soname_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            (stage / "libnvinfer.so.11").write_bytes(b"one library")
            (stage / "libnvinfer.so.11.1.0").write_bytes(b"another library")
            with self.assertRaisesRegex(RuntimeError, "one SONAME"):
                module.write_elf_soname_aliases(stage)

    def test_builder_resources_skip_windows_target_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "tensorrt"
            root.mkdir()
            for name in (
                "libnvinfer_builder_resource_sm90.so.11",
                "libnvinfer_builder_resource_ptx.so.11",
                "libnvinfer_builder_resource_win_sm90.so.11",
                "libnvinfer_builder_resource_win_ptx.so.11",
            ):
                (root / name).write_bytes(name.encode())
            stage = Path(temp) / "stage"
            stage.mkdir()
            module.copy_builder_resources(stage, [root])
            self.assertEqual(
                sorted(path.name for path in stage.iterdir()),
                ["libnvinfer_builder_resource_ptx.so.11", "libnvinfer_builder_resource_sm90.so.11"],
            )

    def test_runtime_copy_still_excludes_all_builder_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "cuda"
            root.mkdir()
            for name in ("libnvrtc.so.12.9.86", "libnvinfer_builder_resource_sm90.so.11"):
                (root / name).write_bytes(name.encode())
            stage = Path(temp) / "stage"
            stage.mkdir()
            module.copy_runtime(stage, [root], "cu129")
            self.assertEqual([path.name for path in stage.iterdir()], ["libnvrtc.so.12.9.86"])

    def test_unused_cuda_families_are_not_staged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sdk"
            root.mkdir()
            for name in (
                "libnvinfer.so.11.1.0",
                "libnvvm.so.4.0.0",
                "libnvJitLink.so.12.9.86",
                "libnvparsers.so.11.1.0",
                "libnvrtc.alt.so.12.9.86",
                "libcudnn_cnn_infer.so.8.9.7",
                "libcudnn_cnn_train.so.8.9.7",
            ):
                (root / name).write_bytes(b"lib")
            stage = Path(temp) / "stage"
            stage.mkdir()
            module.copy_runtime(stage, [root], "cu129")
            self.assertEqual(
                sorted(path.name for path in stage.iterdir()),
                # sorted() compares by code point, so the capital J sorts first.
                ["libcudnn_cnn_infer.so.8.9.7", "libnvJitLink.so.12.9.86", "libnvinfer.so.11.1.0", "libnvvm.so.4.0.0"],
            )

    def test_rtx_onnx_parser_is_staged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sdk"
            root.mkdir()
            # tensorrt_rtx fails to start without its ONNX parser library.
            for name in ("libtensorrt_rtx.so.1.5.0", "libtensorrt_onnxparser_rtx.so.1.5.0", "libtensorrt_shim.so"):
                (root / name).write_bytes(b"lib")
            stage = Path(temp) / "stage"
            stage.mkdir()
            module.copy_runtime(stage, [root], "cu129")
            self.assertIn("libtensorrt_onnxparser_rtx.so.1.5.0", [path.name for path in stage.iterdir()])

    def test_unused_openvino_families_are_not_staged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sdk"
            root.mkdir()
            for name in (
                "libopenvino.so.2024.6.0",
                "libopenvino_onnx_frontend.so.2024.6.0",
                "libopenvino_pytorch_frontend.so.2024.6.0",
                "libopenvino_ir_frontend.so.2024.6.0",
                "libtbb.so.12.13",
                "libtbbbind_2_5.so.3.13",
                "libtbbmalloc.so.2.13",
                "libhwloc.so.15.6.4",
            ):
                (root / name).write_bytes(b"lib")
            stage = Path(temp) / "stage"
            stage.mkdir()
            module.copy_runtime(stage, [root], "generic")
            self.assertEqual(
                sorted(path.name for path in stage.iterdir()),
                [
                    "libhwloc.so.15.6.4",
                    "libopenvino.so.2024.6.0",
                    "libopenvino_onnx_frontend.so.2024.6.0",
                    "libtbb.so.12.13",
                    "libtbbbind_2_5.so.3.13",
                    "libtbbmalloc.so.2.13",
                ],
            )

    def test_soname_replaces_the_versioned_name(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            # The OpenVINO pattern: the file name is year based, the SONAME is ABI based.
            (stage / "libopenvino.so.2024.6.0").write_bytes(minimal_shared_object("libopenvino.so.2460"))
            module.write_elf_soname_aliases(stage)
            self.assertEqual([path.name for path in stage.iterdir()], ["libopenvino.so.2460"])

    def test_builder_resources_keep_the_sdk_name(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            # TensorRT's dispatch loader opens the fully versioned name, and the
            # SDK file records a "do not link" sentinel SONAME, so neither rule
            # may rename it.
            (stage / "libnvinfer_builder_resource_sm86.so.11.1.0").write_bytes(
                minimal_shared_object("do_not_link_against_nvinfer_builder_resource_sm86")
            )
            module.write_elf_soname_aliases(stage)
            self.assertEqual(
                [path.name for path in stage.iterdir()], ["libnvinfer_builder_resource_sm86.so.11.1.0"]
            )

    def test_unversioned_plugin_names_are_left_alone(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            (stage / "vsov.so").write_bytes(minimal_shared_object("libvsov.so"))
            (stage / "libopenvino_intel_cpu_plugin.so").write_bytes(minimal_shared_object(None))
            module.write_elf_soname_aliases(stage)
            self.assertEqual(
                sorted(path.name for path in stage.iterdir()),
                ["libopenvino_intel_cpu_plugin.so", "vsov.so"],
            )

    def test_unresolved_dependency_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            (stage / "vstrt.so").write_bytes(minimal_shared_object("libvstrt.so", ("libnvinfer.so.11", "libc.so.6")))
            with self.assertRaisesRegex(RuntimeError, "unresolved ELF dependencies"):
                module.verify_elf_dependencies(stage)
            (stage / "libnvinfer.so.11").write_bytes(minimal_shared_object("libnvinfer.so.11"))
            module.verify_elf_dependencies(stage)


if __name__ == "__main__":
    unittest.main()
