"""Verify a Linux release payload archive before upload or publication.

One tag publishes one self-contained archive per system, so this checks that the
single archive (or its numbered volumes) carries every plugin, runtime library,
builder helper, and manifest the tag promises.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path, PurePosixPath
import re
import zipfile


_spec = importlib.util.spec_from_file_location(
    "payload_archive", Path(__file__).resolve().parents[1] / "packaging" / "payload_archive.py"
)
assert _spec and _spec.loader
_archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_archive)
open_payload = _archive.open_payload
resolve_volumes = _archive.resolve_volumes


# TensorRT's Linux archive ships these beside the Linux builder resources to
# build engines for a Windows deployment target. No Linux install of this fork
# can load them, so publishing them only inflated the cu129 assets.
WINDOWS_BUILDER_RESOURCE_MARKER = "builder_resource_win"

# A library shipped both under its SONAME and under a fully versioned sibling is
# stored twice; the staging keeps only the SONAME file.
VERSIONED_SIBLING = re.compile(r"(?P<soname>.+\.so\.\d+)\..+$")

PLUGINS = {
    "generic": ("vsncnn", "vsov"),
    "cu121": ("vsncnn", "vsov", "vstrt"),
    "cu129": ("vsncnn", "vsov", "vstrt", "vstrt_rtx"),
}


def payload_volumes(variant: str, asset_dir: Path) -> list[Path]:
    name = f"vs-mlrt-linux-x64-{variant}.zip"
    volumes = resolve_volumes(asset_dir / name)
    if not volumes[0].is_file():
        raise RuntimeError(f"Missing Linux {variant} payload: {volumes[0]}")
    return volumes


def reject_duplicate_libraries(names: set[str]) -> None:
    present = {PurePosixPath(name).name for name in names}
    duplicates = sorted(
        name for name in present if (match := VERSIONED_SIBLING.match(name)) and match.group("soname") in present
    )
    if duplicates:
        raise RuntimeError(f"Linux payload stores a library under two names: {duplicates}")


def members(sources: list[Path]) -> set[str]:
    with open_payload(sources) as archive:
        names = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if not name.startswith("vsmlrt/") or ".." in Path(name).parts:
                raise RuntimeError(f"Invalid Linux payload member: {sources[0].name}: {name}")
            if name.endswith(".dll") or name.endswith(".exe"):
                raise RuntimeError(f"Linux payload contains Windows binary: {sources[0].name}: {name}")
            base = PurePosixPath(name).name
            if WINDOWS_BUILDER_RESOURCE_MARKER in base:
                raise RuntimeError(f"Linux payload contains Windows-target builder resource: {sources[0].name}: {name}")
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise RuntimeError(f"Linux payload asset is stored uncompressed: {sources[0].name}: {name}")
            names.add(name)
        return names


def manifest(sources: list[Path]) -> list[str]:
    with open_payload(sources) as archive:
        try:
            return archive.read("vsmlrt/manifest.vs").decode("ascii").splitlines()
        except KeyError as error:
            raise RuntimeError(f"Linux payload has no manifest.vs: {sources[0].name}") from error


def verify(variant: str, asset_dir: Path) -> None:
    sources = payload_volumes(variant, asset_dir)
    names = members(sources)
    reject_duplicate_libraries(names)

    expected_manifest = ["[VapourSynth Manifest V1]", *PLUGINS[variant]]
    if manifest(sources) != expected_manifest:
        raise RuntimeError(f"Linux {variant} manifest is incorrect: {manifest(sources)!r}")

    required = {f"vsmlrt/{name}.so" for name in PLUGINS[variant]} | {"vsmlrt/manifest.vs"}
    if variant != "generic":
        required |= {"vsmlrt/vsmlrt-cuda/trtexec", "vsmlrt/vsmlrt-cuda/trtexec-build.json"}
    if variant == "cu129":
        required.add("vsmlrt/vsmlrt-cuda/tensorrt_rtx")
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"Linux {variant} payload is missing: {missing}")

    if variant == "generic":
        forbidden = [name for name in names if name.startswith("vsmlrt/vsmlrt-cuda/")]
        if forbidden:
            raise RuntimeError(f"Generic payload contains CUDA files: {forbidden}")
        print("Verified staged Linux generic payload")
        return

    for family in ("libcudart", "libcublas", "libnvinfer", "libnvinfer_plugin", "libnvonnxparser", "libcudnn"):
        if not any(name.startswith(f"vsmlrt/{family}") for name in names):
            raise RuntimeError(f"Linux {variant} payload is missing the {family} library family")
    if not any("builder_resource" in name for name in names):
        raise RuntimeError(f"Linux {variant} payload is missing TensorRT builder resources")
    if variant == "cu129" and not any("libtensorrt_rtx" in name for name in names):
        raise RuntimeError("Linux cu129 payload is missing the TensorRT-RTX library family")
    print(f"Verified staged Linux {variant} payload ({len(sources)} volume(s))")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    parser.add_argument("--asset-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    verify(args.variant, args.asset_dir.resolve())


if __name__ == "__main__":
    main()
