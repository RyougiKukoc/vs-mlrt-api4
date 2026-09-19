"""Verify Linux release overlay archives before upload or publication."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import zipfile


def members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        names = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if not name.startswith("vsmlrt/") or ".." in Path(name).parts:
                raise RuntimeError(f"Invalid Linux payload member: {path.name}: {name}")
            if name.endswith(".dll") or name.endswith(".exe"):
                raise RuntimeError(f"Linux payload contains Windows binary: {path.name}: {name}")
            names.add(name)
        return names


def manifest(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        try:
            return archive.read("vsmlrt/manifest.vs").decode("ascii").splitlines()
        except KeyError as error:
            raise RuntimeError(f"Linux plugin payload has no manifest.vs: {path.name}") from error


def verify(variant: str, asset_dir: Path) -> None:
    generic = asset_dir / "vs-mlrt-linux-x64-generic.zip"
    if not generic.is_file():
        raise RuntimeError(f"Missing generic Linux payload: {generic}")
    generic_names = members(generic)
    required_generic = {"vsmlrt/vsncnn.so", "vsmlrt/vsov.so", "vsmlrt/manifest.vs"}
    if not required_generic.issubset(generic_names):
        raise RuntimeError(f"Generic payload is missing: {sorted(required_generic - generic_names)}")
    if manifest(generic) != ["[VapourSynth Manifest V1]", "vsncnn", "vsov"]:
        raise RuntimeError(f"Generic manifest is incorrect: {manifest(generic)!r}")

    if variant == "generic":
        forbidden = [name for name in generic_names if name.startswith("vsmlrt/vsmlrt-cuda/")]
        if forbidden:
            raise RuntimeError(f"Generic payload contains CUDA files: {forbidden}")
        return

    names = set(generic_names)
    assets = [
        asset_dir / f"vs-mlrt-linux-x64-tensorrt-{variant}.zip",
        asset_dir / f"vs-mlrt-linux-x64-cuda-{variant}.zip",
        asset_dir / f"vs-mlrt-linux-x64-cudnn-{variant}.zip",
        asset_dir / f"vs-mlrt-linux-x64-cudnn-part-2-{variant}.zip",
        asset_dir / f"vs-mlrt-linux-x64-tensorrt-builder-{variant}.zip",
    ]
    if variant == "cu129":
        assets.extend(asset_dir / f"vs-mlrt-linux-x64-tensorrt-builder-resource-{index}-cu129.zip" for index in range(1, 5))
    for asset in assets:
        if not asset.is_file():
            raise RuntimeError(f"Missing Linux {variant} payload: {asset}")
        names.update(members(asset))

    trt_asset = assets[0]
    if manifest(trt_asset) != ["[VapourSynth Manifest V1]", "vstrt"]:
        raise RuntimeError(f"TensorRT manifest is incorrect: {manifest(trt_asset)!r}")

    required = {
        "vsmlrt/vstrt.so",
        "vsmlrt/vsmlrt-cuda/trtexec",
        "vsmlrt/vsmlrt-cuda/trtexec-build.json",
    }
    if not required.issubset(names):
        raise RuntimeError(f"Linux {variant} payload is missing: {sorted(required - names)}")
    for family in ("libcudart", "libcublas", "libnvinfer", "libnvinfer_plugin", "libnvonnxparser"):
        if not any(name.startswith(f"vsmlrt/{family}") for name in names):
            raise RuntimeError(f"Linux {variant} payload is missing the {family} library family")
    if not any(name.startswith("vsmlrt/libcudnn") for name in names):
        raise RuntimeError(f"Linux {variant} payload is missing the cuDNN library family")
    if not any("builder_resource" in name for name in names):
        raise RuntimeError(f"Linux {variant} payload is missing TensorRT builder resources")

    if variant == "cu129":
        rtx = asset_dir / "vs-mlrt-linux-x64-tensorrt-rtx-cu129.zip"
        if not rtx.is_file():
            raise RuntimeError(f"Missing Linux cu129 RTX payload: {rtx}")
        rtx_names = members(rtx)
        if manifest(rtx) != ["[VapourSynth Manifest V1]", "vstrt_rtx"]:
            raise RuntimeError(f"TensorRT-RTX manifest is incorrect: {manifest(rtx)!r}")
        required_rtx = {"vsmlrt/vstrt_rtx.so", "vsmlrt/vsmlrt-cuda/tensorrt_rtx"}
        if not required_rtx.issubset(rtx_names):
            raise RuntimeError(f"Linux cu129 RTX payload is missing: {sorted(required_rtx - rtx_names)}")

    print(f"Verified staged Linux {variant} payload overlays")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    parser.add_argument("--asset-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    verify(args.variant, args.asset_dir.resolve())


if __name__ == "__main__":
    main()
