"""Package one staged Linux payload as a single release archive.

A tag publishes one self-contained archive per system: the generic plugins, the
TensorRT backends, the CUDA/cuDNN runtime, and the builder helpers all travel
together, so an install downloads the model payload plus this archive and
nothing else. Archives that would exceed GitHub's per-asset limit are split
into byte volumes (`vs-mlrt-linux-x64-cu129.zip.001`, `.002`, ...); the build
hook reads the volumes as one stream.
"""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path, PurePosixPath


# Deflate level 1 is the cheapest useful setting and still shrinks the shipped
# libraries far below store mode, which measured 37% for ncnn, 47% for
# TensorRT, 68% for cuDNN, and 75% for builder resources relative to their
# uncompressed size on the payloads themselves.
DEFLATE_LEVEL = 1

# GitHub rejects assets of 2 GiB or larger.
GITHUB_ASSET_LIMIT = 2 * 1024 ** 3
DEFAULT_MAX_BYTES = GITHUB_ASSET_LIMIT - 64 * 1024 ** 2

# Mirrors WINDOWS_BUILDER_RESOURCE_MARKER in packaging/linux_native_build.py.
# TensorRT's Linux archive ships these to build engines for a Windows deployment
# target, which no Linux install of this fork can use.
WINDOWS_BUILDER_RESOURCE_MARKER = "builder_resource_win"

REQUIRED = {
    "vsncnn.so", "vsov.so", "manifest.vs",
    "vstrt.so", "vsmlrt-cuda/trtexec", "vsmlrt-cuda/trtexec-build.json",
}
REQUIRED_CU129 = {"vstrt_rtx.so", "vsmlrt-cuda/tensorrt_rtx"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def split_volumes(archive: Path, max_bytes: int) -> list[Path]:
    """Split an oversized archive into byte volumes and drop the whole file."""
    volumes = []
    with archive.open("rb") as stream:
        while chunk := stream.read(max_bytes):
            volume = archive.with_name(f"{archive.name}.{len(volumes) + 1:03d}")
            volume.write_bytes(chunk)
            volumes.append(volume)
    archive.unlink()
    return volumes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    stage = args.stage_dir.resolve()

    payload = {
        path.relative_to(stage).as_posix(): path
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.relative_to(stage).parts[:1] != ("models",)
    }
    windows_only = sorted(name for name in payload if WINDOWS_BUILDER_RESOURCE_MARKER in PurePosixPath(name).name)
    if windows_only:
        raise RuntimeError(f"Linux payload must not contain Windows-target builder resources: {windows_only}")
    expected = set(REQUIRED)
    if args.variant != "generic":
        if not any("builder_resource" in PurePosixPath(name).name for name in payload):
            raise RuntimeError("Builder resources missing from a CUDA payload")
    else:
        expected -= {"vstrt.so", "vsmlrt-cuda/trtexec", "vsmlrt-cuda/trtexec-build.json"}
    if args.variant == "cu129":
        expected |= REQUIRED_CU129
    missing = sorted(name for name in expected if name not in payload)
    if missing:
        raise RuntimeError(f"Missing required payload files: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=DEFLATE_LEVEL) as archive:
        for name, source in sorted(payload.items()):
            archive.write(source, str(PurePosixPath("vsmlrt") / name))
    volumes: list[Path] = []
    if args.output.stat().st_size > args.max_bytes:
        volumes = split_volumes(args.output, args.max_bytes)
    assets = volumes or [args.output]
    inventory = {
        "variant": args.variant,
        "members": sorted(f"vsmlrt/{name}" for name in payload),
        "assets": [
            {"name": asset.name, "sha256": digest(asset), "size": asset.stat().st_size} for asset in assets
        ],
    }
    if args.inventory:
        args.inventory.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in inventory.items() if k != "members"}, indent=2))


if __name__ == "__main__":
    main()
