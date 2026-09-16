"""Create a checked Linux release payload from a staged vs-mlrt wheel."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PLUGIN_ROOT = PurePosixPath("vapoursynth/plugins/vsmlrt")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_files(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            info.filename.replace("\\", "/"): archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="generic release zip whose identical files are omitted")
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()

    expected = {"generic": {"vsov.so"}, "cu121": {"vstrt.so"}, "cu129": {"vstrt.so", "vstrt_rtx.so"}}[args.variant]
    wheel_files = archive_files(args.wheel)
    prefix = str(PLUGIN_ROOT) + "/"
    payload = {
        name[len(prefix):]: data
        for name, data in wheel_files.items()
        if name.startswith(prefix) and not name.startswith(prefix + "models/")
    }
    # The root hook always regenerates the combined manifest after overlays.
    # Keep an individual manifest in each release zip for direct extraction.
    native = {name for name in payload if name.endswith(".so")}
    if not expected.issubset(native):
        raise RuntimeError(f"{args.variant} wheel is missing expected ELF plugin(s): {sorted(expected - native)}")
    if any(name.endswith(".dll") for name in payload):
        raise RuntimeError("Linux release payload contains a Windows DLL")

    base = archive_files(args.base) if args.base else {}
    base_payload = {
        name.removeprefix("vsmlrt/"): data
        for name, data in base.items()
        if name.startswith("vsmlrt/")
    }
    omitted = []
    if args.base:
        for name in list(payload):
            if name != "manifest.vs" and base_payload.get(name) == payload[name]:
                omitted.append(name)
                del payload[name]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            target = PurePosixPath("vsmlrt") / name
            if target.is_absolute() or ".." in target.parts:
                raise RuntimeError(f"Unsafe package member: {target}")
            archive.writestr(str(target), data)

    inventory = {
        "variant": args.variant,
        "asset": args.output.name,
        "sha256": sha256(args.output.read_bytes()),
        "files": [{"path": f"vsmlrt/{name}", "sha256": sha256(data), "size": len(data)} for name, data in sorted(payload.items())],
        "omitted_identical_to_generic": sorted(omitted),
    }
    if args.inventory:
        args.inventory.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
