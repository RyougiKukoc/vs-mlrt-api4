"""Check a Linux VCS install resolved the staged native overlay set."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    args = parser.parse_args()

    import vsmlrt

    root = Path(vsmlrt.plugins_path).resolve()
    required = [root / "vsncnn.so", root / "vsov.so", root / "manifest.vs"]
    if args.variant != "generic":
        required.extend([root / "vstrt.so", root / "vsmlrt-cuda" / "trtexec", root / "vsmlrt-cuda" / "trtexec-build.json"])
    if args.variant == "cu129":
        required.extend([root / "vstrt_rtx.so", root / "vsmlrt-cuda" / "tensorrt_rtx"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Installed Linux payload is missing:\n" + "\n".join(missing))
    expected = ["[VapourSynth Manifest V1]", "vsncnn", "vsov"]
    if args.variant != "generic":
        expected.append("vstrt")
    if args.variant == "cu129":
        expected.append("vstrt_rtx")
    manifest = (root / "manifest.vs").read_text(encoding="ascii").splitlines()
    if manifest != expected:
        raise SystemExit(f"Unexpected installed manifest: {manifest!r}")
    if args.variant != "generic" and Path(vsmlrt.trtexec_path).resolve() != (root / "vsmlrt-cuda" / "trtexec").resolve():
        raise SystemExit(f"Unexpected trtexec path: {vsmlrt.trtexec_path}")
    if args.variant == "cu129" and Path(vsmlrt.tensorrt_rtx_path).resolve() != (root / "vsmlrt-cuda" / "tensorrt_rtx").resolve():
        raise SystemExit(f"Unexpected TensorRT-RTX path: {vsmlrt.tensorrt_rtx_path}")
    print(f"Verified installed Linux {args.variant} VCS payload: {root}")


if __name__ == "__main__":
    main()
