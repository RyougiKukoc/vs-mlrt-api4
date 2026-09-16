"""Check Linux release zip layout, hashes, and GLIBC requirements."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vsmlrt-payload-") as temp:
        root = Path(temp)
        with zipfile.ZipFile(args.payload) as archive:
            names = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = PurePosixPath(info.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != "vsmlrt":
                    raise RuntimeError(f"Invalid release member: {info.filename}")
                names.append(str(name))
            archive.extractall(root)
        present = {Path(name).name for name in names}
        missing = sorted(set(args.required) - present)
        if missing:
            raise RuntimeError(f"Release payload is missing required file(s): {', '.join(missing)}")
        if any(name.endswith(".dll") for name in names):
            raise RuntimeError("Linux release payload contains a .dll")
        reports = []
        for library in sorted((root / "vsmlrt").glob("*.so")):
            output = subprocess.check_output(["readelf", "--version-info", str(library)], text=True, stderr=subprocess.STDOUT)
            versions = sorted(set(re.findall(r"GLIBC_(\\d+\\.\\d+)", output)))
            reports.append({"path": f"vsmlrt/{library.name}", "sha256": digest(library), "glibc": versions})
        report = {"asset": args.payload.name, "sha256": digest(args.payload), "files": sorted(names), "elf": reports}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
