"""Verify custom logging and the SDK's patched lock code without GPU execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    exe = (args.bin_dir / "trtexec.exe").resolve()
    provenance = json.loads((exe.parent / "trtexec-build.json").read_text())
    if provenance["kind"] != "vsmlrt-custom" or provenance["sha256"] != hashlib.sha256(exe.read_bytes()).hexdigest():
        raise RuntimeError("Custom trtexec provenance does not match executable")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(p.resolve()) for p in args.runtime_dir] + [env["PATH"]])
    with tempfile.TemporaryDirectory(prefix="vsmlrt-trtexec-") as temp:
        directory = Path(temp)
        while len(str(directory)) < 270:
            directory /= "Unicode-\u6d4b\u8bd5-" + "nested-" * 4
        directory.mkdir(parents=True)
        log = directory / "\u65e5\u5fd7.txt"
        env["TRTEXEC_LOG_FILE"] = str(log)
        result = subprocess.run([str(exe), "--help"], env=env, capture_output=True, timeout=60)
        if result.returncode or not log.is_file() or b"--onnx" not in log.read_bytes():
            raise RuntimeError(f"Custom trtexec help/log smoke failed: {result.returncode}\n{result.stdout!r}\n{result.stderr!r}")
        subprocess.run([str(exe.parent / "trtexec_filelock_smoke.exe"), str(directory / "\u7f13\u5b58.cache")], env=env, check=True, timeout=60)
    print("Custom trtexec: --help, Unicode/long log path, and file-lock cleanup passed")


if __name__ == "__main__":
    main()
