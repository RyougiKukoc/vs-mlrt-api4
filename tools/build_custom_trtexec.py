"""Build custom trtexec from the OSS revision matching a pinned binary SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def run(*args: str) -> None:
    print(subprocess.list2cmdline(list(args)), flush=True)
    subprocess.run(args, check=True)


def prepare(source: Path, sdk: Path, version: str) -> None:
    if version not in {"8.6.1.6", "11.1.0.106"}:
        raise ValueError(f"No maintained trtexec source recipe for {version}")
    major, minor, patch, _ = version.split(".")
    for root in [source, sdk]:
        header = (root / "include/NvInferVersion.h").read_text()
        prefix = "NV_TENSORRT" if major == "8" else "TRT"
        suffix = "" if major == "8" else "_ENTERPRISE"
        for name, expected in zip(["MAJOR", "MINOR", "PATCH"], [major, minor, patch]):
            found = re.search(rf"#define\s+{prefix}_{name}{suffix}\s+(\d+)", header)
            if not found or found[1] != expected:
                raise RuntimeError(f"{root}: TensorRT header does not match {version}")
    local = Path(__file__).resolve().parents[1] / "vstrt/trtexec"
    destination = source / "samples/trtexec"
    for name in ["CMakeLists.txt", "logfile.cpp", "filelock_smoke.cpp", "trtexec.manifest"]:
        shutil.copyfile(local / name, destination / name)
    lock_file = source / ("samples/common/common.h" if major == "8" else "shared/utils/fileLock.cpp")
    text = lock_file.read_text(encoding="utf-8")
    old = "CreateFileA(lockFileName.c_str(), GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, 0, NULL)"
    new = "CreateFileW(std::filesystem::u8path(lockFileName).c_str(), GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, FILE_FLAG_DELETE_ON_CLOSE | FILE_ATTRIBUTE_TEMPORARY, NULL)"
    if text.count(old) == 1:
        text = "#include <filesystem>\n" + text.replace(old, new)
        lock_file.write_text(text, encoding="utf-8", newline="\n")
    elif text.count(new) != 1:
        raise RuntimeError(f"{lock_file}: expected exactly one known Windows file-lock operation")
    for name in ["trtexec.cpp", "logfile.cpp", "filelock_smoke.cpp"]:
        if not (destination / name).is_file():
            raise RuntimeError(f"Missing matching OSS source: {name}")
    if major == "11" and not (destination / "trtexec_main.cpp").is_file():
        raise RuntimeError("TensorRT 11.1 requires the separate trtexec_main.cpp entry point")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--architectures", default="75;86-real;89-real")
    parser.add_argument("--build-dir", type=Path, default=Path("vstrt/build_trtexec"))
    parser.add_argument("--install-dir", type=Path, default=Path("vstrt/trtexec-install"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source, sdk = args.source.resolve(), args.sdk.resolve()
    prepare(source, sdk, args.version)
    if args.prepare_only:
        return
    if args.cuda is None:
        parser.error("--cuda is required for compilation")
    cuda = args.cuda.resolve()
    if not (cuda / "include/cuda_profiler_api.h").is_file():
        raise RuntimeError("CUDA installation is missing the cuda_profiler_api component")
    run("cmake", "-S", str(source / "samples/trtexec"), "-B", str(args.build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
        f"-DCUDAToolkit_ROOT={cuda}", f"-DCMAKE_CUDA_COMPILER={cuda / 'bin/nvcc.exe'}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}", f"-DTENSORRT_HOME={sdk}",
        f"-DVSMLRT_TRT_VERSION={args.version}")
    run("cmake", "--build", str(args.build_dir), "--verbose")
    run("cmake", "--install", str(args.build_dir), "--prefix", str(args.install_dir))
    exe = args.install_dir / "bin/trtexec.exe"
    metadata = {"kind": "vsmlrt-custom", "sdk_version": args.version,
                "source_revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                "cuda_root": str(cuda), "architectures": args.architectures,
                "sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}
    (exe.parent / "trtexec-build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
