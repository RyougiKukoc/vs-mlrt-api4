"""Build and stage the native Linux payload used by the root PEP 517 hook.

This script intentionally does not download SDKs. A source-build user must
provide the compatible CMake packages and runtime roots already selected for
their host. It does discover the installed VapourSynth wheel's headers and
pkg-config metadata without clobbering caller configuration.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str], *, env: dict[str, str]) -> None:
    print("vs-mlrt native build:", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def prepend_pkgconfig(env: dict[str, str]) -> Path:
    override = env.get("VSMLRT_VAPOURSYNTH_ROOT")
    if override:
        wheel_root = Path(override).expanduser().resolve()
    else:
        import vapoursynth

        wheel_root = Path(vapoursynth.__file__).resolve().parent
    pkgconfig = wheel_root / "pkgconfig"
    if not pkgconfig.is_dir():
        raise RuntimeError(f"VapourSynth pkg-config directory is missing: {pkgconfig}")
    existing = env.get("PKG_CONFIG_PATH")
    env["PKG_CONFIG_PATH"] = os.pathsep.join((str(pkgconfig), existing)) if existing else str(pkgconfig)
    header_root = wheel_root / "include"
    if not (header_root / "VapourSynth4.h").is_file():
        header_root = header_root / "vapoursynth"
    if not (header_root / "VapourSynth4.h").is_file():
        raise RuntimeError(f"VapourSynth4.h was not found beneath {wheel_root}")
    command(["pkg-config", "--cflags", "vapoursynth"], env=env)
    return header_root


def cmake_build(source: Path, build: Path, install: Path, arguments: list[str], env: dict[str, str]) -> None:
    command(
        ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", "-D", "CMAKE_BUILD_TYPE=Release", *arguments],
        env=env,
    )
    command(["cmake", "--build", str(build), "--verbose"], env=env)
    command(["cmake", "--install", str(build), "--prefix", str(install)], env=env)


def find_library(root: Path, stem: str) -> Path:
    for candidate in root.rglob(f"{stem}.so"):
        return candidate
    for candidate in root.rglob(f"{stem}.dylib"):
        return candidate
    raise FileNotFoundError(f"Could not locate {stem}.so under {root}")


def roots_from_env(env: dict[str, str], *extras: Path | None) -> list[Path]:
    roots = [Path(value).expanduser().resolve() for value in env.get("VSMLRT_RUNTIME_ROOTS", "").split(os.pathsep) if value]
    roots.extend(value.resolve() for value in extras if value and value.exists())
    return list(dict.fromkeys(roots))


def copy_runtime_family(stage: Path, roots: list[Path], prefixes: tuple[str, ...]) -> None:
    for root in roots:
        for pattern in ("*.so*", "*.dylib*"):
            for source in root.rglob(pattern):
                if not source.is_file() or source.is_symlink() or not source.name.startswith(prefixes):
                    continue
                destination = stage / source.name
                if not destination.exists():
                    shutil.copy2(source.resolve(), destination)


def write_elf_soname_aliases(stage: Path) -> None:
    """Create the major-version filenames requested by dynamic ELF loaders.

    NVIDIA SDK archives often contain developer symlinks, which are either
    discarded by archive tooling or inflate a wheel into repeated full copies.
    Stage canonical files plus only the SONAME aliases needed at runtime.
    """
    for library in stage.glob("*.so.*"):
        match = re.match(r"(?P<stem>.+\.so)\.(?P<major>\d+)(?:\..+)?$", library.name)
        if not match:
            continue
        alias = stage / f"{match.group('stem')}.{match.group('major')}"
        if alias == library or alias.exists():
            continue
        # Wheel zip extraction is not guaranteed to preserve symlinks. A hard
        # link retains one inode in the staging directory; archivers may expand
        # it, but the installed wheel still has a valid ELF filename.
        os.link(library, alias)


def copy_runtime(stage: Path, roots: list[Path], variant: str) -> None:
    if variant == "generic":
        copy_runtime_family(stage, roots, ("libopenvino", "libtbb", "libonnx", "libprotobuf"))
    else:
        copy_runtime_family(
            stage,
            roots,
            ("libnvinfer", "libnvonnxparser", "libcudnn", "libcublas", "libcudart", "libnvrtc", "libnvJitLink", "libtensorrt_rtx"),
        )


def write_manifest(stage: Path) -> None:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    plugins = [name for name in ("vsov", "vstrt", "vstrt_rtx") if (stage / f"{name}{suffix}").is_file()]
    (stage / "manifest.vs").write_text("[VapourSynth Manifest V1]\n" + "\n".join(plugins) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform not in {"linux", "darwin"}:
        raise RuntimeError("VSMLRT_FORCE_BUILD native fallback currently requires a POSIX CMake toolchain.")

    stage = args.stage_dir.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    native_suffix = ".dylib" if sys.platform == "darwin" else ".so"
    env = os.environ.copy()
    headers = prepend_pkgconfig(env)
    build_root = ROOT / "build" / "native" / args.variant
    shutil.rmtree(build_root, ignore_errors=True)
    build_root.mkdir(parents=True)

    # CUDA variants are additive: the documented layout contains the generic
    # OpenVINO backend as well as TensorRT. Build it first for every variant.
    openvino_dir_raw = env.get("VSMLRT_OPENVINO_DIR") or env.get("OpenVINO_DIR")
    openvino_dir = Path(openvino_dir_raw).expanduser().resolve() if openvino_dir_raw else None
    generic_args = [f"-DVAPOURSYNTH_INCLUDE_DIRECTORY={headers}"]
    if openvino_dir:
        generic_args.append(f"-DOpenVINO_DIR={openvino_dir}")
    cmake_build(ROOT / "vsov", build_root / "vsov", build_root / "install-vsov", generic_args, env)
    shutil.copy2(find_library(build_root / "install-vsov", "libvsov"), stage / f"vsov{native_suffix}")
    copy_runtime(stage, roots_from_env(env, openvino_dir.parent if openvino_dir else None), "generic")

    if args.variant != "generic":
        trt_home_raw = env.get("VSMLRT_TENSORRT_HOME") or env.get("TENSORRT_HOME")
        if not trt_home_raw:
            raise RuntimeError("VSMLRT_TENSORRT_HOME is required for CUDA source fallback.")
        trt_home = Path(trt_home_raw).expanduser().resolve()
        cuda_root = Path(env["CUDAToolkit_ROOT"]).expanduser().resolve() if env.get("CUDAToolkit_ROOT") else None
        cmake_args = [f"-DVAPOURSYNTH_INCLUDE_DIRECTORY={headers}", f"-DTENSORRT_HOME={trt_home}"]
        if cuda_root:
            cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")
        cmake_build(ROOT / "vstrt", build_root / "vstrt", build_root / "install-vstrt", cmake_args, env)
        shutil.copy2(find_library(build_root / "install-vstrt", "libvstrt"), stage / f"vstrt{native_suffix}")
        runtime_roots = roots_from_env(env, trt_home / "lib", cuda_root / "lib64" if cuda_root else None)
        if args.variant == "cu129" and env.get("VSMLRT_TENSORRT_RTX_HOME"):
            rtx_home = Path(env["VSMLRT_TENSORRT_RTX_HOME"]).expanduser().resolve()
            cmake_build(
                ROOT / "vstrt",
                build_root / "vstrt-rtx",
                build_root / "install-vstrt-rtx",
                [f"-DVAPOURSYNTH_INCLUDE_DIRECTORY={headers}", f"-DTENSORRT_HOME={rtx_home}", *cmake_args[2:]],
                env,
            )
            shutil.copy2(find_library(build_root / "install-vstrt-rtx", "libvstrt_rtx"), stage / f"vstrt_rtx{native_suffix}")
            runtime_roots.extend(roots_from_env(env, rtx_home / "lib"))
        copy_runtime(stage, runtime_roots, args.variant)
    if sys.platform == "linux":
        write_elf_soname_aliases(stage)
    write_manifest(stage)


if __name__ == "__main__":
    main()
