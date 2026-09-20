"""Build and stage the native Linux payload used by the root PEP 517 hook.

This script intentionally does not download SDKs. A source-build user must
provide the compatible CMake packages and runtime roots already selected for
their host. It does discover the installed VapourSynth wheel's headers and
pkg-config metadata without clobbering caller configuration.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# TensorRT's Linux archive also carries builder resources whose names contain
# this marker. They build engines for a Windows deployment target, which this
# fork does not support: the release verifier already rejects Windows binaries
# in Linux payloads, and staging them cost about 5.9 GB of every cu129 release.
WINDOWS_BUILDER_RESOURCE_MARKER = "builder_resource_win"

# Files the published plugins never load, checked by searching the dynamic
# dependencies and the dlopen name strings of every library in the payload:
# - ``.alt.`` NVRTC builds: nothing selects them.
# - ``libnvparsers``: the deprecated Caffe/UFF parser that ``nvonnxparser``
#   replaced; this fork only ever feeds ONNX.
# - cuDNN ``*_train`` libraries: only reachable through the training backends.
# - OpenVINO frontends other than ONNX: Windows ships only the ONNX frontend,
#   and every OV library here reads ONNX models.
# - TensorRT's ``_win_`` builder resources: build engines for a Windows target.
# ``libnvJitLink`` and ``libnvvm`` are deliberately absent: ``libnvinfer``
# names both, so TensorRT loads them while building engines. ``libtbbbind`` and
# ``libtbbmalloc`` stay too: ``libtbb`` names them and they cost 0.5 MB.
UNUSED_RUNTIME_MARKERS = (
    ".alt.",
    "libnvparsers",
    "_train",
    "_paddle_frontend",
    "_pytorch_frontend",
    "_tensorflow_frontend",
    "_tensorflow_lite_frontend",
    "_ir_frontend",
)

# Dynamic dependencies that come from the host rather than the payload. The
# GPU driver, the OpenCL ICD loader, and the C/C++/Fortran runtimes are the
# user's system libraries; everything else has to be inside the payload.
HOST_LIBRARY_NAMES = {
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libgomp.so.1",
    "libm.so.6",
    "libmvec.so.1",
    "libOpenCL.so.1",
    "libpthread.so.0",
    "librt.so.1",
    "libstdc++.so.6",
    "libz.so.1",
    "libcuda.so.1",
    "libnvidia-ml.so.1",
    "libnvidia-ptxjitcompiler.so.1",
}


def command(args: list[str], *, env: dict[str, str]) -> None:
    print("vs-mlrt native build:", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def prepend_pkgconfig(env: dict[str, str]) -> Path:
    include_override = env.get("VSMLRT_VAPOURSYNTH_INCLUDE_DIRECTORY")
    if include_override:
        root = Path(include_override).expanduser().resolve()
        if not (root / "VapourSynth4.h").is_file():
            raise RuntimeError(f"VapourSynth4.h was not found beneath {root}")
        return root
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


def required_package_dir(env: dict[str, str], name: str, alias: str) -> Path:
    value = env.get(name) or env.get(alias)
    if not value:
        raise RuntimeError(f"{name} is required for the Linux source fallback")
    return Path(value).expanduser().resolve()


def copy_runtime_family(
    stage: Path,
    roots: list[Path],
    prefixes: tuple[str, ...],
    *,
    exclude_markers: tuple[str, ...] = (),
) -> None:
    for root in roots:
        for pattern in ("*.so*", "*.dylib*"):
            for source in root.rglob(pattern):
                if not source.is_file() or source.is_symlink() or not source.name.startswith(prefixes):
                    continue
                if any(marker in source.name for marker in exclude_markers):
                    continue
                destination = stage / source.name
                if not destination.exists():
                    shutil.copy2(source.resolve(), destination)


def write_elf_soname_aliases(stage: Path) -> None:
    """Stage exactly one file per library, under the name its dependents request.

    SDK archives ship the real library under its full version next to developer
    symlinks, and dynamic loaders resolve the literal names recorded in
    DT_NEEDED. That name is the SONAME, which cannot be derived from the file
    name in general: OpenVINO 2024.6 names its files after the release year
    while keeping an ABI-based SONAME (``libopenvino.so.2024.6.0`` has SONAME
    ``libopenvino.so.2460``), and ``libnvrtc-builtins.so.12.9.86`` records
    ``libnvrtc-builtins.so.12.9``. Renaming to the SONAME both removes the
    duplicated copy and produces the name the loader looks for; wheel zip
    extraction is not guaranteed to preserve symlinks, so it has to be a
    regular file.

    A SONAME is only trusted when it names the same library. TensorRT's builder
    resources carry a sentinel SONAME instead -- ``libnvinfer_builder_resource_
    sm86.so.11.1.0`` records ``do_not_link_against_nvinfer_builder_resource_
    sm86`` -- and TensorRT opens them as ``<stem>.so.<major>``, so those fall
    back to the file-name rule.
    """
    for library in sorted(stage.glob("*.so.*")):
        match = re.match(r"(?P<stem>.+\.so)\.(?P<major>\d+)(?:\..+)?$", library.name)
        if not match:
            continue
        if "builder_resource" in library.name:
            # TensorRT's dispatch loader asks for the fully versioned name
            # (libLoader.cpp opens libnvinfer_builder_resource_sm86.so.11.1.0),
            # so these keep the name the SDK ships. Only the alias this function
            # would have created is dropped, which is what removes the duplicate.
            continue
        fallback = f"{match.group('stem')}.{match.group('major')}"
        soname = elf_dynamic_names(library)[0]
        alias = stage / (soname if soname and soname.startswith(f"{match.group('stem')}.") else fallback)
        if alias == library:
            continue
        if not alias.exists():
            library.rename(alias)
            continue
        if os.path.samefile(alias, library) or same_contents(alias, library):
            library.unlink()
            continue
        raise RuntimeError(
            f"Staged Linux payload has two different libraries for one SONAME: {alias.name} and {library.name}"
        )


def same_contents(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    return filecmp.cmp(first, second, shallow=False)


def elf_dynamic_names(path: Path) -> tuple[str | None, list[str]]:
    """Return (DT_SONAME, DT_NEEDED) of a 64-bit little-endian ELF shared object."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
            return None, []
        program_offset = struct.unpack("<Q", header[0x20:0x28])[0]
        entry_size, entry_count = struct.unpack("<HH", header[0x36:0x3a])
        stream.seek(program_offset)
        program_headers = stream.read(entry_size * entry_count)
        load_regions: list[tuple[int, int, int]] = []
        dynamic: tuple[int, int, int] | None = None
        for index in range(entry_count):
            entry = program_headers[index * entry_size:(index + 1) * entry_size]
            kind = struct.unpack("<I", entry[0:4])[0]
            if kind == 1:  # PT_LOAD
                vaddr, offset, size = struct.unpack("<QQQ", entry[16:40])
                load_regions.append((vaddr, offset, size))
            elif kind == 2:  # PT_DYNAMIC: offset, vaddr, size
                offset, vaddr, size = struct.unpack("<QQQ", entry[8:32])
                dynamic = (vaddr, offset, size)
        if dynamic is None:
            return None, []

        def file_offset(vaddr: int) -> int | None:
            for base, offset, size in load_regions:
                if base <= vaddr < base + size:
                    return offset + (vaddr - base)
            return None

        stream.seek(dynamic[1])
        table = stream.read(min(dynamic[2], 1 << 16))
        entries: list[tuple[int, int]] = []
        for index in range(0, len(table) - 15, 16):
            tag, value = struct.unpack("<QQ", table[index:index + 16])
            if tag == 0:
                break
            entries.append((tag, value))
        string_address = next((value for tag, value in entries if tag == 5), None)
        string_size = next((value for tag, value in entries if tag == 10), 0)
        offset = file_offset(string_address) if string_address is not None else None
        if offset is None:
            return None, []
        stream.seek(offset)
        strings = stream.read(min(string_size or (1 << 20), 8 << 20))

        def read_string(index: int) -> str:
            if index >= len(strings):
                return ""
            end = strings.find(b"\0", index)
            return strings[index:end if end >= 0 else len(strings)].decode("ascii", "replace")

        soname = next((read_string(value) for tag, value in entries if tag == 14), None)
        needed = [read_string(value) for tag, value in entries if tag == 1]
        return soname or None, [name for name in needed if name]


def write_origin_runpaths(stage: Path) -> None:
    """Point every staged library at its own directory.

    RUNPATH lookups are not transitive, and the SDK libraries ship without one:
    ``libopenvino.so.2460`` records no RUNPATH, so the ``libtbb.so.12`` it links
    resolves only through the main program's search path. VapourSynth loads the
    plugins itself and sets nothing, so the OpenVINO backend failed to load from
    a plain pip install. Relocating prebuilt libraries this way is what wheel
    repair tools do; without ``patchelf`` the payload still works when the caller
    exports ``LD_LIBRARY_PATH``, so the build warns instead of failing.
    """
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        print("vs-mlrt: patchelf is unavailable; staged libraries keep the SDK search paths", file=sys.stderr)
        return
    for library in sorted(stage.glob("*.so*")):
        if library.is_symlink():
            continue
        subprocess.run([patchelf, "--set-rpath", "$ORIGIN", str(library)], check=True)
    for helper in sorted((stage / "vsmlrt-cuda").glob("*")):
        if helper.is_file() and os.access(helper, os.X_OK):
            subprocess.run([patchelf, "--set-rpath", "$ORIGIN:$ORIGIN/..", str(helper)], check=True)


def verify_elf_dependencies(stage: Path) -> None:
    """Every DT_NEEDED of every staged library must resolve inside the payload.

    Names alone are not evidence: the OpenVINO payload looked complete while
    every dependent asked for ``libopenvino.so.2460`` and no such file existed.
    """
    staged = {path.name for path in stage.iterdir() if path.is_file()}
    missing: dict[str, list[str]] = {}
    for library in sorted(stage.glob("*.so*")):
        for needed in elf_dynamic_names(library)[1]:
            if needed in staged or needed in HOST_LIBRARY_NAMES:
                continue
            missing.setdefault(needed, []).append(library.name)
    if missing:
        detail = "; ".join(f"{name} (needed by {', '.join(sorted(set(users)))})" for name, users in sorted(missing.items()))
        raise RuntimeError(f"Staged Linux payload has unresolved ELF dependencies: {detail}")


def copy_runtime(stage: Path, roots: list[Path], variant: str) -> None:
    if variant == "generic":
        copy_runtime_family(
            stage,
            roots,
            # libhwloc is what libtbbbind_2_5 links against for TBB's affinity
            # binding; OpenVINO's runtime ships it under 3rdparty/tbb/lib.
            ("libopenvino", "libtbb", "libhwloc", "libonnx", "libprotobuf", "libncnn"),
            exclude_markers=UNUSED_RUNTIME_MARKERS,
        )
    else:
        copy_runtime_family(
            stage,
            roots,
            ("libnvinfer", "libnvonnxparser", "libcudnn", "libcublas", "libcudart", "libcufft", "libnvblas", "libnvrtc", "libnvvm", "libnvJitLink", "libtensorrt_rtx", "libtensorrt_onnxparser_rtx"),
            exclude_markers=("builder_resource", *UNUSED_RUNTIME_MARKERS),
        )


def copy_builder_resources(stage: Path, roots: list[Path]) -> None:
    copy_runtime_family(
        stage,
        roots,
        ("libnvinfer_builder_resource",),
        exclude_markers=(WINDOWS_BUILDER_RESOURCE_MARKER,),
    )


def write_manifest(stage: Path) -> None:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    plugins = [name for name in ("vsncnn", "vsov", "vstrt", "vstrt_rtx") if (stage / f"{name}{suffix}").is_file()]
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
    ncnn_dir = required_package_dir(env, "VSMLRT_NCNN_DIR", "ncnn_DIR")
    onnx_dir = required_package_dir(env, "VSMLRT_ONNX_DIR", "ONNX_DIR")
    protobuf_dir = required_package_dir(env, "VSMLRT_PROTOBUF_DIR", "protobuf_DIR")
    ncnn_args = [f"-DVAPOURSYNTH_INCLUDE_DIRECTORY={headers}", f"-Dncnn_DIR={ncnn_dir}", f"-DONNX_DIR={onnx_dir}", f"-Dprotobuf_DIR={protobuf_dir}"]
    cmake_build(ROOT / "vsncnn", build_root / "vsncnn", build_root / "install-vsncnn", ncnn_args, env)
    shutil.copy2(find_library(build_root / "install-vsncnn", "libvsncnn"), stage / f"vsncnn{native_suffix}")
    generic_args = [
        f"-DVAPOURSYNTH_INCLUDE_DIRECTORY={headers}",
        f"-DONNX_DIR={onnx_dir}",
        f"-Dprotobuf_DIR={protobuf_dir}",
    ]
    if openvino_dir:
        generic_args.append(f"-DOpenVINO_DIR={openvino_dir}")
    cmake_build(ROOT / "vsov", build_root / "vsov", build_root / "install-vsov", generic_args, env)
    shutil.copy2(find_library(build_root / "install-vsov", "libvsov"), stage / f"vsov{native_suffix}")
    copy_runtime(stage, roots_from_env(env, openvino_dir.parent if openvino_dir else None, ncnn_dir.parent, onnx_dir.parent, protobuf_dir.parent), "generic")

    if args.variant != "generic":
        trt_home_raw = env.get("VSMLRT_TENSORRT_HOME") or env.get("TENSORRT_HOME")
        if not trt_home_raw:
            raise RuntimeError("VSMLRT_TENSORRT_HOME is required for CUDA source fallback.")
        trt_home = Path(trt_home_raw).expanduser().resolve()
        cuda_root = Path(env["CUDAToolkit_ROOT"]).expanduser().resolve() if env.get("CUDAToolkit_ROOT") else None
        cmake_args = [f"-DVAPOURSYNTH_INCLUDE_DIRECTORY={headers}", f"-DTENSORRT_HOME={trt_home}", "-DUSE_NVINFER_PLUGIN=ON"]
        if cuda_root:
            cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")
        cmake_build(ROOT / "vstrt", build_root / "vstrt", build_root / "install-vstrt", cmake_args, env)
        shutil.copy2(find_library(build_root / "install-vstrt", "libvstrt"), stage / f"vstrt{native_suffix}")
        runtime_roots = roots_from_env(env, trt_home / "lib", cuda_root / "lib64" if cuda_root else None)
        if args.variant == "cu129":
            if not env.get("VSMLRT_TENSORRT_RTX_HOME"):
                raise RuntimeError("VSMLRT_TENSORRT_RTX_HOME is required for Linux cu129")
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
        copy_builder_resources(stage, runtime_roots)
        for env_name, filename in (("VSMLRT_TRTEXEC_PATH", "trtexec"), ("VSMLRT_TRTEXEC_BUILD_METADATA", "trtexec-build.json")):
            value = env.get(env_name)
            if not value:
                raise RuntimeError(f"{env_name} is required for Linux CUDA source fallback")
            helper_dir = stage / "vsmlrt-cuda"
            helper_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(value).expanduser().resolve(), helper_dir / filename)
        if args.variant == "cu129":
            value = env.get("VSMLRT_TENSORRT_RTX_PATH")
            if not value:
                raise RuntimeError("VSMLRT_TENSORRT_RTX_PATH is required for Linux cu129")
            shutil.copy2(Path(value).expanduser().resolve(), stage / "vsmlrt-cuda" / "tensorrt_rtx")
    if sys.platform == "linux":
        write_elf_soname_aliases(stage)
        # The dependency check reads DT_NEEDED, which patchelf leaves alone, and
        # runs first because rewriting a dynamic section confuses the reader.
        verify_elf_dependencies(stage)
        write_origin_runpaths(stage)
    write_manifest(stage)


if __name__ == "__main__":
    main()
