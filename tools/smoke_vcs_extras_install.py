from __future__ import annotations

import argparse
import atexit
import os
import re
import site
import subprocess
import sys
from pathlib import Path


_VS_POLICY = None

SUBPROCESS_POLICY_SNIPPET = r"""
import atexit

_policy = None

class IsolatedEnvironmentPolicy:
    def __init__(self, flags):
        self._flags = flags
        self._api = None
        self._environment = None

    def on_policy_registered(self, api):
        self._api = api
        self._environment = api.create_environment(self._flags)

    def on_policy_cleared(self):
        if self._api is not None and self._environment is not None:
            environment = self._environment
            self._environment = None
            self._api.destroy_environment(environment)
        self._api = None

    def get_current_environment(self):
        return self._environment

    def set_environment(self, environment):
        previous = self._environment
        if environment is not None:
            self._environment = environment
        return previous

    def is_alive(self, environment):
        return environment is self._environment

    def close(self):
        if self._api is not None and self._environment is not None:
            environment = self._environment
            self._environment = None
            self._api.destroy_environment(environment)

def _close_policy():
    global _policy
    if _policy is not None:
        _policy.close()
        _policy = None

atexit.register(_close_policy)

if not vs.has_policy():
    _policy = IsolatedEnvironmentPolicy(vs.DISABLE_AUTO_LOADING)
    vs.register_policy(_policy)
"""


class IsolatedEnvironmentPolicy:
    def __init__(self, flags: int) -> None:
        self._flags = flags
        self._api = None
        self._environment = None

    def on_policy_registered(self, api: object) -> None:
        self._api = api
        self._environment = api.create_environment(self._flags)

    def on_policy_cleared(self) -> None:
        if self._api is not None and self._environment is not None:
            environment = self._environment
            self._environment = None
            self._api.destroy_environment(environment)
        self._api = None

    def get_current_environment(self) -> object:
        return self._environment

    def set_environment(self, environment: object) -> object:
        previous = self._environment
        if environment is not None:
            self._environment = environment
        return previous

    def is_alive(self, environment: object) -> bool:
        return environment is self._environment

    def close(self) -> None:
        if self._api is not None and self._environment is not None:
            environment = self._environment
            self._environment = None
            self._api.destroy_environment(environment)


def close_vs_policy() -> None:
    global _VS_POLICY
    if _VS_POLICY is not None:
        _VS_POLICY.close()
        _VS_POLICY = None


atexit.register(close_vs_policy)


def site_roots() -> list[Path]:
    override = os.environ.get("VSMLRT_SMOKE_SITE_ROOTS")
    if override:
        return [Path(item).resolve() for item in override.split(os.pathsep) if item]

    roots = [Path(p) for p in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    return roots


def find_path(roots: list[Path], rel: Path) -> Path | None:
    for root in roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def find_glob(roots: list[Path], pattern: str) -> Path | None:
    for root in roots:
        matches = list(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def require_paths(roots: list[Path], paths: list[Path]) -> dict[Path, Path]:
    found: dict[Path, Path] = {}
    missing = []
    for rel in paths:
        path = find_path(roots, rel)
        if path is None:
            missing.append(str(rel))
        else:
            found[rel] = path
    if missing:
        raise SystemExit("Missing installed files: " + ", ".join(missing))
    return found


def forbid_paths(roots: list[Path], paths: list[Path], reason: str) -> None:
    present = [str(rel) for rel in paths if find_path(roots, rel) is not None]
    if present:
        raise SystemExit(f"{reason}: " + ", ".join(present))


def verify_manifest(
    roots: list[Path],
    plugin_prefix: Path,
    variant: str,
) -> None:
    manifest = find_path(roots, plugin_prefix / "manifest.vs")
    if manifest is None:
        raise SystemExit("Installed payload is missing manifest.vs")

    expected = ["vsncnn", "vsov"]
    if variant != "generic":
        expected.append("vstrt")
        if variant == "cu129":
            expected.append("vstrt_rtx")

    lines = manifest.read_text(encoding="ascii").splitlines()
    expected_lines = ["[VapourSynth Manifest V1]", *expected]
    if lines != expected_lines:
        raise SystemExit(f"manifest.vs entries {lines!r} != {expected_lines!r}")
    print("manifest.vs:", ", ".join(expected))


def parse_variant(raw: str) -> str:
    variant = raw.strip()
    if variant not in {"generic", "cu121", "cu129"}:
        raise SystemExit(f"Unknown install variant: {variant!r}")
    return variant


def import_vsmlrt() -> object:
    global _VS_POLICY
    import vapoursynth as vs

    if not vs.has_policy():
        _VS_POLICY = IsolatedEnvironmentPolicy(vs.DISABLE_AUTO_LOADING)
        vs.register_policy(_VS_POLICY)

    import vsmlrt

    print(f"Imported vsmlrt from {vsmlrt.__file__}")
    return vsmlrt


def check_vsmlrt_paths(
    vsmlrt: object,
    expected_models: Path,
    expected_trtexec: Path | None,
    expected_tensorrt_rtx: Path | None,
) -> None:
    models_path = Path(vsmlrt.models_path).resolve()
    if models_path != expected_models.resolve():
        raise SystemExit(f"vsmlrt.models_path={models_path} != {expected_models.resolve()}")
    print(f"models_path: {models_path}")

    if expected_trtexec is not None:
        trtexec_path = Path(vsmlrt.trtexec_path).resolve()
        if trtexec_path != expected_trtexec.resolve():
            raise SystemExit(f"vsmlrt.trtexec_path={trtexec_path} != {expected_trtexec.resolve()}")
        print(f"trtexec_path: {trtexec_path}")

    if expected_tensorrt_rtx is not None:
        tensorrt_rtx_path = Path(vsmlrt.tensorrt_rtx_path).resolve()
        if tensorrt_rtx_path != expected_tensorrt_rtx.resolve():
            raise SystemExit(
                f"vsmlrt.tensorrt_rtx_path={tensorrt_rtx_path} != {expected_tensorrt_rtx.resolve()}"
            )
        print(f"tensorrt_rtx_path: {tensorrt_rtx_path}")


def smoke_load_generic(plugin_dir: Path) -> None:
    load_script = f"""
from pathlib import Path
import vapoursynth as vs
{SUBPROCESS_POLICY_SNIPPET}

plugin_dir = Path({str(plugin_dir)!r})

core = vs.core
for name in ("vsncnn", "vsov"):
    path = plugin_dir / f"{{name}}.dll"
    print(f"Loading {{path}}")
    try:
        core.std.LoadPlugin(path=str(path))
    except vs.Error as exc:
        if "already loaded" not in str(exc):
            raise
        print(f"Already loaded: {{path}}")

print("ncnn:", core.ncnn.Version())
print("ov:", core.ov.Version())
"""
    completed = subprocess.run(
        [sys.executable, "-c", load_script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if any("failed to preload" in line for line in completed.stderr.splitlines()):
        raise SystemExit("Generic preload emitted loader errors.")


def smoke_autoload_generic() -> None:
    load_script = """
import vapoursynth as vs

core = vs.core
print("ncnn:", core.ncnn.Version())
print("ov:", core.ov.Version())
"""
    completed = subprocess.run(
        [sys.executable, "-c", load_script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if any("failed to load" in line.lower() or "failed to preload" in line.lower() for line in completed.stderr.splitlines()):
        raise SystemExit("Generic autoload emitted loader errors.")


def import_names(path: Path) -> list[str]:
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    names: list[str] = []
    for entry_name in ["IMAGE_DIRECTORY_ENTRY_IMPORT", "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]:
        try:
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY[entry_name]])
        except Exception:
            continue
        attr = "DIRECTORY_ENTRY_DELAY_IMPORT" if entry_name.endswith("DELAY_IMPORT") else "DIRECTORY_ENTRY_IMPORT"
        for entry in getattr(pe, attr, []):
            names.append(entry.dll.decode("ascii", errors="ignore").lower())
    return sorted(set(names))


def verify_generic_imports(plugin_dir: Path) -> None:
    forbidden_tokens = (
        "cublas",
        "cuda",
        "cudnn",
        "cufft",
        "cudart",
        "cupti",
        "nvcuda",
        "nvinfer",
        "nvonnxparser",
        "nvrtc",
    )
    targets = [
        plugin_dir / "vsncnn.dll",
        plugin_dir / "vsov.dll",
    ]
    bad: list[str] = []
    for target in targets:
        for name in import_names(target):
            if any(token in name for token in forbidden_tokens):
                bad.append(f"{target.name} -> {name}")
    if bad:
        raise SystemExit("Generic payload must not import CUDA/NVIDIA DLLs:\n" + "\n".join(sorted(set(bad))))


def verify_cuda_imports(roots: list[Path], plugin_dir: Path, cuda_dir: Path, flavor: str) -> list[str]:
    root_dirs = [plugin_dir, cuda_dir]
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    search_dirs = root_dirs + [
        *roots,
        *(root / "vapoursynth" for root in roots),
        system_root / "System32",
        system_root / "SysWOW64",
    ]

    def resolve_import(name: str) -> Path | None:
        if name.startswith(("api-ms-win-", "ext-ms-win-")):
            return Path("<windows-api-set>")
        for directory in search_dirs:
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def check_import_tree(paths: list[Path]) -> list[str]:
        queue = list(paths)
        seen: set[str] = set()
        missing: list[str] = []
        missing_driver: list[str] = []
        while queue:
            path = queue.pop(0)
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            for name in import_names(path):
                resolved = resolve_import(name)
                if resolved is None:
                    if name == "nvcuda.dll":
                        missing_driver.append(f"{path.name} -> {name}")
                        continue
                    missing.append(f"{path.name} -> {name}")
                    continue
                if resolved.parent in root_dirs and resolved.name.lower().endswith(".dll"):
                    queue.append(resolved)
        if missing:
            raise SystemExit("Missing PE imports:\n" + "\n".join(sorted(set(missing))))
        return sorted(set(missing_driver))

    def check_dll_string_refs(path: Path) -> list[str]:
        data = path.read_bytes()
        refs = sorted(
            set(
                match.group(0).decode("ascii", errors="ignore").lower()
                for match in re.finditer(rb"[A-Za-z0-9_.+-]+\.dll", data)
            )
        )
        tokens = (
            "cublas",
            "cuda",
            "cudnn",
            "cufft",
            "cudart",
            "nvinfer",
            "nvrtc",
            "nv",
            "onnx",
            "protobuf",
            "zlib",
        )
        interesting = [ref for ref in refs if any(token in ref for token in tokens)]
        missing: list[str] = []
        driver_refs: list[str] = []
        print(f"DLL string refs in {path.name}:")
        for ref in interesting:
            status = "found" if resolve_import(ref) is not None else "missing"
            if ref == "nvcuda.dll":
                status = "driver"
                driver_refs.append(f"{path.name} -> {ref}")
            elif status == "missing":
                missing.append(f"{path.name} -> {ref}")
            print(f"  {status}: {ref}")
        if missing:
            raise SystemExit("Missing DLL string refs:\n" + "\n".join(missing))
        return driver_refs

    trt_suffix = "_11" if flavor == "cu129" else ""
    pe_targets = [
        plugin_dir / "vstrt.dll",
        cuda_dir / f"nvinfer{trt_suffix}.dll",
        cuda_dir / f"nvinfer_plugin{trt_suffix}.dll",
    ]
    if flavor == "cu129":
        pe_targets.extend([plugin_dir / "vstrt_rtx.dll", cuda_dir / "tensorrt_rtx_1_5.dll"])

    missing_driver = check_import_tree(pe_targets)
    missing_driver.extend(check_dll_string_refs(cuda_dir / f"nvinfer_plugin{trt_suffix}.dll"))
    return sorted(set(missing_driver))


def smoke_load_cuda(plugin_dir: Path, flavor: str, missing_driver: list[str]) -> None:
    load_script = f"""
from pathlib import Path
import vapoursynth as vs
{SUBPROCESS_POLICY_SNIPPET}

core = vs.core

def load_plugin(path):
    try:
        core.std.LoadPlugin(path=str(path))
    except vs.Error as exc:
        if "already loaded" not in str(exc):
            raise

plugin_dir = Path({str(plugin_dir)!r})
load_plugin(plugin_dir / "vstrt.dll")
if {flavor == "cu129"!r}:
    load_plugin(plugin_dir / "vstrt_rtx.dll")
"""
    completed = subprocess.run(
        [sys.executable, "-c", load_script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)

    stderr_lines = completed.stderr.splitlines()
    if missing_driver:
        allowed_driver_loader_failures = {
            item.split(" -> ", 1)[0].lower()
            for item in missing_driver
            if item.lower().endswith("-> nvcuda.dll")
        }

        def is_allowed_driver_loader_failure(line: str) -> bool:
            lower = line.lower()
            if "failed to preload" in lower:
                return any(name in lower for name in allowed_driver_loader_failures)
            if (
                flavor == "cu129"
                and "vstrt_rtx: tensorrt failed to load" in lower
                and "tensorrt_rtx_1_5.dll" in allowed_driver_loader_failures
            ):
                return True
            return False

        stderr_lines = [line for line in stderr_lines if not is_allowed_driver_loader_failure(line)]
    if any("failed to preload" in line or "TensorRT failed to load" in line for line in stderr_lines):
        raise SystemExit("TensorRT preload emitted loader errors.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="Tag-selected VCS install variant: generic, cu121, or cu129.")
    parser.add_argument("--layout-only", action="store_true", help="Only verify installed files and wrapper paths.")
    args = parser.parse_args()

    variant = parse_variant(args.variant)
    cuda_flavor = None if variant == "generic" else variant
    roots = site_roots()
    for root in reversed(roots):
        sys.path.insert(0, str(root))

    plugin_prefix = Path("vapoursynth/plugins/vsmlrt")
    model_paths = require_paths(
        roots,
        [
            Path("vsmlrt.py"),
            plugin_prefix / "models/dpir",
            plugin_prefix / "models/rife",
            plugin_prefix / "models/RealESRGANv2/animejanaiV2L1.onnx",
            plugin_prefix / "models/RealESRGANv2/animejanaiV3-HD-L1.onnx",
            plugin_prefix / "models/RealESRGANv2/Ani4Kv2-G6i2-Compact.onnx",
        ],
    )
    expected_models = model_paths[plugin_prefix / "models/dpir"].parents[0]

    expected_trtexec: Path | None = None
    expected_tensorrt_rtx: Path | None = None

    generic_paths = require_paths(
        roots,
        [
            plugin_prefix / "vsncnn.dll",
            plugin_prefix / "vsov.dll",
            plugin_prefix / "cache.json",
            plugin_prefix / "openvino.dll",
            plugin_prefix / "tbb12.dll",
        ],
    )
    generic_dir = generic_paths[plugin_prefix / "vsncnn.dll"].parent
    forbid_paths(
        roots,
        [
            plugin_prefix / "vsort.dll",
            plugin_prefix / "vsort",
            plugin_prefix / "onnxruntime.dll",
            plugin_prefix / "DirectML.dll",
            plugin_prefix / "vsov",
        ],
        "Install contains an excluded backend or duplicated OpenVINO directory",
    )

    if cuda_flavor is not None:
        trt_suffix = "_11" if cuda_flavor == "cu129" else ""
        cuda_prefix = plugin_prefix
        cuda_paths = require_paths(
            roots,
            [
                cuda_prefix / "vstrt.dll",
                cuda_prefix / "vsmlrt-cuda" / f"nvinfer{trt_suffix}.dll",
                cuda_prefix / "vsmlrt-cuda" / f"nvinfer_plugin{trt_suffix}.dll",
                cuda_prefix / "vsmlrt-cuda" / "cublas64_12.dll",
                cuda_prefix / "vsmlrt-cuda" / "cublasLt64_12.dll",
                cuda_prefix / "vsmlrt-cuda" / "cudart64_12.dll",
                cuda_prefix / "vsmlrt-cuda" / "trtexec.exe",
            ],
        )
        if cuda_flavor == "cu129":
            require_paths(
                roots,
                [
                    cuda_prefix / "vstrt_rtx.dll",
                    cuda_prefix / "vsmlrt-cuda" / "nvonnxparser_11.dll",
                    cuda_prefix / "vsmlrt-cuda" / "tensorrt_rtx_1_5.dll",
                    cuda_prefix / "vsmlrt-cuda" / "tensorrt_rtx.exe",
                ],
            )
        else:
            forbid_paths(
                roots,
                [plugin_prefix / "vstrt_rtx.dll"],
                "cu121 install unexpectedly contains TensorRT-RTX payload",
            )

        missing_globs = [
            pattern
            for pattern in [
                str(cuda_prefix / "vsmlrt-cuda" / "cudnn*.dll"),
                str(cuda_prefix / "vsmlrt-cuda" / "nvrtc*.dll"),
            ]
            if find_glob(roots, pattern) is None
        ]
        if missing_globs:
            raise SystemExit("Missing installed files matching: " + ", ".join(missing_globs))

        cuda_dir = cuda_paths[cuda_prefix / "vstrt.dll"].parent / "vsmlrt-cuda"
        expected_trtexec = cuda_dir / "trtexec.exe"
        if cuda_flavor == "cu129":
            expected_tensorrt_rtx = cuda_dir / "tensorrt_rtx.exe"
    else:
        forbid_paths(
            roots,
            [
                plugin_prefix / "vstrt.dll",
                plugin_prefix / "vstrt_rtx.dll",
                plugin_prefix / "vsmlrt-cuda",
            ],
            "Generic-only install unexpectedly contains CUDA payload",
        )

    vsmlrt = import_vsmlrt()
    check_vsmlrt_paths(vsmlrt, expected_models, expected_trtexec, expected_tensorrt_rtx)
    verify_manifest(roots, plugin_prefix, variant)

    if args.layout_only:
        print(f"Verified layout for variant: {args.variant}")
        return

    if generic_dir is not None:
        verify_generic_imports(generic_dir)
        if cuda_flavor is None:
            smoke_autoload_generic()
        smoke_load_generic(generic_dir)
        print(f"Verified generic payload under: {generic_dir}")

    if cuda_flavor is not None:
        cuda_plugin_dir = find_path(roots, plugin_prefix / "vstrt.dll")
        assert cuda_plugin_dir is not None
        cuda_plugin_dir = cuda_plugin_dir.parent
        missing_driver = verify_cuda_imports(roots, cuda_plugin_dir, cuda_plugin_dir / "vsmlrt-cuda", cuda_flavor)
        if missing_driver:
            print("Allowed missing NVIDIA driver imports on GitHub runner:")
            print("\n".join(missing_driver))
        smoke_load_cuda(cuda_plugin_dir, cuda_flavor, missing_driver)
        print(f"Verified {cuda_flavor} payload under: {cuda_plugin_dir}")


if __name__ == "__main__":
    main()
