from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API4_PYTHON = Path(r"F:\vpy-api4\vapoursynth-portable-R77\python.exe")
DEFAULT_API3_PYTHON = Path(r"C:\green\WPy64-313110\python\python.exe")

BACKENDS = ("ncnn", "ov_cpu", "ov_gpu", "trt")
MODELS = ("dpir", "waifu2x_cunet_noise3", "animejanaiV3_HD_L1")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def make_worker_command(
    python: Path,
    env_name: str,
    backend: str,
    model: str,
    output_dir: Path,
    width: int,
    height: int,
) -> list[str]:
    return [
        str(python),
        str(Path(__file__).resolve()),
        "--worker",
        "--env-name",
        env_name,
        "--backend",
        backend,
        "--model",
        model,
        "--output-dir",
        str(output_dir),
        "--width",
        str(width),
        "--height",
        str(height),
    ]


def run_worker_case(
    python: Path,
    env_name: str,
    backend: str,
    model: str,
    output_dir: Path,
    width: int,
    height: int,
    timeout: int,
) -> dict:
    command = make_worker_command(python, env_name, backend, model, output_dir, width, height)
    started_at = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        report = {
            "env": env_name,
            "backend": backend,
            "model": model,
            "ok": False,
            "error": f"worker timed out after {timeout} seconds",
        }
        report_path = output_dir / env_name / backend / f"{model}.json"
        report["worker_returncode"] = returncode
        report["worker_elapsed_sec"] = time.monotonic() - started_at
        report["worker_stdout"] = stdout
        report["worker_stderr"] = stderr
        write_json(report_path, report)
        return report
    elapsed = time.monotonic() - started_at
    report_path = output_dir / env_name / backend / f"{model}.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "env": env_name,
            "backend": backend,
            "model": model,
            "ok": False,
            "error": "worker did not write a report",
        }
    report["worker_returncode"] = returncode
    report["worker_elapsed_sec"] = elapsed
    report["worker_stdout"] = stdout
    report["worker_stderr"] = stderr
    if returncode != 0 and report.get("ok", False):
        report["ok"] = False
        report["error"] = f"worker exited with code {returncode}"
    write_json(report_path, report)
    return report


def compare_arrays(api3_report: dict, api4_report: dict) -> dict:
    import numpy as np

    result = {
        "ok": False,
        "exact": False,
        "same_shape": False,
        "max_abs": None,
        "mean_abs": None,
        "rmse": None,
        "plane_diffs": [],
    }
    if not api3_report.get("ok") or not api4_report.get("ok"):
        result["reason"] = "one or both workers failed"
        return result
    if api3_report.get("width") != api4_report.get("width") or api3_report.get("height") != api4_report.get("height"):
        result["reason"] = "output dimensions differ"
        return result
    if api3_report.get("format") != api4_report.get("format"):
        result["reason"] = "output formats differ"
        return result

    api3_npz = Path(api3_report["array_path"])
    api4_npz = Path(api4_report["array_path"])
    a3 = np.load(api3_npz)
    a4 = np.load(api4_npz)
    plane_names = sorted(a3.files)
    if plane_names != sorted(a4.files):
        result["reason"] = "plane sets differ"
        return result

    max_abs = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    exact = True

    for name in plane_names:
        left = a3[name]
        right = a4[name]
        if left.shape != right.shape:
            result["reason"] = f"plane {name} shape differs"
            return result
        diff = left.astype(np.float64) - right.astype(np.float64)
        abs_diff = np.abs(diff)
        plane_max = float(abs_diff.max()) if abs_diff.size else 0.0
        plane_mean = float(abs_diff.mean()) if abs_diff.size else 0.0
        plane_rmse = float(math.sqrt(float((diff * diff).mean()))) if diff.size else 0.0
        plane_exact = bool(np.array_equal(left, right))
        exact = exact and plane_exact
        max_abs = max(max_abs, plane_max)
        total_abs += float(abs_diff.sum())
        total_sq += float((diff * diff).sum())
        total_count += int(diff.size)
        result["plane_diffs"].append(
            {
                "plane": name,
                "shape": list(left.shape),
                "exact": plane_exact,
                "max_abs": plane_max,
                "mean_abs": plane_mean,
                "rmse": plane_rmse,
            }
        )

    result["same_shape"] = True
    result["exact"] = exact
    result["max_abs"] = max_abs
    result["mean_abs"] = total_abs / total_count if total_count else 0.0
    result["rmse"] = math.sqrt(total_sq / total_count) if total_count else 0.0
    result["ok"] = True
    return result


def summarize_comparison(comparison: dict, atol: float) -> str:
    if not comparison.get("ok"):
        return f"FAILED ({comparison.get('reason', 'unknown')})"
    if comparison.get("exact"):
        return "EXACT"
    max_abs = comparison.get("max_abs")
    if max_abs is not None and max_abs <= atol:
        return f"CLOSE max_abs={max_abs:.3g}"
    return f"DIFF max_abs={max_abs:.3g}"


def run_parent(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = {
        "api3": args.api3_python.resolve(),
        "api4": args.api4_python.resolve(),
    }
    for env_name, python in envs.items():
        if not python.exists():
            raise FileNotFoundError(f"{env_name} python not found: {python}")

    summary = {
        "api3_python": str(envs["api3"]),
        "api4_python": str(envs["api4"]),
        "width": args.width,
        "height": args.height,
        "atol": args.atol,
        "cases": [],
    }

    for backend in BACKENDS:
        for model in MODELS:
            print(f"== {backend} / {model} ==", flush=True)
            reports = {}
            for env_name, python in envs.items():
                print(f"  running {env_name}...", flush=True)
                reports[env_name] = run_worker_case(
                    python=python,
                    env_name=env_name,
                    backend=backend,
                    model=model,
                    output_dir=output_dir,
                    width=args.width,
                    height=args.height,
                    timeout=args.timeout,
                )
                status = "ok" if reports[env_name].get("ok") else "error"
                print(f"  {env_name}: {status}", flush=True)

            comparison = compare_arrays(reports["api3"], reports["api4"])
            status = summarize_comparison(comparison, args.atol)
            print(f"  compare: {status}", flush=True)
            summary["cases"].append(
                {
                    "backend": backend,
                    "model": model,
                    "api3": reports["api3"],
                    "api4": reports["api4"],
                    "comparison": comparison,
                    "status": status,
                }
            )

    write_json(output_dir / "summary.json", summary)

    print()
    print("Summary")
    print("-------")
    for case in summary["cases"]:
        print(f"{case['backend']:6s} {case['model']:24s} {case['status']}")
    print()
    print(f"Wrote {output_dir / 'summary.json'}")
    return 0


def backend_object(vsmlrt, backend: str, engine_folder: Path):
    if backend == "ncnn":
        return vsmlrt.BackendV2.NCNN_VK(fp16=False, device_id=0)
    if backend == "ov_cpu":
        return vsmlrt.BackendV2.OV_CPU()
    if backend == "ov_gpu":
        return vsmlrt.BackendV2.OV_GPU(fp16=False, device_id=0)
    if backend == "trt":
        engine_folder.mkdir(parents=True, exist_ok=True)
        return vsmlrt.BackendV2.TRT(
            fp16=False,
            tf32=False,
            use_cuda_graph=False,
            static_shape=True,
            device_id=0,
            engine_folder=str(engine_folder),
        )
    raise ValueError(f"unknown backend: {backend}")


def make_source_clip(vs, core, width: int, height: int):
    import numpy as np

    base = core.std.BlankClip(width=width, height=height, format=vs.RGBS, length=1)

    def make_frame(n, f):
        out = f.copy()
        yy, xx = np.mgrid[0 : out.height, 0 : out.width]
        planes = [
            ((xx * 3 + yy * 5) % 23) / 22.0,
            (xx + 0.25 * ((yy % 7) / 6.0)) / max(out.width - 1, 1),
            (yy + 0.25 * ((xx % 5) / 4.0)) / max(out.height - 1, 1),
        ]
        for plane, values in enumerate(planes):
            np.asarray(out[plane])[:, :] = values.astype(np.float32)
        return out

    return core.std.ModifyFrame(base, base, make_frame)


def run_model(vsmlrt, clip, model: str, backend):
    if model == "dpir":
        return vsmlrt.DPIR(
            clip,
            strength=5.0,
            model=vsmlrt.DPIRModel.drunet_color,
            backend=backend,
        )
    if model == "waifu2x_cunet_noise3":
        return vsmlrt.Waifu2x(
            clip,
            noise=3,
            scale=1,
            model=vsmlrt.Waifu2xModel.cunet,
            backend=backend,
        )
    if model == "animejanaiV3_HD_L1":
        return vsmlrt.RealESRGAN(
            clip,
            model=vsmlrt.RealESRGANModel.animejanaiV3_HD_L1,
            backend=backend,
        )
    raise ValueError(f"unknown model: {model}")


def version_map(core, namespace: str) -> dict:
    if namespace == "ov_cpu" or namespace == "ov_gpu":
        namespace = "ov"
    if namespace == "ncnn":
        namespace = "ncnn"
    if namespace == "trt":
        namespace = "trt"
    if not hasattr(core, namespace):
        return {"available": False}
    plugin = getattr(core, namespace)
    try:
        data = plugin.Version()
    except Exception as exc:
        return {"available": True, "version_error": repr(exc)}
    cleaned = {"available": True}
    for key, value in data.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8", "replace")
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        cleaned[str(key)] = value
    return cleaned


def frame_report(frame) -> tuple[dict, dict]:
    import numpy as np

    arrays = {}
    planes = []
    overall_hash = hashlib.sha256()
    for plane in range(frame.format.num_planes):
        arr = np.ascontiguousarray(np.asarray(frame[plane]))
        arrays[f"p{plane}"] = arr
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        overall_hash.update(arr.tobytes())
        planes.append(
            {
                "plane": plane,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": digest,
                "min": float(arr.min()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
                "mean": float(arr.mean()) if arr.size else 0.0,
                "std": float(arr.std()) if arr.size else 0.0,
            }
        )
    return arrays, {"sha256": overall_hash.hexdigest(), "planes": planes}


def run_worker(args: argparse.Namespace) -> int:
    report_dir = args.output_dir / args.env_name / args.backend
    report_path = report_dir / f"{args.model}.json"
    array_path = report_dir / f"{args.model}.npz"
    report = {
        "env": args.env_name,
        "backend": args.backend,
        "model": args.model,
        "ok": False,
        "array_path": str(array_path),
    }
    try:
        import numpy as np
        import vapoursynth as vs
        import vsmlrt

        core = vs.core
        source = make_source_clip(vs, core, args.width, args.height)
        backend = backend_object(
            vsmlrt,
            args.backend,
            args.output_dir / "trt-engines" / args.env_name / args.backend / args.model,
        )
        output = run_model(vsmlrt, source, args.model, backend)
        started_at = time.monotonic()
        frame = output.get_frame(0)
        render_elapsed = time.monotonic() - started_at
        arrays, stats = frame_report(frame)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(array_path, **arrays)

        report.update(
            {
                "ok": True,
                "python": sys.executable,
                "python_version": sys.version,
                "core": str(core),
                "vsmlrt_path": getattr(vsmlrt, "__file__", None),
                "plugin_version": version_map(core, args.backend),
                "width": output.width,
                "height": output.height,
                "format": output.format.name,
                "render_elapsed_sec": render_elapsed,
                **stats,
            }
        )
    except BaseException as exc:
        report.update(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(report_path, report)
        return 1

    write_json(report_path, report)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare API3/R73 and API4/R77 installed vs-mlrt backend outputs.",
    )
    parser.add_argument("--api3-python", type=Path, default=DEFAULT_API3_PYTHON)
    parser.add_argument("--api4-python", type=Path, default=DEFAULT_API4_PYTHON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "verification-api3-api4-vsmlrt",
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--atol", type=float, default=1e-5)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--env-name", choices=("api3", "api4"), help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODELS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and (not args.env_name or not args.backend or not args.model):
        parser.error("--worker requires --env-name, --backend, and --model")
    return args


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
