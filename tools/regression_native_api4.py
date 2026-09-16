"""Check MLRT artifact ownership and NCNN conversion regressions in fresh processes.

Requires VapourSynth R77. Model checks need NumPy and ONNX. --exercise-ncnn needs
a Vulkan device with fp16 support; --exercise-ort uses the CPU provider.
Nothing is installed or written into the plugin
directory. Every case records the actual DLL hash/version and fails on crashes,
timeouts, unexpected exceptions, non-finite output, or incorrect pixels.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


class IsolatedCores:
    def __init__(self, vs):
        self.vs = vs
        self.environments = []
        self.current = None

    def on_policy_registered(self, api):
        self.api = api
        self.new_environment()

    def on_policy_cleared(self):
        self.close()

    def get_current_environment(self):
        return self.current

    def set_environment(self, environment):
        previous = self.current
        if environment is not None:
            self.current = environment
        return previous

    def is_alive(self, environment):
        return environment in self.environments

    def new_environment(self):
        environment = self.api.create_environment(self.vs.DISABLE_AUTO_LOADING)
        self.environments.append(environment)
        self.current = environment
        return environment

    def destroy(self, environment):
        self.api.destroy_environment(environment)
        self.environments.remove(environment)
        if self.current is environment:
            self.current = self.environments[0] if self.environments else None

    def close(self):
        for environment in self.environments[:]:
            self.destroy(environment)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def decode_path(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def load_core(vs, plugin, namespace, suffix, path=None):
    core = vs.core.core
    core.std.LoadPlugin(
        path or str(plugin), forcens="review" + suffix,
        forceid="io.github.vsmlrt.regression." + namespace + "." + suffix,
    )
    return core, getattr(core, "review" + suffix)


def check_core_handles(vs, policy, args):
    first_environment = policy.current
    first, first_plugin = load_core(vs, args.plugin, args.namespace, "first")
    before = first_plugin.Version()
    first_path = decode_path(before["path"])
    second_environment = policy.new_environment()
    # The OS shares this DLL, while VSPlugin retains the supplied path spelling.
    alternate = str(args.plugin).upper() if os.name == "nt" else str(args.plugin.parent) + "/./" + args.plugin.name
    second, second_plugin = load_core(vs, args.plugin, args.namespace, "second", alternate)
    second_version = second_plugin.Version()
    policy.current = first_environment
    after = first_plugin.Version()
    if decode_path(after["path"]) != first_path:
        raise AssertionError("First core's Version path changed after loading the second core")
    if first_path == decode_path(second_version["path"]):
        raise AssertionError("Alternate path was normalized; the cross-core test is inconclusive")
    del second_plugin, second
    policy.destroy(second_environment)
    gc.collect()
    if decode_path(first_plugin.Version()["path"]) != first_path:
        raise AssertionError("First core's plugin handle did not survive second-core destruction")
    return {"version": before, "second_version": second_version, "forced_ids": True,
            "survived_other_core_destruction": True}


def make_identity_model(path, operator, kernel, width, height):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    attrs = {} if kernel == "omitted" else {"kernel_shape": [1, 1]}
    if kernel == "mismatch":
        attrs["kernel_shape"] = [3, 3]
    if kernel == "zero":
        attrs["kernel_shape"] = [0, 1]
    if kernel == "rank":
        attrs["kernel_shape"] = [1]
    graph = helper.make_graph(
        [helper.make_node(operator, ["input", "weight"], ["output"], **attrs)],
        "identity_convolution",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, height, width])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, height, width])],
        [numpy_helper.from_array(np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=8)
    if kernel in ("explicit", "omitted"):
        onnx.checker.check_model(model)
    onnx.save(model, path)


def check_identity_model(vs, policy, args, directory):
    import numpy as np

    case = args.worker
    operator = "ConvTranspose" if case.startswith("deconv") else "Conv"
    kernel = next((name for name in ("omitted", "mismatch", "zero", "rank") if name in case), "explicit")
    fp16 = "fp16" in case
    half_input = "half-input" in case
    half_output = "half-output" in case
    width = height = 8 if "aligned-control" in case else 9
    model_path = directory / "identity.onnx"
    missing_model = case == "missing-model"
    if not missing_model:
        make_identity_model(model_path, operator, kernel, width, height)
    core, plugin = load_core(vs, args.plugin, args.namespace, "model")
    version = plugin.Version()
    base = core.std.BlankClip(width=width, height=height, format=vs.RGBH if half_input else vs.RGBS, length=4)

    def pattern(n, f):
        frame = f.copy()
        y, x = np.mgrid[:height, :width]
        for plane in range(3):
            np.asarray(frame[plane])[:] = (plane + 1) / 8 + (x % 4) / 64 + (y % 4) / 256 + n / 32
        return frame

    source = core.std.ModifyFrame(base, base, pattern)
    arguments = {"network_path": str(model_path), "fp16": fp16, "tilesize": [width, height]}
    if args.namespace == "ov":
        arguments["device"] = "CPU"
    elif args.namespace in {"trt", "trt_rtx"}:
        if args.trt_engine is None:
            raise RuntimeError("--trt-engine is required for TensorRT model execution")
        arguments = {"engine_path": str(args.trt_engine), "tilesize": [width, height],
                     "device_id": args.device_id, "num_streams": 2}
    else:
        arguments.update(output_format=int(half_output), device_id=args.device_id, num_streams=2)
    if args.namespace == "ort":
        arguments["provider"] = "CPU"
    if case == "conv-builtin":
        arguments.update(network_path=model_path.name, builtin=True,
                         builtindir=os.path.relpath(directory, args.plugin.parent))
    if missing_model and args.namespace not in {"trt", "trt_rtx"}:
        arguments["network_path"] = str(directory / "does-not-exist.onnx")
    elif missing_model:
        arguments["engine_path"] = str(directory / "does-not-exist.engine")
    if missing_model or kernel in ("mismatch", "zero", "rank"):
        try:
            rejected = plugin.Model(source, **arguments)
            rejected.get_frame(0)
        except vs.Error as error:
            return {"version": version, "expected_error": str(error)}
        raise AssertionError("Invalid model input was not rejected")

    def make_output():
        if case == "conv-flexible":
            value = plugin.Model(source, flexible_output_prop="regression", **arguments)
            if not isinstance(value, dict) or value["num_planes"] != 3:
                raise AssertionError("Missing or incorrect flexible output metadata")
            channels = [value["clip"].std.PropToClip(prop=f"regression{plane}") for plane in range(3)]
            return core.std.ShufflePlanes(channels, [0, 0, 0], vs.RGB)
        return plugin.Model(source, **arguments)

    # Overlapping filter instances and outstanding requests exercise the
    # allocator pool as instances are destroyed and resources are reacquired.
    held = make_output()
    frame_reports = []
    for iteration in range(3):
        output = make_output()
        pending = [output.get_frame_async(n) for n in range(4)]
        held_pending = held.get_frame_async(iteration)
        for n, future in enumerate(pending):
            with future.result() as frame, source.get_frame(n) as expected:
                planes = []
                for plane in range(3):
                    actual = np.asarray(frame[plane])
                    if not np.isfinite(actual).all():
                        raise AssertionError(f"Non-finite output in plane {plane}, frame {n}")
                    reference = np.asarray(expected[plane])
                    if not np.array_equal(actual, reference):
                        difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
                        position = np.unravel_index(int(difference.argmax()), difference.shape)
                        raise AssertionError(
                            f"Incorrect identity output in plane {plane}, frame {n}: "
                            f"max_abs={difference.max():.9g}, mean_abs={difference.mean():.9g}, "
                            f"at={position}, actual={float(actual[position]):.9g}, "
                            f"expected={float(reference[position]):.9g}, dtype={actual.dtype}"
                        )
                    planes.append({
                        "sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
                        "min": float(actual.min()),
                        "max": float(actual.max()),
                        "average": float(actual.mean()),
                    })
                if iteration == 0:
                    frame_reports.append({"n": n, "planes": planes})
        held_pending.result().close()
        del output, pending, held_pending, future
        gc.collect()
    del held, source, base, plugin, core
    gc.collect()
    return {"version": version, "width": width, "height": height,
            "frames_per_instance": 4, "iterations": 3, "exact_finite_identity": True,
            "frame_reports": frame_reports}


def worker(args):
    import vapoursynth as vs

    report = {"case": args.worker, "python": sys.executable, "plugin": str(args.plugin),
              "plugin_sha256": hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
              "namespace": args.namespace, "autoload_disabled": True, "ok": False}
    with contextlib.ExitStack() as stack:
        if os.name == "nt":
            for path in [args.plugin.parent, *args.dll_dir]:
                stack.enter_context(os.add_dll_directory(str(path)))
        policy = IsolatedCores(vs)
        try:
            vs.register_policy(policy)
            report["core"] = str(vs.core)
            if args.worker == "multicore":
                report.update(check_core_handles(vs, policy, args))
            else:
                with tempfile.TemporaryDirectory(prefix="model-", dir=args.output.parent) as temporary:
                    report.update(check_identity_model(vs, policy, args, Path(temporary)))
            report["ok"] = True
        except Exception:
            report["error"] = traceback.format_exc()
        finally:
            policy.close()
        write_json(args.output, report)
    return 0 if report["ok"] else 1


def parent(args):
    cases = ["multicore"]
    if args.exercise_model:
        cases += ["conv-explicit", "missing-model"]
    if args.exercise_ncnn:
        if args.namespace != "ncnn":
            raise ValueError("--exercise-ncnn requires --namespace ncnn")
        cases += ["conv-explicit", "conv-omitted", "deconv-explicit", "deconv-omitted",
                  "conv-fp16", "conv-fp16-half-output", "conv-fp16-half-input",
                  "conv-fp16-half-input-half-output", "conv-fp16-aligned-control",
                  "conv-builtin", "conv-mismatch", "conv-zero", "conv-rank", "deconv-rank"]
    if args.exercise_ort:
        if args.namespace != "ort":
            raise ValueError("--exercise-ort requires --namespace ort")
        cases += ["conv-explicit", "conv-omitted", "conv-fp16", "conv-fp16-half-input",
                  "conv-fp16-half-output", "conv-builtin", "conv-flexible"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="native-cases-", dir=args.output.parent) as temporary:
        for case in cases:
            destination = Path(temporary) / f"{case}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--plugin", str(args.plugin),
                       "--namespace", args.namespace, "--worker", case, "--output", str(destination),
                       "--device-id", str(args.device_id)]
            if args.trt_engine is not None:
                command.extend(["--trt-engine", str(args.trt_engine)])
            for path in args.dll_dir:
                command.extend(["--dll-dir", str(path)])
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                result = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {
                    "case": case, "ok": False, "error": "Worker did not write its report"}
                result.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
                result["ok"] = result.get("ok", False) and process.returncode == 0
            except subprocess.TimeoutExpired as error:
                result = {"case": case, "ok": False, "error": f"Timed out after {args.timeout}s",
                          "stdout": str(error.stdout), "stderr": str(error.stderr)}
            results.append(result)
            print(f"{case}: {'PASS' if result['ok'] else 'FAIL'}", flush=True)
    report = {"ok": all(result["ok"] for result in results), "cases": results,
              "plugin": str(args.plugin), "plugin_sha256": hashlib.sha256(args.plugin.read_bytes()).hexdigest()}
    write_json(args.output, report)
    print(f"Report: {args.output}")
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--namespace", choices=["ncnn", "ov", "ort", "trt", "trt_rtx"], required=True)
    parser.add_argument("--output", type=Path, default=Path("verification-native-api4.json"))
    parser.add_argument("--exercise-ncnn", action="store_true")
    parser.add_argument("--exercise-ort", action="store_true")
    parser.add_argument("--exercise-model", action="store_true", help="render an identity ONNX model and invalid-model error path")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--trt-engine", type=Path)
    parser.add_argument("--dll-dir", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.plugin = args.plugin.resolve(strict=True)
    args.output = args.output.resolve()
    if args.trt_engine is not None:
        args.trt_engine = args.trt_engine.resolve(strict=True)
    args.dll_dir = [path.resolve(strict=True) for path in args.dll_dir]
    return worker(args) if args.worker else parent(args)


if __name__ == "__main__":
    sys.exit(main())
