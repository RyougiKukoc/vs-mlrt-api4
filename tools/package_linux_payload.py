"""Create a checked Linux release payload from a staged native directory."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    parser.add_argument("--component", choices=["generic", "tensorrt", "cuda", "cudnn", "cudnn-part", "builder", "builder-tools", "builder-resource", "rtx", "all"], default="all")
    parser.add_argument("--resource-index", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--part-index", type=int, choices=[1, 2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    stage = args.stage_dir.resolve()
    files = {
        p.relative_to(stage).as_posix(): p
        for p in stage.rglob("*")
        if p.is_file() and p.relative_to(stage).parts[:1] != ("models",)
    }
    if args.component == "builder-resource" and args.resource_index is None:
        raise RuntimeError("--resource-index is required for builder-resource payloads")
    if args.component == "cudnn-part" and args.part_index is None:
        raise RuntimeError("--part-index is required for cudnn-part payloads")
    builder_resources = sorted(
        (name, path) for name, path in files.items() if "builder_resource" in PurePosixPath(name).name
    )
    resource_slots: list[list[str]] = [[], [], [], []]
    resource_sizes = [0, 0, 0, 0]
    for name, path in sorted(builder_resources, key=lambda item: item[1].stat().st_size, reverse=True):
        slot = min(range(4), key=resource_sizes.__getitem__)
        resource_slots[slot].append(name)
        resource_sizes[slot] += path.stat().st_size
    selected_resources = set(resource_slots[(args.resource_index or 1) - 1])
    cudnn_files = sorted(
        (name, path) for name, path in files.items() if PurePosixPath(name).name.startswith("libcudnn")
    )
    cudnn_slots: list[list[str]] = [[], []]
    cudnn_sizes = [0, 0]
    for name, path in sorted(cudnn_files, key=lambda item: item[1].stat().st_size, reverse=True):
        slot = min(range(2), key=cudnn_sizes.__getitem__)
        cudnn_slots[slot].append(name)
        cudnn_sizes[slot] += path.stat().st_size
    selected_cudnn = set(cudnn_slots[(args.part_index or 1) - 1])

    def keep(name: str) -> bool:
        base = PurePosixPath(name).name
        if args.component == "generic": return base in {"vsncnn.so", "vsov.so", "manifest.vs"} or base.startswith(("libopenvino", "libtbb", "libonnx", "libprotobuf", "libncnn"))
        if args.component == "tensorrt": return base == "vstrt.so" or (base.startswith(("libnvinfer", "libnvonnxparser", "libnvparsers")) and "builder_resource" not in base)
        if args.component == "cuda": return base.startswith(("libcublas", "libcudart", "libcufft", "libnvblas", "libnvrtc", "libnvJitLink", "libnvvm"))
        if args.component == "cudnn": return base.startswith("libcudnn") and (args.part_index is None or name in selected_cudnn)
        if args.component == "cudnn-part": return base.startswith("libcudnn") and name in selected_cudnn
        if args.component == "builder": return name in {"vsmlrt-cuda/trtexec", "vsmlrt-cuda/trtexec-build.json"} or "builder_resource" in base
        if args.component == "builder-tools": return name in {"vsmlrt-cuda/trtexec", "vsmlrt-cuda/trtexec-build.json"}
        if args.component == "builder-resource": return name in selected_resources
        if args.component == "rtx": return base == "vstrt_rtx.so" or base.startswith("libtensorrt_rtx") or name == "vsmlrt-cuda/tensorrt_rtx"
        return True
    payload = {n: p for n, p in files.items() if keep(n)}
    manifests = {"generic": b"[VapourSynth Manifest V1]\nvsncnn\nvsov\n", "tensorrt": b"[VapourSynth Manifest V1]\nvstrt\n", "rtx": b"[VapourSynth Manifest V1]\nvstrt_rtx\n"}
    if args.component in manifests: payload["manifest.vs"] = manifests[args.component]
    expected = {"generic": {"vsncnn.so", "vsov.so"}, "tensorrt": {"vstrt.so"}, "rtx": {"vstrt_rtx.so"}}.get(args.component, set())
    if not expected.issubset(payload): raise RuntimeError(f"Missing required files: {sorted(expected - set(payload))}")
    if args.component in {"builder", "builder-resource"} and not any("builder_resource" in n for n in payload): raise RuntimeError("Builder resources missing")
    if args.variant != "generic" and args.component in {"all", "builder", "builder-tools"} and not {"vsmlrt-cuda/trtexec", "vsmlrt-cuda/trtexec-build.json"}.issubset(payload): raise RuntimeError("trtexec builder files missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, source in sorted(payload.items()):
            archive_name = str(PurePosixPath("vsmlrt") / name)
            if isinstance(source, bytes):
                archive.writestr(archive_name, source)
            else:
                archive.write(source, archive_name)
    inventory = {"variant": args.variant, "component": args.component, "asset": args.output.name, "sha256": digest(args.output), "files": sorted(f"vsmlrt/{n}" for n in payload)}
    if args.inventory: args.inventory.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory, indent=2))
if __name__ == "__main__": main()
