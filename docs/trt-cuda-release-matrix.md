# TensorRT CUDA Release Matrix

This document describes the Windows TensorRT payloads published by this fork
and the rules for keeping those payloads in sync with the default-branch VCS
installer.

The short version:

- Users install from the default branch and select payloads with pip extras.
- Release tags such as `cu121`, `cu129`, `generic`, and `models` are binary
  asset slots consumed by the build hooks.
- `cu121` and `cu129` must not be installed together in one Python
  environment.
- `generic` is NVIDIA-free and may be combined with either CUDA extra.

## User-Facing Matrix

| Extra | Native plugins | CUDA / GPU line | Main dependency versions | Intended users |
| --- | --- | --- | --- | --- |
| `generic` | `vsncnn`, `vsov` | No NVIDIA CUDA payload | ncnn/Vulkan and OpenVINO runtime DLLs | Intel, AMD, or NVIDIA users who want non-TensorRT backends. |
| `cu121` | `vstrt` | CUDA 12.1.1 redist | TensorRT 8.6.1.6, cuDNN 8.9.7 | NVIDIA systems pinned to drivers that can run CUDA 12.1/12.2-era user-mode libraries. |
| `cu129` | `vstrt`, `vstrt_rtx` | CUDA 12.9.1 redist | TensorRT 11.1.0.106, TensorRT-RTX 1.5.0.114, cuDNN 9.19 | Current NVIDIA systems with drivers new enough for CUDA 12.9 user-mode libraries. |

`cu121` does not include `vstrt_rtx`. NVIDIA did not publish a matching
Windows TensorRT-RTX package for the CUDA 12.1/TensorRT 8.6 line.

The selected extra changes only the installed native payload. The Python entry
point remains stable:

```python
import vsmlrt

out = vsmlrt.DPIR(clip, strength=5.0, backend=vsmlrt.Backend.TRT(fp16=True))
```

## Install Mapping

Normal installs target this repository's default branch:

```powershell
pip install "vs-mlrt[generic] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu121] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu129] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu121,generic] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu129,generic] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
```

Do not document direct CUDA release-tag checkouts as the primary install path.
The CUDA tag names are Release tags for prebuilt assets. The default branch
contains the selector packages and dependency graph that make combinations such
as `cu121,generic` work without duplicating `vsmlrt.py` or `models/`.

Each CUDA or generic extra depends on `vs-mlrt-models`, so the shared model
payload is installed once even when multiple extras are selected.

## Release Asset Slots

The build hooks download these assets from GitHub Releases:

| Release tag | Assets |
| --- | --- |
| `models` | `models.zip` |
| `generic` | `vs-mlrt-windows-x64-generic.zip` |
| `cu121` | `vs-mlrt-windows-x64-tensorrt-cu121.zip`, `vs-mlrt-windows-x64-cuda-cu121.zip`, `vs-mlrt-windows-x64-cudnn-cu121.zip` |
| `cu129` | `vs-mlrt-windows-x64-tensorrt-cu129.zip`, `vs-mlrt-windows-x64-cuda-cu129.zip`, `vs-mlrt-windows-x64-cudnn-cu129.zip`, `vs-mlrt-windows-x64-tensorrt-core-cu129.zip`, `vs-mlrt-windows-x64-tensorrt-plugin-cu129.zip`, `vs-mlrt-windows-x64-tensorrt-extra-cu129.zip`, `vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip` |

All native payload zips are rooted at `vsmlrt/`. After pip installation, the
selected payloads overlay into:

```text
site-packages/vapoursynth/plugins/vsmlrt/
```

Important layout details:

- `models.zip` supplies `models/` and is shared by all extras.
- CUDA payloads place CUDA, cuDNN, TensorRT, and helper executables under
  `vsmlrt/vsmlrt-cuda/`.
- `vstrt.dll` lives at the plugin root. `vstrt_rtx.dll` is installed only by
  `cu129`.
- `generic` contains no CUDA, TensorRT, ORT, or DirectML payload. It contains
  only `vsncnn`, `vsov`, and their support DLLs.
- OpenVINO support files live once at the plugin root. There is no duplicated
  `vsov/` runtime directory.
- Every release line includes a `manifest.vs` for manual single-line installs.
  The cu129 manifest lists both `vstrt` and `vstrt_rtx` because its split
  assets form one required payload set.
  Pip payload wheels discard those per-release copies; the main `vs-mlrt`
  wheel owns one shared manifest and regenerates its entries from the selected
  extras. Thus `generic,cu121` lists `vsncnn`, `vsov`, and `vstrt` without two
  wheels overwriting the same file.

The `cu129` TensorRT payload is split because GitHub Release assets must stay
below 2 GiB.

## Publishing Workflow

Windows release assets are produced by these workflows:

| Workflow | Purpose |
| --- | --- |
| `.github/workflows/windows-vcs-models.yml` | Build and publish the shared `models` asset. |
| `.github/workflows/windows-vcs-generic.yml` | Build and publish the NVIDIA-free `generic` asset. |
| `.github/workflows/windows-vcs-package.yml` | Build and publish the `cu121` and `cu129` TensorRT assets. |
| `.github/workflows/windows-vcs-install-smoke.yml` | Install from the default branch with all supported extra combinations and verify the installed layout. |

Recommended maintainer loop:

1. Change source, packaging, or workflow files on the default branch.
2. Rebuild the affected release asset slot with `publish=true`.
3. Run `windows-vcs-install-smoke.yml` after release assets have been refreshed.
4. Treat stale smoke results from before the asset refresh as non-authoritative.

When debugging CUDA packaging, build one line at a time. The package workflow
defaults to `cu121`; use `cu129` after the `cu121` path is healthy. `all` is
useful for final publication but noisy while isolating one payload line.

## Validation Boundary

GitHub-hosted Windows runners do not provide the NVIDIA display driver.
Therefore CUDA smoke tests allow `nvcuda.dll` to be missing. Other missing PE
imports, missing TensorRT string-loaded DLLs, or unexpected payload files are
packaging errors.

A green hosted CUDA smoke verifies:

- the VCS extras dependency graph;
- release asset download and overlay order;
- plugin directory layout;
- bundled CUDA/cuDNN/TensorRT DLL coverage;
- explicit VapourSynth plugin loading up to the driver boundary.

It does not prove real GPU inference. Final TRT runtime checks still need a
machine with a compatible NVIDIA driver.

For `generic`, hosted smoke installs the Vulkan SDK so `vsncnn.dll` can load on
the runner. Real ncnn and OpenVINO inference still depends on the user's GPU
driver/runtime.

## CUDA-Sensitive Maintenance Points

Most C++ code is sensitive to TensorRT major/version APIs rather than CUDA
minor versions. Review these areas when changing TensorRT lines:

- `vstrt/trt_utils.h`: `NV_TENSORRT_MAJOR`, `NV_TENSORRT_MINOR`,
  `NV_TENSORRT_PATCH`, and `TRT_MAJOR_RTX` select binding, tensor shape,
  datatype, context-memory, and enqueue APIs.
- `vstrt/win32.cpp`: dynamic DLL names differ between TensorRT 8.x,
  TensorRT 11.x, and TensorRT-RTX.
- `vstrt/vs_tensorrt.cpp`: TensorRT-RTX diagnostics and engine validation are
  guarded by TensorRT version macros.
- `vstrt/trtexec/`: the custom `trtexec` build must track the TensorRT OSS
  branch matching the selected SDK.
- `scripts/vsmlrt.py`: `parse_trt_version()`, `trtexec()`,
  `tensorrt_rtx()`, and model workarounds choose CLI flags by TensorRT
  version. TensorRT 8.6 and TensorRT 11.1 do not accept the same flag set.

CUDA-minor-sensitive pieces live mostly in CI and packaging:

- CUDA redist version and component names, for example `12.1.1` vs `12.9.1`.
- `CMAKE_CUDA_ARCHITECTURES`; `cu121` omits Blackwell `120-real`, while
  `cu129` includes it.
- TensorRT and TensorRT-RTX download URLs.
- `TENSORRT_LIBRARY_SUFFIX`; TensorRT 8.6 uses unversioned Windows DLL names,
  while TensorRT 11 uses names such as `nvinfer_11.dll`.
- Runtime DLL split and overlay order under `vsmlrt/vsmlrt-cuda/`.

## Model Payload Notes

The `models` release is assembled separately from upstream model releases. It
is not CUDA-sensitive and should be reused by `generic`, `cu121`, and `cu129`.

The model workflow currently assembles the payload from upstream
`model-20211209`, `model-20220923`, and `contrib-models`, and checks for core
and contributed model files such as `dpir`, `rife`, `animejanaiV2L1.onnx`,
`animejanaiV3-HD-L1.onnx`, and `Ani4Kv2-G6i2-Compact.onnx`.
