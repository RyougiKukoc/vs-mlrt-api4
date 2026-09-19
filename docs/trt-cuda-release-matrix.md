# TensorRT CUDA Release Matrix

This document describes the Windows and Linux TensorRT payloads published by
this fork and the rules for keeping those payloads in sync with the tag-selected
VCS installer.

The short version:

- Users install from one of three VCS tags: `generic`, `cu121`, or `cu129`.
- Release tags such as `cu121`, `cu129`, `generic`, and `models` are binary
  asset slots consumed by the build hooks.
- `cu121` and `cu129` must not be installed together in one Python
  environment.
- `cu121` and `cu129` automatically install the `generic` native payload in
  addition to their TRT-side assets.

## User-Facing Matrix

| Tag | Native plugins | CUDA / GPU line | Main dependency versions | Intended users |
| --- | --- | --- | --- | --- |
| `generic` | `vsncnn`, `vsov` | No NVIDIA CUDA payload | ncnn/Vulkan and OpenVINO runtime DLLs | Intel, AMD, or NVIDIA users who want non-TensorRT backends. |
| `cu121` | `vsncnn`, `vsov`, `vstrt` | CUDA 12.1.1 redist | TensorRT 8.6.1.6, cuDNN 8.9.7 | NVIDIA systems pinned to drivers that can run CUDA 12.1/12.2-era user-mode libraries. |
| `cu129` | `vsncnn`, `vsov`, `vstrt`, `vstrt_rtx` | CUDA 12.9.1 redist | TensorRT 11.1.0.106, TensorRT-RTX 1.5.0.114, cuDNN 9.19 | Current NVIDIA systems with drivers new enough for CUDA 12.9 user-mode libraries. |

`cu121` does not include `vstrt_rtx`. NVIDIA did not publish a matching
Windows TensorRT-RTX package for the CUDA 12.1/TensorRT 8.6 line.

The selected tag changes only the installed native payload. The Python entry
point remains stable:

```python
import vsmlrt

out = vsmlrt.DPIR(clip, strength=5.0, backend=vsmlrt.Backend.TRT(fp16=True))
```

## Install Mapping

Normal installs target one of this repository's VCS tags:

```powershell
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@generic"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu121"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu129"
```

`@generic` installs the generic payload plus models. `@cu121` and `@cu129`
install models, download the matching TRT-side release assets, and also pull
the `generic` release asset so the installed plugin directory always includes
`vsncnn` and `vsov`.

Keep the `generic`, `cu121`, and `cu129` tags on distinct commits. The build
hook resolves the active payload line from the exact tag(s) pointing at `HEAD`,
and ambiguous multi-tag commits are treated as an error.

## Release Asset Slots

The build hooks download these assets from GitHub Releases:

| Release tag | Assets |
| --- | --- |
| `models` | `models.zip` |
| `generic` | `vs-mlrt-windows-x64-generic.zip`, `vs-mlrt-linux-x64-generic.zip` |
| `cu121` | Windows `vs-mlrt-windows-x64-tensorrt-cu121.zip`, `vs-mlrt-windows-x64-cuda-cu121.zip`, `vs-mlrt-windows-x64-cudnn-cu121.zip`, `vs-mlrt-windows-x64-tensorrt-builder-cu121.zip`; Linux adds `vs-mlrt-linux-x64-cudnn-part-2-cu121.zip` to the matching TensorRT, CUDA, cuDNN, and builder assets |
| `cu129` | Windows standard, split TensorRT, CUDA, cuDNN, builder tool plus three builder-resource, and RTX assets; Linux additionally splits cuDNN into two assets and uses four builder-resource overlays to account for ELF SONAME aliases |

All native payload zips are rooted at `vsmlrt/`. After pip installation, the
selected payloads overlay into:

```text
site-packages/vapoursynth/plugins/vsmlrt/
```

Important layout details:

- `models.zip` supplies `models/` and is shared by all three install tags.
- CUDA payloads place CUDA, cuDNN, TensorRT, and helper executables under
  `vsmlrt/vsmlrt-cuda/`.
- Builder overlays contain `trtexec`, its provenance JSON, and TensorRT builder
  resources required by public `Backend.TRT` ONNX conversion. Windows `cu129`
  publishes the tool overlay plus three resource overlays, while Linux uses
  four resource overlays because staged ELF SONAME aliases are retained. Both
  stay below GitHub's 2 GiB per-asset limit.
- `vstrt.dll` lives at the plugin root. `vstrt_rtx.dll` is installed only by
  `cu129`.
- `generic` contains no CUDA, TensorRT, ORT, or DirectML payload. It contains
  only `vsncnn`, `vsov`, and their support DLLs.
- OpenVINO support files live once at the plugin root. There is no duplicated
  `vsov/` runtime directory.
- Plugin-bearing overlays include `manifest.vs`; CUDA, cuDNN, and builder
  overlays contain support libraries and helpers only. The root `vs-mlrt`
  wheel regenerates one shared manifest after overlaying selected assets, so
  `@cu121` ends up with `vsncnn`, `vsov`, and `vstrt`, while `@cu129` adds
  `vstrt_rtx`.

The `cu129` TensorRT payload is split because GitHub Release assets must stay
below 2 GiB.

## Publishing Workflow

Windows and Linux release assets are produced by these workflows:

| Workflow | Purpose |
| --- | --- |
| `.github/workflows/windows-vcs-models.yml` | Build and publish the shared `models` asset. |
| `.github/workflows/windows-vcs-generic.yml` | Build and publish the NVIDIA-free `generic` asset. |
| `.github/workflows/windows-vcs-package.yml` | Build and publish Windows TensorRT, CUDA, builder, and RTX assets. |
| `.github/workflows/linux-vcs-package.yml` | Build and publish matching Linux assets. |
| `.github/workflows/windows-vcs-install-smoke.yml` | Manual check of the already published `generic`, `cu121`, and `cu129` VCS tags. |

The generic and pinned TensorRT workflows also build pull requests, without
publishing. Each job installs its own staged zip files through the wheel build
hook, checks installed file hashes against those zips, then runs load/layout
smoke. CUDA jobs use the currently published generic and model dependencies,
whose hashes are recorded separately; they do not consume a new generic build
from another job in the same pull request. The job
writes `payload-provenance-<variant>.json` with its source commit, asset hashes,
dependency identities, and verification result.

Tag/manual publication happens only after that installation gate succeeds. A
post-upload check compares GitHub's asset digests with the tested zips. The
separate manual VCS smoke remains useful for checking tag selection and remote
delivery; it is not the success gate for a concurrent build of new assets.

Recommended maintainer loop:

1. Change source, packaging, or workflow files on the default branch.
2. Review the PR's native build, regression, and staged-install results.
3. Rebuild the affected release asset slot with `publish=true`; require its
   staged-install and published-digest checks to pass.
   If generic also changed, publish generic first, then run the CUDA jobs so
   their recorded generic dependency hash identifies that new payload.
4. Run `windows-vcs-install-smoke.yml` after release assets have been refreshed.
5. Treat stale smoke results from before the asset refresh as non-authoritative.

When debugging CUDA packaging, build one line at a time. The package workflow
defaults to `cu121`; use `cu129` after the `cu121` path is healthy. `all` is
useful for final publication but noisy while isolating one payload line.

## Validation Boundary

GitHub-hosted Windows runners do not provide the NVIDIA display driver.
Therefore CUDA smoke tests allow `nvcuda.dll` to be missing. Other missing PE
imports, missing TensorRT string-loaded DLLs, or unexpected payload files are
packaging errors.

A green hosted CUDA smoke verifies:

- the tag-selected VCS build hook;
- release asset download and overlay order;
- plugin directory layout;
- bundled CUDA/cuDNN/TensorRT DLL coverage;
- explicit VapourSynth plugin loading up to the driver boundary.

It does not prove real GPU inference. Final TRT runtime checks still need a
machine with a compatible NVIDIA driver.

## Maintainer GPU Verification

Do not mark a CUDA line unavailable merely because a hosted runner lacks an
NVIDIA driver. First probe the maintainer host with `nvidia-smi` and use an
isolated VapourSynth/Python environment when a compatible GPU is present.
Validate `cu121` and `cu129` in separate environments or extracted payload
directories; they must never be overlaid into the same plugin directory.

For each refreshed CUDA Release slot:

1. Download every selected payload zip and compare its SHA-256 with GitHub's
   Release asset digest before extraction.
2. Explicitly load `vstrt.dll` from the extracted package after adding both
   the plugin root and `vsmlrt-cuda/` to the Windows DLL search path. Check
   `core.trt.Version()` and `core.trt.DeviceProperties(0)` against the
   intended CUDA/TensorRT line and physical GPU.
3. For `cu129`, explicitly load `vstrt_rtx.dll` too. Build a small identity
   ONNX engine with the bundled `tensorrt_rtx.exe`, then render an identity
   `GRAY_S` frame through `core.trt_rtx.Model`; require matching dimensions,
   format, and frame hash. This proves a real GPU engine build and plugin
   inference rather than just DLL discovery.
4. Treat `cu121` engine execution as a separate gate using the matching
   builder overlay. Verify the packaged `trtexec` and builder resource create a
   deterministic engine, then verify `vstrt` renders a real frame.

These are CUDA runtime/package checks, not an API3/API4 behavior comparison.
Record the driver, GPU, runtime versions, Release digests, and any engine/frame
hashes in the release notes or maintenance log.

## Compression Policy

All `vs-mlrt` release archive creation uses `-mx=0` (store mode). NVIDIA
runtime DLLs, models, and payload zips are already compressed or do not repay
CI CPU time with a slower compression level. Keep Actions artifact uploads at
`compression-level: 0` as well. Retain the existing GitHub per-asset size
checks; faster storage must not bypass the 2 GiB release limit.

For `generic`, hosted smoke installs the Vulkan SDK so `vsncnn.dll` can load on
the runner. Real ncnn and OpenVINO inference still depends on the user's GPU
driver/runtime.

## CUDA-Sensitive Maintenance Points

The Python distribution version `16.2.2` identifies the complete plugin release
aligned with upstream tag `v16.2.test1`. The separate wrapper API version is not
used as the package version. Most C++ code is sensitive to TensorRT
major/version APIs rather than CUDA minor versions. Review these areas when
changing TensorRT lines:

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
- Host toolset: `cu121` selects MSVC 14.38 so an updated hosted runner does not
  silently make CUDA 12.1's compiler unsupported.
- `CMAKE_CUDA_ARCHITECTURES`; `cu121` omits Blackwell `120-real`, while
  `cu129` includes it.
- TensorRT and TensorRT-RTX download URLs.
- `TENSORRT_LIBRARY_SUFFIX`; TensorRT 8.6 uses unversioned Windows DLL names,
  while TensorRT 11 uses names such as `nvinfer_11.dll`.
- Runtime DLL split and overlay order under `vsmlrt/vsmlrt-cuda/`.

`tools/build_custom_trtexec.py` maintains separate source lists for TensorRT
8.6 and 11.1 and validates the selected headers. Its profiler headers, UTF-8
manifest, log redirection, and SDK file-lock patch are part of the custom tool
contract. `tools/test_custom_trtexec.py` checks the executable's recorded hash,
`--help`, Unicode/long log paths, and real file-lock cleanup without a GPU.
A failed custom build stops the package job; it no longer substitutes an
untested vendor executable.

## Model Payload Notes

The `models` release is assembled separately from upstream model releases. It
is not CUDA-sensitive and should be reused by `generic`, `cu121`, and `cu129`.

The model workflow currently assembles the payload from upstream
`model-20211209`, `model-20220923`, and `contrib-models`, and checks for core
and contributed model files such as `dpir`, `rife`, `animejanaiV2L1.onnx`,
`animejanaiV3-HD-L1.onnx`, and `Ani4Kv2-G6i2-Compact.onnx`.
