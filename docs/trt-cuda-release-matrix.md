# TensorRT CUDA Release Matrix

This fork intentionally publishes only two Windows TensorRT payload lines:

| Tag | CUDA toolkit | TensorRT | TensorRT-RTX | Notes |
| --- | --- | --- | --- | --- |
| `cu121` | 12.1.1 | 8.6.1.6, Windows CUDA 12.0 package | Not included | Targets older drivers that cannot run CUDA 12.9+ user-mode libraries. NVIDIA did not publish a matching TensorRT-RTX 1.x CUDA 12.1 Windows package. |
| `cu129` | 12.9.1 | 11.1.0.106, Windows CUDA 12.9 package | 1.5.0.114, Windows CUDA 12.9 package | Current high-end line, includes both `trt` and `trt_rtx`. |

`vsmlrt.py` remains the stable Python entry point. Users still call functions such as `vsmlrt.DPIR(..., backend=vsmlrt.Backend.TRT(...))`; the selected Git tag only changes which native TensorRT payload is installed.

## VCS Install Mapping

The package name is `vs-mlrt`, and the import remains `import vsmlrt`.

```powershell
pip install "vs-mlrt @ git+https://github.com/<owner>/vs-mlrt.git@cu129"
pip install "vs-mlrt @ git+https://github.com/<owner>/vs-mlrt.git@cu121"
```

The build hook downloads and overlays the matching release assets:

- `cu121`: `vs-mlrt-windows-x64-tensorrt-cu121.zip`, `vs-mlrt-windows-x64-cuda-cu121.zip`, `vs-mlrt-windows-x64-cudnn-cu121.zip`, and `vs-mlrt-windows-x64-models.zip`
- `cu129`: `vs-mlrt-windows-x64-tensorrt-cu129.zip`, `vs-mlrt-windows-x64-cuda-cu129.zip`, `vs-mlrt-windows-x64-cudnn-cu129.zip`, `vs-mlrt-windows-x64-tensorrt-core-cu129.zip`, `vs-mlrt-windows-x64-tensorrt-plugin-cu129.zip`, `vs-mlrt-windows-x64-tensorrt-extra-cu129.zip`, `vs-mlrt-windows-x64-models.zip`, and `vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip`

Each zip is rooted at `vsmlrt/`. The TensorRT zip contains `manifest.vs`, `vstrt.dll`, and `trtexec.exe`. The CUDA and cuDNN overlays contain secondary TensorRT dependencies such as cuBLAS, cuFFT, NVRTC, cuDNN, and CUPTI DLLs. The cu129 core, plugin, and extra overlays contain the regular TensorRT DLLs split by role, including `nvinfer_11.dll`, `nvinfer_plugin_11.dll`, and parser/dispatch DLLs such as `nvonnxparser_11.dll`. The models zip contains the bundled `models/` directory. The cu129 RTX overlay contains `vstrt_rtx.dll`, `tensorrt_rtx_1_5.dll`, and a manifest that enables both `vstrt` and `vstrt_rtx`; extract it after the base TensorRT zip for manual installs. VCS installs perform the overlay automatically into `site-packages/vapoursynth/plugins/vsmlrt` and install `vsmlrt.py` as the importable wrapper module.

The payload is split because GitHub Release assets must stay below 2 GiB, while the full cu129 TensorRT + TensorRT-RTX + model bundle exceeds that limit.

The model directory is assembled in CI from the upstream `model-20211209`, `model-20220923`, and `contrib-models` releases unless workflow inputs override those tags.

Do not point both tags at the same commit. If both tags share one commit, a pip VCS checkout may not preserve which tag the user requested. The `packaging/cuda-tag.txt` file must read `cu121` in the `cu121` tag commit and `cu129` in the `cu129` tag commit.

## Validation Notes

Focus validation on one CUDA line at a time. The workflow dispatch defaults are intentionally `cu121`; run `cu129` only after `cu121` has passed packaging and VCS install smoke.

GitHub-hosted Windows runners do not provide the NVIDIA display driver, so `nvcuda.dll` is an allowed missing dependency in smoke tests. Any other missing PE import or DLL string reference is treated as a packaging error. A passing runner smoke verifies the VCS install path, package layout, TensorRT redist DLL coverage, and VapourSynth plugin load up to the driver boundary; it does not prove GPU inference. Final DPIR/TRT runtime inference still needs a CUDA machine.

Only Windows prebuilt release payloads are published. Non-Windows VCS installs do not download these release assets and should use the source-build path instead.

## Process Self-Review

Keep the validation loop narrow. First run `windows-vcs-package.yml` and `windows-vcs-install-smoke.yml` with `cuda-flavor=cu121`; only then repeat for `cu129`. Avoid `cuda-flavor=all` while debugging one line.

Treat green CI cautiously. Earlier smoke tests passed while TensorRT printed loader errors; strict smoke now captures plugin stderr, checks recursive PE imports, and scans TensorRT DLL string references for dynamically loaded dependencies.

Keep `vsmlrt.py` as the immutable entry point. Packaging changes may alter installed native payloads, but users must still call models through `import vsmlrt` and functions such as `DPIR(..., backend=...)`.

## CUDA-Sensitive Code

The C++ TRT plugin is mostly keyed on TensorRT API macros, not CUDA minor versions. These blocks should move only when the selected TensorRT release changes:

- `vstrt/trt_utils.h`: `NV_TENSORRT_MAJOR`, `NV_TENSORRT_MINOR`, `NV_TENSORRT_PATCH`, and `TRT_MAJOR_RTX` select TensorRT binding, tensor shape, datatype, context memory, and enqueue APIs.
- `vstrt/win32.cpp`: TensorRT DLL names change for TensorRT 10+ and TensorRT-RTX, for example `nvinfer_11.dll` and `tensorrt_rtx_1_5.dll`.
- `vstrt/vs_tensorrt.cpp`: TensorRT-RTX engine validity diagnostics are guarded by `TRT_MAJOR_RTX` and `NV_TENSORRT_VERSION`.
- `vstrt/trtexec/`: the custom `trtexec` build is coupled to the TensorRT OSS branch matching the selected TensorRT SDK.
- `scripts/vsmlrt.py`: `parse_trt_version()`, `trtexec()`, `tensorrt_rtx()`, and a few model-specific workarounds choose CLI flags by TensorRT version. For example, cu121/TRT 8.6 still uses pre-TRT-11 precision and tactic-source flags, while cu129/TRT 11.1 uses the newer build path.

The CUDA-minor-sensitive pieces are in CI/package assembly:

- CUDA installer version and component suffix, such as `12.1.1` with `nvcc_12.1`.
- `CMAKE_CUDA_ARCHITECTURES`; `cu121` omits Blackwell `120-real`, while `cu129` includes it.
- Runtime DLL payload under `vsmlrt/vsmlrt-cuda/` in the TensorRT and optional TensorRT-RTX release zips.
- TensorRT download URLs and `TENSORRT_LIBRARY_SUFFIX`, because TRT 8.6 uses unversioned Windows DLL names while TRT 11 uses names such as `nvinfer_11.dll`.
