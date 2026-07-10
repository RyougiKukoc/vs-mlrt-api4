# vs-mlrt

VapourSynth ML runtime plugins and a Python wrapper for model-driven video filters.
This fork keeps the public Python entry point stable as `import vsmlrt` while adding
release-backed Windows payloads that can be installed directly from Git tags.

The upstream backend source layout is preserved, but this fork's prebuilt release
work currently focuses on Windows x64 `generic`, `cu121`, and `cu129` install
lines.

## Quick Install: Windows

Requirements:

- Windows x64.
- Python 3.9 or newer.
- A VapourSynth R75+ Python environment.
- For `generic`: a working GPU/runtime stack for the backend you use
  (`ncnn`/Vulkan, `ov`/OpenVINO, or `ort`/DirectML or CPU).
- For `cu121` and `cu129`: an NVIDIA driver compatible with the selected CUDA
  payload.

Install one payload line by choosing the matching Git tag:

```powershell
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@generic"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu121"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu129"
```

The package installs `vsmlrt.py` plus the native plugin payload under:

```text
site-packages/vapoursynth/plugins/vsmlrt/
```

Users still call models through the same wrapper:

```python
import vsmlrt

out = vsmlrt.DPIR(clip, strength=5.0, backend=vsmlrt.Backend.TRT(fp16=True))
```

`vsmlrt.py` is the stable public entry point. Do not call backend DLLs or model
files directly from user scripts.

The build hook prints progress while downloading large release assets. If your
pip frontend suppresses build-backend output, add `-v` to the install command.

On non-Windows platforms, this VCS package does not install a native prebuilt
plugin payload. Use upstream packages or build the required backend from source.

## Choosing an Install Tag

All install tags reuse the shared `models` release asset. You do not need to
download models separately when using the VCS install commands above.

| Tag | Native plugins | Extra runtime payload | Use when |
| --- | --- | --- | --- |
| `generic` | `vsncnn`, `vsov`, `vsort` | ncnn/OpenVINO/ONNX Runtime support DLLs, no CUDA/TensorRT | The machine should use non-TensorRT backends such as Vulkan ncnn, OpenVINO, or ONNX Runtime DML/CPU. |
| `cu121` | `vstrt` | CUDA 12.1.1, TensorRT 8.6.1.6, cuDNN | The machine is limited to CUDA 12.1/12.2-era drivers or you need the older TRT 8 line. |
| `cu129` | `vstrt`, `vstrt_rtx` | CUDA 12.9.1, TensorRT 11.1.0.106, TensorRT-RTX 1.5.0.114, cuDNN | The machine has a current driver and should use the newer TRT 11 plus `trt_rtx` payload. |

If a machine's driver only supports up to CUDA 12.2, install `@cu121`. Installing
`@cu129` on that machine is expected to fail at runtime because the bundled CUDA
12.9 user-mode libraries still need a compatible NVIDIA driver.

## Release Asset Layout

Native plugin assets are published on their install tags:

- `generic`: `vs-mlrt-windows-x64-generic.zip`.
- `cu121`: `vs-mlrt-windows-x64-tensorrt-cu121.zip`,
  `vs-mlrt-windows-x64-cuda-cu121.zip`, and
  `vs-mlrt-windows-x64-cudnn-cu121.zip`.
- `cu129`: `vs-mlrt-windows-x64-tensorrt-cu129.zip`,
  `vs-mlrt-windows-x64-cuda-cu129.zip`,
  `vs-mlrt-windows-x64-cudnn-cu129.zip`,
  `vs-mlrt-windows-x64-tensorrt-core-cu129.zip`,
  `vs-mlrt-windows-x64-tensorrt-plugin-cu129.zip`,
  `vs-mlrt-windows-x64-tensorrt-extra-cu129.zip`, and
  `vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip`.

Models are published once on the separate `models` tag as
`models.zip`. The VCS build hook downloads the selected native payload plus this
shared model payload, so model files are not duplicated across `generic`,
`cu121`, and `cu129` releases.

The model payload is assembled from upstream `model-20211209`,
`model-20220923`, and `contrib-models`. It includes contributed RealESRGAN
models such as `animejanaiV2L1.onnx`, `animejanaiV3-HD-L1.onnx`, and
`Ani4Kv2-G6i2-Compact.onnx`.

For more detail, see [docs/trt-cuda-release-matrix.md](docs/trt-cuda-release-matrix.md).

## Manual Installation

Manual installs are possible but easier to get wrong than VCS installs. Extract
the selected tag assets and the shared `models` asset into the same
VapourSynth plugin directory, preserving the top-level `vsmlrt/` folder.

For `generic`, extract `vs-mlrt-windows-x64-generic.zip` and `models.zip`.

For `cu129`, extract all `cu129` assets. If you want `vstrt_rtx` available,
extract `vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip` last so its manifest enables
both `vstrt` and `vstrt_rtx`. The VCS installer performs this overlay
automatically.

## Backend Source Layout

- `scripts/vsmlrt.py`: Python wrapper and model-facing API.
- `vstrt/`: TensorRT and TensorRT-RTX plugin source plus custom `trtexec` build files.
- `vsort/`: ONNX Runtime backend source.
- `vsov/`: OpenVINO backend source.
- `vsncnn/`: ncnn Vulkan backend source.
- `vsmigx/`: MIGraphX backend source, retained from upstream but not a focus of this fork.
- `common/`: shared helper code used by native plugins.
- `.github/workflows/`: CI, packaging, smoke tests, and release publication.

## Development And Validation

There is no root CMake project. Build the backend you are changing:

```powershell
cmake -S vstrt -B vstrt/build -G Ninja -D CMAKE_BUILD_TYPE=Release
cmake --build vstrt/build --verbose
cmake --install vstrt/build --prefix vstrt/install
```

Windows releases are validated in GitHub Actions:

- `.github/workflows/windows-vcs-models.yml` builds and publishes the shared
  `models` release asset.
- `.github/workflows/windows-vcs-generic.yml` builds the `generic` native payload
  with `vsncnn`, `vsov`, and `vsort`.
- `.github/workflows/windows-vcs-package.yml` builds the `cu121` and `cu129`
  native payload assets.
- `.github/workflows/windows-vcs-generic-install-smoke.yml` installs from the
  published `generic` tag with `pip` and verifies the generic plugin layout.
- `.github/workflows/windows-vcs-install-smoke.yml` installs from the published
  CUDA tags with `pip` and verifies the VapourSynth plugin layout.

GitHub-hosted runners do not provide the NVIDIA display driver, so `nvcuda.dll`
is the only allowed missing dependency in smoke tests. A green smoke test proves
package layout and plugin loading up to the driver boundary; final GPU inference
still needs a real CUDA machine.

For `generic`, GitHub Actions installs the Vulkan SDK during smoke tests so
`vsncnn.dll` can load on the hosted runner. Real ncnn inference still depends on
the user's installed GPU driver and Vulkan support.

## Useful Overrides

The build hook supports a few environment variables for maintainers:

- `VSMLRT_SKIP_PREBUILT=1`: skip release payload download.
- `VSMLRT_PAYLOAD_TAG=generic|cu121|cu129`: force a native payload tag.
- `VSMLRT_CUDA_TAG=cu121|cu129`: legacy alias for CUDA payload tests.
- `VSMLRT_RELEASE_REPO=owner/repo`: download native assets from another repo.
- `VSMLRT_MODELS_TAG=models`: override the shared model tag.
- `VSMLRT_MODELS_RELEASE_REPO=owner/repo`: download models from another repo.
- `VSMLRT_PREBUILT_PATHS=zip1;zip2;...`: use local payload zips instead of releases.
- `VSMLRT_DOWNLOAD_PROGRESS=0`: hide build-hook download progress.
- `VSMLRT_DOWNLOAD_PROGRESS_INTERVAL=5`: change progress report interval in seconds.
- `VSMLRT_PROGRESS_CONSOLE=0`: do not try direct Windows console progress output.

Use these only for packaging tests. Normal users should install from `@generic`,
`@cu121`, or `@cu129`.
