# vs-mlrt

VapourSynth ML runtime plugins and a Python wrapper for model-driven video filters.
This fork keeps the public Python entry point stable as `import vsmlrt` while adding
release-backed Windows TensorRT packages that can be installed directly from Git
tags.

The upstream backend source layout is preserved, but this fork's prebuilt release
work focuses on Windows x64 TensorRT. Other backend directories remain useful for
source work and reference builds.

## Quick Install: Windows TensorRT

Requirements:

- Windows x64.
- Python 3.9 or newer.
- A VapourSynth R75+ Python environment.
- An NVIDIA driver compatible with the selected CUDA payload.

Install one CUDA line by choosing the matching Git tag:

```powershell
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

On non-Windows platforms, this VCS package does not install a native prebuilt
plugin payload. Use upstream packages or build the required backend from source.

## Choosing a CUDA Tag

The prebuilt package bundles CUDA user-mode runtime DLLs, TensorRT, cuDNN, and
models. You do not need a local CUDA Toolkit for `pip install`, but your NVIDIA
driver must be new enough for the selected CUDA runtime.

| Tag | CUDA payload | TensorRT | TensorRT-RTX | Use when |
| --- | --- | --- | --- | --- |
| `cu121` | CUDA 12.1.1 | 8.6.1.6 | Not included | The machine is limited to CUDA 12.1/12.2-era drivers or you need the older TRT 8 line. |
| `cu129` | CUDA 12.9.1 | 11.1.0.106 | 1.5.0.114 | The machine has a current driver and should use the newer TRT 11 plus `trt_rtx` payload. |

If a machine's driver only supports up to CUDA 12.2, install `@cu121`. Installing
`@cu129` on that machine is expected to fail at runtime because the bundled CUDA
12.9 user-mode libraries still need a compatible NVIDIA driver.

## Release Asset Layout

CUDA-specific plugin assets are published on the CUDA tags:

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
`vs-mlrt-windows-x64-models.zip`. The VCS build hook downloads the selected CUDA
payload plus this shared model payload, so model files are not duplicated across
`cu121` and `cu129` releases.

The model payload is assembled from upstream `model-20211209`,
`model-20220923`, and `contrib-models`. It includes contributed RealESRGAN
models such as `animejanaiV2L1.onnx`, `animejanaiV3-HD-L1.onnx`, and
`Ani4Kv2-G6i2-Compact.onnx`.

For more detail, see [docs/trt-cuda-release-matrix.md](docs/trt-cuda-release-matrix.md).

## Manual Installation

Manual installs are possible but easier to get wrong than VCS installs. Extract
the selected CUDA tag assets and the shared `models` asset into the same
VapourSynth plugin directory, preserving the top-level `vsmlrt/` folder.

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

Windows TensorRT releases are validated in GitHub Actions:

- `.github/workflows/windows-vcs-models.yml` builds and publishes the shared
  `models` release asset.
- `.github/workflows/windows-vcs-package.yml` builds the `cu121` and `cu129`
  native payload assets.
- `.github/workflows/windows-vcs-install-smoke.yml` installs from the published
  tags with `pip` and verifies the VapourSynth plugin layout.

GitHub-hosted runners do not provide the NVIDIA display driver, so `nvcuda.dll`
is the only allowed missing dependency in smoke tests. A green smoke test proves
package layout and plugin loading up to the driver boundary; final GPU inference
still needs a real CUDA machine.

## Useful Overrides

The build hook supports a few environment variables for maintainers:

- `VSMLRT_SKIP_PREBUILT=1`: skip release payload download.
- `VSMLRT_CUDA_TAG=cu121|cu129`: force a CUDA payload tag.
- `VSMLRT_RELEASE_REPO=owner/repo`: download CUDA assets from another repo.
- `VSMLRT_MODELS_TAG=models`: override the shared model tag.
- `VSMLRT_MODELS_RELEASE_REPO=owner/repo`: download models from another repo.
- `VSMLRT_PREBUILT_PATHS=zip1;zip2;...`: use local payload zips instead of releases.

Use these only for packaging tests. Normal users should install from `@cu121` or
`@cu129`.
