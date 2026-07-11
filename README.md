# vs-mlrt

VapourSynth ML runtime plugins and the `vsmlrt.py` Python wrapper.

This fork is API4-oriented and publishes Windows x64 binary payloads through
GitHub Releases. Users install from the default Git branch and select payloads
with pip extras; Release tags are binary asset slots, not the normal install
entry point.

## Quick Install: Windows

Requirements:

- Windows x64.
- Python 3.9 or newer.
- A VapourSynth R75+ Python environment.
- For `generic`: a working runtime for the backend you use:
  `ncnn`/Vulkan, `ov`/OpenVINO, or `ort`/DirectML or CPU.
- For `cu121` and `cu129`: an NVIDIA driver compatible with the selected CUDA
  payload.

Install the wrapper plus one or more payload extras:

```powershell
pip install "vs-mlrt[generic] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu121] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu129] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu121,generic] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
pip install "vs-mlrt[cu129,generic] @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git"
```

Do not install `cu121` and `cu129` into the same environment. Use `cu121` for
machines limited to CUDA 12.1/12.2-era drivers, and use `cu129` for machines
with a current enough NVIDIA driver for CUDA 12.9 user-mode libraries.

The package keeps the public Python entry point stable:

```python
import vsmlrt

out = vsmlrt.DPIR(clip, strength=5.0, backend=vsmlrt.Backend.TRT(fp16=True))
```

If pip hides build-backend output while large assets download, add `-v` to the
install command.

On non-Windows platforms this VCS package does not install native prebuilt
payloads. Use upstream packages or build the required backend from source.

## Payload Extras

All extras install the shared model payload once through the `vs-mlrt-models`
selector package. The native payloads are separate wheels, so combinations such
as `cu121,generic` do not duplicate `vsmlrt.py` or the model files.

| Extra | Native plugins | Runtime payload | Installed plugin directory |
| --- | --- | --- | --- |
| `generic` | `vsncnn`, `vsov`, `vsort` | ncnn, OpenVINO, ONNX Runtime, DirectML support DLLs | `site-packages/vapoursynth/plugins/vsmlrt-generic/` |
| `cu121` | `vstrt` | CUDA 12.1.1, TensorRT 8.6.1.6, cuDNN | `site-packages/vapoursynth/plugins/vsmlrt-cu121/` |
| `cu129` | `vstrt`, `vstrt_rtx` | CUDA 12.9.1, TensorRT 11.1.0.106, TensorRT-RTX 1.5.0.114, cuDNN | `site-packages/vapoursynth/plugins/vsmlrt-cu129/` |

Models install under:

```text
site-packages/vsmlrt_models/models/
```

`vsmlrt.py` resolves models from that shared package first, then falls back to
legacy layouts for compatibility. TensorRT helper executables are resolved from
the selected CUDA payload directory.

## Release Asset Layout

The default branch controls installation. These Release tags remain as binary
asset slots consumed by the selector packages:

- `models`: `models.zip`.
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

The model payload is assembled from upstream `model-20211209`,
`model-20220923`, and `contrib-models`. It includes contributed RealESRGAN
models such as `animejanaiV2L1.onnx`, `animejanaiV3-HD-L1.onnx`, and
`Ani4Kv2-G6i2-Compact.onnx`.

For the CUDA dependency matrix, see
[docs/trt-cuda-release-matrix.md](docs/trt-cuda-release-matrix.md).

## Manual Installation

Manual installation is possible but easier to get wrong than pip installation.
The release zips are rooted at `vsmlrt/`; when manually combining payloads, put
each payload into its own VapourSynth plugin directory, for example:

```text
vapoursynth/plugins/vsmlrt-generic/
vapoursynth/plugins/vsmlrt-cu121/
vapoursynth/plugins/vsmlrt-cu129/
```

For normal users, prefer the pip extras above. They install `manifest.vs`, native
DLLs, support DLLs, models, and `vsmlrt.py` in the layout expected by the
wrapper.

## Backend Source Layout

- `scripts/vsmlrt.py`: Python wrapper and model-facing API.
- `vstrt/`: TensorRT and TensorRT-RTX plugin source plus custom `trtexec` build files.
- `vsort/`: ONNX Runtime backend source.
- `vsov/`: OpenVINO backend source.
- `vsncnn/`: ncnn Vulkan backend source.
- `vsmigx/`: MIGraphX backend source, retained from upstream but not a focus of this fork.
- `common/`: shared helper code used by native plugins.
- `packaging/payloads/`: pip selector packages for `generic`, `cu121`, `cu129`, and `models`.
- `.github/workflows/`: CI, packaging, smoke tests, and release publication.

## Development And Validation

There is no root CMake project. Build the backend you are changing, for example:

```powershell
cmake -S vstrt -B vstrt/build -G Ninja -D CMAKE_BUILD_TYPE=Release
cmake --build vstrt/build --verbose
cmake --install vstrt/build --prefix vstrt/install
```

Windows release workflows:

- `.github/workflows/windows-vcs-models.yml` builds and publishes the shared
  `models` release asset.
- `.github/workflows/windows-vcs-generic.yml` builds the `generic` native
  payload with `vsncnn`, `vsov`, and `vsort`.
- `.github/workflows/windows-vcs-package.yml` builds the `cu121` and `cu129`
  TensorRT payload assets.
- `.github/workflows/windows-vcs-install-smoke.yml` installs from the default
  branch using `generic`, `cu121`, `cu129`, `cu121,generic`, and
  `cu129,generic` extras, then verifies the installed layout.

GitHub-hosted runners do not provide the NVIDIA display driver, so `nvcuda.dll`
is the only allowed missing dependency in CUDA smoke tests. A green smoke test
proves package layout and plugin loading up to the driver boundary; final GPU
inference still needs a real CUDA machine.

For `generic`, GitHub Actions installs the Vulkan SDK during smoke tests so
`vsncnn.dll` can load on the hosted runner. Real ncnn inference still depends on
the user's installed GPU driver and Vulkan support.

## Useful Overrides

Selector package build hooks support local and URL overrides for maintainer
testing:

- `VSMLRT_GENERIC_PREBUILT_PATH` or `VSMLRT_GENERIC_PREBUILT_URL`.
- `VSMLRT_CU121_PREBUILT_PATH` or `VSMLRT_CU121_PREBUILT_URL`.
- `VSMLRT_CU129_PREBUILT_PATH` or `VSMLRT_CU129_PREBUILT_URL`.
- `VSMLRT_MODELS_PREBUILT_PATH` or `VSMLRT_MODELS_PREBUILT_URL`.
- `VSMLRT_PREBUILT_PATHS` or `VSMLRT_PREBUILT_URLS`: generic fallback for the
  selector currently being built.
- `VSMLRT_RELEASE_REPO=owner/repo`: download native assets from another repo.
- `VSMLRT_MODELS_RELEASE_REPO=owner/repo`: download model assets from another
  repo.
- `VSMLRT_MODELS_TAG=models`: override the shared model release tag.
- `VSMLRT_SKIP_PREBUILT=1`: build a wrapper-only wheel.
- `VSMLRT_DOWNLOAD_PROGRESS=0`: hide build-hook download progress.
- `VSMLRT_DOWNLOAD_PROGRESS_INTERVAL=5`: change progress report interval in seconds.
- `VSMLRT_PROGRESS_CONSOLE=0`: do not try direct Windows console progress output.

The root build hook still accepts legacy single-payload overrides such as
`VSMLRT_PAYLOAD_TAG`, `VSMLRT_CUDA_TAG`, and `VSMLRT_PREBUILT_PATHS`. Those are
for maintainer compatibility checks; normal installs should use pip extras from
the default branch.
