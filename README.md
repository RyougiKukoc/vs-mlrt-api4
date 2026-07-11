# vs-mlrt

VapourSynth ML runtime plugins and the `vsmlrt.py` Python wrapper.

This fork is API4-oriented and publishes Windows x64 binary payloads through
GitHub Releases. Users install from the default Git branch and select payloads
with pip extras; Release tags are binary asset slots, not the normal install
entry point.

## Quick Install: Windows

Requirements:

- Windows x64.
- Python 3.12 or newer.
- A VapourSynth R75+ Python environment.
- For `generic`: a working runtime for the backend you use:
  `ncnn`/Vulkan or `ov`/OpenVINO.
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

To update an existing VCS installation, rerun the same command with `-U`. The
`3.23.2.post1` packaging revision upgrades the generic payload selector, so pip
uninstalls the old duplicated `vsov/` layout before installing the flat layout.
The wrapper still reports upstream API version `3.23.2`.

On non-Windows platforms this VCS package does not install native prebuilt
payloads. Use upstream packages or build the required backend from source.

## Uninstall

Pip extras install separate payload wheels. Therefore
`pip uninstall vs-mlrt` removes only the Python wrapper package; it does not
remove models or native backend payloads that were installed by extras.

Uninstall the wrapper together with the payload packages you installed:

```powershell
pip uninstall -y vs-mlrt vs-mlrt-models vs-mlrt-payload-generic
pip uninstall -y vs-mlrt vs-mlrt-models vs-mlrt-payload-cu121
pip uninstall -y vs-mlrt vs-mlrt-models vs-mlrt-payload-cu129
pip uninstall -y vs-mlrt vs-mlrt-models vs-mlrt-payload-cu121 vs-mlrt-payload-generic
pip uninstall -y vs-mlrt vs-mlrt-models vs-mlrt-payload-cu129 vs-mlrt-payload-generic
```

For a full cleanup regardless of which extras were installed, it is safe to
name all known vs-mlrt packages:

```powershell
pip uninstall -y vs-mlrt vs-mlrt-models vs-mlrt-payload-generic vs-mlrt-payload-cu121 vs-mlrt-payload-cu129
```

## Payload Extras

All extras install the shared model payload once through the `vs-mlrt-models`
selector package. The native payloads are separate wheels, so combinations such
as `cu121,generic` do not duplicate `vsmlrt.py` or the model files.

| Extra | Native plugins | Runtime payload | Installed plugin directory |
| --- | --- | --- | --- |
| `generic` | `vsncnn`, `vsov` | ncnn and OpenVINO support DLLs | `site-packages/vapoursynth/plugins/vsmlrt/` |
| `cu121` | `vstrt` | CUDA 12.1.1, TensorRT 8.6.1.6, cuDNN | `site-packages/vapoursynth/plugins/vsmlrt/` |
| `cu129` | `vstrt`, `vstrt_rtx` | CUDA 12.9.1, TensorRT 11.1.0.106, TensorRT-RTX 1.5.0.114, cuDNN | `site-packages/vapoursynth/plugins/vsmlrt/` |

All selected payloads overlay into the same `vsmlrt` plugin directory, matching
the upstream integrated release layout:

```text
site-packages/vapoursynth/plugins/vsmlrt/
  manifest.vs
  models/
  vsmlrt-cuda/
  cache.json
  openvino.dll
  tbb12.dll
  vsncnn.dll
  vsov.dll
  vstrt.dll
  vstrt_rtx.dll
```

Only the files for the selected extras are present. `vstrt_rtx.dll` is installed
only by `cu129`. `vsmlrt.py` resolves models from `vsmlrt/models`, and
TensorRT helper executables are resolved from `vsmlrt/vsmlrt-cuda`. The wheel
also installs a small DLL search-path helper. OpenVINO runtime files live only
at the plugin root; the package does not ship a second copy under `vsov/`.

`manifest.vs` prevents VapourSynth from trying to load support DLLs as plugins.
The main `vs-mlrt` package owns this shared file and refreshes it from the
native plugin DLLs actually installed. For example, `generic,cu121` produces a
manifest containing `vsncnn`, `vsov`, and `vstrt`, independent of pip's payload
installation order.

## Backend Scope

This fork intentionally publishes a smaller Windows payload set than upstream.
The released VCS extras are:

- `generic`: `vsncnn` and `vsov`.
- `cu121`: `vstrt` built for the CUDA 12.1/TensorRT 8.6 line.
- `cu129`: `vstrt` plus `vstrt_rtx` built for the CUDA 12.9/TensorRT 11 line.

The `vsort`/ONNX Runtime backend was migrated to API4 in source, but this fork
does not publish it in the `generic` payload. For our target Windows users,
ncnn/Vulkan has no obvious practical disadvantage compared with ORT/DirectML,
while ORT substantially increases package size and DLL placement complexity.
Users who explicitly need ORT should use upstream packages or build `vsort`
from source.

`vsmigx` and the CoreML path are not part of this fork's migration/release
scope. The maintainer does not have suitable ROCm/MIGraphX or Apple/CoreML
hardware for meaningful build and runtime validation, so those paths are left
close to upstream instead of being presented as supported payloads.

Anyone who wants to continue those backends can use the published migration
skill and this fork's patch as the starting point:

- Skill repository:
  <https://github.com/RyougiKukoc/vapoursynth-api3-to-api4-skill>
- Upstream comparison base:
  `AmusementClub/vs-mlrt` tag `v15.16`,
  commit `885e8bb827fc431fce8e3109e7d60b0c38aa2035`.

A useful local comparison command is:

```powershell
git remote add upstream https://github.com/AmusementClub/vs-mlrt.git 2>$null
git fetch upstream tag v15.16
git diff 885e8bb827fc431fce8e3109e7d60b0c38aa2035..HEAD
```

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
The release zips are rooted at `vsmlrt/`; when manually combining payloads,
overlay all selected assets into one VapourSynth plugin directory:

```text
vapoursynth/plugins/vsmlrt/
```

Each individual release line includes its own `manifest.vs`. When manually
overlaying `generic` with a CUDA release, merge the plugin names in the
manifest as well. Pip extras do this automatically and are the supported
combination path.

For normal users, prefer the pip extras above. They install native DLLs, support
DLLs, models, the DLL search-path helper, and `vsmlrt.py` in the layout expected
by the wrapper.

## Backend Source Layout

- `scripts/vsmlrt.py`: Python wrapper and model-facing API.
- `vstrt/`: TensorRT and TensorRT-RTX plugin source plus custom `trtexec` build files.
- `vsort/`: ONNX Runtime backend source, API4-migrated but not published by this fork.
- `vsov/`: OpenVINO backend source.
- `vsncnn/`: ncnn Vulkan backend source.
- `vsmigx/`: MIGraphX backend source, retained from upstream and not modified or published by this fork.
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
  payload with `vsncnn` and `vsov`.
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
