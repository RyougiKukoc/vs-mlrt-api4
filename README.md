# vs-mlrt

VapourSynth ML runtime plugins and the `vsmlrt.py` Python wrapper.

This fork is API4-oriented and publishes tested Windows and Linux x86_64 binary
payloads through GitHub Releases. Users install from one of three VCS refs,
`generic`, `cu121`, or `cu129`. The CUDA refs install their own TensorRT
payload plus the `generic` plugin payload automatically, while GitHub Releases
stay trimmed so `cu121` and `cu129` publish only the TRT-side assets.

## Quick Install

Requirements:

- Windows or Linux x86_64.
- Python 3.12 or newer.
- A VapourSynth R75+ Python environment.
- For `generic`: a working runtime for the backend you use:
  `ncnn`/Vulkan or `ov`/OpenVINO.
- For `cu121` and `cu129`: an NVIDIA driver compatible with the selected CUDA
  payload.

Install from the tag that matches the payload line you want:

```powershell
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@generic"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu121"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu129"
```

Do not install `cu121` and `cu129` into the same environment. Use `cu121` for
machines limited to CUDA 12.1/12.2-era drivers, and use `cu129` for machines
with a current enough NVIDIA driver for CUDA 12.9 user-mode libraries.
`@cu121` installs `vsncnn`, `vsov`, and `vstrt`; `@cu129` installs `vsncnn`,
`vsov`, `vstrt`, and `vstrt_rtx`.

The package keeps the public Python entry point stable:

```python
import vsmlrt

out = vsmlrt.DPIR(clip, strength=5.0, backend=vsmlrt.Backend.TRT(fp16=True))
```

If pip hides build-backend output while large assets download, add `-v` to the
install command.

To update an existing VCS installation, rerun the same command with `-U`. The
The Python distribution version `16.2.2` identifies the complete plugin release
aligned with upstream tag `v16.2.test1`. The separate `vsmlrt.__version__` value
identifies only the Python wrapper API and is not the package version. The
Windows install flow continues to use the three tag-selected variants and the
flattened OpenVINO runtime layout.

On Linux x86_64, the build hook downloads the matching tested Linux Release
payload by default and reports `vs-mlrt: using Release asset`. The Linux
`generic` payload carries the CPU-capable OpenVINO backend (`vsov`); the CUDA
payloads overlay the matching TensorRT backend on it. Linux does not ship the
Vulkan/NCNN binary because its Vulkan driver requirements are materially less
portable than the OpenVINO CPU payload. Windows continues to ship both NCNN and
OpenVINO in `generic`.

Set `VSMLRT_FORCE_BUILD=1` to bypass Release assets. The fallback invokes the
repository CMake projects and requires a compatible VapourSynth wheel SDK plus
the selected backend SDKs. It prepends the installed wheel's
`vapoursynth/pkgconfig` directory to `PKG_CONFIG_PATH` without discarding an
existing value. Generic builds require OpenVINO, ONNX, and Protobuf CMake
packages; CUDA builds additionally require the matching CUDA, TensorRT, and,
for `cu129` RTX, TensorRT-RTX SDKs. Set `VSMLRT_OPENVINO_DIR`,
`VSMLRT_TENSORRT_HOME`, `VSMLRT_TENSORRT_RTX_HOME`, `CUDAToolkit_ROOT`, and
`VSMLRT_RUNTIME_ROOTS` as appropriate for a non-standard SDK layout.

## Uninstall

The VCS install produces a single `vs-mlrt` wheel, so normal cleanup is:

```powershell
pip uninstall -y vs-mlrt
```

## Installed Layout

Each tag installs one integrated package directory:

| Tag | Native plugins | Runtime payload | Installed plugin directory |
| --- | --- | --- | --- |
| `generic` | `vsncnn`, `vsov` | ncnn and OpenVINO support DLLs | `site-packages/vapoursynth/plugins/vsmlrt/` |
| `cu121` | `vsncnn`, `vsov`, `vstrt` | generic payload plus CUDA 12.1.1, TensorRT 8.6.1.6, cuDNN | `site-packages/vapoursynth/plugins/vsmlrt/` |
| `cu129` | `vsncnn`, `vsov`, `vstrt`, `vstrt_rtx` | generic payload plus CUDA 12.9.1, TensorRT 11.1.0.106, TensorRT-RTX 1.5.0.114, cuDNN | `site-packages/vapoursynth/plugins/vsmlrt/` |

All selected files land in the same `vsmlrt` plugin directory, matching the
upstream integrated release layout:

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

Only the files for the selected tag are present. `vstrt_rtx.dll` is installed
only by `cu129`. `vsmlrt.py` resolves models from `vsmlrt/models`, and
TensorRT helper executables are resolved from `vsmlrt/vsmlrt-cuda`. The wheel
also installs a small DLL search-path helper. OpenVINO runtime files live only
at the plugin root; the package does not ship a second copy under `vsov/`.

`manifest.vs` prevents VapourSynth from trying to load support DLLs as plugins.
The `vs-mlrt` wheel owns this file and regenerates it from the native plugin
DLLs actually present. That means `@cu121` produces `vsncnn`, `vsov`, and
`vstrt`, while `@cu129` adds `vstrt_rtx`.

## Backend Scope

This fork intentionally publishes a smaller Windows payload set than upstream.
The released VCS tags are:

- `generic`: `vsncnn` and `vsov`.
- `cu121`: `vstrt` built for the CUDA 12.1/TensorRT 8.6 line.
- `cu129`: `vstrt` plus `vstrt_rtx` built for the CUDA 12.9/TensorRT 11 line.

On Linux x86_64, the matching release asset names are
`vs-mlrt-linux-x64-generic.zip`,
`vs-mlrt-linux-x64-tensorrt-<variant>.zip`,
`vs-mlrt-linux-x64-cuda-<variant>.zip`,
`vs-mlrt-linux-x64-cudnn-<variant>.zip`. The CUDA assets overlay in
that order after the generic asset. Every zip is rooted at `vsmlrt/` and
contains ELF `.so` files only; the shared `models.zip` payload is installed for
all three refs. Linux CUDA payloads currently carry the standard `vstrt`
backend; `vstrt_rtx` remains Windows-only until a separately validated Linux
TensorRT-RTX runtime is available. Generic Linux wheels use the VapourSynth
R79 baseline tag `manylinux_2_27_x86_64`. The CUDA 12.1 and 12.9 TensorRT
plugins are tagged `manylinux_2_34_x86_64`: `readelf --version-info` on their
final `vstrt.so` records `GLIBC_2.34`, so presenting those SDK-bound wheels as
`manylinux_2_27` would be false. The VapourSynth runtime remains an additional
end-to-end compatibility requirement.

The `vsort`/ONNX Runtime backend was migrated to API4 in source, but this fork
does not publish it in the `generic` payload. For our target Windows users,
ncnn/Vulkan has no obvious practical disadvantage compared with ORT/DirectML,
while ORT substantially increases package size and DLL placement complexity.
Users who explicitly need ORT should use upstream packages or build `vsort`
from source.

`windows-vcs-generic.yml` and `windows-vcs-package.yml` maintain the Windows
release payloads. The older individual NCNN, OpenVINO, TensorRT, and TRT-RTX
workflows are manual/reusable compatibility builds. `windows-ort.yml` remains
an independent compatibility build with its existing CUDA/DirectML dependency
versions; it uses API4 headers and R77 to check plugin ownership and CPU model
execution. Its CPU tests do not verify CUDA or DirectML inference.

To run the artifact regressions with an R77 Python environment:

```text
python tools/regression_native_api4.py --plugin path/to/vsncnn.dll --namespace ncnn --exercise-ncnn --output verification/ncnn.json
python tools/regression_native_api4.py --plugin path/to/vsort.dll --namespace ort --exercise-ort --output verification/ort.json
```

Model cases require NumPy and ONNX. NCNN execution additionally requires a
Vulkan device supporting fp16. Omit the `--exercise-*` option to check only
registration and two independent core lifetimes; repeat `--dll-dir` when
support DLLs are outside the plugin's normal package directories.

`vsmigx` and the CoreML path are not part of this fork's migration/release
scope. The maintainer does not have suitable ROCm/MIGraphX or Apple/CoreML
hardware for meaningful build and runtime validation, so those paths are left
close to upstream instead of being presented as supported payloads.

Anyone who wants to continue those backends can use the published migration
skill and this fork's patch as the starting point:

- Skill repository:
  <https://github.com/RyougiKukoc/vapoursynth-api3-to-api4-skill>
- Upstream comparison base:
  `AmusementClub/vs-mlrt` tag `v16.2.test1`,
  commit `9e4d0c9dbbcaa28275772d30520330e69a58307c`.

A useful local comparison command is:

```powershell
git remote add upstream https://github.com/AmusementClub/vs-mlrt.git 2>$null
git fetch upstream tag v16.2.test1
git diff 9e4d0c9dbbcaa28275772d30520330e69a58307c..HEAD
```

## Paired Model Evidence

The API3 R73 Windows baseline and the `generic` API4 Release payloads were
compared with the same current `scripts/vsmlrt.py` wrapper, deterministic
64x64 `RGBS` input, and `BackendV2.OV_CPU()`. The three model files were
copied into extracted test payload directories only; they are not additions to
the `generic` Release asset.

| Wrapper case | Model file | SHA-256 | Output | Result |
| --- | --- | --- | --- | --- |
| `RealESRGANModel.animejanaiV3_HD_L1` | `RealESRGANv2/animejanaiV3-HD-L1.onnx` | `d328ff0b2fc36145af167093d951ac6fd577e8be26fe0e48764557ac94e03877` | 128x128 RGBS | API3 Windows, API4 Windows, and API4 Linux bytes match. |
| `Waifu2xModel.cunet`, noise 3, scale 1 | `waifu2x/cunet/noise3_model.onnx` | `1c2439403f8f2c6ac5f95d9be780d257e910d17dd89d79b41da0fe1b7ad11b21` | 64x64 RGBS | API3 Windows, API4 Windows, and API4 Linux bytes match. |
| `DPIRModel.drunet_color`, strength 5 | `dpir/drunet_color.onnx` | `ae6af55252e268e9dd3f567e66b81227c98fd2846b3cb7febd9d0a6bbabb4617` | 64x64 RGBS | API3/API4 Windows bytes match; Linux OpenVINO CPU differs by at most `4.172325134277344e-07` (`mean_abs=5.6869614202999706e-08`), below the `1e-5` comparison limit. |

Each case renders a real frame and compares contiguous float32 planes, rather
than treating model creation or plugin version reporting as inference evidence.

## Release Asset Layout

These Release tags remain as binary asset slots consumed by the root build
hook:

- `models`: `models.zip`.
- `generic`: `vs-mlrt-windows-x64-generic.zip` and
  `vs-mlrt-linux-x64-generic.zip`.
- `cu121`: `vs-mlrt-windows-x64-tensorrt-cu121.zip`,
  `vs-mlrt-windows-x64-cuda-cu121.zip`, and
  `vs-mlrt-windows-x64-cudnn-cu121.zip`, plus
  `vs-mlrt-linux-x64-tensorrt-cu121.zip`,
  `vs-mlrt-linux-x64-cuda-cu121.zip`,
  `vs-mlrt-linux-x64-cudnn-cu121.zip`.
- `cu129`: `vs-mlrt-windows-x64-tensorrt-cu129.zip`,
  `vs-mlrt-windows-x64-cuda-cu129.zip`,
  `vs-mlrt-windows-x64-cudnn-cu129.zip`,
  `vs-mlrt-windows-x64-tensorrt-core-cu129.zip`,
  `vs-mlrt-windows-x64-tensorrt-plugin-cu129.zip`,
  `vs-mlrt-windows-x64-tensorrt-extra-cu129.zip`, and
  `vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip`, plus
  `vs-mlrt-linux-x64-tensorrt-cu129.zip`,
  `vs-mlrt-linux-x64-cuda-cu129.zip`,
  `vs-mlrt-linux-x64-cudnn-cu129.zip`.

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

Each individual release line includes its own `manifest.vs`. The `generic`
release lists `vsncnn` and `vsov`; the CUDA releases list only their TRT-side
plugins. When manually overlaying `generic` with a CUDA release, merge the
plugin names in the manifest as well.

For normal users, prefer the tag-based pip installs above. They install native DLLs, support
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
- `packaging/payloads/`: retained payload-wheel builders for maintainers and release fixtures.
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
- `.github/workflows/windows-vcs-install-smoke.yml` installs from the
  `generic`, `cu121`, and `cu129` VCS tags, then verifies the installed layout.

GitHub-hosted runners do not provide the NVIDIA display driver, so `nvcuda.dll`
is the only allowed missing dependency in CUDA smoke tests. A green smoke test
proves package layout and plugin loading up to the driver boundary; final GPU
inference still needs a real CUDA machine.

For `generic`, GitHub Actions installs the Vulkan SDK during smoke tests so
`vsncnn.dll` can load on the hosted runner. Real ncnn inference still depends on
the user's installed GPU driver and Vulkan support.

## Useful Overrides

The build hooks support local and URL overrides for maintainer testing:

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

The root build hook accepts maintainer overrides such as
`VSMLRT_PAYLOAD_TAG`, `VSMLRT_CUDA_TAG`, and `VSMLRT_PREBUILT_PATHS`. Normal
installs should use the tag-based VCS commands above. Keep `generic`, `cu121`,
and `cu129` on distinct commits so a VCS checkout resolves one install variant
unambiguously.
