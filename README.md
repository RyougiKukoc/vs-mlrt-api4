# vs-mlrt

VapourSynth ML runtime plugins and the `vsmlrt.py` Python wrapper.

This fork is API4-oriented and publishes tested Windows and Linux x86_64 payloads
through GitHub Releases. Users install one of three VCS refs, `generic`, `cu121`,
or `cu129`. Each ref publishes one self-contained archive per system that carries
everything the ref installs, so a pip install downloads the model payload plus
that single archive.

## Quick Install

Requirements:

- Windows or Linux x86_64.
- Python 3.12 or newer.
- A VapourSynth API4 environment (the release tests run on R77 and newer; the
  Linux generic wheel targets the R79 baseline).
- For `generic`: a working `ncnn`/Vulkan or `ov`/OpenVINO runtime.
- For `cu121` and `cu129`: an NVIDIA driver compatible with that CUDA line.

```powershell
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@generic"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu121"
pip install "vs-mlrt @ git+https://github.com/RyougiKukoc/vs-mlrt-api4.git@cu129"
```

Do not install `cu121` and `cu129` into the same environment. Use `cu121` for
machines limited to CUDA 12.1/12.2-era drivers, and `cu129` for machines with a
current enough NVIDIA driver for CUDA 12.9 user-mode libraries. To update an
existing install, rerun the same command with `-U`; add `-v` if pip hides the
build hook's download progress.

The distribution version `16.2.2` identifies this plugin release, which follows
upstream tag `v16.2.test1`. `vsmlrt.__version__` is the wrapper API version, not
the package version.

The wrapper resolves helper executables in this order: the explicit
`VSMLRT_TRTEXEC_PATH`, `VSMLRT_MIGRAPHX_DRIVER_PATH`, or
`VSMLRT_TENSORRT_RTX_PATH` override; the selected package payload; then the host
`PATH`. It preserves the parent process environment when launching those tools.
On Linux the package libraries carry `RUNPATH=$ORIGIN`, so plugins resolve the
libraries next to them when VapourSynth loads them.

Set `VSMLRT_FORCE_BUILD=1` to bypass the Release payload and build from source
instead; see [Development And Validation](#development-and-validation).

## Installed Layout

Every tag installs one integrated plugin directory:

| Tag | Native plugins | Runtime payload |
| --- | --- | --- |
| `generic` | `vsncnn`, `vsov` | ncnn and OpenVINO support libraries |
| `cu121` | `vsncnn`, `vsov`, `vstrt` | the generic payload plus CUDA 12.1.1, TensorRT 8.6.1.6, cuDNN |
| `cu129` | `vsncnn`, `vsov`, `vstrt`, `vstrt_rtx` | the generic payload plus CUDA 12.9.1, TensorRT 11.1.0.106, TensorRT-RTX 1.5.0.114, cuDNN |

```text
site-packages/vapoursynth/plugins/vsmlrt/
  manifest.vs
  models/
  vsmlrt-cuda/
  cache.json
  openvino, tbb12, vsncnn, vsov, vstrt[, vstrt_rtx]   (.dll on Windows, .so on Linux)
```

Only the files for the selected tag are present. `vsmlrt.py` resolves models
from `vsmlrt/models` and TensorRT helpers from `vsmlrt/vsmlrt-cuda`. OpenVINO
runtime files live at the plugin root only; there is no second copy under
`vsov/`. `manifest.vs` stops VapourSynth from loading support libraries as
plugins: the wheel owns it and regenerates it from the native plugins actually
installed, so `@cu121` yields `vsncnn`, `vsov`, and `vstrt`, and `@cu129` adds
`vstrt_rtx`.

## Backend Scope

This fork publishes a deliberately small, tested backend set:

- `generic`: `vsncnn` and `vsov`.
- `cu121`: adds `vstrt` from the CUDA 12.1 / TensorRT 8.6 line.
- `cu129`: adds `vstrt` and `vstrt_rtx` from the CUDA 12.9 / TensorRT 11 line.

The `vsort`/ONNX Runtime backend is migrated to API4 in source but not published:
for the target users ncnn/Vulkan has no practical disadvantage, while ORT adds
substantially to package size and library placement complexity. `vsmigx` and the
CoreML path stay close to upstream and are not published either, because there is
no ROCm/MIGraphX or Apple hardware available for meaningful validation. Users who
need those backends should use upstream packages or build them from this source;
the migration skill and the upstream comparison base are listed in
[docs/backend-evidence.md](docs/backend-evidence.md).

## Release Asset Layout

Each tag is a binary asset slot consumed by the root build hook:

| Release tag | Assets |
| --- | --- |
| `models` | `models.zip` |
| `generic` | `vs-mlrt-windows-x64-generic.zip`, `vs-mlrt-linux-x64-generic.zip` |
| `cu121` | `vs-mlrt-windows-x64-cu121.zip`, `vs-mlrt-linux-x64-cu121.zip` |
| `cu129` | `vs-mlrt-windows-x64-cu129.zip`, `vs-mlrt-linux-x64-cu129.zip` |

A pip install downloads `models.zip` plus the archive for its tag and platform,
and nothing else: the CUDA archives embed the generic plugins, so they never
download the `generic` asset separately. An archive above GitHub's 2 GiB
per-asset limit is published as numbered volumes of one stream
(`vs-mlrt-linux-x64-cu129.zip.001`, `.002`, ...); the hook probes for volumes,
downloads them, and reads them as a single archive. Everything is deflated at
level 1.

The payload rules live in [docs/trt-cuda-release-matrix.md](docs/trt-cuda-release-matrix.md):
which libraries the archives keep, why builder resources are not optional, and
why a staged Linux library is named after its ELF `DT_SONAME`.

## Manual Installation

Prefer the pip installs above. Manual use means extracting the archive for your
tag and platform into one VapourSynth plugin directory:

```text
vapoursynth/plugins/vsmlrt/
```

The archive is rooted at `vsmlrt/`, and the wheel regenerates `manifest.vs` after
extraction; a manually assembled directory needs a matching `manifest.vs` of its
own.

## Development And Validation

There is no root CMake project. Build the backend you are changing:

```powershell
cmake -S vstrt -B vstrt/build -G Ninja -D CMAKE_BUILD_TYPE=Release
cmake --build vstrt/build --verbose
cmake --install vstrt/build --prefix vstrt/install
```

Payload workflows:

- `.github/workflows/windows-vcs-models.yml` publishes the shared `models` asset.
- `.github/workflows/windows-vcs-generic.yml` builds the `generic` payload.
- `.github/workflows/windows-vcs-package.yml` and
  `.github/workflows/linux-vcs-package.yml` build the `cu121`/`cu129` archives,
  verify the staged payload, install it through the wheel hook, and publish only
  after the staged install and the published-digest checks pass.
- `.github/workflows/windows-vcs-install-smoke.yml` installs from the three tags
  and verifies the installed layout. Run it after refreshing assets.

The release tests in `tools/` cover the parts that fail silently otherwise:
`verify_staged_payload.py` and `verify_linux_staged_payload.py` check payload
content and compression, `test_linux_native_staging.py`,
`test_linux_payload_packaging.py`, `test_linux_staged_payload.py`,
`test_payload_verification.py`, `test_hook_overlay_validation.py`, and
`test_vsmlrt_paths.py` cover staging, packaging, and the installed path helper.
Run any of them directly with `python tools/<name>.py`.

GitHub-hosted runners have no NVIDIA display driver, so a green CUDA smoke test
proves layout and plugin loading up to the driver boundary only; real GPU
inference needs a CUDA machine. On a GPU host, with the package's `vsmlrt-cuda`
directory placed before any system `trtexec` on `PATH`:

```text
python tools/smoke_trtexec.py
```

It builds a small engine through the public `vsmlrt.trtexec()` API and checks
that the wrapper resolved the same executable as `PATH`. GPU-level model and
engine checks are recorded in
[docs/trt-cuda-release-matrix.md](docs/trt-cuda-release-matrix.md) and
[docs/backend-evidence.md](docs/backend-evidence.md).

`VSMLRT_FORCE_BUILD=1` builds from source instead of the Release payload. It
invokes the repository CMake projects and needs a compatible VapourSynth wheel
SDK plus the selected backend SDKs: ncnn, OpenVINO, ONNX, and Protobuf for
`generic`, plus the matching CUDA, TensorRT, and, for `cu129`, TensorRT-RTX SDKs.
Set `VSMLRT_OPENVINO_DIR`, `VSMLRT_NCNN_DIR`, `VSMLRT_ONNX_DIR`,
`VSMLRT_PROTOBUF_DIR`, `VSMLRT_TENSORRT_HOME`, `VSMLRT_TENSORRT_RTX_HOME`,
`CUDAToolkit_ROOT`, and `VSMLRT_RUNTIME_ROOTS` for a non-standard SDK layout.

## Source Layout

- `scripts/vsmlrt.py`: Python wrapper and model-facing API.
- `vstrt/`: TensorRT and TensorRT-RTX plugin source plus custom `trtexec` build files.
- `vsncnn/`, `vsov/`: ncnn Vulkan and OpenVINO backend source.
- `vsort/`, `vsmigx/`: API4-migrated sources that this fork does not publish.
- `common/`: shared helper code used by the native plugins.
- `packaging/`: the PEP 517 build hook, the Linux staging script, and the
  retained payload-wheel builders under `packaging/payloads/`.
- `tools/`: payload packaging, verification, smoke, and regression tools.
- `docs/`: the payload rules and the recorded backend evidence.

## Useful Overrides

Build-hook overrides, for maintainer testing only:

- `VSMLRT_PREBUILT_PATHS` or `VSMLRT_PREBUILT_URLS`: payload archives for the
  selected tag (a base archive, a whole archive, or any one of its volumes).
- `VSMLRT_MODELS_PREBUILT_PATH` or `VSMLRT_MODELS_PREBUILT_URL`: the model payload.
- `VSMLRT_PAYLOAD_TAG`: select `generic`, `cu121`, or `cu129` explicitly;
  `VSMLRT_CUDA_TAG` is the legacy form.
- `VSMLRT_RELEASE_REPO`, `VSMLRT_MODELS_RELEASE_REPO`, `VSMLRT_MODELS_TAG`:
  download from another repository or model tag.
- `VSMLRT_SKIP_PREBUILT=1`: build a wrapper-only wheel.
- `VSMLRT_DOWNLOAD_PROGRESS=0`, `VSMLRT_DOWNLOAD_PROGRESS_INTERVAL`,
  `VSMLRT_PROGRESS_CONSOLE=0`: control download progress output.

The payload-wheel builders under `packaging/payloads/` accept per-tag
`VSMLRT_<TAG>_PREBUILT_PATH`/`_URL` overrides; normal installs use the tag-based
commands above. Keep `generic`, `cu121`, and `cu129` on distinct commits so a
VCS checkout resolves one install variant unambiguously.
