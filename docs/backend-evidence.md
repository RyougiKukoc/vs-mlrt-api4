# Backend evidence and porting notes

Recorded runtime evidence for the published payloads, plus the starting points
for the backends this fork does not publish.

## Porting starting points

`vsmigx` and the CoreML path are outside this fork's migration and release scope.
The maintainer has no ROCm/MIGraphX or Apple/CoreML hardware for meaningful build
and runtime validation, so those paths are left close to upstream instead of
being presented as supported payloads. Anyone continuing them can start from:

- Skill repository:
  <https://github.com/RyougiKukoc/vapoursynth-api3-to-api4-skill>
- Upstream comparison base:
  `AmusementClub/vs-mlrt` tag `v16.2.test1`,
  commit `9e4d0c9dbbcaa28275772d30520330e69a58307c`.

```powershell
git remote add upstream https://github.com/AmusementClub/vs-mlrt.git 2>$null
git fetch upstream tag v16.2.test1
git diff 9e4d0c9dbbcaa28275772d30520330e69a58307c..HEAD
```

The bundled `trtexec` is built from the matching NVIDIA TensorRT OSS tag with
this fork's pinned CMake recipe and its Windows file-lock and logging fixes. It
is not a from-scratch implementation, and it is not the untouched prebuilt
NVIDIA executable.

## Paired model evidence

The API3 R73 Windows baseline and the `generic` API4 Release payloads were
compared with the same current `scripts/vsmlrt.py` wrapper, deterministic 64x64
`RGBS` input, and `BackendV2.OV_CPU()`. The three model files were copied into
extracted test payload directories only; they are not additions to the `generic`
Release asset.

| Wrapper case | Model file | SHA-256 | Output | Result |
| --- | --- | --- | --- | --- |
| `RealESRGANModel.animejanaiV3_HD_L1` | `RealESRGANv2/animejanaiV3-HD-L1.onnx` | `d328ff0b2fc36145af167093d951ac6fd577e8be26fe0e48764557ac94e03877` | 128x128 RGBS | API3 Windows, API4 Windows, and API4 Linux bytes match. |
| `Waifu2xModel.cunet`, noise 3, scale 1 | `waifu2x/cunet/noise3_model.onnx` | `1c2439403f8f2c6ac5f95d9be780d257e910d17dd89d79b41da0fe1b7ad11b21` | 64x64 RGBS | API3 Windows, API4 Windows, and API4 Linux bytes match. |
| `DPIRModel.drunet_color`, strength 5 | `dpir/drunet_color.onnx` | `ae6af55252e268e9dd3f567e66b81227c98fd2846b3cb7febd9d0a6bbabb4617` | 64x64 RGBS | API3/API4 Windows bytes match; Linux OpenVINO CPU differs by at most `4.172325134277344e-07` (`mean_abs=5.6869614202999706e-08`), below the `1e-5` comparison limit. |

Each case renders a real frame and compares contiguous float32 planes, rather
than treating model creation or plugin version reporting as inference evidence.
`tools/compare_api3_api4_vsmlrt_backends.py` reproduces this comparison; it needs
NumPy and ONNX and the two payload layouts side by side.

## Accelerator backend evidence

The same RTX 3090 Ti (SM 8.6, driver API 13040) was used for the CUDA cases. The
standard TensorRT and TensorRT-RTX lines were kept in isolated API3 and cu129
API4 payload directories.

- `BackendV2.OV_GPU()` was invoked with AnimeJanai on API3 Windows, API4 Windows,
  and API4 Linux. All three correctly report no supported OpenVINO GPU device.
  This host has NVIDIA hardware only; OpenVINO's GPU plugin needs a compatible
  Intel GPU. The Linux check was repeated with an OpenCL dispatch loader and PoCL
  ICD, so the result is a device capability boundary rather than a missing-loader
  claim.
- Standard TRT API3 built an RTX 3090 Ti engine for AnimeJanai with the baseline
  TensorRT 10.14.1 `trtexec` and rendered a 128x128 RGBS frame. The cu129 API4
  builder overlay created a static 1x1x16x16 identity engine, then the extracted
  `vstrt.so` runtime payload loaded it and rendered an exact 16x16 GrayS identity
  frame (`3550e6853d980fa61e6e0c9b0acb00e60f1594784ec83b46d5e977d0080f6f23`).
- TensorRT-RTX built isolated engines and rendered all three model cases with the
  baseline 1.1.1 and cu129 API4 1.5.0 runtimes. AnimeJanai matched bytes. CUNet
  noise3 had `max_abs=0.0005944371223449707` and DPIR had
  `max_abs=0.00029768049716949463`; both are finite, same-shaped GPU outputs from
  distinct RTX engine/compiler versions and are recorded as numerical
  differences, not strict matches.

Do not move `trtexec` or the builder resources into a runtime-only payload to
make a model-building convenience path work. Keep builder and runtime evidence
separate, and always request a frame after loading a generated engine.
