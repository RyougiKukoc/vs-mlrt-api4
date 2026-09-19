# Linux / Windows payload audit and repair plan

The current published Linux assets are materially smaller than Windows. The
requested target is Windows capability parity; package size is secondary.

Required repairs:

1. Linux generic must contain `vsncnn.so` as well as `vsov.so`. Upstream has a
   working `.github/workflows/linux-ncnn.yml`; the fork's Linux packaging had
   simply omitted its artifact. This automatically repairs generic, `cu121`,
   and `cu129`, because CUDA refs overlay generic.
2. Linux standard TensorRT must enable `USE_NVINFER_PLUGIN=ON` and carry the
   Windows-equivalent TensorRT parser, plugin, dispatch/lean and CUDA support
   families. The goal is behavior parity, not the smallest runtime closure.
3. Linux `cu129` must include `vstrt_rtx.so`, the TensorRT-RTX runtime and the
   `tensorrt_rtx` builder helper. NVIDIA publishes the matching
   TensorRT-RTX 1.5.0.114 Linux CUDA 12.9 archive as a `.tar.zst`.
4. Both CUDA refs must publish a dedicated builder overlay containing the
   version-matched `trtexec`, provenance JSON, and TensorRT builder resources.
   Linux CUDA overlays also split oversized CUDA/cuDNN and cu129
   builder-resource payloads into separate assets under GitHub's 2 GiB limit.
   `Backend.TRT` invokes `trtexec` to convert ONNX; host PATH fallback is not a
   complete VCS installation.

`trtexec` is built from the matching NVIDIA TensorRT OSS tag with this fork's
pinned CMake recipe and Windows file-lock/logging fixes. It is not a wholly
from-scratch implementation, and it is not the untouched prebuilt NVIDIA
executable.

The overlays are intentionally separate: generic, standard TensorRT, CUDA,
cuDNN, builder, and (for `cu129`) RTX. The root build hook downloads all
selected overlays and regenerates one final manifest.

Verification requirements:

- component packaging tests must prove builder resources do not leak into the
  standard runtime overlay;
- Linux `readelf`/`ldd` checks must cover every published ELF family;
- clean API4 Linux and Windows installs must load the plugins;
- deterministic `Backend.TRT` ONNX conversion must use the packaged builder;
- `cu129` must build an RTX engine with the packaged `tensorrt_rtx` and render a
  real frame on a compatible NVIDIA GPU.

The exact source provenance previously audited was generic
`e328c6641ca4d133d0f18d7ab337ca7b1b3a7b7a`, `cu121`
`e6a7150d8fd917cd8765e0e008c08d48f427ee80`, and `cu129`
`b44ca11dfac276c99104e465b15282cfa0618aa9`.
