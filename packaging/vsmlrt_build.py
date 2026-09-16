"""Build a variant-specific vs-mlrt wheel from release payloads or source.

The three VCS refs carry ``packaging/payload-tag.txt``. Keeping that marker in
the checkout makes a PEP 517 build deterministic even when pip performs a
shallow detached checkout and does not retain annotated tags.
"""
from __future__ import annotations

import filecmp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


PAYLOAD_TAGS = {"generic", "cu121", "cu129"}
CUDA_TAGS = {"cu121", "cu129"}
GENERIC_TAG = "generic"
MODELS_TAG = "models"
MODELS_ASSET = "models.zip"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_PROGRESS_INTERVAL = 5.0
MANIFEST_HEADER = "[VapourSynth Manifest V1]"
PLUGIN_BASENAMES = ("vsncnn", "vsov", "vsort", "vstrt", "vstrt_rtx", "vsmigx")
LINUX_MACHINE_NAMES = {"amd64", "x86_64"}


class CustomBuildHook(BuildHookInterface):
    """Attach a tested native plugin payload to a VCS-built wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        del version
        if self.target_name != "wheel":
            return

        payload_tag = self._detect_payload_tag()
        force_include = build_data.setdefault("force_include", {})
        stage_dir = Path(self.root) / "build" / "vsmlrt_payload"
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True)

        if self._skip_prebuilt():
            force_include[str(Path(self.root) / "packaging" / "manifest.vs")] = (
                "vapoursynth/plugins/vsmlrt/manifest.vs"
            )
            return

        used_prebuilt = False
        if not self._force_build() and self._has_platform_release_payload():
            try:
                self._stage_prebuilt_payloads(stage_dir, payload_tag)
                used_prebuilt = True
            except Exception as error:
                print(
                    "vs-mlrt: matching release payload was unavailable; "
                    f"falling back to local native build ({error})",
                    file=sys.stderr,
                    flush=True,
                )

        if not used_prebuilt:
            self._stage_local_build(stage_dir, payload_tag)

        # Models are platform-neutral data. A native force build still needs
        # them for the public Python wrapper, so acquire them separately.
        self._stage_models(stage_dir)
        self._prepare_plugin_dir(stage_dir)
        self._validate_plugin_dir(stage_dir, payload_tag)
        force_include[str(stage_dir)] = "vapoursynth/plugins/vsmlrt"
        build_data["tag"] = self._wheel_tag(payload_tag)
        mode = "Release asset" if used_prebuilt else "local native build"
        print(f"vs-mlrt: using {mode} for {payload_tag} on {platform.system()}", file=sys.stderr, flush=True)

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        del version, build_data, artifact_path
        shutil.rmtree(Path(self.root) / "build" / "vsmlrt_payload", ignore_errors=True)

    def _skip_prebuilt(self) -> bool:
        return self._truthy(os.environ.get("VSMLRT_SKIP_PREBUILT"))

    def _force_build(self) -> bool:
        return self._truthy(os.environ.get("VSMLRT_FORCE_BUILD"))

    @staticmethod
    def _truthy(value: str | None) -> bool:
        return bool(value and value.strip().lower() not in {"", "0", "false", "no", "off"})

    def _wheel_tag(self, payload_tag: str) -> str:
        override = os.environ.get("VSMLRT_PLATFORM_TAG")
        if override:
            return f"py3-none-{override}"
        if platform.system() == "Windows" and platform.machine().lower() in LINUX_MACHINE_NAMES:
            return "py3-none-win_amd64"
        if platform.system() == "Linux" and platform.machine().lower() in LINUX_MACHINE_NAMES:
            # The generic OpenVINO build uses the VapourSynth R79 baseline.
            # CUDA plugins compiled against current NVIDIA SDKs require
            # GLIBC_2.34, verified from their final vstrt.so ELF metadata.
            platform_tag = "manylinux_2_27_x86_64" if payload_tag == GENERIC_TAG else "manylinux_2_34_x86_64"
            return f"py3-none-{platform_tag}"
        return "py3-none-any"

    def _detect_payload_tag(self) -> str:
        explicit = os.environ.get("VSMLRT_PAYLOAD_TAG")
        if explicit:
            if explicit not in PAYLOAD_TAGS:
                raise RuntimeError(
                    f"Unsupported VSMLRT_PAYLOAD_TAG={explicit!r}; expected generic, cu121, or cu129."
                )
            return explicit

        marker = Path(self.root) / "packaging" / "payload-tag.txt"
        if marker.is_file():
            selected = marker.read_text(encoding="ascii").strip()
            if selected in PAYLOAD_TAGS:
                return selected
            raise RuntimeError(f"Invalid payload marker {marker}: {selected!r}.")

        # Compatibility with old checkout markers. New variant commits always
        # carry payload-tag.txt and do not depend on this fallback.
        legacy = os.environ.get("VSMLRT_CUDA_TAG")
        if legacy in CUDA_TAGS:
            return legacy
        return GENERIC_TAG

    def _has_platform_release_payload(self) -> bool:
        system = platform.system()
        machine = platform.machine().lower()
        return machine in LINUX_MACHINE_NAMES and system in {"Windows", "Linux"}

    def _stage_prebuilt_payloads(self, stage_dir: Path, payload_tag: str) -> None:
        for payload in self._resolve_payload_paths(payload_tag):
            self._safe_extract_payload(payload, stage_dir)

    def _stage_models(self, stage_dir: Path) -> None:
        models = self._resolve_models_payload()
        extracted = Path(self.root) / "build" / "vsmlrt_models"
        shutil.rmtree(extracted, ignore_errors=True)
        extracted.mkdir(parents=True)
        self._safe_extract_payload(models, extracted)
        for candidate in (extracted / "vsmlrt" / "models", extracted / "models"):
            if candidate.is_dir():
                shutil.copytree(candidate, stage_dir / "models", dirs_exist_ok=True)
                return
        raise RuntimeError("Model payload is missing vsmlrt/models/ or models/.")

    def _stage_local_build(self, stage_dir: Path, payload_tag: str) -> None:
        command = [
            sys.executable,
            str(Path(self.root) / "packaging" / "linux_native_build.py"),
            "--variant",
            payload_tag,
            "--stage-dir",
            str(stage_dir),
        ]
        env = os.environ.copy()
        env["VSMLRT_PAYLOAD_TAG"] = payload_tag
        subprocess.run(command, cwd=self.root, env=env, check=True)

    def _prepare_plugin_dir(self, plugin_dir: Path) -> None:
        # The legacy Windows payload stores OpenVINO support files in vsov/.
        # Linux payloads keep a flat directory to match their $ORIGIN RPATH.
        if platform.system() == "Windows":
            self._flatten_openvino_runtime(plugin_dir)
        self._write_manifest(plugin_dir)

    def _flatten_openvino_runtime(self, plugin_dir: Path) -> None:
        support_dir = plugin_dir / "vsov"
        if not support_dir.is_dir():
            return
        for source in support_dir.rglob("*"):
            if not source.is_file():
                continue
            destination = plugin_dir / source.name
            if destination.exists():
                if not filecmp.cmp(source, destination, shallow=False):
                    raise RuntimeError(f"Conflicting OpenVINO runtime files: {source} and {destination}.")
                source.unlink()
            else:
                shutil.move(str(source), destination)
        shutil.rmtree(support_dir)

    def _write_manifest(self, plugin_dir: Path) -> None:
        suffix = self._native_suffix()
        plugins = [name for name in PLUGIN_BASENAMES if (plugin_dir / f"{name}{suffix}").is_file()]
        (plugin_dir / "manifest.vs").write_text(
            "\n".join((MANIFEST_HEADER, *plugins, "")), encoding="ascii", newline="\n"
        )

    def _validate_plugin_dir(self, plugin_dir: Path, payload_tag: str) -> None:
        suffix = self._native_suffix()
        expected = {"vsov"} if platform.system() == "Linux" else {"vsncnn", "vsov"}
        if payload_tag in CUDA_TAGS:
            expected.add("vstrt")
        # Linux publishes standard TensorRT only. TensorRT-RTX needs its own
        # Linux SDK/runtime and supported RTX validation before it can join a
        # Linux payload; Windows retains the existing cu129 RTX package.
        if payload_tag == "cu129" and platform.system() == "Windows":
            expected.add("vstrt_rtx")
        missing = [name for name in sorted(expected) if not (plugin_dir / f"{name}{suffix}").is_file()]
        if missing:
            raise RuntimeError(
                f"Selected {payload_tag} payload is missing native {suffix} plugin(s): {', '.join(missing)}."
            )
        if not (plugin_dir / "models").is_dir():
            raise RuntimeError("Selected payload is missing models/.")

    def _native_suffix(self) -> str:
        return {"Windows": ".dll", "Linux": ".so", "Darwin": ".dylib"}.get(platform.system(), ".so")

    def _resolve_payload_paths(self, payload_tag: str) -> list[Path]:
        explicit = os.environ.get("VSMLRT_PREBUILT_PATHS") or os.environ.get("VSMLRT_PREBUILT_PATH")
        if explicit:
            paths = [Path(value).expanduser().resolve() for value in explicit.split(os.pathsep) if value.strip()]
            if not paths:
                raise RuntimeError("VSMLRT_PREBUILT_PATHS did not contain any paths.")
            return paths
        explicit_urls = os.environ.get("VSMLRT_PREBUILT_URLS") or os.environ.get("VSMLRT_PREBUILT_URL")
        if explicit_urls:
            return self._download_urls([value for value in explicit_urls.split(os.pathsep) if value.strip()])
        return self._download_release_payloads(payload_tag)

    def _resolve_models_payload(self) -> Path:
        explicit = os.environ.get("VSMLRT_MODELS_PREBUILT_PATH")
        if explicit:
            return Path(explicit).expanduser().resolve()
        explicit_url = os.environ.get("VSMLRT_MODELS_PREBUILT_URL")
        if explicit_url:
            return self._download_urls([explicit_url])[0]
        repo = os.environ.get("VSMLRT_MODELS_RELEASE_REPO") or self._detect_github_repo()
        tag = os.environ.get("VSMLRT_MODELS_TAG") or MODELS_TAG
        return self._download_urls([f"https://github.com/{repo}/releases/download/{tag}/{MODELS_ASSET}"])[0]

    def _download_release_payloads(self, payload_tag: str) -> list[Path]:
        repo = os.environ.get("VSMLRT_RELEASE_REPO") or self._detect_github_repo()
        system = platform.system()
        if system == "Windows":
            if payload_tag == GENERIC_TAG:
                assets = ["vs-mlrt-windows-x64-generic.zip"]
                tags = [GENERIC_TAG]
            else:
                assets = [
                    "vs-mlrt-windows-x64-generic.zip",
                    f"vs-mlrt-windows-x64-tensorrt-{payload_tag}.zip",
                    f"vs-mlrt-windows-x64-cuda-{payload_tag}.zip",
                    f"vs-mlrt-windows-x64-cudnn-{payload_tag}.zip",
                ]
                tags = [GENERIC_TAG, payload_tag, payload_tag, payload_tag]
                if payload_tag == "cu129":
                    assets.extend(
                        [
                            "vs-mlrt-windows-x64-tensorrt-core-cu129.zip",
                            "vs-mlrt-windows-x64-tensorrt-plugin-cu129.zip",
                            "vs-mlrt-windows-x64-tensorrt-extra-cu129.zip",
                            "vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip",
                        ]
                    )
                    tags.extend([payload_tag] * 4)
        elif system == "Linux":
            assets = ["vs-mlrt-linux-x64-generic.zip"]
            tags = [GENERIC_TAG]
            if payload_tag in CUDA_TAGS:
                assets.extend(
                    [
                        f"vs-mlrt-linux-x64-tensorrt-{payload_tag}.zip",
                        f"vs-mlrt-linux-x64-cuda-{payload_tag}.zip",
                        f"vs-mlrt-linux-x64-cudnn-{payload_tag}.zip",
                    ]
                )
                tags.extend([payload_tag] * 3)
        else:
            raise RuntimeError(f"No tested release payload exists for {system} {platform.machine()}.")
        return self._download_urls(
            [f"https://github.com/{repo}/releases/download/{tag}/{asset}" for tag, asset in zip(tags, assets)]
        )

    def _download_urls(self, urls: list[str]) -> list[Path]:
        download_dir = Path(self.root) / "build" / "vsmlrt_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for url in dict.fromkeys(urls):
            asset = url.rsplit("/", 1)[-1].split("?", 1)[0]
            destination = download_dir / asset
            self._download_url(url, destination, asset)
            result.append(destination)
        return result

    def _download_url(self, url: str, destination: Path, asset: str) -> None:
        part = destination.with_name(f"{destination.name}.part")
        request = urllib.request.Request(url, headers={"User-Agent": "vs-mlrt-build-hook"})
        self._emit_download_progress(f"downloading {asset}")
        started = time.monotonic()
        downloaded = 0
        try:
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as output:
                total = int(response.headers.get("Content-Length") or 0)
                last_report = started
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if time.monotonic() - last_report >= self._download_progress_interval():
                        self._report_download_progress(asset, downloaded, total, started)
                        last_report = time.monotonic()
            part.replace(destination)
        except Exception:
            part.unlink(missing_ok=True)
            raise
        self._report_download_progress(asset, downloaded, downloaded, started, done=True)

    def _safe_extract_payload(self, archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename.replace("\\", "/"))
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Unsafe payload member: {member.filename}")
                if not member_path.parts or member_path.parts[0] != "vsmlrt":
                    raise RuntimeError(f"Payload must have a top-level vsmlrt/ directory: {member.filename}")
            archive.extractall(destination.parent)
            source = destination.parent / "vsmlrt"
            if not source.is_dir():
                raise RuntimeError(f"Payload did not extract vsmlrt/: {archive_path}")
            if source.resolve() != destination.resolve():
                shutil.copytree(source, destination, dirs_exist_ok=True)
                shutil.rmtree(source)

    def _report_download_progress(self, asset: str, downloaded: int, total: int, started: float, done: bool = False) -> None:
        elapsed = max(time.monotonic() - started, 0.001)
        if done:
            self._emit_download_progress(f"downloaded {asset}: {self._format_bytes(downloaded)} in {elapsed:.1f}s")
            return
        if total:
            self._emit_download_progress(
                f"downloading {asset}: {downloaded / total * 100:5.1f}% "
                f"{self._format_bytes(downloaded)}/{self._format_bytes(total)}"
            )

    def _emit_download_progress(self, message: str) -> None:
        if self._truthy(os.environ.get("VSMLRT_DOWNLOAD_PROGRESS", "1")):
            print(f"vs-mlrt: {message}", file=sys.stderr, flush=True)

    def _download_progress_interval(self) -> float:
        try:
            return max(float(os.environ.get("VSMLRT_DOWNLOAD_PROGRESS_INTERVAL", DOWNLOAD_PROGRESS_INTERVAL)), 0.5)
        except ValueError:
            return DOWNLOAD_PROGRESS_INTERVAL

    @staticmethod
    def _format_bytes(value: int) -> str:
        size = float(value)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if size < 1024 or unit == "GiB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GiB"

    def _detect_github_repo(self) -> str:
        try:
            remote = subprocess.check_output(
                ["git", "remote", "get-url", "origin"], cwd=self.root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            remote = ""
        match = re.search(r"github\\.com[:/](?P<repo>[^/]+/[^/.]+)(?:\\.git)?$", remote)
        return match.group("repo") if match else "RyougiKukoc/vs-mlrt-api4"
