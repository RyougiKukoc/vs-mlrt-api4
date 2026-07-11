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
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


CUDA_TAGS = {"cu121", "cu129"}
GENERIC_TAG = "generic"
MODELS_TAG = "models"
MODELS_ASSET = "models.zip"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_PROGRESS_INTERVAL = 5.0


class ReleasePayloadBuildHook(BuildHookInterface):
    payload_tag = ""
    install_name = ""
    models_payload = False

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return

        if os.environ.get("VSMLRT_SKIP_PREBUILT") == "1":
            return

        if platform.system() != "Windows":
            return

        build_data["tag"] = "py3-none-win_amd64"

        payload_zip_paths = self._resolve_payload_paths()

        extract_dir = Path(self.root) / "build" / "vsmlrt_prebuilt"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)

        for payload_zip_path in payload_zip_paths:
            with zipfile.ZipFile(payload_zip_path) as archive:
                archive.extractall(extract_dir)

        force_include = build_data.setdefault("force_include", {})
        if self.models_payload:
            models_dir = self._find_models_dir(extract_dir)
            force_include[str(models_dir)] = "vapoursynth/plugins/vsmlrt/models"
            return

        plugin_dir = extract_dir / "vsmlrt"
        if not plugin_dir.is_dir():
            raise RuntimeError("Prebuilt payload is missing vsmlrt/.")
        if not self.install_name:
            raise RuntimeError("Payload build hook is missing install_name.")
        (plugin_dir / "manifest.vs").unlink(missing_ok=True)
        self._prepare_plugin_dir(plugin_dir)
        force_include[str(plugin_dir)] = f"vapoursynth/plugins/{self.install_name}"

    def _prepare_plugin_dir(self, plugin_dir: Path) -> None:
        if self.payload_tag != GENERIC_TAG:
            return

        support_dir = plugin_dir / "vsov"
        if not support_dir.is_dir():
            return

        for source in support_dir.rglob("*"):
            if not source.is_file():
                continue
            destination = plugin_dir / source.name
            if destination.exists():
                if not filecmp.cmp(source, destination, shallow=False):
                    raise RuntimeError(
                        f"Conflicting OpenVINO runtime files: {source} and {destination}."
                    )
                source.unlink()
            else:
                shutil.move(str(source), destination)
        shutil.rmtree(support_dir)

    def _find_models_dir(self, extract_dir: Path) -> Path:
        candidates = [
            extract_dir / "vsmlrt" / "models",
            extract_dir / "models",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise RuntimeError("Prebuilt model payload is missing models/.")

    def _resolve_payload_paths(self) -> list[Path]:
        explicit_paths = self._read_env("PREBUILT_PATHS") or self._read_env("PREBUILT_PATH")
        if explicit_paths:
            paths = [item.strip() for item in explicit_paths.split(os.pathsep) if item.strip()]
            if not paths:
                raise RuntimeError("Prebuilt path override did not contain any paths.")
            return [Path(item).expanduser().resolve() for item in paths]

        explicit_urls = self._read_env("PREBUILT_URLS") or self._read_env("PREBUILT_URL")
        if explicit_urls:
            urls = [item.strip() for item in explicit_urls.split(os.pathsep) if item.strip()]
            if not urls:
                raise RuntimeError("Prebuilt URL override did not contain any URLs.")
            return self._download_urls(urls)

        return self._download_release_payloads()

    def _read_env(self, key: str) -> str:
        specific = os.environ.get(f"VSMLRT_{self._env_prefix()}_{key}")
        if specific:
            return specific
        return os.environ.get(f"VSMLRT_{key}", "")

    def _env_prefix(self) -> str:
        tag = self.payload_tag or ("models" if self.models_payload else "payload")
        return re.sub(r"[^A-Za-z0-9]+", "_", tag).upper()

    def _download_release_payloads(self) -> list[Path]:
        repo = os.environ.get("VSMLRT_RELEASE_REPO") or self._detect_github_repo()
        if self.models_payload:
            models_repo = os.environ.get("VSMLRT_MODELS_RELEASE_REPO") or repo
            models_tag = os.environ.get("VSMLRT_MODELS_TAG") or MODELS_TAG
            return self._download_urls(
                [f"https://github.com/{models_repo}/releases/download/{models_tag}/{MODELS_ASSET}"]
            )

        tag = self.payload_tag
        if tag == GENERIC_TAG:
            urls = [
                f"https://github.com/{repo}/releases/download/{GENERIC_TAG}/vs-mlrt-windows-x64-generic.zip"
            ]
        elif tag in CUDA_TAGS:
            urls = [
                f"https://github.com/{repo}/releases/download/{tag}/vs-mlrt-windows-x64-tensorrt-{tag}.zip",
                f"https://github.com/{repo}/releases/download/{tag}/vs-mlrt-windows-x64-cuda-{tag}.zip",
                f"https://github.com/{repo}/releases/download/{tag}/vs-mlrt-windows-x64-cudnn-{tag}.zip",
            ]
            if tag == "cu129":
                urls.extend(
                    [
                        "https://github.com/"
                        f"{repo}/releases/download/cu129/vs-mlrt-windows-x64-tensorrt-core-cu129.zip",
                        "https://github.com/"
                        f"{repo}/releases/download/cu129/vs-mlrt-windows-x64-tensorrt-plugin-cu129.zip",
                        "https://github.com/"
                        f"{repo}/releases/download/cu129/vs-mlrt-windows-x64-tensorrt-extra-cu129.zip",
                        "https://github.com/"
                        f"{repo}/releases/download/cu129/vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip",
                    ]
                )
        else:
            raise RuntimeError(f"Unsupported payload tag: {tag!r}.")

        return self._download_urls(urls)

    def _download_urls(self, urls: list[str]) -> list[Path]:
        download_dir = Path(self.root) / "build" / "vsmlrt_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        destinations = []
        seen_urls = set()
        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            asset = url.rsplit("/", 1)[-1].split("?", 1)[0]
            destination = download_dir / asset
            self._download_url(url, destination, asset)
            destinations.append(destination)
        return destinations

    def _download_url(self, url: str, destination: Path, asset: str) -> None:
        part_destination = destination.with_name(f"{destination.name}.part")
        request = urllib.request.Request(url, headers={"User-Agent": "vs-mlrt-payload-build-hook"})
        self._emit_download_progress(f"downloading {asset}")
        started_at = time.monotonic()
        last_report_at = started_at
        progress_interval = self._download_progress_interval()
        downloaded = 0

        try:
            with urllib.request.urlopen(request) as response:
                total = int(response.headers.get("Content-Length") or 0)
                with part_destination.open("wb") as output:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report_at >= progress_interval:
                            self._report_download_progress(asset, downloaded, total, started_at)
                            last_report_at = now

            part_destination.replace(destination)
        except Exception:
            part_destination.unlink(missing_ok=True)
            raise

        self._report_download_progress(asset, downloaded, downloaded, started_at, done=True)

    def _report_download_progress(
        self,
        asset: str,
        downloaded: int,
        total: int,
        started_at: float,
        done: bool = False,
    ) -> None:
        elapsed = max(time.monotonic() - started_at, 0.001)
        speed = downloaded / elapsed
        if done:
            self._emit_download_progress(
                f"downloaded {asset}: {self._format_bytes(downloaded)} in {elapsed:.1f}s"
            )
            return

        if total > 0:
            percent = min(downloaded / total, 1.0)
            bar_width = 20
            filled = int(percent * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            message = (
                f"downloading {asset} [{bar}] {percent * 100:5.1f}% "
                f"{self._format_bytes(downloaded)}/{self._format_bytes(total)} "
                f"at {self._format_bytes(speed)}/s"
            )
        else:
            message = (
                f"downloading {asset}: {self._format_bytes(downloaded)} "
                f"at {self._format_bytes(speed)}/s"
            )
        self._emit_download_progress(message)

    def _emit_download_progress(self, message: str) -> None:
        if os.environ.get("VSMLRT_DOWNLOAD_PROGRESS", "1").lower() in {"0", "false", "no"}:
            return

        line = f"vs-mlrt: {message}"
        if os.name == "nt" and os.environ.get("VSMLRT_PROGRESS_CONSOLE", "1").lower() not in {"0", "false", "no"}:
            try:
                with open("CONOUT$", "w", encoding="utf-8", errors="replace") as console:
                    print(line, file=console, flush=True)
                return
            except OSError:
                pass

        print(line, file=sys.stderr, flush=True)

    def _download_progress_interval(self) -> float:
        raw = os.environ.get("VSMLRT_DOWNLOAD_PROGRESS_INTERVAL")
        if raw:
            try:
                return max(float(raw), 0.5)
            except ValueError:
                pass
        return DOWNLOAD_PROGRESS_INTERVAL

    def _format_bytes(self, value: float) -> str:
        units = ("B", "KiB", "MiB", "GiB")
        size = float(value)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{size:.0f} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GiB"

    def _detect_github_repo(self) -> str:
        try:
            remote = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                cwd=self.root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            remote = ""

        match = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/.]+)(?:\.git)?$", remote)
        if match:
            return match.group("repo")

        return "RyougiKukoc/vs-mlrt-api4"
