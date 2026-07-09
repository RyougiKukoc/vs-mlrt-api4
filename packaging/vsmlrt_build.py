from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


CUDA_TAGS = {"cu121", "cu129"}
DEFAULT_CUDA_TAG_FILE = Path("packaging") / "cuda-tag.txt"


class CustomBuildHook(BuildHookInterface):
    """Attach the tested native plugin payload to VCS-built wheels."""

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return

        build_data["tag"] = "py3-none-win_amd64"

        if os.environ.get("VSMLRT_SKIP_PREBUILT") == "1":
            return

        if platform.system() != "Windows":
            raise RuntimeError("vs-mlrt release-backed VCS installs currently target Windows.")

        payload_zip_paths = self._resolve_payload_paths()

        extract_dir = Path(self.root) / "build" / "vsmlrt_prebuilt"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)

        for payload_zip_path in payload_zip_paths:
            with zipfile.ZipFile(payload_zip_path) as archive:
                archive.extractall(extract_dir)

        vapoursynth_dir = extract_dir / "vapoursynth"
        plugin_dir = extract_dir / "vsmlrt"

        force_include = build_data.setdefault("force_include", {})
        if vapoursynth_dir.is_dir():
            force_include[str(vapoursynth_dir)] = "vapoursynth"
        elif plugin_dir.is_dir():
            force_include[str(plugin_dir)] = "vapoursynth/plugins/vsmlrt"
        else:
            raise RuntimeError("Prebuilt payload is missing vsmlrt/ or vapoursynth/.")

    def _resolve_payload_paths(self) -> list[Path]:
        explicit_paths = os.environ.get("VSMLRT_PREBUILT_PATHS") or os.environ.get("VSMLRT_PREBUILT_PATH")
        if explicit_paths:
            paths = [item.strip() for item in explicit_paths.split(os.pathsep) if item.strip()]
            if not paths:
                raise RuntimeError("VSMLRT_PREBUILT_PATHS did not contain any paths.")
            return [Path(item).expanduser().resolve() for item in paths]

        explicit_urls = os.environ.get("VSMLRT_PREBUILT_URLS") or os.environ.get("VSMLRT_PREBUILT_URL")
        if explicit_urls:
            urls = [item.strip() for item in explicit_urls.split(os.pathsep) if item.strip()]
            if not urls:
                raise RuntimeError("VSMLRT_PREBUILT_URLS did not contain any URLs.")
            return self._download_urls(urls)

        return self._download_release_payloads()

    def _download_release_payloads(self) -> list[Path]:
        cuda_tag = self._detect_cuda_tag()
        repo = os.environ.get("VSMLRT_RELEASE_REPO") or self._detect_github_repo()
        assets = [
            f"vs-mlrt-windows-x64-tensorrt-{cuda_tag}.zip",
            f"vs-mlrt-windows-x64-cuda-{cuda_tag}.zip",
            f"vs-mlrt-windows-x64-cudnn-{cuda_tag}.zip",
            "vs-mlrt-windows-x64-models.zip",
        ]
        if cuda_tag == "cu129":
            assets.extend(
                [
                    "vs-mlrt-windows-x64-tensorrt-core-cu129.zip",
                    "vs-mlrt-windows-x64-tensorrt-plugin-cu129.zip",
                    "vs-mlrt-windows-x64-tensorrt-extra-cu129.zip",
                    "vs-mlrt-windows-x64-tensorrt-rtx-cu129.zip",
                ]
            )

        urls = [f"https://github.com/{repo}/releases/download/{cuda_tag}/{asset}" for asset in assets]
        return self._download_urls(urls)

    def _download_urls(self, urls: list[str]) -> list[Path]:
        download_dir = Path(self.root) / "build" / "vsmlrt_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        destinations = []
        for url in urls:
            asset = url.rsplit("/", 1)[-1].split("?", 1)[0]
            destination = download_dir / asset
            urllib.request.urlretrieve(url, destination)
            destinations.append(destination)
        return destinations

    def _detect_cuda_tag(self) -> str:
        explicit = os.environ.get("VSMLRT_CUDA_TAG")
        if explicit:
            if explicit not in CUDA_TAGS:
                raise RuntimeError(f"Unsupported VSMLRT_CUDA_TAG={explicit!r}; expected cu121 or cu129.")
            return explicit

        ref_name = os.environ.get("GITHUB_REF_NAME")
        if ref_name in CUDA_TAGS:
            return ref_name

        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=self.root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            tag = ""

        if tag in CUDA_TAGS:
            return tag

        configured = self._read_default_cuda_tag()
        if configured:
            return configured

        return "cu129"

    def _read_default_cuda_tag(self) -> str:
        tag_file = Path(self.root) / DEFAULT_CUDA_TAG_FILE
        if not tag_file.is_file():
            return ""

        tag = tag_file.read_text(encoding="utf-8").strip()
        if tag not in CUDA_TAGS:
            raise RuntimeError(f"Unsupported default CUDA tag in {DEFAULT_CUDA_TAG_FILE}: {tag!r}.")
        return tag

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

        return "AmusementClub/vs-mlrt"
