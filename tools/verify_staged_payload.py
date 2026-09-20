"""Install staged release zips and prove their installed bytes before publishing.

Unchanged generic/model dependencies are fetched from their existing releases.
The native zips built by this job always come from --asset-dir. Publication can
then verify the released asset digests against this exact tested input set.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
import urllib.request
import zipfile


_spec = importlib.util.spec_from_file_location(
    "payload_archive", Path(__file__).resolve().parents[1] / "packaging" / "payload_archive.py"
)
assert _spec and _spec.loader
_archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_archive)
open_payload = _archive.open_payload
resolve_volumes = _archive.resolve_volumes


# Release payloads are deflated, but tiny metadata members may legally be
# stored when deflate cannot shrink them.
STORED_MEMBER_LIMIT = 4096


def digest_stream(stream) -> str:
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return digest_stream(stream)


def request(url: str):
    headers = {"User-Agent": "vs-mlrt-payload-verification"}
    token = os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def release(repo: str, tag: str) -> dict:
    with request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}") as stream:
        return json.load(stream)


def download_dependency(repo: str, tag: str, name: str, folder: Path) -> tuple[Path, dict]:
    asset = next(item for item in release(repo, tag)["assets"] if item["name"] == name)
    path = folder / name
    expected = asset.get("digest", "")
    if not path.exists() or not expected or expected != "sha256:" + digest(path):
        temporary = path.with_suffix(".part")
        with request(asset["browser_download_url"]) as stream, temporary.open("wb") as out:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                out.write(chunk)
        temporary.replace(path)
    actual = digest(path)
    if expected and expected != "sha256:" + actual:
        raise RuntimeError(f"Release dependency digest mismatch: {name}")
    return path, {"name": name, "sha256": actual, "asset_id": asset["id"], "updated_at": asset["updated_at"], "tag": tag}


def staged_names(variant: str, asset_dir: Path) -> list[str]:
    """Names of this variant's published assets, split volumes included."""
    if variant == "generic":
        return ["vs-mlrt-windows-x64-generic.zip"]
    return [path.name for path in sorted(asset_dir.glob(f"vs-mlrt-windows-x64-{variant}.zip*"))]


def staged_archives(variant: str, asset_dir: Path) -> list[list[Path]]:
    names = staged_names(variant, asset_dir)
    if not names:
        raise RuntimeError(f"No staged {variant} payload archive in {asset_dir}")
    base = (asset_dir / names[0]).resolve()
    return [resolve_volumes(base)]


def verify_installed(archives: list[list[Path]], site: Path, variant: str) -> int:
    count = 0
    builder_resource_found = False
    for sources in archives:
        with open_payload(sources) as archive:
            names = [info.filename.replace("\\", "/") for info in archive.infolist() if not info.is_dir()]
            builder_resource_found = builder_resource_found or any("builder_resource" in name.lower() for name in names)
            for member in archive.infolist():
                if member.is_dir():
                    continue
                # Store mode made every asset as large as its payload; level 1
                # removes 18% to 63% per family. 7-Zip may still store a member
                # that deflate cannot shrink, so only real payload files are
                # required to be deflated.
                if member.compress_type != zipfile.ZIP_DEFLATED and member.file_size > STORED_MEMBER_LIMIT:
                    raise RuntimeError(f"Release asset is stored uncompressed: {sources[0].name}: {member.filename}")
                rel = Path(member.filename)
                if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "vsmlrt":
                    raise RuntimeError(f"Unexpected package path: {member.filename}")
                # The existing build hook deliberately merges manifests and flattens OV.
                if rel.name == "manifest.vs":
                    continue
                if len(rel.parts) > 2 and rel.parts[1] == "vsov":
                    rel = Path("vsmlrt") / rel.name
                installed = site / "vapoursynth/plugins" / rel
                with archive.open(member) as stream:
                    expected = digest_stream(stream)
                if not installed.is_file() or digest(installed) != expected:
                    raise RuntimeError(f"Installed payload differs from staged {sources[0].name}: {rel}")
                count += 1
    if variant != "generic" and not builder_resource_found:
        raise RuntimeError("Builder payload is missing TensorRT builder resources")
    return count


def verify_published(repo: str, variant: str, evidence: Path) -> None:
    recorded = json.loads(evidence.read_text())
    if not recorded.get("ok") or recorded["variant"] != variant:
        raise RuntimeError("Publication requires successful matching staged installation evidence")
    assets = {item["name"]: item for item in release(repo, variant)["assets"]}
    for item in recorded["staged_assets"]:
        remote = assets.get(item["name"])
        if remote is None or remote.get("digest") != "sha256:" + item["sha256"]:
            raise RuntimeError(f"Published release does not match tested asset: {item['name']}")
    print("Published asset SHA-256 values match the staged installation evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["generic", "cu121", "cu129"], required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--asset-dir", type=Path, default=Path("."))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--verify-published-only", action="store_true")
    args = parser.parse_args()
    if args.verify_published_only:
        verify_published(args.repo, args.variant, args.evidence)
        return
    project = Path(__file__).resolve().parents[1]
    asset_dir = args.asset_dir.resolve()
    archives = staged_archives(args.variant, asset_dir)
    volumes = [path for sources in archives for path in sources]
    evidence = {"variant": args.variant, "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
                "staged_assets": [{"name": p.name, "sha256": digest(p), "size": p.stat().st_size} for p in volumes],
                "dependency_assets": [], "ok": False}
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    dependencies = project / "build/verification-dependencies"
    dependencies.mkdir(parents=True, exist_ok=True)
    # Every payload archive is self-contained, so the only shared download is the
    # model payload. The generic asset the payload embedded is recorded for
    # traceability but deliberately not installed: a payload that forgot to
    # embed it has to fail here.
    if args.variant != "generic":
        _, info = download_dependency(args.repo, "generic", "vs-mlrt-windows-x64-generic.zip", dependencies)
        evidence["dependency_assets"].append(info)
    paths = list(volumes)
    models, info = download_dependency(args.repo, "models", "models.zip", dependencies)
    paths.append(models)
    evidence["dependency_assets"].append(info)
    env = os.environ.copy()
    env["VSMLRT_PREBUILT_PATHS"] = os.pathsep.join(str(p) for p in paths)
    env["VSMLRT_PAYLOAD_TAG"] = args.variant
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--no-cache-dir", str(project)], env=env, check=True)
    evidence["installed_file_count"] = verify_installed(archives, Path(sysconfig.get_path("purelib")), args.variant)
    subprocess.run([sys.executable, str(project / "tools/smoke_vcs_extras_install.py"), "--variant", args.variant], check=True)
    # Ensure inputs were not replaced between installation and publication.
    for path, info in zip(volumes, evidence["staged_assets"]):
        if digest(path) != info["sha256"]:
            raise RuntimeError(f"Staged payload changed during verification: {path}")
    evidence["ok"] = True
    evidence["verified_at_unix"] = time.time()
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Installed {evidence['installed_file_count']} files from exact staged {args.variant} payloads")


if __name__ == "__main__":
    main()
