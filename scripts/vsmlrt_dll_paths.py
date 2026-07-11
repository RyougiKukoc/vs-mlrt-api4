from __future__ import annotations

import os
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_MANIFEST_HEADER = "[VapourSynth Manifest V1]"
_PLUGIN_BASENAMES = ("vsncnn", "vsov", "vsort", "vstrt", "vstrt_rtx", "vsmigx")


def _sync_vsmlrt_manifest(plugin_root: Path) -> None:
    if not plugin_root.is_dir():
        return

    plugins = [name for name in _PLUGIN_BASENAMES if (plugin_root / f"{name}.dll").is_file()]
    contents = "\n".join((_MANIFEST_HEADER, *plugins, ""))
    manifest = plugin_root / "manifest.vs"
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    try:
        if manifest.is_file() and manifest.read_text(encoding="ascii") == contents:
            return
        temporary.write_text(contents, encoding="ascii", newline="\n")
        os.replace(temporary, manifest)
    except (OSError, UnicodeError):
        # A read-only site-packages directory keeps the wheel's empty manifest.
        # Explicit plugin loading still remains available in that environment.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _add_dll_directory(path: Path) -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None or not path.is_dir():
        return

    path_text = str(path)
    for handle_path, _handle in _DLL_DIRECTORY_HANDLES:
        if handle_path == path_text:
            return
    _DLL_DIRECTORY_HANDLES.append((path_text, add_dll_directory(path_text)))


def _configure_vsmlrt_dll_paths() -> None:
    if os.name != "nt":
        return

    site_dir = Path(__file__).resolve().parent
    plugin_roots = [
        site_dir / "vapoursynth" / "plugins" / "vsmlrt",
    ]
    support_names = ("vsov", "vsort", "vsmlrt-cuda")

    for plugin_root in plugin_roots:
        _sync_vsmlrt_manifest(plugin_root)
        _add_dll_directory(plugin_root)
        for support_name in support_names:
            _add_dll_directory(plugin_root / support_name)


_configure_vsmlrt_dll_paths()
