from __future__ import annotations

import os
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


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
        _add_dll_directory(plugin_root)
        for support_name in support_names:
            _add_dll_directory(plugin_root / support_name)


_configure_vsmlrt_dll_paths()
