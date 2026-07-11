from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from payload_build import ReleasePayloadBuildHook


class CustomBuildHook(ReleasePayloadBuildHook):
    payload_tag = "generic"
    install_name = "vsmlrt-generic"


def get_build_hook():
    return CustomBuildHook
