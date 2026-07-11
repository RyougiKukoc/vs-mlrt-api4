from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from payload_build import ReleasePayloadBuildHook


class CustomBuildHook(ReleasePayloadBuildHook):
    payload_tag = "cu129"
    install_name = "vsmlrt-cu129"


def get_build_hook():
    return CustomBuildHook
