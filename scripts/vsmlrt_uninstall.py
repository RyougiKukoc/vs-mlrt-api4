from __future__ import annotations

import subprocess
import sys


PACKAGES = (
    "vs-mlrt",
    "vs-mlrt-models",
    "vs-mlrt-payload-generic",
    "vs-mlrt-payload-cu121",
    "vs-mlrt-payload-cu129",
)


def main() -> int:
    return subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", *PACKAGES])


if __name__ == "__main__":
    raise SystemExit(main())
