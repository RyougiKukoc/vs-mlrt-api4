"""Read a release payload that may be split into numbered archive volumes.

A tag publishes one self-contained archive per system. Archives above GitHub's
per-asset limit are stored as byte volumes named `<archive>.zip.001`, `.002`,
and so on, exactly like upstream's `.7z.001` split, so the volumes have to be
read as one continuous stream.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path


class VolumeStream(io.RawIOBase):
    """Present numbered archive volumes as one seekable byte stream."""

    def __init__(self, volumes: list[Path]):
        self.volumes = volumes
        self.sizes = [volume.stat().st_size for volume in volumes]
        self.total = sum(self.sizes)
        self.offsets: list[int] = []
        running = 0
        for size in self.sizes:
            self.offsets.append(running)
            running += size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        else:
            self.position = self.total + offset
        return self.position

    def readinto(self, buffer) -> int:
        wanted = len(buffer)
        written = 0
        while written < wanted and self.position < self.total:
            index = max(i for i, offset in enumerate(self.offsets) if offset <= self.position)
            local = self.position - self.offsets[index]
            with self.volumes[index].open("rb") as stream:
                stream.seek(local)
                chunk = stream.read(wanted - written)
            if not chunk:
                break
            buffer[written:written + len(chunk)] = chunk
            written += len(chunk)
            self.position += len(chunk)
        return written


def volume_names(archive: Path) -> list[Path]:
    """Return the volumes of an archive, whether split or whole."""
    volumes = sorted(archive.parent.glob(f"{archive.name}.[0-9][0-9][0-9]"))
    return volumes or [archive]


def resolve_volumes(path: Path) -> list[Path]:
    """Accept either a whole archive or any one of its volumes."""
    if re.fullmatch(r".*\.[0-9]{3}", path.name):
        path = path.with_name(path.name[:-4])
    return volume_names(path)


def open_payload(sources: list[Path]) -> zipfile.ZipFile:
    """Open one payload archive from its single file or its volumes."""
    if len(sources) == 1:
        return zipfile.ZipFile(sources[0])
    return zipfile.ZipFile(io.BufferedReader(VolumeStream(sources)))
