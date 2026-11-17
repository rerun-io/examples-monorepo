"""TartanAir V2 by HTTP range extraction: one trajectory's members out of an environment-wide zip.

Hugging Face ships each environment and difficulty as one zip per modality (``<Env>/Data_<easy|hard>/image_<cam>.zip``)
that holds every trajectory. A trajectory is a small part of each zip, so ``extract_members`` reads the zip's central
directory through HTTP ``Range`` requests and fetches only the members under that trajectory's directory. Python's
``zipfile`` does the parsing (ZIP64, deflate, the CRC check of each member) over a seekable file whose reads are range
requests. A member already on disk at its uncompressed size is not fetched again, so an interrupted download restarts
where it stopped.
"""

from __future__ import annotations

import io
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import httpx

HF_REVISION: str = "0d2d145e973832742a2aaa04b7d2ebffc8d82817"
"""The ``theairlabcmu/tartanair2`` revision COLMAP's TartanAir V2 benchmark pins."""
HF_BASE_URL: str = f"https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/{HF_REVISION}/"
"""Prefix of every archive URL; no login needed."""
SMALL_READ: int = 65_536
"""Reads shorter than this (zip headers) fetch this many bytes at once, so one member costs two requests, not four."""
ATTEMPTS: int = 5
"""Tries per range request before the download fails."""

RangeFetch = Callable[[int, int], bytes]
"""``fetch(start, stop)`` returns bytes ``[start, stop)`` of the remote file."""


class RangeFile(io.RawIOBase):
    """A read-only seekable file whose reads are ``fetch`` calls, with one small read-ahead block."""

    def __init__(self, fetch: RangeFetch, size: int) -> None:
        super().__init__()
        self.fetch: RangeFetch = fetch
        self.size: int = size
        self.position: int = 0
        self.block_start: int = 0
        self.block: bytes = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base: int = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer: memoryview | bytearray) -> int:  # pyrefly: ignore[bad-override]
        stop: int = min(self.position + len(buffer), self.size)
        if stop <= self.position:
            return 0
        offset: int = self.position - self.block_start
        if not (offset >= 0 and stop <= self.block_start + len(self.block)):
            if stop - self.position < SMALL_READ:
                self.block_start, self.block = self.position, self.fetch(self.position, min(self.position + SMALL_READ, self.size))
            else:
                self.block_start, self.block = self.position, self.fetch(self.position, stop)
            offset = 0
        count: int = stop - self.position
        buffer[:count] = self.block[offset : offset + count]
        self.position = stop
        return count


def extract_members(fetch: RangeFetch, size: int, prefix: str, destination: Path, *, workers: int = 1) -> tuple[int, int]:
    """Write every member under ``prefix`` to ``destination/<member name>``, skipping members already complete.

    Args:
        fetch: Range reader of the remote zip.
        size: The zip's size in bytes.
        prefix: Member-name prefix to extract, e.g. ``"ArchVizTinyHouseDay/Data_easy/P006/"``.
        destination: Root the member names are written under.
        workers: Members fetched at once; each worker parses the central directory once.

    Returns:
        (members under ``prefix``, bytes written this call).

    Raises:
        ValueError: A member name leaves ``destination``, or a written file's size is not the member's.
        zipfile.BadZipFile: A member's CRC does not match.
    """
    local: threading.local = threading.local()

    def archive() -> zipfile.ZipFile:
        if not hasattr(local, "zip"):
            local.zip = zipfile.ZipFile(RangeFile(fetch, size))
        return local.zip

    members: list[zipfile.ZipInfo] = [info for info in archive().infolist() if info.filename.startswith(prefix) and not info.is_dir()]
    for info in members:
        if ".." in PurePosixPath(info.filename).parts or PurePosixPath(info.filename).is_absolute():
            raise ValueError(f"zip member {info.filename!r} would land outside {destination}")

    def one(info: zipfile.ZipInfo) -> int:
        target: Path = destination / info.filename
        if target.is_file() and target.stat().st_size == info.file_size:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        staged: Path = target.with_name(target.name + ".part")
        staged.write_bytes(archive().read(info))  # zipfile checks the CRC on a full read
        if staged.stat().st_size != info.file_size:
            raise ValueError(f"{staged} holds {staged.stat().st_size} bytes, the zip says {info.file_size}")
        staged.replace(target)
        return info.file_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        written: int = sum(pool.map(one, members))
    return len(members), written


def http_range(client: httpx.Client, url: str) -> tuple[RangeFetch, int]:
    """A retrying range reader for ``url`` and the file's size (from the first response's ``Content-Range``).

    The Hub answers ``resolve`` URLs with a redirect to a signed CDN URL and rate-limits the ``resolve`` calls (3,000 per
    5 minutes per IP), so the redirect is followed once and every range goes to the CDN URL; an expired signature
    (401/403) resolves it again. A 429 waits the ``Retry-After`` the Hub asks for.
    """
    resolved: list[str] = []

    def resolve() -> httpx.Response:
        while True:
            probe: httpx.Response = client.get(url, headers={"Range": "bytes=0-0"})
            if probe.status_code != 429:
                break
            time.sleep(float(probe.headers.get("retry-after", "60")) + 1.0)
        if probe.status_code != 206:
            raise OSError(f"{url}: HTTP {probe.status_code} to a range request")
        resolved[:] = [str(probe.url)]
        return probe

    def fetch(start: int, stop: int) -> bytes:
        problem: str = ""
        for attempt in range(ATTEMPTS):
            try:
                response: httpx.Response = client.get(resolved[0], headers={"Range": f"bytes={start}-{stop - 1}"})
                if response.status_code == 206 and len(response.content) == stop - start:
                    return response.content
                problem = f"HTTP {response.status_code}, {len(response.content)} of {stop - start} bytes"
                if response.status_code in (401, 403):
                    resolve()
            except httpx.HTTPError as error:
                problem = repr(error)
            if attempt + 1 < ATTEMPTS:
                time.sleep(2.0**attempt)
        raise OSError(f"{url} bytes {start}-{stop - 1}: {problem} after {ATTEMPTS} attempts")

    return fetch, int(resolve().headers["content-range"].rsplit("/", 1)[1])
