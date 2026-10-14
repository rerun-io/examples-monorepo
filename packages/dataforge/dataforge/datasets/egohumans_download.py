"""EgoHumans download: the release's Google Drive folder (one tar.gz per sequence) and the per-sequence configs on GitHub.

The Drive folder (``assets/DOWNLOAD.md`` in rawalkhirodkar/egohumans) is listed once into ``egohumans_drive.json`` (file ids
and sizes; Drive publishes no checksums), so a download needs no Google API and no folder scraping. Drive refuses popular
files for up to a day ("Quota exceeded: too many users have viewed or downloaded this file recently") and answers with an HTML
page instead of the bytes: ``fetch_drive_file`` recognises that page, keeps whatever part it already has, and raises
``DriveQuotaError`` so a caller retries later. It never slices a file into small ranges to get around the quota.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests
from serde import serde

from dataforge import transports, writing
from dataforge.egohumans_layout import CaptureConfig
from dataforge.records import decode, read_json

MANIFEST: Path = Path(__file__).with_name("egohumans_drive.json")
DRIVE_URL: str = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
CONFIG_REPO_REVISION: str = "d7875dd4b3e3cc4e306af386d5c4febf1290a3e9"
"""rawalkhirodkar/egohumans main as of 2026-10-13; ``egohumans/configs/<activity>/<sequence>.yaml`` per sequence."""
CONFIG_URL: str = "https://raw.githubusercontent.com/rawalkhirodkar/egohumans/{revision}/egohumans/{path}"
CHUNK_BYTES: int = 8 << 20
HEADERS: dict[str, str] = {**transports.IDENTITY, "User-Agent": "dataforge (https://github.com/rerun-io/examples-monorepo)"}
"""Uncompressed transfer (so ``Range`` and sizes count file bytes), and a name for the client."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DriveFile:
    """One file of the release folder."""

    path: str
    """Folder-relative path, also the raw-root-relative destination (``01_tagging/001_tagging.tar.gz``)."""
    file_id: str
    """Drive file id."""
    size_bytes: int
    """Exact size Drive reports for it."""

    @property
    def key(self) -> str:
        """Sequence key: ``01_tagging/001_tagging``."""
        return self.path.removesuffix(".tar.gz")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DriveManifest:
    """The release folder as listed on ``listed``."""

    folder_id: str
    """The folder DOWNLOAD.md links."""
    listed: str
    """Date of the listing (ISO)."""
    files: list[DriveFile]
    """Every file, sequence archives and the benchmark annotations."""

    @property
    def sequences(self) -> list[DriveFile]:
        """The per-sequence archives; the COCO-format benchmark files are derived from them and not converted."""
        return [file for file in self.files if file.path.endswith(".tar.gz")]


def read_manifest() -> DriveManifest:
    return read_json(MANIFEST, DriveManifest)


class DriveQuotaError(RuntimeError):
    """Drive answered its quota page instead of the file; retry later (the quota resets within a day)."""


def fetch_drive_file(file: DriveFile, dest: Path, *, timeout_s: float = 120.0) -> bool:
    """Fetch one Drive file to ``dest``, resuming a ``<dest>.part``; return whether bytes were fetched.

    ``dest`` only ever holds a complete file (size checked, mode 0644 set before the rename). A server that ignores the
    resume range, or answers another range than the one asked, restarts the part instead of appending to it (and a body
    that does not start the gzip stream is refused); a part already complete (a run killed before its rename) is renamed
    without a request, and one longer than the file is discarded.

    Raises:
        DriveQuotaError: Drive served its HTML quota page; the part is kept for the next attempt.
        ValueError: The transfer ended short or long of the listed size, or a fresh transfer is not gzip.
    """
    if dest.is_file() and dest.stat().st_size == file.size_bytes:
        return False
    part: Path = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have: int = part.stat().st_size if part.is_file() else 0
    if have > file.size_bytes:
        part.unlink()
        have = 0
    if have < file.size_bytes:
        headers: dict[str, str] = {**HEADERS, "Range": f"bytes={have}-"} if have else HEADERS
        with requests.get(DRIVE_URL.format(file_id=file.file_id), headers=headers, stream=True, timeout=timeout_s) as response:
            response.raise_for_status()
            # The quota page is the only HTML Drive answers with; the archive's own type is not documented, so it is not required.
            if response.headers.get("Content-Type", "").startswith("text/html"):
                raise DriveQuotaError(f"{file.path}: Drive served its quota page instead of the file (too many downloads); retry later")
            # A resume only when the range starts where the part ends; any other answer must be the whole file from byte 0.
            resuming: bool = have > 0 and response.status_code == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {have}-")
            blocks = response.iter_content(chunk_size=CHUNK_BYTES)
            first: bytes = next(blocks, b"")
            if not resuming and not first.startswith(b"\x1f\x8b"):
                raise ValueError(f"{file.path}: Drive served {response.headers.get('Content-Type')} that is not gzip")
            with part.open("ab" if resuming else "wb") as sink:
                sink.write(first)
                for block in blocks:
                    sink.write(block)
                os.fchmod(sink.fileno(), 0o644)
    written: int = part.stat().st_size
    if written != file.size_bytes:
        raise ValueError(f"{file.path}: {written} of {file.size_bytes} bytes after the transfer; rerun to resume")
    part.rename(dest)
    return True


def config_path(key: str) -> str:
    """Raw-root-relative path of a sequence's upstream config: ``configs/<activity>/<sequence>.yaml``."""
    sequence: str = key.split("/")[-1]
    return f"configs/{sequence.split('_', 1)[1]}/{sequence}.yaml"


def fetch_config(key: str, root: Path) -> bool:
    """Fetch a sequence's upstream config at the pinned revision unless present; return whether it was fetched."""
    dest: Path = root / config_path(key)
    if dest.is_file():
        return False
    url: str = CONFIG_URL.format(revision=CONFIG_REPO_REVISION, path=config_path(key))
    text: str = transports.http_text(url)
    decode(CaptureConfig, text, source=url, fmt="yaml")  # refuse a page that is not a config before it lands
    with writing.atomic_write(dest) as temp:
        temp.write_text(text)
    return True
