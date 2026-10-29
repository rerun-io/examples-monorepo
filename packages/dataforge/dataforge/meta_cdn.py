"""Meta's signed CDN URL files (projectaria.com: HOT3D, Aria Gen2 Pilot): parse them, fetch with SHA-1, extract zips.

A user downloads ``<Dataset>_download_urls*.json`` from the dataset's page; each entry is a CDN file with its
size, SHA-1 and a signed URL that expires (the ``oe`` query parameter). A signed URL is a secret: none is printed,
logged or written; the manifests a dataset's ``download`` leaves behind carry names, sizes and SHA-1s only.
"""

import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from serde import serde

from dataforge import transports, writing
from dataforge.records import decode


@dataclass(frozen=True, slots=True)
class CdnSource:
    """Which dataset's URL file this is, for the messages a user acts on."""

    name: str
    """Dataset name as messages print it, e.g. ``HOT3D``."""
    renew: str
    """How a user gets (or renews) a URL file."""


@serde
@dataclass(frozen=True, slots=True)
class CdnFile:
    """One entry of a URL file (third-party format)."""

    filename: str
    """CDN file name."""
    sha1sum: str
    """Hex SHA-1 of the complete file."""
    file_size_bytes: int
    """Size of the complete file."""
    download_url: str = field(repr=False)
    """Signed CDN URL: a secret, kept out of ``repr`` and every message."""


@serde
@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """The URL file's dataset, release and zip inventory."""

    dataset_name: str
    """E.g. ``Hot3DAria``, ``Hot3DAssets``, ``AriaGen2PilotDataset``."""
    data_groups: dict[str, list[str]]
    """Files each listed zip extracts to, by group (not every zip is listed)."""
    release: str | None = None
    """Upstream release; the HOT3D assets file has none."""


@serde
@dataclass(frozen=True, slots=True)
class UrlFile:
    """``<Dataset>_download_urls*.json`` as projectaria.com hands it out."""

    sequences: dict[str, dict[str, CdnFile]]
    """Files per sequence, by data type."""
    sequence_config: SequenceConfig
    """Release and zip inventory."""


def read_url_file(path: Path, source: CdnSource) -> UrlFile:
    """Parse a URL file; a decode error names only its type, since pyserde's message quotes the rejected value (a URL)."""
    return decode(UrlFile, path.read_text(), source=str(path), redact=f"not a {source.name} URL file; {source.renew}")


def url_file_at(path: Path | None, flag: str, source: CdnSource) -> Path:
    """The URL file a config field names, or the fix when it is unset or absent."""
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{source.name} needs {flag} <URL file> (got {path}); {source.renew}")
    return path


def expiry(url: str) -> datetime | None:
    """When a signed fbcdn URL stops working (its ``oe`` parameter, hex Unix seconds), if it says."""
    stamps: list[str] = parse_qs(urlparse(url).query).get("oe", [])
    return datetime.fromtimestamp(int(stamps[0], 16), UTC) if stamps else None


def fetch_verified(entry: CdnFile, dest: Path, source: CdnSource) -> None:
    """Fetch into ``<dest>.part`` (resuming it), check size and sha1, then rename onto ``dest``.

    An expired link fails before any request. A complete ``.part`` whose sha1 is wrong is removed so the next
    run refetches it; nothing else is ever deleted.
    """
    stamp: datetime | None = expiry(entry.download_url)
    if stamp is not None and stamp <= datetime.now(UTC):
        raise ValueError(f"the {source.name} URL file's link for {entry.filename} expired on {stamp:%Y-%m-%d %H:%M} UTC; {source.renew}")
    part: Path = dest.with_name(f"{dest.name}.part")
    transports.http_fetch(entry.download_url, dest=part, label=f"{source.name} {entry.filename}")
    size: int = part.stat().st_size
    if size != entry.file_size_bytes:
        raise ValueError(f"{part} holds {size} bytes, the URL file lists {entry.file_size_bytes}; rerun to resume")
    try:
        transports.publish_verified(part, dest, transports.FileIntegrity(entry.file_size_bytes, "sha1", entry.sha1sum))
    except transports.IntegrityError:
        part.unlink()
        raise ValueError(f"{entry.filename}: sha1 differs from the URL file; removed the download, rerun to refetch") from None


def extract(archive: Path, members: list[str], into: Path) -> None:
    """Write the named members the archive has atomically beneath ``into``, in order, then remove the archive.

    An absent member is not an error here: whether the sequence is complete is discovery's verdict.
    """
    with zipfile.ZipFile(archive) as zipped:
        listed: set[str] = set(zipped.namelist())
        for name in (name for name in members if name in listed):
            with writing.atomic_write(into / name) as temp_path, zipped.open(name) as member, temp_path.open("wb") as sink:
                while block := member.read(transports.CHUNK_BYTES):
                    sink.write(block)
    archive.unlink()


def fetch_file(entry: CdnFile, dest: Path, report: transports.FetchReport, source: CdnSource) -> None:
    """Fetch one plain file unless it is already there at its listed size."""
    fetched: bool = not (dest.is_file() and dest.stat().st_size == entry.file_size_bytes)
    if fetched:
        fetch_verified(entry, dest, source)
    report.count(fetched, entry.file_size_bytes)


def fetch_zip(entry: CdnFile, members: list[str], required: list[str], into: Path, report: transports.FetchReport, source: CdnSource) -> None:
    """Fetch one zip and extract ``members`` unless every ``required`` one is already there.

    A zip left under its final name by an interrupted extraction was verified before its rename, so it is reused.
    """
    archive: Path = into / entry.filename
    if all((into / name).is_file() for name in required):
        archive.unlink(missing_ok=True)
        report.count(False, entry.file_size_bytes)
        return
    if not (archive.is_file() and archive.stat().st_size == entry.file_size_bytes):
        fetch_verified(entry, archive, source)
        report.count(True, entry.file_size_bytes)
    extract(archive, members, into)
