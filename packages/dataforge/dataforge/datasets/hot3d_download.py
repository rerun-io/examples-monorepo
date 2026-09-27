"""Fetch HOT3D from Meta's CDN through the signed URL files a user downloads from projectaria.com.

Only what the converter reads is fetched: the VRS, the ``ground_truth`` and ``hand_data`` zips (members
extracted, the zip then removed) and, once for both devices, the object GLBs. The preview mp4 and every MPS
archive are skipped (Aria: 195 GB of 526 GB). A signed URL is a secret: none is printed, logged or written;
the manifest ``download`` leaves behind carries names, sizes and sha1sums only.
"""

import fcntl
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from serde import serde
from serde.json import to_json

from dataforge import transports, writing
from dataforge.datasets.base import RemoteSequence
from dataforge.datasets.hot3d_source import (
    DEVICES,
    Device,
    ListedFile,
    Manifest,
    Metadata,
    manifest_path,
    required_members,
    sequence_files,
)
from dataforge.records import decode, read_json

URL_FILE_HELP: str = "download a fresh URL file from https://www.projectaria.com/datasets/hot3D/ (links expire after about 3 weeks)"
"""How a user gets (or renews) a URL file."""

FETCHED_TYPES: tuple[str, ...] = ("main_vrs", "ground_truth", "hand_data")
"""Per-sequence data types the converter reads; ``video_main_rgb`` and ``mps_*`` are never fetched."""

UNREAD_MEMBERS: frozenset[str] = frozenset({"box2d_hands.csv", "box2d_objects.csv", "license.txt"})
"""Zip members the converter never reads, so they are not extracted."""


@serde
@dataclass(frozen=True, slots=True)
class CdnFile:
    """One entry of a HOT3D URL file (third-party format)."""

    filename: str
    """CDN file name."""
    sha1sum: str
    """Hex SHA-1 of the complete file."""
    file_size_bytes: int
    """Size of the complete file."""
    download_url: str = field(repr=False)
    """Signed CDN URL: a secret, kept out of ``repr`` and every message."""

    def listed(self) -> ListedFile:
        """The entry without its URL."""
        return ListedFile(self.filename, self.sha1sum, self.file_size_bytes)


@serde
@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """The URL file's dataset, release and zip inventory."""

    dataset_name: str
    """``Hot3DAria``, ``Hot3DQuest`` or ``Hot3DAssets``."""
    data_groups: dict[str, list[str]]
    """Files each zip extracts to, by group."""
    release: str | None = None
    """Upstream release; the assets file has none."""


@serde
@dataclass(frozen=True, slots=True)
class UrlFile:
    """``Hot3D{Aria,Quest,Assets}_download_urls*.json`` as projectaria.com hands it out."""

    sequences: dict[str, dict[str, CdnFile]]
    """Files per sequence (the assets file has one pseudo-sequence ``assets``), by data type."""
    sequence_config: SequenceConfig
    """Release and zip inventory."""


def read_url_file(path: Path) -> UrlFile:
    """Parse a URL file; a decode error names only its type, since pyserde's message quotes the rejected value (a URL)."""
    return decode(UrlFile, path.read_text(), source=str(path), redact=f"not a HOT3D URL file; {URL_FILE_HELP}")


def url_file_at(path: Path | None, flag: str) -> Path:
    """The URL file a config field names, or the fix when it is unset or absent."""
    if path is None or not path.is_file():
        raise FileNotFoundError(f"HOT3D needs {flag} <URL file> (got {path}); {URL_FILE_HELP}")
    return path


def expiry(url: str) -> datetime | None:
    """When a signed fbcdn URL stops working (its ``oe`` parameter, hex Unix seconds), if it says."""
    stamps: list[str] = parse_qs(urlparse(url).query).get("oe", [])
    return datetime.fromtimestamp(int(stamps[0], 16), UTC) if stamps else None


def manifest_of(urls: UrlFile, device: Device) -> Manifest:
    """The URL file's fetched types and read members, without URLs; the file must be the device's."""
    expected: str = f"Hot3D{DEVICES[device].url_label}"
    if urls.sequence_config.dataset_name != expected:
        raise ValueError(f"the URL file is {urls.sequence_config.dataset_name}'s; hot3d-{device} needs {expected}_download_urls*.json")
    return Manifest(
        release=str(urls.sequence_config.release),
        data_groups={group: [name for name in urls.sequence_config.data_groups[group] if name not in UNREAD_MEMBERS] for group in FETCHED_TYPES[1:]},
        sequences={name: {kind: files[kind].listed() for kind in FETCHED_TYPES} for name, files in sorted(urls.sequences.items())},
    )


def remote_sequences(manifest: Manifest, device: Device) -> list[RemoteSequence]:
    """One listing row per sequence: what ``download`` fetches and what it leaves on disk."""
    return [
        RemoteSequence(
            key=name,
            size_bytes=sum(entry.file_size_bytes for entry in files.values()),
            files=tuple(sequence_files(manifest, device, name)),
        )
        for name, files in manifest.sequences.items()
    ]


def fetch_verified(entry: CdnFile, dest: Path) -> None:
    """Fetch into ``<dest>.part`` (resuming it), check size and sha1, then rename onto ``dest``.

    An expired link fails before any request. A complete ``.part`` whose sha1 is wrong is removed so the next
    run refetches it; nothing else is ever deleted.
    """
    stamp: datetime | None = expiry(entry.download_url)
    if stamp is not None and stamp <= datetime.now(UTC):
        raise ValueError(f"the HOT3D URL file's link for {entry.filename} expired on {stamp:%Y-%m-%d %H:%M} UTC; {URL_FILE_HELP}")
    part: Path = dest.with_name(f"{dest.name}.part")
    transports.http_fetch(entry.download_url, dest=part, label=f"HOT3D {entry.filename}")
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


def fetch_file(entry: CdnFile, dest: Path, report: transports.FetchReport) -> None:
    """Fetch one plain file unless it is already there at its listed size."""
    fetched: bool = not (dest.is_file() and dest.stat().st_size == entry.file_size_bytes)
    if fetched:
        fetch_verified(entry, dest)
    report.count(fetched, entry.file_size_bytes)


def fetch_zip(entry: CdnFile, members: list[str], required: list[str], into: Path, report: transports.FetchReport) -> None:
    """Fetch one zip and extract ``members`` unless every ``required`` one is already there.

    A zip left under its final name by an interrupted extraction was verified before its rename, so it is reused.
    """
    archive: Path = into / entry.filename
    if all((into / name).is_file() for name in required):
        archive.unlink(missing_ok=True)
        report.count(False, entry.file_size_bytes)
        return
    if not (archive.is_file() and archive.stat().st_size == entry.file_size_bytes):
        fetch_verified(entry, archive)
        report.count(True, entry.file_size_bytes)
    extract(archive, members, into)


def fetch_assets(root: Path, assets_url_file: Path | None, report: transports.FetchReport) -> None:
    """Fetch the shared GLBs once for both devices, under a lock so concurrent device downloads do not race.

    ``instance.json`` is extracted last, so it marks a finished extraction (it also lists three objects that ship no
    GLB, so it cannot serve as the GLB inventory). The assets URL file is needed only before that (or when given).
    """
    folder: Path = root / "assets"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if assets_url_file is None and (folder / "instance.json").is_file():
            return
        urls: UrlFile = read_url_file(url_file_at(assets_url_file, "--assets-url-file"))
        if urls.sequence_config.dataset_name != "Hot3DAssets":
            raise ValueError(f"the assets URL file is {urls.sequence_config.dataset_name}'s; --assets-url-file needs Hot3DAssets_download_urls*.json")
        members: list[str] = sorted(
            (name for name in urls.sequence_config.data_groups["assets"] if name not in UNREAD_MEMBERS), key=lambda name: name == "instance.json"
        )
        fetch_zip(urls.sequences["assets"]["assets"], members, members, folder, report)


def download(root: Path, device: Device, *, url_file: Path | None, assets_url_file: Path | None, sequences: tuple[str, ...] | None) -> None:
    """Write the manifest, then fetch the selected sequences and the shared assets; rerunning resumes."""
    urls: UrlFile = read_url_file(url_file_at(url_file, "--url-file"))
    manifest: Manifest = manifest_of(urls, device)
    selected: list[str] = sorted(urls.sequences) if sequences is None else sorted(set(sequences))
    unknown: list[str] = [name for name in selected if name not in urls.sequences]
    if unknown:
        raise ValueError(f"hot3d-{device}: unknown sequence(s) {unknown} in the URL file; see `dataforge-download hot3d-{device} --list-remote`")
    with writing.atomic_write(manifest_path(root, device)) as temp_path:
        temp_path.write_text(to_json(manifest))

    report: transports.FetchReport = transports.FetchReport()
    fetch_assets(root, assets_url_file, report)
    for name in selected:
        folder: Path = root / device / name
        files: dict[str, CdnFile] = urls.sequences[name]
        fetch_file(files["main_vrs"], folder / "recording.vrs", report)
        # ground_truth first: its metadata.json says whether hand_data is needed (test-split sequences ship none).
        for group in FETCHED_TYPES[1:]:
            metadata: Metadata | None = read_json(folder / "metadata.json", Metadata) if (folder / "metadata.json").is_file() else None
            required: list[str] = [member for member in manifest.data_groups[group] if member in required_members(manifest, metadata)]
            fetch_zip(files[group], manifest.data_groups[group], required, folder, report)
        print(f"  {name}: done")
    print(f"hot3d-{device}: {len(selected)} sequence(s); {report.summary()} → {root}")
