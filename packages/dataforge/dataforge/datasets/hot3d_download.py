"""Fetch HOT3D from Meta's CDN through the signed URL files a user downloads from projectaria.com.

Only what the converter reads is fetched: the VRS, the ``ground_truth`` and ``hand_data`` zips (members
extracted, the zip then removed) and, once for both devices, the object GLBs. The preview mp4 and every MPS
archive are skipped (Aria: 195 GB of 526 GB). A signed URL is a secret: none is printed, logged or written;
the manifest ``download`` leaves behind carries names, sizes and sha1sums only.
"""

import fcntl
from pathlib import Path

from serde.json import to_json

from dataforge import meta_cdn, transports, writing
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
from dataforge.meta_cdn import CdnFile, UrlFile
from dataforge.records import read_json

URL_FILE_HELP: str = "download a fresh URL file from https://www.projectaria.com/datasets/hot3D/ (links expire after about 3 weeks)"
"""How a user gets (or renews) a URL file."""
SOURCE: meta_cdn.CdnSource = meta_cdn.CdnSource("HOT3D", URL_FILE_HELP)
"""Names HOT3D in every CDN message."""

FETCHED_TYPES: tuple[str, ...] = ("main_vrs", "ground_truth", "hand_data")
"""Per-sequence data types the converter reads; ``video_main_rgb`` and ``mps_*`` are never fetched."""

UNREAD_MEMBERS: frozenset[str] = frozenset({"box2d_hands.csv", "box2d_objects.csv", "license.txt"})
"""Zip members the converter never reads, so they are not extracted."""


def read_url_file(path: Path) -> UrlFile:
    """Parse a HOT3D URL file without ever quoting a URL."""
    return meta_cdn.read_url_file(path, SOURCE)


def url_file_at(path: Path | None, flag: str) -> Path:
    """The URL file a config field names, or the fix when it is unset or absent."""
    return meta_cdn.url_file_at(path, flag, SOURCE)


def manifest_of(urls: UrlFile, device: Device) -> Manifest:
    """The URL file's fetched types and read members, without URLs; the file must be the device's."""
    expected: str = f"Hot3D{DEVICES[device].url_label}"
    if urls.sequence_config.dataset_name != expected:
        raise ValueError(f"the URL file is {urls.sequence_config.dataset_name}'s; hot3d-{device} needs {expected}_download_urls*.json")
    return Manifest(
        release=str(urls.sequence_config.release),
        data_groups={group: [name for name in urls.sequence_config.data_groups[group] if name not in UNREAD_MEMBERS] for group in FETCHED_TYPES[1:]},
        sequences={
            name: {kind: ListedFile(files[kind].filename, files[kind].sha1sum, files[kind].file_size_bytes) for kind in FETCHED_TYPES}
            for name, files in sorted(urls.sequences.items())
        },
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
        meta_cdn.fetch_zip(urls.sequences["assets"]["assets"], members, members, folder, report, SOURCE)


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
        meta_cdn.fetch_file(files["main_vrs"], folder / "recording.vrs", report, SOURCE)
        # ground_truth first: its metadata.json says whether hand_data is needed (test-split sequences ship none).
        for group in FETCHED_TYPES[1:]:
            metadata: Metadata | None = read_json(folder / "metadata.json", Metadata) if (folder / "metadata.json").is_file() else None
            required: list[str] = [member for member in manifest.data_groups[group] if member in required_members(manifest, metadata)]
            meta_cdn.fetch_zip(files[group], manifest.data_groups[group], required, folder, report, SOURCE)
        print(f"  {name}: done")
    print(f"hot3d-{device}: {len(selected)} sequence(s); {report.summary()} → {root}")
