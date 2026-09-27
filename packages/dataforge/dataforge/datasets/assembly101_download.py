"""Assembly101 transport: pinned HF listings, video downloads and range-read pose members.

Three sources, all on the Hub at pinned revisions: the 720p mirror's videos, nimble calibration and
manifest; single members of the mirror's 72 GB ``AssemblyPoses.zip`` (read over HTTP ranges, never the
whole archive); and the gated official repo's ``annotations/``.
"""

import io
import shutil
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, cast
from zipfile import ZipFile, ZipInfo

import httpx
from huggingface_hub import HfApi, HfFileSystem, RepoFile, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from dataforge.datasets.assembly101_source import (
    MIRROR_REPO,
    MIRROR_REVISION,
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    POSE_MEMBERS,
    SHARED_POSE_MEMBER,
    VIDEO_DIR,
)
from dataforge.transports import STAGING_DIR, FetchReport, FileIntegrity, IntegrityError, hf_file_info, matches_file, publish_verified

POSES_ZIP: str = "AssemblyPoses.zip"
MIRROR_SHARED_DIRS: tuple[str, ...] = ("assemblyhands-toolkit/calib/nimble_json_calib",)
MIRROR_SHARED_FILES: tuple[str, ...] = ("manifests/sequences.csv",)
Source = Literal["mirror", "zip", "official"]


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One file the download lands, at its path relative to the source repo."""

    source: Source
    """Mirror file, mirror zip member, or official annotation."""
    path: str
    """Repo path; a zip member name is also its raw-root path."""
    integrity: FileIntegrity
    """Bytes on disk once fetched (the uncompressed size for a zip member) and the Hub's hash; a zip member has
    none, its CRC-32 is checked on read."""


@dataclass(frozen=True, slots=True)
class RemoteIndex:
    """What the pinned sources offer, split into per-sequence and shared files."""

    sequences: dict[str, list[RemoteFile]]
    """Videos and pose members only that sequence uses, by sequence directory."""
    shared: list[RemoteFile]
    """Manifest, nimble calibration and every sequence's fixed extrinsics (the calibration-session lookup)."""


def repo_file(source: Source, entry: RepoFile) -> RemoteFile:
    return RemoteFile(source, entry.path, hf_file_info(entry).integrity)


def list_mirror(api: HfApi, path: str) -> list[RemoteFile]:
    tree = api.list_repo_tree(MIRROR_REPO, repo_type="dataset", revision=MIRROR_REVISION, path_in_repo=path, recursive=True)
    return [repo_file("mirror", entry) for entry in tree if isinstance(entry, RepoFile)]


def open_poses_zip(filesystem: HfFileSystem) -> ZipFile:
    """Range-read the pinned archive; 16 MiB blocks keep big members fast and small ones cheap."""
    handle: io.IOBase = filesystem.open(f"datasets/{MIRROR_REPO}@{MIRROR_REVISION}/{POSES_ZIP}", "rb", block_size=16 * 1024 * 1024)
    return ZipFile(cast(IO[bytes], handle))


def transfer_error(where: str, error: HfHubHTTPError | httpx.HTTPError) -> RuntimeError:
    """Name the file and the status only: the Hub's own message carries the signed CDN URL of the redirect."""
    response: httpx.Response | None = error.response if isinstance(error, HfHubHTTPError | httpx.HTTPStatusError) else None
    status: str = f"HTTP {response.status_code}" if response is not None else "no response"
    return RuntimeError(f"{where}: {type(error).__name__} ({status})")


def remote_index(api: HfApi, filesystem: HfFileSystem) -> RemoteIndex:
    """List the mirror's videos, calibration and the zip's central directory; downloads no content."""
    sequences: dict[str, list[RemoteFile]] = {}
    for remote in list_mirror(api, VIDEO_DIR):
        sequences.setdefault(remote.path.split("/")[2], []).append(remote)
    shared: list[RemoteFile] = [remote for folder in MIRROR_SHARED_DIRS for remote in list_mirror(api, folder)]
    shared += [
        repo_file("mirror", entry)
        for entry in api.get_paths_info(MIRROR_REPO, list(MIRROR_SHARED_FILES), repo_type="dataset", revision=MIRROR_REVISION)
        if isinstance(entry, RepoFile)
    ]
    with open_poses_zip(filesystem) as archive:
        members: list[ZipInfo] = [info for info in archive.infolist() if not info.is_dir()]
    for info in members:
        _, member, name = info.filename.split("/")
        sequence: str = name.removesuffix(".json")
        entry: RemoteFile = RemoteFile("zip", info.filename, FileIntegrity(info.file_size, None, None))
        if member == SHARED_POSE_MEMBER:
            shared.append(entry)
        elif member in POSE_MEMBERS and sequence in sequences:
            sequences[sequence].append(entry)
    return RemoteIndex(dict(sorted(sequences.items())), shared)


def list_annotations(api: HfApi) -> list[RemoteFile]:
    """The gated official repo's annotations; needs a Hub login that accepted its terms."""
    tree = api.list_repo_tree(OFFICIAL_REPO, repo_type="dataset", revision=OFFICIAL_REVISION, path_in_repo="annotations", recursive=True)
    return [repo_file("official", entry) for entry in tree if isinstance(entry, RepoFile)]


def fetch_hub_files(repo: str, revision: str, files: list[RemoteFile], local_dir: Path, report: FetchReport) -> None:
    """Eight files at a time; the first failure cancels the queue.

    Each file lands under ``local_dir/.dataforge-staging`` and is published by ``publish_verified`` only after
    its size and hash match, so a destination only ever holds verified bytes and the caller's size-only skip
    cannot accept an unverified file. A staged file left by a run killed before publishing is verified and
    published without a new transfer. hf_hub_download writes each attempt to a fresh ``.incomplete`` file and
    renames it in place, so an interrupted transfer restarts from zero; the temp files a killed run leaves
    behind are deleted first.
    """
    staging: Path = local_dir / STAGING_DIR
    for stale in (staging / ".cache/huggingface/download").rglob("*.incomplete"):
        stale.unlink()

    def fetch(remote: RemoteFile) -> None:
        staged: Path = staging / remote.path
        if not matches_file(staged, remote.integrity):
            try:
                # force_download only matters for a staged file that fails its check; hf_hub_download never leaves one.
                hf_hub_download(repo, remote.path, repo_type="dataset", revision=revision, local_dir=str(staging), force_download=staged.exists())
            except (HfHubHTTPError, httpx.HTTPError) as error:
                raise transfer_error(f"{repo}@{revision[:8]}/{remote.path}", error) from None
        try:
            publish_verified(staged, local_dir / remote.path, remote.integrity)
        except IntegrityError as error:
            staged.unlink()
            raise ValueError(f"{error}; deleted") from None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures: dict[Future[None], RemoteFile] = {pool.submit(fetch, remote): remote for remote in files}
        wait(futures, return_when=FIRST_EXCEPTION)
        pool.shutdown(cancel_futures=True)
    for future, remote in futures.items():
        if not future.cancelled():
            future.result()
            report.count(True, remote.integrity.size_bytes)


def fetch_members(filesystem: HfFileSystem, files: list[RemoteFile], root: Path, report: FetchReport) -> None:
    """Extract members in archive order through a ``.part`` file; a partial member restarts, CRC-32 is checked on read."""
    if not files:
        return
    try:
        with open_poses_zip(filesystem) as archive:
            infos: list[ZipInfo] = sorted((archive.getinfo(remote.path) for remote in files), key=lambda info: info.header_offset)
            by_path: dict[str, RemoteFile] = {remote.path: remote for remote in files}
            for info in infos:
                target: Path = root / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary: Path = target.with_name(f"{target.name}.part")
                with archive.open(info) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                try:
                    publish_verified(temporary, target, by_path[info.filename].integrity)
                except IntegrityError as error:
                    temporary.unlink()
                    raise ValueError(f"{error}; deleted") from None
                report.count(True, info.file_size)
    except (HfHubHTTPError, httpx.HTTPError) as error:
        raise transfer_error(f"{MIRROR_REPO}@{MIRROR_REVISION[:8]}/{POSES_ZIP}", error) from None
