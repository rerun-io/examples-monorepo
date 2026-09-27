"""UmeTrack's GitHub source: a pinned file index and a verified, resumable fetch of selected recordings.

The corpus lives in ``github.com/facebookresearch/UmeTrack_data`` under ``raw_data/``: one label JSON (a git
blob) and one stacked MP4 (a git-LFS object) per recording. The index is one recursive tree listing at the
pinned commit plus the 2,410 LFS pointers, which carry each MP4's sha256 and size; it is cached beside the
recordings so later listings and downloads cost no request.
"""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import requests
from serde import serde
from serde.json import to_json

from dataforge.records import decode, read_json
from dataforge.transports import DigestAlgorithm, FetchReport, FileIntegrity, IntegrityError, StaleLocalFile, http_fetch, http_text, publish_verified
from dataforge.writing import atomic_write

REPOSITORY: str = "facebookresearch/UmeTrack_data"
REVISION: str = "950bda2ab602d0ca5591476e04c993ec6f3cac4f"
"""``main`` since 2023-05-03 (the repository's only commit); every URL below names it, so the bytes cannot move."""

RECORDINGS: int = 2410
"""Recordings at ``REVISION``, real and synthetic; a listing with another count is refused."""

CONCURRENCY: int = 8
"""Parallel requests to GitHub, for pointers and files alike."""

INDEX_NAME: str = f"umetrack_index_{REVISION[:12]}.json"
"""Cached index, written under the raw root; the only shared file the corpus has."""

PATH_RE: re.Pattern[str] = re.compile(r"[a-z_]+/[a-z_]+/[a-z_]+/user_\d+/recording_\d+\.(json|mp4)")
"""``<domain>/<interaction>/<split>/user_NN/recording_NN.<ext>``: the layout ``discover()`` globs, and no way out of the root."""

DIGEST_RE: re.Pattern[str] = re.compile(r"sha256:[0-9a-f]{64}|git-sha1:[0-9a-f]{40}")


@serde
@dataclass(frozen=True, slots=True)
class TreeEntry:
    """One row of GitHub's git tree API (third-party schema: unknown fields allowed)."""

    path: str
    type: str
    sha: str
    size: int | None = None


@serde
@dataclass(frozen=True, slots=True)
class Tree:
    """GitHub's recursive tree response."""

    truncated: bool
    tree: list[TreeEntry]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One raw file at ``REVISION``."""

    path: str
    """Raw-root-relative path, e.g. ``real/hand_hand/testing/user_05/recording_00.mp4``."""
    size_bytes: int
    digest: str
    """``sha256:<hex>`` of the content for LFS objects, ``git-sha1:<hex>`` (git's blob hash) for JSONs."""

    def __post_init__(self) -> None:
        if not PATH_RE.fullmatch(self.path) or not DIGEST_RE.fullmatch(self.digest) or self.size_bytes < 0:
            raise ValueError(f"malformed UmeTrack index row: {self.path!r} {self.size_bytes} {self.digest!r}")

    @property
    def integrity(self) -> FileIntegrity:
        algorithm, digest = self.digest.split(":", 1)
        return FileIntegrity(self.size_bytes, cast(DigestAlgorithm, algorithm), digest)

    @property
    def url(self) -> str:
        # An LFS object (sha256) is served by the media host; a plain git blob (git-sha1) by the raw host.
        match self.digest.split(":", 1)[0]:
            case "sha256":
                return f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/raw_data/{self.path}"
            case "git-sha1":
                return f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/raw_data/{self.path}"
            case kind:
                raise ValueError(f"unknown digest kind {kind!r} for {self.path}")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RemoteIndex:
    """Every raw file at one revision (the cached ``INDEX_NAME``)."""

    revision: str
    files: list[RemoteFile]

    def __post_init__(self) -> None:
        if self.revision != REVISION:
            raise ValueError(f"index is for {self.revision}, not {REVISION}")
        if len({remote.path for remote in self.files}) != len(self.files):
            raise ValueError("index lists a path twice")
        unpaired: list[str] = [key for key, files in self.recordings().items() if [f.path.rsplit(".", 1)[1] for f in files] != ["json", "mp4"]]
        if unpaired:
            raise ValueError(f"index recordings without exactly one JSON and one MP4: {unpaired[:5]}")

    def recordings(self) -> dict[str, list[RemoteFile]]:
        """Files grouped by recording key (the path without its extension), keys and files sorted."""
        grouped: dict[str, list[RemoteFile]] = {}
        for remote in sorted(self.files, key=lambda f: f.path):
            grouped.setdefault(remote.path.rsplit(".", 1)[0], []).append(remote)
        return dict(sorted(grouped.items()))


def lfs_pointer(text: str) -> tuple[str, int]:
    """Return ``(sha256 hex, size)`` from a git-lfs pointer file."""
    fields: dict[str, str] = dict(line.split(" ", 1) for line in text.splitlines() if " " in line)
    if not fields.get("version", "").startswith("https://git-lfs.github.com/spec/") or not fields.get("oid", "").startswith("sha256:"):
        raise ValueError(f"not a git-lfs pointer: {text[:80]!r}")
    return fields["oid"].removeprefix("sha256:"), int(fields["size"])


def build_index() -> RemoteIndex:
    """List ``raw_data/`` at ``REVISION``: one tree request, then the MP4s' LFS pointers in parallel."""
    tree: Tree = decode(
        Tree,
        http_text(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"),
        source=f"GitHub tree response for {REPOSITORY}@{REVISION}",
    )
    if tree.truncated:
        raise ValueError(f"GitHub truncated the tree of {REPOSITORY}@{REVISION}")
    entries: list[TreeEntry] = [e for e in tree.tree if e.type == "blob" and e.path.startswith("raw_data/") and e.path.endswith((".json", ".mp4"))]
    videos: list[TreeEntry] = [e for e in entries if e.path.endswith(".mp4")]
    with requests.Session() as session, ThreadPoolExecutor(CONCURRENCY) as pool:
        session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=CONCURRENCY))
        pointers: list[str] = list(
            pool.map(lambda e: http_text(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{e.path}", session=session), videos)
        )
    lfs: dict[str, tuple[str, int]] = {e.path: lfs_pointer(text) for e, text in zip(videos, pointers, strict=True)}
    files: list[RemoteFile] = []
    for e in entries:
        relative: str = e.path.removeprefix("raw_data/")
        if e.path in lfs:
            sha256, size = lfs[e.path]
            files.append(RemoteFile(relative, size, f"sha256:{sha256}"))
        elif e.size is not None:
            files.append(RemoteFile(relative, e.size, f"git-sha1:{e.sha}"))
        else:
            raise ValueError(f"GitHub tree lists {e.path} without a size")
    return RemoteIndex(REVISION, sorted(files, key=lambda f: f.path))


def load_index(root: Path) -> RemoteIndex:
    """Read the cached index under ``root``, building and caching it first if absent; either must list every recording."""
    path: Path = root / INDEX_NAME
    if path.is_file():
        try:
            index: RemoteIndex = read_json(path, RemoteIndex)
        except ValueError as error:
            raise ValueError(f"unreadable UmeTrack index {error}; delete it to rebuild") from error
    else:
        index = build_index()
        with atomic_write(path) as staging:
            staging.write_text(to_json(index))
    if len(index.files) != 2 * RECORDINGS:
        raise ValueError(f"UmeTrack index {path} lists {len(index.files)} raw files, expected {2 * RECORDINGS}; delete it to rebuild")
    return index


@contextmanager
def exclusive(root: Path) -> Iterator[None]:
    """Hold an exclusive lock on the raw root directory itself (no lock file); a second download there fails at once."""
    root.mkdir(parents=True, exist_ok=True)
    descriptor: int = os.open(root, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another UmeTrack download is writing under {root}; let it finish") from error
        yield
    finally:
        os.close(descriptor)


def fetch_file(root: Path, remote: RemoteFile) -> bool:
    """Fetch one file unless it is already complete; return whether bytes were transferred.

    The transfer lands in ``<name>.partial`` (resumed on the next run) and is renamed only once its size and
    hash match, so a file under its final name is always complete. A final file of the wrong size is refused,
    never overwritten; a ``.partial`` proven wrong is removed so the next run restarts it.
    """
    dest: Path = root / remote.path
    if dest.is_file():
        if dest.stat().st_size != remote.size_bytes:
            raise ValueError(f"{dest} holds {dest.stat().st_size} bytes, {REVISION[:12]} has {remote.size_bytes}; delete it and rerun")
        return False
    staging: Path = dest.with_name(dest.name + ".partial")
    try:
        http_fetch(remote.url, dest=staging)
    except StaleLocalFile as error:
        staging.unlink()
        raise ValueError(f"{error}; removed it, the next run restarts it") from error
    try:
        publish_verified(staging, dest, remote.integrity)
    except IntegrityError as error:
        staging.unlink()
        raise ValueError(f"{error}; removed it, the next run restarts it") from None
    return True


def fetch_files(root: Path, files: list[RemoteFile]) -> None:
    """Fetch ``files`` with ``CONCURRENCY`` parallel transfers, print a one-line summary, then raise listing every failure."""
    report: FetchReport = FetchReport()
    failures: list[str] = []
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        futures = {pool.submit(fetch_file, root, remote): remote for remote in files}
        for future in as_completed(futures):
            try:
                report.count(future.result(), futures[future].size_bytes)
            except (ValueError, RuntimeError, OSError) as failure:
                failures.append(f"{futures[future].path}: {failure}")
    print(f"umetrack: {report.summary()}; {len(failures)} failed")
    if failures:
        raise RuntimeError(f"{len(failures)} UmeTrack files failed; rerun to resume:\n" + "\n".join(sorted(failures)))
