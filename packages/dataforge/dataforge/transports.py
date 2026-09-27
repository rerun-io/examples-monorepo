"""Download transports: ``local_verify`` for corpora already on disk (robocap's download
verb is verify-only), ``hf_fetch`` for HuggingFace snapshots (msd), ``hf_fetch_verified``
for single LFS archives checked against the Hub's sha256 (hocap), and the resuming
``http_fetch`` plus ``http_text`` and its Apache index reader (lamaria, umetrack).

Every dataset download shares the integrity toolkit: ``FileIntegrity`` (size plus sha256, sha1
or git blob sha1) with ``content_digest`` / ``matches_file`` / ``verify_file``, one
``publish_verified`` (stage, verify, atomic replace), the ``hf_file_info`` adapter and one
``FetchReport`` for the summary line. Listing, skip rules, what happens to a rejected file and
whether a partial resumes stay with each dataset.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import requests
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile

CHUNK_BYTES: int = 1 << 20
"""Streaming block size: 1 MiB, small enough to resume cheaply, big enough to saturate a link."""

IDENTITY: dict[str, str] = {"Accept-Encoding": "identity"}
"""Ask for the bytes as stored: a gzip transfer encoding (raw.githubusercontent.com sends one by default)
makes ``Content-Length`` and ``Range`` count compressed bytes while the body arrives decoded."""

ATTEMPTS: int = 4
"""How many times a transport tries one URL before giving up."""

RETRY_BACKOFF_S: tuple[float, ...] = (5.0, 15.0, 45.0)
"""Waits before each retry: four attempts spread over about a minute, which is
what a stalled transfer or a moment of packet loss needs. A longer outage — the
LaMAria archive was down for hours during development — is not something to wait
out in-process: the run gives up, keeps the bytes, and the next one resumes for
free. The last wait repeats if a caller asks for more attempts than there are
entries."""


class IndexEntry(NamedTuple):
    """One row of a remote directory listing."""

    name: str
    """File name, exactly as the page links it."""
    display_bytes: int
    """The size the page *displays*, rounded to three significant digits (``897M``
    → 940 572 672). Good for a budget, a progress line or a summary, and useless
    for verification: only the server's ``Content-Length`` says what to expect."""


class StaleLocalFile(ValueError):
    """The file on disk cannot be reconciled with the remote one, so no retry helps."""


APACHE_ROW_RE: re.Pattern[str] = re.compile(
    r'<a href="(?P<href>[^"]+)">(?P<name>[^<]*)</a>\s*</td>\s*<td[^>]*>[^<]*</td>\s*<td[^>]*>\s*(?P<size>[0-9.]+[KMGT]?|-)\s*</td>'
)
"""One Apache ``IndexOptions FancyIndexing`` table row: the link, its mtime cell, its size cell."""

SIZE_SUFFIX_BYTES: dict[str, int] = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
"""Apache abbreviates with binary multiples, so ``897M`` means 897 MiB."""


def local_verify(root: Path, *, required: Iterable[str]) -> list[str]:
    """Return the ``required`` glob patterns (relative to ``root``) with no match."""
    return [pattern for pattern in required if not any(root.glob(pattern))]


def hf_fetch(
    repo_id: str,
    *,
    allow_patterns: Sequence[str],
    local_dir: Path,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> Path:
    """Fetch a subset of a HuggingFace repo into a plain directory tree.

    ``local_dir`` mode on purpose: files land at ``local_dir/<path-in-repo>``
    rather than in the symlinked hub cache, so a converter globs the raw tree
    exactly as it would a locally recorded corpus, and a partial fetch of a
    multi-hundred-GB dataset costs one copy of what it asked for. Transfer
    acceleration is not set here but is a launch-environment knob
    (``HF_XET_HIGH_PERFORMANCE=1``), which the dataforge pixi feature sets in
    ``[feature.dataforge.activation.env]``.

    Args:
        repo_id: Hub repo, e.g. ``"collabora/monado-slam-datasets"``.
        allow_patterns: Glob patterns of repo-relative paths to fetch; an empty
            sequence fetches nothing (``snapshot_download``'s own semantics).
        local_dir: Destination directory; created by ``snapshot_download``.
        repo_type: ``"dataset"`` (default), ``"model"``, or ``"space"``.
        revision: Branch, tag, or commit; ``None`` takes the default branch.

    Returns:
        ``local_dir``, so callers can chain the fetch into a glob.
    """
    snapshot_download(
        repo_id,
        repo_type=repo_type,
        allow_patterns=list(allow_patterns),
        local_dir=str(local_dir),
        revision=revision,
    )
    return local_dir


def hf_fetch_files(repo_id: str, paths: Sequence[str], *, local_dir: Path, revision: str) -> Path:
    """Fetch known dataset paths without listing the Hub repository."""
    for path in paths:
        hf_hub_download(repo_id, path, repo_type="dataset", local_dir=str(local_dir), revision=revision)
    return local_dir


STAGING_DIR: str = ".dataforge-staging"
"""Subdirectory of ``local_dir`` where ``hf_fetch_verified`` lands and checks a file before publishing it."""

DigestAlgorithm = Literal["sha256", "sha1", "git-sha1"]
"""How a source hashes its files: plain sha256 (LFS objects), plain sha1 (HOT3D's CDN) or git's blob sha1 (files stored in git)."""


class IntegrityError(ValueError):
    """A file's size or content hash differs from what its source lists."""


@dataclass(frozen=True, slots=True)
class FileIntegrity:
    """What a source promises about one file: its exact size and, when it ships one, its content hash."""

    size_bytes: int
    """Exact size in bytes."""
    algorithm: DigestAlgorithm | None
    """Hash the source ships; ``None`` checks the size only (a zip member's CRC-32 is checked on read)."""
    digest: str | None
    """Hex digest under ``algorithm``; ``None`` exactly when ``algorithm`` is."""

    def __post_init__(self) -> None:
        if (self.algorithm is None) != (self.digest is None):
            raise ValueError(f"a digest needs its algorithm and an algorithm its digest, got {self.algorithm!r} {self.digest!r}")


def content_digest(path: Path, algorithm: DigestAlgorithm) -> str:
    """Hex digest of ``path`` under ``algorithm``, streamed in ``CHUNK_BYTES`` blocks."""
    match algorithm:
        case "sha256":
            hasher = hashlib.sha256()
        case "sha1":
            hasher = hashlib.sha1()
        case "git-sha1":
            # Git hashes a "blob <size>\0" header before the content.
            hasher = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        while block := source.read(CHUNK_BYTES):
            hasher.update(block)
    return hasher.hexdigest()


def verify_file(path: Path, expected: FileIntegrity) -> None:
    """Raise ``IntegrityError`` naming the first difference, size before one sequential hash pass."""
    size: int = path.stat().st_size
    if size != expected.size_bytes:
        raise IntegrityError(f"{path} holds {size} bytes, the source lists {expected.size_bytes}")
    if expected.algorithm is not None:
        found: str = content_digest(path, expected.algorithm)
        if found != expected.digest:
            raise IntegrityError(f"{path}: {expected.algorithm} {found} differs from the source's {expected.digest}")


def matches_file(path: Path, expected: FileIntegrity) -> bool:
    """Whether ``path`` is a file holding exactly the bytes ``expected`` describes."""
    if not path.is_file():
        return False
    try:
        verify_file(path, expected)
    except IntegrityError:
        return False
    return True


def publish_verified(staged: Path, dest: Path, expected: FileIntegrity, *, set_aside_replaced: bool = False) -> None:
    """Rename ``staged`` onto ``dest`` once it matches ``expected``, so ``dest`` only ever holds verified bytes.

    A mismatch raises ``IntegrityError`` and touches neither file: whether the rejected copy is deleted,
    quarantined or kept to resume is the caller's policy. ``set_aside_replaced`` keeps an existing ``dest`` as
    ``<name>.stale-<n>`` instead of replacing it; if the rename onto ``dest`` then fails, the old file moves back.
    """
    verify_file(staged, expected)
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept: Path | None = set_aside(dest, "stale") if set_aside_replaced and dest.exists() else None
    try:
        os.replace(staged, dest)
    except OSError:
        if kept is not None:
            kept.rename(dest)
        raise


@dataclass(frozen=True, slots=True)
class HfFileInfo:
    """One file of a pinned Hub listing, with both hashes the Hub reports."""

    path: str
    """Repo-relative path, also the ``local_dir``-relative destination."""
    size_bytes: int
    """Exact size in bytes."""
    lfs_sha256: str | None
    """LFS content hash; ``None`` for files stored in git."""
    blob_id: str
    """Git blob sha1: the content hash of a file stored in git (of its pointer for an LFS file)."""

    @property
    def integrity(self) -> FileIntegrity:
        """The content hash the Hub ships: the LFS sha256, else the git blob sha1."""
        if self.lfs_sha256 is not None:
            return FileIntegrity(self.size_bytes, "sha256", self.lfs_sha256)
        return FileIntegrity(self.size_bytes, "git-sha1", self.blob_id)


def hf_file_info(entry: RepoFile) -> HfFileInfo:
    """Adapt one ``RepoFile`` of a Hub listing."""
    return HfFileInfo(entry.path, entry.size, entry.lfs.sha256 if entry.lfs is not None else None, entry.blob_id)


def hf_lfs_files(repo_id: str, names: Sequence[str], *, revision: str) -> list[HfFileInfo]:
    """The named dataset files at ``revision``, in the order given; each must be an LFS file."""
    found: dict[str, HfFileInfo] = {}
    for info in HfApi().get_paths_info(repo_id, list(names), repo_type="dataset", revision=revision):
        if isinstance(info, RepoFile) and info.lfs is not None:
            found[info.path] = hf_file_info(info)
    missing: list[str] = [name for name in names if name not in found]
    if missing:
        raise FileNotFoundError(f"{repo_id}@{revision} has no LFS file {missing}")
    return [found[name] for name in names]


def set_aside(path: Path, reason: str) -> Path:
    """Rename ``path`` to the first free ``<name>.<reason>-<n>``; nothing is deleted or overwritten."""
    n: int = 0
    while (kept := path.with_name(f"{path.name}.{reason}-{n}")).exists():
        n += 1
    path.rename(kept)
    return kept


def hf_fetch_verified(repo_id: str, file: HfFileInfo, *, local_dir: Path, revision: str) -> bool:
    """Fetch one file unless ``local_dir`` already holds its exact bytes; return whether bytes were fetched.

    The file lands under ``local_dir/.dataforge-staging`` and is published by ``publish_verified`` only after
    its size and hash match, so the destination only ever holds verified bytes. A destination that does not
    match is kept as ``<name>.stale-<n>`` until the replacement is ready; a staged file that fails its check is
    kept as ``<name>.<algorithm>-mismatch-<n>`` (``sha256`` for an LFS file) and the call raises, so the next
    run fetches again.

    huggingface_hub (1.28) does not resume: an interrupted transfer restarts from zero. hf_xet's chunk cache
    (``HF_XET_CACHE``, default ``~/.cache/huggingface/xet``, outside ``local_dir``) can make a quick retry cheap.
    """
    dest: Path = local_dir / file.path
    if matches_file(dest, file.integrity):
        return False
    staged: Path = local_dir / STAGING_DIR / file.path
    if not matches_file(staged, file.integrity):
        hf_fetch_files(repo_id, [file.path], local_dir=local_dir / STAGING_DIR, revision=revision)
    try:
        publish_verified(staged, dest, file.integrity, set_aside_replaced=True)
    except IntegrityError:
        algorithm: str | None = file.integrity.algorithm
        kept: Path = set_aside(staged, f"{algorithm}-mismatch")
        raise ValueError(f"{repo_id}@{revision} {file.path}: fetched bytes differ from the shipped size/{algorithm}; kept as {kept}, rerun") from None
    return True


@dataclass
class FetchReport:
    """What one download run moved; every dataset's summary line ends with ``summary()``."""

    fetched: int = 0
    """Files transferred."""
    skipped: int = 0
    """Files already complete on disk."""
    fetched_bytes: int = 0
    """Bytes transferred, as the source lists them."""
    started_s: float = field(default_factory=time.perf_counter)
    """``perf_counter`` when the run started."""

    def count(self, fetched: bool, size_bytes: int) -> None:
        """Count one file: transferred (``size_bytes`` of it) or already complete on disk."""
        if fetched:
            self.fetched += 1
            self.fetched_bytes += size_bytes
        else:
            self.skipped += 1

    def summary(self) -> str:
        """``fetched <n> file(s), <GB> in <s> (<MB/s>); <n> already complete``."""
        seconds: float = time.perf_counter() - self.started_s
        return (
            f"fetched {self.fetched} file(s), {self.fetched_bytes / 1e9:.2f} GB in {seconds:.0f} s "
            f"({self.fetched_bytes / 1e6 / max(seconds, 1e-9):.1f} MB/s); {self.skipped} already complete"
        )


def attempts_of(url: str, attempts: int) -> Iterator[int]:
    """Yield attempt numbers ``1..attempts``, waiting ``RETRY_BACKOFF_S`` before each retry.

    ``url`` is only for the waiting line it prints, to stderr so a caller's stdout stays machine-readable;
    pass a label in its place to keep a signed URL out of logs.
    """
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            delay_s: float = RETRY_BACKOFF_S[min(attempt - 2, len(RETRY_BACKOFF_S) - 1)]
            print(f"  waiting {delay_s:g} s before attempt {attempt}/{attempts} at {url}", file=sys.stderr)
            time.sleep(delay_s)
        yield attempt


def http_fetch(url: str, *, dest: Path, timeout_s: float = 60.0, attempts: int = ATTEMPTS, label: str | None = None) -> Path:
    """Fetch one URL to one path over plain HTTP, resuming and retrying a partial file.

    Written for archives that are big and servers that are flaky (LaMAria's
    multi-GB VRS files behind an Apache index whose TLS handshakes stall for
    hours), so the transport is built to be re-run:

    * A ``dest`` that already holds the server's full ``Content-Length`` is
      returned untouched — no body is transferred.
    * A shorter ``dest`` is resumed with a ``Range`` header and appended to. A
      server that ignores the range (answering 200 instead of 206) restarts the
      file rather than corrupting it by appending a second copy of the head.
    * Nothing is ever deleted: a stalled transfer **leaves the bytes on disk**,
      so the next attempt — this call's own, or the next run's — resumes.

    ``dest`` is returned so a caller can chain the fetch into a read; its parent
    directories are created, and ``timeout_s`` bounds each request so a stalled
    read raises instead of hanging forever. ``label`` replaces the URL in every
    message, and then failures report only their type and HTTP status: a signed
    URL (HOT3D's CDN links) is a secret, and ``requests`` errors quote the URL.

    Raises:
        StaleLocalFile: ``dest`` is longer than the remote file, which no retry
            can fix — delete it and refetch.
        RuntimeError: Every attempt failed; the partial file is kept.
    """
    shown: str = url if label is None else label
    last_failure: str = ""
    for attempt in attempts_of(shown, attempts):
        try:
            head: requests.Response = requests.head(url, headers=IDENTITY, timeout=timeout_s, allow_redirects=True)
            head.raise_for_status()
            announced: str | None = head.headers.get("Content-Length")
            total: int | None = None if announced is None else int(announced)

            dest.parent.mkdir(parents=True, exist_ok=True)
            have: int = dest.stat().st_size if dest.is_file() else 0
            if total is not None:
                if have == total:
                    return dest
                if have > total:
                    raise StaleLocalFile(f"{dest} holds {have} bytes but {shown} is only {total}; delete it and refetch")

            headers: dict[str, str] = {**IDENTITY, "Range": f"bytes={have}-"} if have else IDENTITY
            with requests.get(url, headers=headers, stream=True, timeout=timeout_s) as response:
                response.raise_for_status()
                # A 200 to a Range request means the server sent the whole file again.
                resuming: bool = have > 0 and response.status_code == 206
                with dest.open("ab" if resuming else "wb") as sink:
                    for block in response.iter_content(chunk_size=CHUNK_BYTES):
                        sink.write(block)

            written: int = dest.stat().st_size
            if total is not None and written != total:
                raise ValueError(f"{dest} holds {written} of {total} bytes after fetching {shown}; resuming from there")
            return dest
        except StaleLocalFile:
            raise
        except (requests.RequestException, ValueError) as failure:
            if label is not None and isinstance(failure, requests.RequestException):
                status: str = "" if failure.response is None else f" (HTTP {failure.response.status_code})"
                last_failure = f"{type(failure).__name__}{status}"
            else:
                last_failure = f"{type(failure).__name__}: {failure}"
            landed: int = dest.stat().st_size if dest.is_file() else 0
            print(f"  warning: attempt {attempt}/{attempts} stalled at {landed / 1e9:.2f} GB ({last_failure}); resuming")
    raise RuntimeError(f"{attempts} attempts at {shown} all stalled, last: {last_failure}")


def http_text(url: str, *, session: requests.Session | None = None, timeout_s: float = 30.0, attempts: int = ATTEMPTS) -> str:
    """GET a small text document, retrying the way ``http_fetch`` does; a shared ``session`` keeps connections alive.

    Warnings go to stderr, so a caller's stdout stays machine-readable.

    Raises:
        RuntimeError: Every attempt failed. A server that answers 4xx/5xx is worth failing on rather than
            treating as an empty document.
    """
    last_failure: str = ""
    for attempt in attempts_of(url, attempts):
        try:
            response: requests.Response = (session or requests).get(url, timeout=timeout_s)
            response.raise_for_status()
            return response.text
        except requests.RequestException as failure:
            last_failure = f"{type(failure).__name__}: {failure}"
            print(f"  warning: attempt {attempt}/{attempts} at {url} failed ({last_failure})", file=sys.stderr)
    raise RuntimeError(f"{attempts} attempts at {url} all failed, last: {last_failure}")


def http_index(url: str, *, timeout_s: float = 30.0, attempts: int = ATTEMPTS) -> list[IndexEntry]:
    """Read one Apache fancy-index page with ``http_text``; entries come back in page order.

    ``url`` needs its trailing slash, and ``timeout_s`` is per request — a page is a few kilobytes.
    """
    return parse_apache_index(http_text(url, timeout_s=timeout_s, attempts=attempts))


def parse_apache_index(html: str) -> list[IndexEntry]:
    """List an Apache fancy-index page as ``IndexEntry`` rows, in listing order.

    Rows without a file (the ``Parent Directory`` link, the ``?C=N;O=D`` sort
    headers, subdirectories) and rows whose size cell is Apache's ``-`` are
    dropped, so every entry names something fetchable, in the page's own order.
    """
    listed: list[IndexEntry] = []
    for row in APACHE_ROW_RE.finditer(html):
        href: str = row["href"]
        size: str = row["size"]
        if size == "-" or href.startswith(("?", "/")) or href.endswith("/"):
            continue
        multiple: int = SIZE_SUFFIX_BYTES.get(size[-1], 1)
        digits: str = size[:-1] if size[-1] in SIZE_SUFFIX_BYTES else size
        listed.append(IndexEntry(name=row["name"], display_bytes=int(float(digits) * multiple)))
    return listed


def repo_revision(repo_id: str, revision: str | None = None) -> str | None:
    """Resolve a branch/tag to the commit sha stamped into every converted rrd.

    Resolve without listing files so a dataset can pin all fetches once per run.
    """
    return HfApi().repo_info(repo_id, repo_type="dataset", revision=revision).sha
