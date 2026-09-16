"""Download transports. v1 shipped ``local_verify``; ``hf_fetch`` lands with MSD.

Planned surface (from the design report): ``hf_fetch`` (HuggingFace snapshots),
``http_fetch``, ``gdrive_fetch``, ``api_fetch``, and ``local_verify`` for
datasets that are already on disk (robocap's download verb is verify-only).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeAlias

import requests
from huggingface_hub import snapshot_download

CHUNK_BYTES: int = 1 << 20
"""Streaming block size: 1 MiB, small enough to resume cheaply, big enough to saturate a link."""

IndexEntry: TypeAlias = tuple[str, int]
"""One row of a remote directory listing: file name and its size in bytes."""

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


def http_fetch(url: str, *, dest: Path, expected_size: int | None = None, timeout_s: float = 60.0) -> Path:
    """Fetch one URL to one path over plain HTTP, resuming a partial file.

    Written for archives that are big and servers that are flaky (LaMAria's
    multi-GB VRS files behind an Apache index whose TLS handshakes stall for
    hours), so the transport is built to be re-run:

    * A ``dest`` that already holds the server's full ``Content-Length`` is
      returned untouched — no body is transferred.
    * A shorter ``dest`` is resumed with a ``Range`` header and appended to. A
      server that ignores the range (answering 200 instead of 206) restarts the
      file rather than corrupting it by appending a second copy of the head.
    * Nothing is ever deleted on failure: a size mismatch raises and **leaves
      the bytes on disk**, so the next call resumes instead of starting over.

    Args:
        url: Absolute ``http(s)://`` URL of one file.
        dest: Local path to write; parent directories are created.
        expected_size: Size in bytes the caller believes the file has (e.g. from
            a manifest). Checked against the server's ``Content-Length`` before
            any transfer, so a stale manifest fails in a second, not an hour.
        timeout_s: Per-request connect/read timeout. A stalled read raises
            rather than hanging forever; the caller retries and resumes.

    Returns:
        ``dest``, so callers can chain the fetch into a read.

    Raises:
        ValueError: If ``expected_size`` disagrees with ``Content-Length``, if
            ``dest`` is longer than the remote file, or if the bytes on disk
            after the transfer do not match the announced size.
        requests.HTTPError: If either request answers 4xx/5xx.
    """
    head: requests.Response = requests.head(url, timeout=timeout_s, allow_redirects=True)
    head.raise_for_status()
    announced: str | None = head.headers.get("Content-Length")
    remote_size: int | None = None if announced is None else int(announced)
    if expected_size is not None and remote_size is not None and expected_size != remote_size:
        raise ValueError(f"{url} is {remote_size} bytes but {expected_size} were expected; the manifest is stale")
    total: int | None = remote_size if remote_size is not None else expected_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    have: int = dest.stat().st_size if dest.is_file() else 0
    if total is not None:
        if have == total:
            return dest
        if have > total:
            raise ValueError(f"{dest} holds {have} bytes but {url} is only {total}; delete it and refetch")

    headers: dict[str, str] = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        # A 200 to a Range request means the server sent the whole file again.
        resuming: bool = have > 0 and response.status_code == 206
        with dest.open("ab" if resuming else "wb") as sink:
            for block in response.iter_content(chunk_size=CHUNK_BYTES):
                sink.write(block)

    written: int = dest.stat().st_size
    if total is not None and written != total:
        raise ValueError(f"{dest} holds {written} of {total} bytes after fetching {url}; rerun to resume from there")
    return dest


def parse_apache_index(html: str) -> list[IndexEntry]:
    """List an Apache fancy-index page as ``(name, size_bytes)`` in listing order.

    The sizes are the **displayed**, three-significant-digit ones (``897M``,
    ``1.9G``), so they are good for a manifest, a progress bar, or a disk-budget
    check, and useless for verification — a fetch validates against the
    ``Content-Length`` the server reports for the file itself.

    Rows without a file (the ``Parent Directory`` link, the ``?C=N;O=D`` sort
    headers, subdirectories) and rows whose size cell is Apache's ``-`` are
    dropped, so every entry names something fetchable.

    Args:
        html: The index page's body.

    Returns:
        One entry per listed file, in the page's order.
    """
    listed: list[IndexEntry] = []
    for row in APACHE_ROW_RE.finditer(html):
        href: str = row["href"]
        size: str = row["size"]
        if size == "-" or href.startswith(("?", "/")) or href.endswith("/"):
            continue
        multiple: int = SIZE_SUFFIX_BYTES.get(size[-1], 1)
        digits: str = size[:-1] if size[-1] in SIZE_SUFFIX_BYTES else size
        listed.append((row["name"], int(float(digits) * multiple)))
    return listed


def gdrive_fetch() -> None:
    """Google Drive fetch — not needed by any v1 dataset yet."""
    raise NotImplementedError("gdrive_fetch is not needed by any v1 dataset")
