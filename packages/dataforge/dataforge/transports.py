"""Download transports. v1 shipped ``local_verify``; ``hf_fetch`` lands with MSD.

Planned surface (from the design report): ``hf_fetch`` (HuggingFace snapshots),
``http_fetch``, ``gdrive_fetch``, ``api_fetch``, and ``local_verify`` for
datasets that are already on disk (robocap's download verb is verify-only).
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import requests
from huggingface_hub import snapshot_download

CHUNK_BYTES: int = 1 << 20
"""Streaming block size: 1 MiB, small enough to resume cheaply, big enough to saturate a link."""

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


def attempts_of(url: str, attempts: int) -> Iterator[int]:
    """Yield attempt numbers ``1..attempts``, waiting ``RETRY_BACKOFF_S`` before each retry.

    ``url`` is only for the waiting line it prints.
    """
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            delay_s: float = RETRY_BACKOFF_S[min(attempt - 2, len(RETRY_BACKOFF_S) - 1)]
            print(f"  waiting {delay_s:g} s before attempt {attempt}/{attempts} at {url}")
            time.sleep(delay_s)
        yield attempt


def http_fetch(url: str, *, dest: Path, timeout_s: float = 60.0, attempts: int = ATTEMPTS) -> Path:
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
    read raises instead of hanging forever.

    Raises:
        StaleLocalFile: ``dest`` is longer than the remote file, which no retry
            can fix — delete it and refetch.
        RuntimeError: Every attempt failed; the partial file is kept.
    """
    last_failure: str = ""
    for attempt in attempts_of(url, attempts):
        try:
            head: requests.Response = requests.head(url, timeout=timeout_s, allow_redirects=True)
            head.raise_for_status()
            announced: str | None = head.headers.get("Content-Length")
            total: int | None = None if announced is None else int(announced)

            dest.parent.mkdir(parents=True, exist_ok=True)
            have: int = dest.stat().st_size if dest.is_file() else 0
            if total is not None:
                if have == total:
                    return dest
                if have > total:
                    raise StaleLocalFile(f"{dest} holds {have} bytes but {url} is only {total}; delete it and refetch")

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
                raise ValueError(f"{dest} holds {written} of {total} bytes after fetching {url}; resuming from there")
            return dest
        except StaleLocalFile:
            raise
        except (requests.RequestException, ValueError) as failure:
            last_failure = f"{type(failure).__name__}: {failure}"
            landed: int = dest.stat().st_size if dest.is_file() else 0
            print(f"  warning: attempt {attempt}/{attempts} stalled at {landed / 1e9:.2f} GB ({last_failure}); resuming")
    raise RuntimeError(f"{attempts} attempts at {url} all stalled, last: {last_failure}")


def http_index(url: str, *, timeout_s: float = 30.0, attempts: int = ATTEMPTS) -> list[IndexEntry]:
    """Read one Apache fancy-index page, retrying the way ``http_fetch`` does.

    Entries come back in page order. ``url`` needs its trailing slash, and
    ``timeout_s`` is per request — a page is a few kilobytes.

    Raises:
        RuntimeError: Every attempt failed. An archive that answers 4xx/5xx is
            worth failing on rather than treating as an empty directory.
    """
    last_failure: str = ""
    for attempt in attempts_of(url, attempts):
        try:
            page: requests.Response = requests.get(url, timeout=timeout_s)
            page.raise_for_status()
            return parse_apache_index(page.text)
        except requests.RequestException as failure:
            last_failure = f"{type(failure).__name__}: {failure}"
            print(f"  warning: attempt {attempt}/{attempts} at the index {url} failed ({last_failure})")
    raise RuntimeError(f"{attempts} attempts at {url} all failed, last: {last_failure}")


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


def gdrive_fetch() -> None:
    """Google Drive fetch — not needed by any v1 dataset yet."""
    raise NotImplementedError("gdrive_fetch is not needed by any v1 dataset")
