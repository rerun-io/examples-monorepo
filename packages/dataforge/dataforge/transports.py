"""Download transports. v1 ships ``local_verify``; the fetchers land with HOCap.

Planned surface (from the design report): ``hf_fetch`` (HuggingFace snapshots),
``http_fetch``, ``gdrive_fetch``, ``api_fetch``, and ``local_verify`` for
datasets that are already on disk (robocap's download verb is verify-only).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def local_verify(root: Path, *, required: Iterable[str]) -> list[str]:
    """Return the ``required`` glob patterns (relative to ``root``) with no match."""
    return [pattern for pattern in required if not any(root.glob(pattern))]


def hf_fetch() -> None:
    """HuggingFace snapshot fetch — lands with the HOCap dataset (next PR)."""
    raise NotImplementedError("hf_fetch lands with the HOCap dataset")


def http_fetch() -> None:
    """Plain HTTP fetch — not needed by any v1 dataset yet."""
    raise NotImplementedError("http_fetch is not needed by any v1 dataset")


def gdrive_fetch() -> None:
    """Google Drive fetch — not needed by any v1 dataset yet."""
    raise NotImplementedError("gdrive_fetch is not needed by any v1 dataset")
