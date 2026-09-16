"""Download transports. v1 shipped ``local_verify``; ``hf_fetch`` lands with MSD.

Planned surface (from the design report): ``hf_fetch`` (HuggingFace snapshots),
``http_fetch``, ``gdrive_fetch``, ``api_fetch``, and ``local_verify`` for
datasets that are already on disk (robocap's download verb is verify-only).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from huggingface_hub import snapshot_download


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


def http_fetch() -> None:
    """Plain HTTP fetch — not needed by any v1 dataset yet."""
    raise NotImplementedError("http_fetch is not needed by any v1 dataset")


def gdrive_fetch() -> None:
    """Google Drive fetch — not needed by any v1 dataset yet."""
    raise NotImplementedError("gdrive_fetch is not needed by any v1 dataset")
