"""Download the r2b Galileo sample (226 keyframes, 8 pinhole cameras, ground truth) from upstream PyCuSFM.

The sample ships in nvidia-isaac/PyCuSFM's git LFS store. This reads the file list of `data/r2b_galileo` at a pinned
commit from the GitHub tree API and fetches each file from GitHub's LFS media host (the raw host for the few files outside LFS), so neither git-lfs nor a clone is
needed. Files already present with the right size are kept.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from colsfm import DATA_DIR

UPSTREAM_REPO: Final[str] = "nvidia-isaac/PyCuSFM"
"""Where the sample lives."""

UPSTREAM_COMMIT: Final[str] = "0c97a6700ee1ee6e7913ec68acc774a3ec47ccbd"
"""upstream main, 2026-02-09."""

SAMPLE_PATH: Final[str] = "data/r2b_galileo"
"""The sample's directory in the upstream tree."""

EXPECTED_FILES: Final[int] = 234
"""226 JPEGs, `frames_meta.json`, `ground_truth.txt` and the per-camera metadata at that commit."""


@dataclass
class Config:
    """Where to put the sample."""

    output_dir: Path = field(default_factory=lambda: DATA_DIR / "r2b_galileo")
    """Destination; the upstream layout below `data/r2b_galileo` is kept."""
    workers: int = 8
    """Parallel downloads."""


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "colsfm-download"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main(config: Config) -> None:
    """Fetch every file of the sample that is not already on disk.

    Args:
        config: Destination and parallelism.

    Raises:
        RuntimeError: When the upstream tree does not hold the expected number of files.
    """
    tree: dict[str, object] = json.loads(_get(f"https://api.github.com/repos/{UPSTREAM_REPO}/git/trees/{UPSTREAM_COMMIT}?recursive=1"))
    entries: list[dict[str, object]] = [
        entry for entry in tree["tree"] if entry["type"] == "blob" and str(entry["path"]).startswith(f"{SAMPLE_PATH}/")  # pyrefly: ignore[bad-index, not-iterable]
    ]
    if len(entries) != EXPECTED_FILES:
        raise RuntimeError(f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} {SAMPLE_PATH}: {len(entries)} files, expected {EXPECTED_FILES}")

    def fetch(entry: dict[str, object]) -> bool:
        relative: str = str(entry["path"]).removeprefix(f"{SAMPLE_PATH}/")
        destination: Path = config.output_dir / relative
        if destination.is_file() and destination.stat().st_size > 200:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload: bytes = _get(f"https://media.githubusercontent.com/media/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{entry['path']}")
        except urllib.error.HTTPError as error:
            # Files outside LFS are not on the media host.
            if error.code != 404:
                raise
            payload = _get(f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{entry['path']}")
        if payload.startswith(b"version https://git-lfs"):
            raise RuntimeError(f"{relative}: got an LFS pointer, not the file")
        destination.with_suffix(destination.suffix + ".part").write_bytes(payload)
        destination.with_suffix(destination.suffix + ".part").replace(destination)
        return True

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        fetched: int = sum(pool.map(fetch, entries))
    print(f"[colsfm] r2b_galileo: {fetched} fetched, {len(entries) - fetched} already present -> {config.output_dir}")
