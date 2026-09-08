"""Resolve the reference artifacts that are too large to check in.

Eight of the ten segments carry their basalt trajectory in the repository. The
two long-tier ones do not: `MIO14_moving_props` is 3.4 MB of CSV plus 3.6 MB of
per-frame digests and `MIPT03_thrillofthefight_fight_2` is 4.8 MB plus 5.1 MB,
which is not what a git history is for. Those live in a **reference bundle** — a
directory laid out as ``<segment>/basalt_traj.csv`` — that a developer points at
with ``SLAM_RS_REFERENCE_DIR`` or drops in ``packages/slam-rs/data/reference``.

There is no downloader yet: the bundle is machine-local until it is published.
Everything here therefore reports a *reason* rather than fetching, so a test can
skip with a message that says exactly what to do.
"""

import os
from dataclasses import dataclass
from pathlib import Path

BUNDLE_DIR_VARIABLE: str = "SLAM_RS_REFERENCE_DIR"
"""Environment variable naming the bundle root."""
DEFAULT_BUNDLE_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "reference"
"""Where the bundle is looked for when the variable is unset; gitignored."""
TRAJECTORY_CSV: str = "basalt_traj.csv"
"""The C++ trajectory inside a segment's bundle directory, named as basalt writes it."""
RUN_JSON: str = "run.json"
"""The run record beside it: what the C++ was configured with and what it measured."""


@dataclass(slots=True, frozen=True)
class BundleFile:
    """One artifact resolved against the bundle."""

    path: Path
    """Where the file would be, whether or not it exists."""
    reason: str | None
    """Why it is unusable, or None when it is present and readable."""

    @property
    def available(self) -> bool:
        """Whether the file is there."""
        return self.reason is None


def bundle_root() -> Path:
    """The bundle directory: ``SLAM_RS_REFERENCE_DIR`` when set, else the package default."""
    override: str | None = os.environ.get(BUNDLE_DIR_VARIABLE)
    return Path(override).expanduser() if override else DEFAULT_BUNDLE_DIR


def resolve(segment_id: str, filename: str) -> BundleFile:
    """Look for one artifact of one segment in the bundle.

    Args:
        segment_id: Segment directory inside the bundle.
        filename: Artifact name, e.g. ``basalt_traj.csv``.

    Returns:
        The path it would occupy, and why it cannot be used if it is missing.
    """
    root: Path = bundle_root()
    path: Path = root / segment_id / filename
    if path.is_file():
        return BundleFile(path=path, reason=None)
    source: str = f"{BUNDLE_DIR_VARIABLE}={root}" if os.environ.get(BUNDLE_DIR_VARIABLE) else f"the default bundle directory {root}"
    return BundleFile(
        path=path,
        reason=(
            f"{segment_id}/{filename} is not in {source}. It is a long-tier artifact kept out of the "
            f"repository for its size; copy the reference bundle there or set {BUNDLE_DIR_VARIABLE} to a "
            f"directory laid out as <segment>/{filename}."
        ),
    )
