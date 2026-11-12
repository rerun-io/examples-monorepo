"""The example clips, and the one every entry point uses by default: the run CLI's ``--input``, the app's first,
preloaded example and the ZeroGPU Space's (which orders its own short cuts with :func:`default_first`)."""

from pathlib import Path

DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
"""``packages/metrictrail/data``: the example clips (the ``_download-metrictrail-*data`` tasks fetch them)."""
DEFAULT_CLIP: str = "IMG_0497.mp4"
"""File name of the default clip: a walk along a stone balustrade (dpvo's ``movies/IMG_0497.mp4``, 1:58, 3,532 frames;
the default 1,500-frame cap tracks 1,177 of them). Pablo's pick on 2026-11-12. A cut of it keeps this name."""
DEFAULT_INPUT: Path = DATA_DIR / "movies" / DEFAULT_CLIP
"""The default clip in :data:`DATA_DIR`."""


def default_first(clips: list[Path]) -> list[Path]:
    """``clips`` with the ones named :data:`DEFAULT_CLIP` moved to the front, the rest in their order."""
    return [clip for clip in clips if clip.name == DEFAULT_CLIP] + [clip for clip in clips if clip.name != DEFAULT_CLIP]


def example_videos(data_dir: Path = DATA_DIR) -> list[Path]:
    """The example clips under ``data_dir``: the default clip, then the iPhone ones (``iphone/``), then dpvo's ``movies/``."""
    return default_first(sorted((data_dir / "iphone").glob("*.mp4")) + sorted((data_dir / "movies").glob("*.mp4")))
