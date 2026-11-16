"""Canonical target-video paths shared by publication and result consumers."""

from __future__ import annotations

from pathlib import Path



def target_video_path(camera_id: int) -> Path:
    """Return the relative publication path for a target camera."""

    return Path("videos") / f"{camera_id:02d}.mp4"


