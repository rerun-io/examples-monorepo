"""Adapter for MAMMA-format captures: ``meta/*.npz`` calibration + per-camera MP4s.

Matches the layout of the crossing_arms golden inputs::

    <root>/
      meta/{A001,B001,C001,D001}.npz + global.npz
      videos_light/{A001,B001,C001,D001}.mp4   (or videos/)
"""

from __future__ import annotations

from pathlib import Path

from mamma.calibration.npz_contract import CameraCalibration, GlobalCaptureInfo, load_camera_npz, load_global_npz
from mamma.datasets.sequence import MultiViewSequence

_VIDEO_DIR_CANDIDATES: tuple[str, ...] = ("videos_light", "videos", "videos_crf24")


def load_mamma_sequence(root: Path, camera_names: list[str] | None = None) -> MultiViewSequence:
    """Load a MAMMA-format capture directory into a :class:`MultiViewSequence`.

    Args:
        root: Capture directory containing ``meta/`` and a videos directory.
        camera_names: Optional subset/ordering of cameras; defaults to all
            per-camera NPZs found in ``meta/`` sorted by name.
    """
    meta_dir: Path = root / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"missing calibration dir: {meta_dir}")

    video_dir: Path | None = next((root / d for d in _VIDEO_DIR_CANDIDATES if (root / d).is_dir()), None)
    if video_dir is None:
        raise FileNotFoundError(f"no videos dir among {_VIDEO_DIR_CANDIDATES} under {root}")

    global_info: GlobalCaptureInfo = load_global_npz(meta_dir / "global.npz")

    if camera_names is None:
        camera_names = sorted(p.stem for p in meta_dir.glob("*.npz") if p.stem != "global")
    if not camera_names:
        raise FileNotFoundError(f"no per-camera NPZs in {meta_dir}")

    cameras: list[CameraCalibration] = [load_camera_npz(meta_dir / f"{name}.npz") for name in camera_names]
    video_paths: list[Path] = []
    for name in camera_names:
        video_path: Path = video_dir / f"{name}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"missing video for camera {name}: {video_path}")
        video_paths.append(video_path)

    frame_count: int = global_info.frame_end - global_info.frame_start
    return MultiViewSequence(
        name=global_info.sequence_name,
        cameras=cameras,
        video_paths=video_paths,
        fps=global_info.fps,
        frame_count=frame_count,
        frame_start=global_info.frame_start,
    )
