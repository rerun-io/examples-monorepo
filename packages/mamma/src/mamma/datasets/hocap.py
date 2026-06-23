"""HOCap dataset adapter: simplecv's exo loader -> :class:`MultiViewSequence`.

HOCap ships per-camera JPEG sequences; simplecv's loader encodes them to an
``output.mp4`` per camera on first use (one-time dataset prep, outside the
streaming loop). Calibration comes from the dataset's intrinsics/extrinsics
YAMLs with the tag-anchored world frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exo.hocap_exo import HocapExoSequence
from simplecv.data.exoego.hocap import HocapConfig

from mamma.calibration.npz_contract import CameraCalibration
from mamma.datasets.sequence import MultiViewSequence

HOCAP_FPS: float = 30.0
"""HOCap RealSense capture rate."""


def pinhole_to_calibration(pinhole: PinholeParameters) -> CameraCalibration:
    """simplecv ``PinholeParameters`` -> mamma calibration (w2c convention)."""
    k_matrix: Float64[ndarray, "3 3"] = np.asarray(pinhole.intrinsics.k_matrix, dtype=np.float64)
    world_to_cam: Float64[ndarray, "4 4"] = np.asarray(pinhole.extrinsics.cam_T_world, dtype=np.float64)
    return CameraCalibration(
        name=pinhole.name,
        k_matrix=k_matrix,
        world_to_cam=world_to_cam,
        width=pinhole.intrinsics.width,
        height=pinhole.intrinsics.height,
    )


def load_hocap_sequence(
    root_directory: Path,
    subject_id: str = "8",
    sequence_name: str = "20231024_180733",
    max_cameras: int | None = None,
) -> MultiViewSequence:
    """Load a HOCap subject/sequence as a :class:`MultiViewSequence`.

    Args:
        root_directory: HOCap sample root (contains ``calibration/`` and
            ``subject_*/``).
        subject_id: HOCap subject.
        sequence_name: Capture timestamp directory.
        max_cameras: Optional cap on exo cameras (e.g. 4 for the realtime
            envelope); cameras are taken in loader order.
    """
    exo = HocapExoSequence(HocapConfig(root_directory=root_directory, subject_id=subject_id, sequence_name=sequence_name))
    cameras: list[CameraCalibration] = [pinhole_to_calibration(p) for p in exo.exo_cam_list if p is not None]
    video_paths: list[Path] = list(exo.exo_video_paths)
    if max_cameras is not None:
        cameras = cameras[:max_cameras]
        video_paths = video_paths[:max_cameras]
    frame_count: int = len(exo.exo_video_readers)
    return MultiViewSequence(
        name=f"hocap_subject{subject_id}_{sequence_name}",
        cameras=cameras,
        video_paths=video_paths,
        fps=HOCAP_FPS,
        frame_count=frame_count,
    )
