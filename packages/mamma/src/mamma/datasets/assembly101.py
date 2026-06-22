"""Assembly101 dataset adapter: simplecv's exo loader -> :class:`MultiViewSequence`."""

from __future__ import annotations

from pathlib import Path

from simplecv.data.exo.assembly101_exo import Assembly101ExoSequence
from simplecv.data.exoego.assembly101 import Assembly101Config

from mamma.calibration.npz_contract import CameraCalibration
from mamma.datasets.hocap import pinhole_to_calibration
from mamma.datasets.sequence import MultiViewSequence


def load_assembly101_sequence(
    root_directory: Path,
    sequence_name: str = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724",
    max_cameras: int | None = None,
) -> MultiViewSequence:
    """Load an Assembly101 sequence (fixed exo cameras) as a :class:`MultiViewSequence`.

    Args:
        root_directory: Sample root containing ``videos/av1-720-new`` and
            ``assembly101_camera_and_hand_poses``.
        sequence_name: Sequence directory/JSON stem.
        max_cameras: Optional cap on exo cameras, taken in loader order.
    """
    exo = Assembly101ExoSequence(Assembly101Config(root_directory=root_directory, sequence_name=sequence_name))
    cameras: list[CameraCalibration] = [pinhole_to_calibration(p) for p in exo.exo_cam_list if p is not None]
    video_paths: list[Path] = list(exo.exo_video_paths)
    if max_cameras is not None:
        cameras = cameras[:max_cameras]
        video_paths = video_paths[:max_cameras]
    fps: float = float(exo.exo_video_readers.video_readers[0].fps)
    frame_count: int = len(exo.exo_video_readers)
    return MultiViewSequence(
        name=f"assembly101_{sequence_name[:40]}",
        cameras=cameras,
        video_paths=video_paths,
        fps=fps,
        frame_count=frame_count,
    )
