from pathlib import Path

import cv2
import numpy as np
import pytest
from jaxtyping import UInt8
from numpy import ndarray

from mv_api.data.synced_videos import SyncedVideoExoEgoConfig


def test_synced_video_config_default_root_directory_is_not_developer_local() -> None:
    config: SyncedVideoExoEgoConfig = SyncedVideoExoEgoConfig()

    assert config.root_directory == Path()


def _write_test_mp4(path: Path, *, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc: int = int(cv2.VideoWriter_fourcc(*"mp4v"))  # pyrefly: ignore[missing-attribute]
    writer: cv2.VideoWriter = cv2.VideoWriter(str(path), fourcc, 5.0, (16, 12))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v writer is unavailable in this environment.")

    bgr: UInt8[ndarray, "h w 3"] = np.full((12, 16, 3), rgb[::-1], dtype=np.uint8)
    for _ in range(2):
        writer.write(bgr)
    writer.release()


def test_synced_video_config_builds_sequence_from_exo_ego_mp4s(tmp_path: Path) -> None:
    root_directory: Path = tmp_path / "lg"
    _write_test_mp4(root_directory / "exo" / "LEFT CAM-synced.mp4", rgb=(255, 0, 0))
    _write_test_mp4(root_directory / "exo" / "RIGHT CAM-synced.mp4", rgb=(0, 255, 0))
    _write_test_mp4(root_directory / "ego" / "TOP CAM-synced.mp4", rgb=(0, 0, 255))

    config: SyncedVideoExoEgoConfig = SyncedVideoExoEgoConfig(root_directory=root_directory)
    sequence = config.setup()

    assert sequence.exo_sequence is not None
    assert sequence.ego_sequence is not None
    assert sequence.exo_sequence.exo_video_names == ["LEFT_CAM-synced", "RIGHT_CAM-synced"]
    assert sequence.ego_sequence.ego_video_names == ["TOP_CAM-synced"]
    assert set(sequence.stream_timestamps_ns) == {
        "exo/LEFT_CAM-synced",
        "exo/RIGHT_CAM-synced",
        "ego/TOP_CAM-synced",
    }

    sample = sequence[0]
    assert sample.exo_bgr_list is not None
    assert sample.ego_bgr_list is not None
    assert len(sample.exo_bgr_list) == 2
    assert len(sample.ego_bgr_list) == 1
