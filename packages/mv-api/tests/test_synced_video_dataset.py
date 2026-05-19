from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
from jaxtyping import Int, UInt8
from numpy import ndarray

from mv_api.data.synced_videos import SyncedVideoExoEgoConfig, SyncedVideoExoEgoSequence


def test_synced_video_config_requires_root_directory_before_setup() -> None:
    config: SyncedVideoExoEgoConfig = SyncedVideoExoEgoConfig()

    assert config.root_directory is None
    with pytest.raises(ValueError, match="root_directory"):
        config.setup()


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


class _GetItemOnlyReader:
    def __getitem__(self, frame_idx: int) -> UInt8[ndarray, "h w 3"]:
        assert frame_idx == 0
        return np.full((12, 16, 3), 127, dtype=np.uint8)


class _FakeEgoVideoReaders:
    def __init__(self) -> None:
        self.video_readers: list[_GetItemOnlyReader] = [_GetItemOnlyReader()]


class _FakeEgoSequence:
    def __init__(self) -> None:
        self.ego_video_readers: _FakeEgoVideoReaders = _FakeEgoVideoReaders()


def test_synced_video_sequence_samples_ego_frames_with_reader_getitem() -> None:
    sequence: SyncedVideoExoEgoSequence = object.__new__(SyncedVideoExoEgoSequence)
    sequence.ego_sequence = cast(Any, _FakeEgoSequence())
    sequence._ego_stream_names = ["ego/top"]
    timestamps: Int[ndarray, "n_frames"] = np.array([0], dtype=np.int64)
    sequence.stream_timestamps_ns = {"ego/top": timestamps}

    ego_bgr_list = sequence._sample_ego_frames(timestamp_ns=0)

    assert ego_bgr_list is not None
    assert len(ego_bgr_list) == 1
    assert int(ego_bgr_list[0][0, 0, 0]) == 127
