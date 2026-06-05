from __future__ import annotations

from pathlib import Path
from typing import Any

from simplecv.apis.view_hocap_torchcodec_multiview import HOCAP_EXO_CAMERA_NAMES, build_blueprint, find_hocap_video_paths


def test_find_hocap_video_paths_returns_sorted_camera_outputs(tmp_path: Path) -> None:
    """HOCAP camera videos are found in stable camera-name order."""
    sequence_dir: Path = tmp_path / "subject_8" / "sequence"
    second_camera: str = HOCAP_EXO_CAMERA_NAMES[1]
    first_camera: str = HOCAP_EXO_CAMERA_NAMES[0]
    (sequence_dir / second_camera).mkdir(parents=True)
    (sequence_dir / first_camera).mkdir(parents=True)
    (sequence_dir / second_camera / "output.mp4").touch()
    (sequence_dir / first_camera / "output.mp4").touch()

    video_paths: list[Path] = find_hocap_video_paths(tmp_path, subject_id="8", sequence_name="sequence", max_videos=2)

    assert [path.parent.name for path in video_paths] == [first_camera, second_camera]


def test_build_blueprint_creates_video_and_decoded_views() -> None:
    """The Rerun layout places native video and decoded images side by side."""
    blueprint: Any = build_blueprint(["cam_a"])
    time_selection: Any = blueprint.time_panel.time_selection

    assert blueprint is not None
    assert blueprint.time_panel.timeline == "video_time"
    assert blueprint.time_panel.loop_mode == "selection"
    assert time_selection is not None
    assert time_selection.min.value == 0
