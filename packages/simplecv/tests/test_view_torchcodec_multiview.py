from __future__ import annotations

from pathlib import Path
from typing import Any

from simplecv.apis.view_torchcodec_multiview import build_blueprint, discover_video_paths, video_entity_names


def test_discover_video_paths_returns_sorted_mp4s_recursively(tmp_path: Path) -> None:
    """Video directories are discovered by sorted relative mp4 paths."""
    first_video_path: Path = tmp_path / "cam_a" / "output.mp4"
    second_video_path: Path = tmp_path / "cam_b" / "output.mp4"
    third_video_path: Path = tmp_path / "root.mp4"
    ignored_video_path: Path = tmp_path / "cam_c" / "ignore.mov"
    first_video_path.parent.mkdir(parents=True)
    second_video_path.parent.mkdir(parents=True)
    ignored_video_path.parent.mkdir(parents=True)
    first_video_path.touch()
    second_video_path.touch()
    third_video_path.touch()
    ignored_video_path.touch()

    video_paths: list[Path] = discover_video_paths(tmp_path, max_videos=0)
    entity_names: list[str] = video_entity_names(tmp_path, video_paths)

    assert video_paths == [first_video_path, second_video_path, third_video_path]
    assert entity_names == ["cam_a/output", "cam_b/output", "root"]


def test_build_blueprint_creates_video_and_decoded_views() -> None:
    """The Rerun layout places native video and decoded images side by side."""
    blueprint: Any = build_blueprint(["cam_a/output"])
    time_selection: Any = blueprint.time_panel.time_selection

    assert blueprint is not None
    assert blueprint.time_panel.timeline == "video_time"
    assert blueprint.time_panel.loop_mode == "selection"
    assert time_selection is not None
    assert time_selection.min.value == 0
