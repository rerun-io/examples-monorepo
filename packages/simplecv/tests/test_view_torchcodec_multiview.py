from __future__ import annotations

from pathlib import Path

from simplecv.apis.view_torchcodec_multiview import discover_video_paths, video_entity_names


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
