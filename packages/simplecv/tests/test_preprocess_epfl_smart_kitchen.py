from pathlib import Path

import pytest

from simplecv.apis.preprocess_epfl_smart_kitchen import (
    PreprocessConfig,
    VideoProbe,
    _validate_target_probe,
    build_mirror_plan,
    dataset_card_text,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MONOREPO_ROOT: Path = PROJECT_ROOT.parents[1]


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")


def test_epfl_av1_mirror_plan_preserves_layout_and_excludes_generated_outputs(tmp_path: Path) -> None:
    source_root: Path = tmp_path / "source"
    target_root: Path = tmp_path / "target"
    session_root: Path = source_root / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23"
    _touch(session_root / "videos" / "hololens.mp4")
    _touch(session_root / "videos" / "output0.mp4")
    _touch(session_root / "videos_depth" / "hololens_depth.mp4")
    _touch(session_root / "videos" / "hololens.rerun_h264.mp4")
    _touch(source_root / "Public_release_videos.zip")
    _touch(
        source_root
        / "Public_release_videos"
        / "train"
        / "YH2002"
        / "2023_12_04_10_15_23-av1-test"
        / "videos"
        / "hololens.mp4"
    )
    _touch(session_root / "meta_data" / "camera_matrix.json")
    _touch(
        source_root
        / "Public_release_pose"
        / "train"
        / "YH2002"
        / "2023_12_04_10_15_23"
        / "pose_3d"
        / "pose3d_mano.csv"
    )

    plan = build_mirror_plan(source_root=source_root, target_root=target_root)

    video_rels: set[str] = {job.relative_path.as_posix() for job in plan.video_jobs}
    copy_rels: set[str] = {job.relative_path.as_posix() for job in plan.copy_jobs}

    assert video_rels == {
        "Public_release_videos/train/YH2002/2023_12_04_10_15_23/videos/hololens.mp4",
        "Public_release_videos/train/YH2002/2023_12_04_10_15_23/videos/output0.mp4",
        "Public_release_videos/train/YH2002/2023_12_04_10_15_23/videos_depth/hololens_depth.mp4",
    }
    assert "Public_release_videos/train/YH2002/2023_12_04_10_15_23/videos/hololens.rerun_h264.mp4" not in video_rels
    assert all("-av1-test" not in rel for rel in video_rels)
    assert "Public_release_videos.zip" not in copy_rels
    assert "Public_release_videos/train/YH2002/2023_12_04_10_15_23/meta_data/camera_matrix.json" in copy_rels
    assert "Public_release_pose/train/YH2002/2023_12_04_10_15_23/pose_3d/pose3d_mano.csv" in copy_rels
    assert all(not rel.endswith(".mp4") for rel in copy_rels)
    assert all(job.target_path == target_root / job.relative_path for job in plan.video_jobs)


def test_epfl_av1_dataset_card_has_requested_language() -> None:
    card_text: str = dataset_card_text()

    assert "license: cc-by-nc-4.0" in card_text
    assert "AV1-transcoded SimpleCV-compatible mirror" in card_text
    assert "Not endorsed by EPFL" not in card_text
    assert "Unofficial AV1-transcoded" not in card_text
    assert "not the official upstream release" not in card_text


def test_epfl_av1_preprocess_task_is_registered() -> None:
    pixi_text: str = (MONOREPO_ROOT / "pixi.toml").read_text()

    assert "[feature.simplecv.tasks.simplecv-preprocess-epfl-smart-kitchen]" in pixi_text
    assert "tools/preprocess_epfl_smart_kitchen.py" in pixi_text


def test_epfl_av1_nvenc_default_uses_single_worker() -> None:
    config = PreprocessConfig()

    assert config.num_workers == 1


def test_epfl_av1_validate_target_probe_warns_on_frame_count_mismatch() -> None:
    source_probe = VideoProbe(
        path="/tmp/source.mp4",
        codec_name="h264",
        width=640,
        height=480,
        pix_fmt="yuv420p",
        duration_sec=10.0,
        frame_count=300,
        size_bytes=100,
    )
    target_probe = VideoProbe(
        path="/tmp/target.mp4",
        codec_name="av1",
        width=640,
        height=480,
        pix_fmt="yuv420p",
        duration_sec=10.0,
        frame_count=299,
        size_bytes=50,
    )

    with pytest.warns(UserWarning, match="Target frame count does not match source"):
        message = _validate_target_probe(
            source_probe=source_probe,
            target_probe=target_probe,
            duration_tolerance_sec=0.25,
        )

    assert message == "frame_count_mismatch"
