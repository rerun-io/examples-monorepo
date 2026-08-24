from pathlib import Path

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from mv_api.mamma_pose_capture import (
    compute_reprojection_summary,
    load_calibrated_mamma_capture,
    project_keypoints,
)


def _write_capture(root: Path) -> None:
    meta_dir: Path = root / "meta"
    video_dir: Path = root / "videos"
    meta_dir.mkdir(parents=True)
    video_dir.mkdir()
    np.savez(
        meta_dir / "global.npz",
        seq_name="test-eight-view",
        fps=25.0,
        frame_start=0,
        frame_end=121,
    )
    for name, x in (("cam_b", 1.0), ("cam_a", 0.0)):
        world_to_cam: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
        world_to_cam[0, 3] = x
        np.savez(
            meta_dir / f"{name}.npz",
            cam_name=name,
            cam_int=np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]]),
            cam_ext=world_to_cam,
            cam_img_w=64,
            cam_img_h=48,
        )
        (video_dir / f"{name}.mp4").touch()


def test_load_calibrated_mamma_capture_preserves_sorted_camera_alignment(tmp_path: Path) -> None:
    _write_capture(tmp_path)

    capture = load_calibrated_mamma_capture(tmp_path, expected_frame_count=121)

    assert capture.sequence.camera_names == ["cam_a", "cam_b"]
    assert [path.name for path in capture.sequence.video_paths] == ["cam_a.mp4", "cam_b.mp4"]
    assert capture.sequence.frame_count == 121
    assert capture.sequence.fps == 25.0
    assert len(capture.pinholes) == 2
    np.testing.assert_allclose(capture.pinholes[1].projection_matrix, capture.projection_matrices[1])


def test_project_keypoints_and_summary_ignore_low_confidence_observations() -> None:
    projection_matrices: Float32[ndarray, "camera 3 4"] = np.array(
        [
            [[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[100.0, 0.0, 0.0, 100.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    joints3d: Float32[ndarray, "frame joint 4"] = np.array(
        [[[0.0, 0.0, 2.0, 0.9], [1.0, 0.0, 2.0, 0.9]]], dtype=np.float32
    )
    projected: Float32[ndarray, "frame camera joint 2"] = project_keypoints(
        joints3d=joints3d,
        projection_matrices=projection_matrices,
    )
    observed: Float32[ndarray, "frame camera joint 3"] = np.concatenate(
        [projected, np.ones((*projected.shape[:-1], 1), dtype=np.float32)], axis=-1
    )
    observed[0, 0, 0, 0] += 3.0
    observed[0, 0, 1, 1] += 4.0
    observed[0, 1, 1, :2] += 100.0
    observed[0, 1, 1, 2] = 0.1

    summary = compute_reprojection_summary(
        observed_joints2d=observed,
        projected_joints2d=projected,
        confidence_threshold=0.7,
    )

    assert summary.valid_observation_count == 3
    assert summary.mean_px == 7.0 / 3.0
    assert summary.median_px == 3.0
    np.testing.assert_allclose(summary.per_camera_mean_px, np.array([3.5, 0.0], dtype=np.float32))
