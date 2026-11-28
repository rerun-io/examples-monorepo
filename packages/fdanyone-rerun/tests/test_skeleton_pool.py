"""A skeleton video drawn in a spawned pool process is the one drawn in this process (CPU, synthetic keypoints)."""

from fractions import Fraction
from pathlib import Path

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from fdanyone.skeleton.pipeline import _skeleton_video_pool, _SkeletonVideoJob, _write_skeleton_video
from fdanyone.video import read_rgb_video


def _job(path: Path) -> _SkeletonVideoJob:
    generator: np.random.Generator = np.random.default_rng(0)
    frames: int = 6
    base: Float32[ndarray, "70 2"] = np.stack([generator.uniform(150, 350, 70), generator.uniform(150, 850, 70)], axis=1).astype(np.float32)
    keypoints: Float32[ndarray, "6 70 2"] = np.stack([base + 4.0 * index for index in range(frames)])
    depths: Float32[ndarray, "6 70"] = generator.uniform(2.0, 3.0, (frames, 70)).astype(np.float32)
    return _SkeletonVideoJob(
        keypoints_2d=keypoints,
        keypoint_depths=depths,
        body_scales=np.full(frames, 400.0, dtype=np.float32),
        canvas_height=1024,
        canvas_width=512,
        output_height=512,
        output_width=256,
        fps=Fraction(25),
        path=path,
    )


def test_pooled_skeleton_video_matches_the_in_process_one(tmp_path: Path) -> None:
    direct: Path = _write_skeleton_video(_job(tmp_path / "direct.mp4"))
    with _skeleton_video_pool(2) as pool:
        pooled: Path = pool.submit(_write_skeleton_video, _job(tmp_path / "pooled.mp4")).result()
    expected = read_rgb_video(direct, expected_frames=6)
    actual = read_rgb_video(pooled, expected_frames=6)
    assert any(frame.any() for frame in expected)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
