"""Unit tests for the identity/geometry pieces of the causal tracker (CPU-only)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float32, Float64
from numpy import ndarray
from ultralytics.engine.results import Boxes

from mamma.models.detection import Yolo12xDetector
from mamma.ops.identity import (
    FeatureBank,
    assign_hungarian,
    epipolar_score,
    fundamental_matrix,
    resolve_epipolar_px,
)


def _look_at_extrinsics(camera_position: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """World->camera transform for a camera at ``camera_position`` looking at the origin."""
    forward: Float64[ndarray, "3"] = -camera_position / np.linalg.norm(camera_position)
    up_hint: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0])
    right: Float64[ndarray, "3"] = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    down: Float64[ndarray, "3"] = np.cross(forward, right)
    rotation: Float64[ndarray, "3 3"] = np.stack([right, down, forward], axis=0)
    world_to_cam: Float64[ndarray, "4 4"] = np.eye(4)
    world_to_cam[:3, :3] = rotation
    world_to_cam[:3, 3] = -rotation @ camera_position
    return world_to_cam


def _project(k: Float64[ndarray, "3 3"], w2c: Float64[ndarray, "4 4"], point: Float64[ndarray, "3"]) -> Float64[ndarray, "3"]:
    cam_point: Float64[ndarray, "3"] = w2c[:3, :3] @ point + w2c[:3, 3]
    pixel: Float64[ndarray, "3"] = k @ cam_point
    return np.array([pixel[0] / pixel[2], pixel[1] / pixel[2], 1.0])


def test_epipolar_score_consistent_point() -> None:
    """A 3D point projected into two cameras scores ~1; a far-off point scores 0."""
    k: Float64[ndarray, "3 3"] = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    w2c_a: Float64[ndarray, "4 4"] = _look_at_extrinsics(np.array([3.0, 0.0, 1.5]))
    w2c_b: Float64[ndarray, "4 4"] = _look_at_extrinsics(np.array([0.0, 3.0, 1.5]))

    f_matrix: Float64[ndarray, "3 3"] | None = fundamental_matrix(k, w2c_a, k, w2c_b)
    assert f_matrix is not None

    point: Float64[ndarray, "3"] = np.array([0.1, -0.2, 1.0])
    x_a: Float64[ndarray, "3"] = _project(k, w2c_a, point)
    x_b: Float64[ndarray, "3"] = _project(k, w2c_b, point)

    sigma_px: float
    max_dist_px: float
    sigma_px, max_dist_px = resolve_epipolar_px(1280, 720)
    score_good: float = epipolar_score(f_matrix, x_a, x_b, sigma_px, max_dist_px)
    assert score_good > 0.99

    x_b_wrong: Float64[ndarray, "3"] = x_b + np.array([500.0, 500.0, 0.0])
    score_bad: float = epipolar_score(f_matrix, x_a, x_b_wrong, sigma_px, max_dist_px)
    assert score_bad < score_good * 0.5


def test_feature_bank_dedup_and_eviction() -> None:
    bank: FeatureBank = FeatureBank(max_size=3)
    base: Float32[torch.Tensor, "512"] = torch.randn(512)
    assert bank.append(0, base)
    assert not bank.append(0, base * 2.0)  # same direction => cosine 1.0 => duplicate
    for _ in range(5):
        bank.append(0, torch.randn(512))
    assert len(bank._bank[0]) <= 3
    assert bank.obj_ids == [0]


def test_feature_bank_similarity_identifies_match() -> None:
    bank: FeatureBank = FeatureBank()
    person_a: Float32[torch.Tensor, "512"] = torch.randn(512)
    person_b: Float32[torch.Tensor, "512"] = torch.randn(512)
    bank.append(0, person_a)
    bank.append(1, person_b)
    dets: Float32[torch.Tensor, "2 512"] = torch.stack([person_b + 0.01 * torch.randn(512), person_a + 0.01 * torch.randn(512)])
    scores: Float32[ndarray, "2 2"] = bank.similarity(dets)
    matches: dict[int, int] = assign_hungarian(scores, min_score=0.5)
    assert matches == {0: 1, 1: 0}


def test_detector_keeps_confident_persons_drops_overlapping_duplicates_and_puts_the_best_first() -> None:
    """Per frame, in frame order: class 0 at confidence >= 0.5 only; a box overlapping a better one by IoU > 0.75 goes."""
    rows: list[list[list[float]]] = [  # x1 y1 x2 y2 confidence class, per frame (Ultralytics' result.boxes)
        [[0, 0, 10, 10, 0.6, 0], [0, 0, 10, 9.5, 0.9, 0], [20, 20, 30, 30, 0.7, 0], [0, 0, 5, 5, 0.95, 2], [40, 40, 50, 50, 0.4, 0]], [], [[1, 1, 4, 4, 0.8, 0]],
    ]  # fmt: skip
    results = iter([SimpleNamespace(boxes=Boxes(torch.tensor(r).reshape(-1, 6), orig_shape=(4, 4))) for r in rows])
    detector = object.__new__(Yolo12xDetector)  # the YOLO model replaced by its canned results
    detector.model, detector.confidence, detector.dedup_iou = lambda *_image, **_options: [next(results)], 0.5, 0.75  # type: ignore[assignment]
    found = detector(torch.zeros((3, 4, 4, 3), dtype=torch.uint8))
    assert found.xyxy.tolist() == [[0, 0, 10, 9.5], [20, 20, 30, 30], [1, 1, 4, 4]] and found.frame_indices.tolist() == [0, 0, 2]
    np.testing.assert_allclose(found.scores.numpy(), [0.9, 0.7, 0.8], rtol=1e-6)


@settings(max_examples=60, deadline=None)
@given(score=arrays(np.float32, st.tuples(st.integers(0, 5), st.integers(0, 5)), elements=st.floats(0.0, 1.0, width=32)), min_score=st.floats(0.0, 1.0))
def test_identity_assignment_is_one_to_one_above_the_threshold(score: np.ndarray, min_score: float) -> None:
    """The per-tick identity map: each subject takes at most one detection and each detection at most one subject."""
    matches: dict[int, int] = assign_hungarian(score, min_score=min_score)
    assert len(set(matches.values())) == len(matches)
    assert all(0 <= r < score.shape[0] and 0 <= c < score.shape[1] and score[r, c] >= min_score for r, c in matches.items())
    if score.size and min_score == 0.0:
        assert len(matches) == min(score.shape)
