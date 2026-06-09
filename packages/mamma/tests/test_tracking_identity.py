"""Unit tests for the identity/geometry pieces of the causal tracker (CPU-only)."""

from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float32, Float64
from numpy import ndarray

from mamma.tracking.detection import bbox_iou_xyxy
from mamma.tracking.identity import (
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


def test_bbox_iou_xyxy() -> None:
    box: Float32[ndarray, "4"] = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    assert bbox_iou_xyxy(box, box) == 1.0
    half: Float32[ndarray, "4"] = np.array([5.0, 0.0, 15.0, 10.0], dtype=np.float32)
    assert abs(bbox_iou_xyxy(box, half) - (50.0 / 150.0)) < 1e-6
    disjoint: Float32[ndarray, "4"] = np.array([20.0, 20.0, 30.0, 30.0], dtype=np.float32)
    assert bbox_iou_xyxy(box, disjoint) == 0.0
