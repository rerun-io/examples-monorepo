"""The batched GPU triangulation and wearer selection against the per-frame reference they replaced."""

from itertools import combinations

import cv2
import numpy as np
import pytest
from jaxtyping import Bool, Float64
from numpy import ndarray

from selfcap_curate.keypoints import INLIER_PX, Camera, Triangulated, project, select_wearers, triangulate_frames


# Independent per-frame SVD reference from 9d7032286. Acceptance follows the
# current policy (>=0.3); the old >=0.7 display threshold discarded valid support.
def reference_select_wearer(camera: Camera, anchors: Float64[ndarray, "3 3"], boxes: Float64[ndarray, "p 4"], scores: Float64[ndarray, "p"]) -> int:
    """Nearest head/wrist rectangle distance, head weight two, reject cost >=0.5."""
    projected = project(camera, anchors)
    good = np.isfinite(projected[:, :2]).all(1) & (projected[:, 2] > 0.0)
    if not good.any() or not len(boxes):
        return -1
    uv, weights = projected[good, :2], np.array([2.0, 1.0, 1.0])[good]
    costs = np.full(len(boxes), np.inf)
    for i, (box, score) in enumerate(zip(boxes, scores, strict=True)):
        if score < 0.3 or not np.isfinite(box).all() or np.any(box[2:] - box[:2] <= 0.0):
            continue
        diagonal = max(float(np.linalg.norm(box[2:] - box[:2])), 1.0)
        outside = np.maximum(np.maximum(box[:2] - uv, uv - box[2:]), 0.0)
        centre = np.linalg.norm(uv - (box[:2] + box[2:]) / 2.0, axis=1) / diagonal
        costs[i] = np.average(np.linalg.norm(outside, axis=1) / diagonal + 0.05 * centre, weights=weights) - 0.01 * float(score)
    best = int(np.argmin(costs))
    return best if costs[best] < 0.5 else -1


def reference_triangulate(
    cameras: list[Camera], uv: Float64[ndarray, "v j 2"], scores: Float64[ndarray, "v j"], threshold: float = INLIER_PX
) -> Triangulated:
    """All view pairs, >=0.5° ray separation, weighted DLT and two gated refits."""
    views, joints = scores.shape
    xyz, counts, confidence = np.full((joints, 3), np.nan, np.float32), np.zeros(joints, np.uint8), np.zeros(joints, np.float32)
    if views < 2:
        return Triangulated(xyz, counts, confidence)
    projections = np.stack([np.linalg.inv(c.world_T_cam)[:3] for c in cameras])
    norm = np.stack(
        [
            cv2.undistortPoints(
                np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1, 2), c.K, np.array([*c.radial, 0.0, 0.0, 0.0])
            ).reshape(joints, 2)
            for c, p in zip(cameras, uv, strict=True)
        ]
    )
    valid = (scores >= 0.3) & np.isfinite(scores) & np.isfinite(uv).all(2) & np.isfinite(norm).all(2)
    norm, clean = np.where(np.isfinite(norm), norm, 0.0), np.where(valid, scores, 0.0)
    if not np.any(valid.sum(0) >= 2):
        return Triangulated(xyz, counts, confidence)

    def fit(mask: Bool[ndarray, "v j"]) -> Float64[ndarray, "j 3"]:
        rows = np.stack(
            [
                norm[..., 0, None] * projections[:, None, 2] - projections[:, None, 0],
                norm[..., 1, None] * projections[:, None, 2] - projections[:, None, 1],
            ],
            axis=2,
        )
        rows *= np.sqrt(np.maximum(clean * mask, 0.0))[:, :, None, None]
        _, _, vh = np.linalg.svd(rows.transpose(1, 0, 2, 3).reshape(joints, -1, 4), full_matrices=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            return vh[:, -1, :3] / vh[:, -1, 3:4]

    def residual(points: Float64[ndarray, "j 3"]) -> Float64[ndarray, "v j"]:
        projected = [project(c, points) for c in cameras]
        return np.stack(
            [
                np.where((p[:, 2] > 1e-5) & np.isfinite(r := np.linalg.norm(p[:, :2] - observed, axis=1)), r, np.inf)
                for p, observed in zip(projected, uv, strict=True)
            ]
        )

    best_mask, best_count, best_weight, best_error = np.zeros((views, joints), bool), np.zeros(joints, int), np.zeros(joints), np.full(joints, np.inf)
    rays = np.concatenate([norm, np.ones((views, joints, 1))], axis=2)
    rays = np.stack([r @ c.world_T_cam[:3, :3].T for c, r in zip(cameras, rays, strict=True)])
    rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-12)
    for a, b in combinations(range(views), 2):
        seed = np.zeros((views, joints), bool)
        seed[a] = seed[b] = valid[a] & valid[b] & (np.sum(rays[a] * rays[b], axis=1) < np.cos(np.deg2rad(0.5)))
        res = residual(fit(seed))
        inliers = valid & (res <= threshold) & seed[a][None, :]
        count, weight = inliers.sum(0), (inliers * clean).sum(0)
        mean = np.sum(np.where(inliers, res, 0.0), axis=0) / np.maximum(count, 1)
        better = (count > best_count) | (
            (count == best_count) & ((weight > best_weight + 1e-6) | ((np.abs(weight - best_weight) <= 1e-6) & (mean < best_error)))
        )
        best_mask[:, better] = inliers[:, better]
        best_count[better], best_weight[better], best_error[better] = count[better], weight[better], mean[better]
    inliers = best_mask & valid & (residual(fit(best_mask)) <= threshold)
    points = fit(inliers)
    res = residual(points)
    inliers &= res <= threshold
    count = inliers.sum(0)
    good = (count >= 2) & np.isfinite(points).all(1)
    xyz[good], counts[good] = points[good], count[good]
    confidence[good] = (np.where(inliers, clean, 0.0).sum(0) / np.maximum(count, 1))[good]
    return Triangulated(xyz, counts, confidence)


def cameras(views: int, rng: np.random.Generator) -> list[Camera]:
    result = []
    for v in range(views):
        angle = 2 * np.pi * v / views + rng.normal(0, 0.1)
        position = np.array([3.0 * np.sin(angle), rng.normal(0, 0.2), -3.0 * np.cos(angle)])
        forward = -position / np.linalg.norm(position)
        right = np.cross([0.0, 1.0, 0.0], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = np.stack([right, down, forward], 1), position
        k1 = rng.choice([0.0, -0.12])
        result.append(
            Camera(
                np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]]), pose, (k1, 0.02 if k1 else 0.0), f"/world/rig_{v:02d}/cam_00"
            )
        )
    return result


@pytest.mark.parametrize("chunk", [7, 2048])
@pytest.mark.parametrize("seed", range(6))
def test_batched_triangulation_matches_the_per_frame_reference(seed: int, chunk: int) -> None:
    rng = np.random.default_rng(seed)
    views, frames, joints = int(rng.integers(2, 6)), 40, 133
    cams = cameras(views, rng)
    truth = rng.normal(0, 0.4, (frames, joints, 3))
    uv = np.stack([np.stack([project(c, truth[f])[:, :2] for c in cams]) for f in range(frames)])  # f v j 2
    uv += rng.normal(0, 1.5, uv.shape)
    outliers = rng.random(uv.shape[:3]) < 0.08
    uv[outliers] += rng.normal(0, 60, (int(outliers.sum()), 2))
    scores = rng.uniform(0.0, 1.0, uv.shape[:3])
    uv[rng.random(uv.shape[:3]) < 0.03] = np.nan
    scores[rng.random(scores.shape) < 0.02] = np.nan
    batch = triangulate_frames(cams, uv, scores, chunk=chunk)
    xyz, counts, confidence = batch.xyz, batch.n_views, batch.confidence
    for f in range(frames):
        expected = reference_triangulate(cams, uv[f], scores[f])
        np.testing.assert_array_equal(counts[f], expected.n_views)
        np.testing.assert_array_equal(np.isfinite(xyz[f]), np.isfinite(expected.xyz))
        np.testing.assert_allclose(xyz[f], expected.xyz, atol=1e-6, equal_nan=True)  # float32 output; float64 agrees to ~1e-13 m
        np.testing.assert_allclose(confidence[f], expected.confidence, atol=1e-6)


def test_batched_wearer_selection_matches_the_per_frame_reference() -> None:
    rng = np.random.default_rng(3)
    camera = cameras(4, rng)[0]
    anchors = rng.normal(0, 0.3, (200, 3, 3))
    anchors[::17] = np.nan
    centre = np.stack([project(camera, a)[:, :2].mean(0) if np.isfinite(a).all() else np.zeros(2) for a in anchors])
    boxes = np.concatenate([centre[:, None] + rng.normal(0, 150, (200, 3, 2)), centre[:, None] + 200 + rng.normal(0, 150, (200, 3, 2))], -1)
    boxes[rng.random((200, 3)) < 0.1] = np.nan
    box_scores = rng.uniform(0, 1, (200, 3))
    batched = select_wearers(camera, anchors, boxes, box_scores)
    np.testing.assert_array_equal(batched, [reference_select_wearer(camera, anchors[f], boxes[f], box_scores[f]) for f in range(200)])
