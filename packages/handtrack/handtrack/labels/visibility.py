"""Per-keypoint visibility from the ground-truth hand meshes: is a keypoint hidden behind the OTHER hand in a camera?

A keypoint is seen along the ray from the camera centre to it. It is hidden when a triangle of the other hand's mesh crosses that
ray more than ``OTHER_HAND_MARGIN_M`` in front of it. This is the failure the tracker needs to know about: under the other hand,
KeyNet's keypoints slide onto that hand. Self-occlusion (the hand's own fingers or palm in front) is optional (``self_occlusion``)
and off for the labels: on 2026-09-29 the strict rule marked fingertips hidden in 94-99 % of a two-hand clip's frames and the palm
centre always (the landmarks sit on or under the skin), while KeyNet predicts self-occluded keypoints well from its prior. With it
on, the own hand hides a keypoint when it crosses the ray more than the keypoint's flesh margin in front (twice the rest distance to
the nearest mesh vertex, plus ``FLESH_SLACK_M``).

Visibility depends on the camera only, not on a crop: a perspective crop camera shares the camera's centre, so it sees the
keypoint along the same ray. Objects are not modelled (UmeTrack has none; SHOW3D's and HOT3D's objects can still hide a
keypoint that this marks visible).
"""

import torch
from jaxtyping import Bool, Float32, Int64
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

OTHER_HAND_MARGIN_M: float = 0.005
"""The other hand hides a keypoint when its surface is at least this far in front of it (touching hands stay visible)."""
FLESH_SLACK_M: float = 0.003
"""Added to twice the rest-pose flesh depth of each landmark (mesh sampling and the skinning's stretch)."""
_EPS: float = 1e-9


def flesh_margin(model: HandModelTorch) -> Float32[Tensor, "21"]:
    """Per landmark: 2 × (rest distance to the nearest mesh vertex) + slack, in metres, clamped to 5-35 mm."""
    landmarks_mm: Float32[Tensor, "21 3"] = model.landmark_rest_positions.float()
    vertices_mm: Float32[Tensor, "v 3"] = model.mesh_vertices.float()
    nearest_mm: Float32[Tensor, "21"] = torch.cdist(landmarks_mm, vertices_mm).min(dim=-1).values
    return (2.0 * nearest_mm / 1000.0 + FLESH_SLACK_M).clamp(0.005, 0.035)


def ray_blocked(points: Float32[Tensor, "b r 3"], limit: Float32[Tensor, "b r"], triangles: Float32[Tensor, "b t 3 3"]) -> Bool[Tensor, "b r"]:
    """For rays from the origin towards ``points``: does any triangle cross the ray at a distance in (0, ``limit``)?

    Möller-Trumbore on every (ray, triangle) pair of the batch item. Degenerate triangles (all vertices equal, e.g. a missing hand
    zeroed out) never hit.
    """
    distance: Float32[Tensor, "b r"] = points.norm(dim=-1).clamp_min(_EPS)
    direction: Float32[Tensor, "b r 3"] = points / distance[..., None]
    v0: Float32[Tensor, "b t 3"] = triangles[:, :, 0]
    edge1: Float32[Tensor, "b t 3"] = triangles[:, :, 1] - v0
    edge2: Float32[Tensor, "b t 3"] = triangles[:, :, 2] - v0
    pvec: Float32[Tensor, "b r t 3"] = torch.cross(direction[:, :, None].expand(-1, -1, edge2.shape[1], -1), edge2[:, None].expand(-1, direction.shape[1], -1, -1), dim=-1)
    det: Float32[Tensor, "b r t"] = (edge1[:, None] * pvec).sum(-1)
    safe: Float32[Tensor, "b r t"] = torch.where(det.abs() > _EPS, det, torch.ones_like(det))
    tvec: Float32[Tensor, "b t 3"] = -v0  # ray origin (the camera centre) minus v0
    u: Float32[Tensor, "b r t"] = (tvec[:, None] * pvec).sum(-1) / safe
    qvec: Float32[Tensor, "b t 3"] = torch.cross(tvec, edge1, dim=-1)
    v: Float32[Tensor, "b r t"] = (direction[:, :, None] * qvec[:, None]).sum(-1) / safe
    hit_at: Float32[Tensor, "b r t"] = (edge2 * qvec).sum(-1)[:, None] / safe
    hit: Bool[Tensor, "b r t"] = (det.abs() > _EPS) & (u >= 0) & (v >= 0) & (u + v <= 1) & (hit_at > _EPS) & (hit_at < limit[..., None])
    return hit.any(dim=-1)


def keypoints_hidden(
    points_cam: Float32[Tensor, "b 2 21 3"],
    vertices_cam: Float32[Tensor, "b 2 v 3"],
    faces: Int64[Tensor, "f 3"],
    margin: Float32[Tensor, "21"],
    chunk: int = 64,
    self_occlusion: bool = False,
) -> Bool[Tensor, "b 2 21"]:
    """Both hands' keypoints hidden by the other hand's mesh (and, with ``self_occlusion``, by their own), per camera view b. NaN
    hands (no pose) hide nothing and are hidden nowhere (their keypoints come back False: no statement)."""
    device: torch.device = points_cam.device
    faces = faces.to(device)
    margin = margin.to(device)
    hidden: Bool[Tensor, "b 2 21"] = torch.zeros(points_cam.shape[:3], dtype=torch.bool, device=device)
    for start in range(0, points_cam.shape[0], chunk):
        points: Float32[Tensor, "n 2 21 3"] = points_cam[start : start + chunk]
        valid_points: Bool[Tensor, "n 2 21"] = torch.isfinite(points).all(dim=-1)
        meshes: Float32[Tensor, "n 2 v 3"] = torch.nan_to_num(vertices_cam[start : start + chunk], nan=0.0)
        triangles: Float32[Tensor, "n 2 f 3 3"] = meshes[:, :, faces]
        safe_points: Float32[Tensor, "n 2 21 3"] = torch.nan_to_num(points, nan=1.0)
        distance: Float32[Tensor, "n 2 21"] = safe_points.norm(dim=-1)
        for side in (0, 1):
            other: Bool[Tensor, "n 21"] = ray_blocked(safe_points[:, side], distance[:, side] - OTHER_HAND_MARGIN_M, triangles[:, 1 - side])
            if self_occlusion:
                other = other | ray_blocked(safe_points[:, side], distance[:, side] - margin, triangles[:, side])
            hidden[start : start + chunk, side] = other & valid_points[:, side]
    return hidden
