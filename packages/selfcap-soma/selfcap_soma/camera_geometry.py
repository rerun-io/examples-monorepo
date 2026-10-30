"""Projection and robust triangulation reused from selfcap-curate/keypoints.py.

Copied without behavioral changes from local commit d2be70660. This small
inference subset avoids installing the detector stack in the fitting env.
The parity test compares source AST and numeric projection against upstream.
"""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor

INLIER_PX: float = 8.0
EGO_POSE_MAX_GAP_S: float = 0.1


@dataclass(frozen=True, slots=True)
class Camera:
    K: Float64[ndarray, "3 3"]  # Encoded-image pinhole matrix.
    world_T_cam: Float64[ndarray, "4 4"]  # RDF camera to Quest world, metres.
    radial: tuple[float, float]  # OpenCV k1,k2; refined iOS pinholes are rectified (zero).
    entity: str  # Camera node for the per-view overlay.


def project(camera: Camera, xyz: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    """Project world points to u,v,depth with the camera's radial distortion."""
    inverse = np.linalg.inv(camera.world_T_cam)
    q = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[:, :2] / q[:, 2:3]
        radius = np.sum(xy * xy, axis=1)
        uv = (xy * (1.0 + camera.radial[0] * radius + camera.radial[1] * radius**2)[:, None]) @ camera.K[:2, :2].T + camera.K[:2, 2]
    return np.c_[uv, q[:, 2]]


@dataclass(frozen=True, slots=True)
class EgoCamera:
    rig: int  # OAK rig index.
    cam: int  # 0 rgb, 1 left, 2 right.
    K: Float64[ndarray, "3 3"]  # Encoded-image pinhole matrix.
    distortion: Float64[ndarray, "c"]  # OpenCV order (k1, k2, p1, p2, k3, k4, k5, k6, s1-s4, tx, ty); the viewer's Pinhole ignores it.
    size: tuple[int, int]  # Image width, height.
    rig_T_cam: Float64[ndarray, "4 4"]  # Factory stereo extrinsic; identity for the rgb rig frame.
    times_s: Float64[ndarray, "p"]  # video_time of the calibrated rig poses (Quest head samples shifted by the OAK offset).
    world_T_rig: Float64[ndarray, "p 4 4"]  # Head pose x OAK mount, RDF rig to Quest world.


def project_ego(camera: EgoCamera, times_s: Float64[ndarray, "n"], xyz: Float32[ndarray, "n j 3"]) -> Float32[ndarray, "n j 2"]:
    """World points at each time through the rig pose (lerp + slerp) and the full OAK lens model; missing when unprojectable.

    Points behind the camera or outside the calibrated field (beyond the image corners' rays, where the rational model can fold
    back into the image) stay NaN, as do times outside the pose track or across a pose gap.
    """
    from scipy.spatial.transform import Rotation, Slerp

    right = np.clip(np.searchsorted(camera.times_s, times_s), 1, len(camera.times_s) - 1)
    left = right - 1
    gap = camera.times_s[right] - camera.times_s[left]
    usable = (times_s >= camera.times_s[0]) & (times_s <= camera.times_s[-1]) & (gap <= EGO_POSE_MAX_GAP_S)
    clipped = np.clip(times_s, camera.times_s[0], camera.times_s[-1])
    alpha = ((clipped - camera.times_s[left]) / np.maximum(gap, 1e-12))[:, None]
    translation = (1.0 - alpha) * camera.world_T_rig[left, :3, 3] + alpha * camera.world_T_rig[right, :3, 3]
    rotation = Slerp(camera.times_s, Rotation.from_matrix(camera.world_T_rig[:, :3, :3]))(clipped).as_matrix()
    world_T_cam_r = rotation @ camera.rig_T_cam[:3, :3]
    world_T_cam_t = translation + np.einsum("nij,j->ni", rotation, camera.rig_T_cam[:3, 3])
    local = np.einsum("njk,nkl->njl", xyz.astype(np.float64) - world_T_cam_t[:, None, :], world_T_cam_r)  # R^T (x - t), row vectors
    width, height = camera.size
    corners = np.array([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]]).reshape(-1, 1, 2)
    max_radius = 1.05 * float(np.linalg.norm(cv2.undistortPoints(corners, camera.K, camera.distortion).reshape(-1, 2), axis=1).max())
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = local[..., :2] / local[..., 2:3]
    good = usable[:, None] & (local[..., 2] > 1e-3) & np.isfinite(normalized).all(-1) & (np.linalg.norm(normalized, axis=-1) <= max_radius)
    uv = np.full((*xyz.shape[:2], 2), np.nan, dtype=np.float32)
    if good.any():
        pixels, _ = cv2.projectPoints(local[good].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.distortion)
        uv[good] = pixels.reshape(-1, 2)
    return uv


def triangulate_frames(
    cameras: list[Camera], uv: Float64[ndarray, "n v j 2"], scores: Float64[ndarray, "n v j"], threshold: float = INLIER_PX, chunk: int = 2048
) -> tuple[Float32[ndarray, "n j 3"], UInt8[ndarray, "n j"], Float32[ndarray, "n j"]]:
    """All view pairs, >=0.5° ray separation, weighted DLT and two gated refits, for every frame and joint at once.

    float64 on the GPU when there is one: each DLT is the smallest eigenvector of its 4x4 normal matrix, which agrees
    with the per-frame SVD it replaced to ~1e-13 m (cc808233 ep001: 7 s -> ~0.2 s per episode).
    """
    frames, views, joints = scores.shape
    xyz, counts, confidence = (
        np.full((frames, joints, 3), np.nan, np.float32),
        np.zeros((frames, joints), np.uint8),
        np.zeros((frames, joints), np.float32),
    )
    if views < 2 or not frames:
        return xyz, counts, confidence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor(array: ndarray) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64, device=device)

    inverse = tensor(np.stack([np.linalg.inv(c.world_T_cam) for c in cameras]))
    rotation = tensor(np.stack([c.world_T_cam[:3, :3] for c in cameras]))
    lens = tensor(np.stack([c.K[:2, :2] for c in cameras]))
    centre = tensor(np.stack([c.K[:2, 2] for c in cameras]))
    radial = tensor(np.array([c.radial for c in cameras]))
    cos_min = float(np.cos(np.deg2rad(0.5)))

    def solve(observed_np: Float64[ndarray, "c v j 2"], scores_np: Float64[ndarray, "c v j"]) -> tuple[ndarray, ndarray, ndarray]:
        norm_np = np.stack(
            [
                cv2.undistortPoints(
                    np.nan_to_num(observed_np[:, v], nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1, 2), c.K, np.array([*c.radial, 0.0, 0.0, 0.0])
                ).reshape(len(observed_np), joints, 2)
                for v, c in enumerate(cameras)
            ],
            axis=1,
        )
        observed, score, norm = tensor(observed_np), tensor(scores_np), tensor(norm_np)
        valid = (score >= 0.3) & torch.isfinite(score) & torch.isfinite(observed).all(-1) & torch.isfinite(norm).all(-1)
        norm = torch.where(torch.isfinite(norm), norm, 0.0)
        clean = torch.where(valid, score, 0.0)
        root = torch.sqrt(clean)
        rows = torch.stack(
            [
                norm[..., 0, None] * inverse[None, :, None, 2] - inverse[None, :, None, 0],
                norm[..., 1, None] * inverse[None, :, None, 2] - inverse[None, :, None, 1],
            ],
            dim=3,
        )  # f v j 2 4

        def fit(mask: Tensor) -> Tensor:
            weighted = rows * torch.where(mask, root, 0.0)[..., None, None]
            gram = torch.einsum("fvjri,fvjrk->fjik", weighted, weighted).reshape(-1, 4, 4)
            vectors = torch.cat([torch.linalg.eigh(part)[1][..., 0] for part in gram.split(32768)])
            vectors = vectors.reshape(len(observed_np), joints, 4)
            return vectors[..., :3] / vectors[..., 3:4]

        def residual(points: Tensor) -> Tensor:
            q = torch.einsum("vik,fjk->fvji", inverse[:, :3, :3], points) + inverse[None, :, None, :3, 3]
            xy = q[..., :2] / q[..., 2:3]
            r2 = (xy * xy).sum(-1)
            distortion = 1.0 + radial[None, :, None, 0] * r2 + radial[None, :, None, 1] * r2 * r2
            pixels = torch.einsum("vik,fvjk->fvji", lens, xy * distortion[..., None]) + centre[None, :, None, :]
            error = torch.linalg.norm(pixels - observed, dim=-1)
            return torch.where((q[..., 2] > 1e-5) & torch.isfinite(error), error, torch.inf)

        rays = torch.einsum("vik,fvjk->fvji", rotation, torch.cat([norm, torch.ones_like(norm[..., :1])], -1))
        rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True).clamp(min=1e-12)
        best_mask = torch.zeros_like(valid)
        best_count = torch.zeros(valid.shape[0], joints, dtype=torch.int64, device=device)
        best_weight = torch.zeros(valid.shape[0], joints, dtype=torch.float64, device=device)
        best_error = torch.full((valid.shape[0], joints), torch.inf, dtype=torch.float64, device=device)
        for a, b in combinations(range(views), 2):
            pair = valid[:, a] & valid[:, b] & ((rays[:, a] * rays[:, b]).sum(-1) < cos_min)
            seed = torch.zeros_like(valid)
            seed[:, a], seed[:, b] = pair, pair
            res = residual(fit(seed))
            inliers = valid & (res <= threshold) & pair[:, None, :]
            count, weight = inliers.sum(1), (inliers * clean).sum(1)
            mean = torch.where(inliers, res, 0.0).sum(1) / count.clamp(min=1)
            better = (count > best_count) | (
                (count == best_count) & ((weight > best_weight + 1e-6) | (((weight - best_weight).abs() <= 1e-6) & (mean < best_error)))
            )
            best_mask = torch.where(better[:, None, :], inliers, best_mask)
            best_count, best_weight, best_error = (
                torch.where(better, new, old) for new, old in ((count, best_count), (weight, best_weight), (mean, best_error))
            )
        inliers = best_mask & valid & (residual(fit(best_mask)) <= threshold)
        points = fit(inliers)
        inliers &= residual(points) <= threshold
        count = inliers.sum(1)
        good = (count >= 2) & torch.isfinite(points).all(-1)
        return (
            torch.where(good[..., None], points, torch.nan).float().cpu().numpy(),
            torch.where(good, count, 0).to(torch.uint8).cpu().numpy(),
            torch.where(good, torch.where(inliers, clean, 0.0).sum(1) / count.clamp(min=1), 0.0).float().cpu().numpy(),
        )

    for begin in range(0, frames, chunk):
        stop = min(begin + chunk, frames)
        xyz[begin:stop], counts[begin:stop], confidence[begin:stop] = solve(uv[begin:stop], scores[begin:stop])
    return xyz, counts, confidence
