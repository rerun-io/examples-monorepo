"""Pure camera geometry shared by selfcap-curate and selfcap-soma: projection through pinhole and OAK lens models, and robust
multi-view triangulation. It holds no catalog or layer I/O."""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from torch import Tensor

from dataforge.interpolation import interpolate_poses

INLIER_PX: float = 8.0
"""A view supports a point when it reprojects within this many pixels."""
MIN_DETECTION_SCORE: float = 0.3
"""Permissive detection acceptance; triangulation still requires two 8 px inliers."""
HEADSET_COVERED: Int64[ndarray, "k"] = np.r_[0:3, 23:91]
"""Body nose/eyes and the 68 face landmarks: every wearer has a Quest on, and RTMW and Sapiens2 both place these on its
front plate with high confidence (cc808233 ep001 head crops), so 3D omits them and readers of 2D rows must not use them."""
EGO_POSE_MAX_GAP_S: float = 0.1
"""Rig poses further apart than this are not interpolated across (the overlay is left missing)."""
MIN_RAY_SEPARATION_DEG: float = 5.0
"""Smallest angle between the lines of two supporting rays: phones at almost one viewpoint, or facing each other across the
wearer (f28dc7af rig_01/rig_03, rays 179.5 degrees apart), leave depth free along the line. Near-parallel pairs pass the pixel
gate with depth errors of metres; at 5 degrees and 2 m a 5 px error moves depth ~0.13 m."""
CONFIDENCE_LAMBDA: float = 40.0
"""Kineo's reprojection decay (arXiv 2510.24464 eq. 12): a miss of 1% of the focal length scales a view's score by exp(-0.4)."""
ROBUST_ROUNDS: int = 6
"""Reweighting rounds over every view after the consensus seed."""
ROBUST_OUTER_GATE: float = 3.0
"""Views farther than this many inlier gates (24 px) from the point get no weight: a wrong person or a wild guess does not
pull it. f28dc7af: RTMW-x ears 154/20 -> 17/17 mm from SAM3DFast, steady on the head (4/5 mm); shoulders 14/39 -> 11/10 mm."""


@dataclass(frozen=True, slots=True)
class Camera:
    K: Float64[ndarray, "3 3"]
    """Encoded-image pinhole matrix."""
    world_T_cam: Float64[ndarray, "4 4"]
    """RDF camera to Quest world, metres."""
    radial: tuple[float, float]
    """OpenCV k1,k2; refined iOS pinholes are rectified (zero)."""
    entity: str
    """Camera node for the per-view overlay."""
    size: tuple[int, int] = (1920, 1080)
    """Calibrated image width and height."""


@dataclass(frozen=True, slots=True)
class TriangulationBatch:
    xyz: Float32[ndarray, "n j 3"]
    """World metres; NaN for missing or degenerate joints."""
    n_views: UInt8[ndarray, "n j"]
    """Positive-depth inlier count, zero when unusable."""
    confidence: Float32[ndarray, "n j"]
    """Kineo pairwise reprojection score over the views that saw the joint (0-1)."""
    inliers: Bool[ndarray, "n v j"]
    """The views supporting each joint, in camera order."""
    max_reprojection_px: Float32[ndarray, "n j"]
    """Largest error over the supporting views, NaN when missing."""


def reprojection_confidence(
    errors: Float64[ndarray, "n v j"], scores: Float64[ndarray, "n v j"], focal: Float64[ndarray, "v"], seen: Bool[ndarray, "n v j"]
) -> Float32[ndarray, "n j"]:
    """Kineo's 3D confidence (arXiv 2510.24464 eq. 11-14): each view's score times exp(-lambda x miss / focal), each pair of
    views the geometric mean of the two, averaged over the pairs of views that saw the joint; 0 with fewer than two. The
    paper averages over every camera pair; here a view that did not detect the joint does not lower it."""
    shrunk = np.where(seen, scores * np.exp(-CONFIDENCE_LAMBDA * np.nan_to_num(errors, nan=1e9, posinf=1e9) / focal[:, None]), 0.0)
    roots = np.sqrt(shrunk)
    pair_sum = (roots.sum(1) ** 2 - shrunk.sum(1)) / 2.0  # sum over i<j of sqrt(a_i a_j)
    count = seen.sum(1)
    pairs = count * (count - 1) / 2.0
    return np.where(count >= 2, pair_sum / np.maximum(pairs, 1.0), 0.0).astype(np.float32)


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
    rig: int
    """OAK rig index."""
    cam: int
    """0 rgb, 1 left, 2 right."""
    K: Float64[ndarray, "3 3"]
    """Encoded-image pinhole matrix."""
    distortion: Float64[ndarray, "c"]
    """OpenCV order (k1, k2, p1, p2, k3, k4, k5, k6, s1-s4, tx, ty); the viewer's Pinhole ignores it."""
    size: tuple[int, int]
    """Image width, height."""
    rig_T_cam: Float64[ndarray, "4 4"]
    """Factory stereo extrinsic; identity for the rgb rig frame."""
    times_s: Float64[ndarray, "p"]
    """video_time of the calibrated rig poses (Quest head samples shifted by the OAK offset)."""
    world_T_rig: Float64[ndarray, "p 4 4"]
    """Head pose x OAK mount, RDF rig to Quest world."""


def camera_transforms(
    times_s: Float64[ndarray, "n"],
    stamps: Float64[ndarray, "p"],
    world_T_rig: Float64[ndarray, "p 4 4"],
    rig_T_cam: Float64[ndarray, "4 4"],
) -> Float64[ndarray, "n 4 4"]:
    """Interpolate rig translation/rotation, then apply the camera lever arm; gaps over 100 ms stay missing."""
    return interpolate_poses(stamps, world_T_rig[:, :3, 3], Rotation.from_matrix(world_T_rig[:, :3, :3]), times_s, EGO_POSE_MAX_GAP_S).transforms @ rig_T_cam


def project_ego(camera: EgoCamera, times_s: Float64[ndarray, "n"], xyz: Float32[ndarray, "n j 3"]) -> Float32[ndarray, "n j 2"]:
    """World points at each time through the rig pose (lerp + slerp) and the full OAK lens model; missing when unprojectable.

    Points behind the camera or outside the calibrated field (beyond the image corners' rays, where the rational model can fold
    back into the image) stay NaN, as do times outside the pose track or across a pose gap.
    """
    transforms: Float64[ndarray, "n 4 4"] = camera_transforms(times_s, camera.times_s, camera.world_T_rig, camera.rig_T_cam)
    local = np.einsum("njk,nkl->njl", xyz.astype(np.float64) - transforms[:, None, :3, 3], transforms[:, :3, :3])
    width, height = camera.size
    corners = np.array([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]]).reshape(-1, 1, 2)
    max_radius = 1.05 * float(np.linalg.norm(cv2.undistortPoints(corners, camera.K, camera.distortion).reshape(-1, 2), axis=1).max())
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = local[..., :2] / local[..., 2:3]
    good = (local[..., 2] > 1e-3) & np.isfinite(normalized).all(-1) & (np.linalg.norm(normalized, axis=-1) <= max_radius)
    uv = np.full((*xyz.shape[:2], 2), np.nan, dtype=np.float32)
    if good.any():
        pixels, _ = cv2.projectPoints(local[good].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.distortion)
        uv[good] = pixels.reshape(-1, 2)
    return uv


def triangulate_frames(
    cameras: list[Camera], uv: Float64[ndarray, "n v j 2"], scores: Float64[ndarray, "n v j"], threshold: float = INLIER_PX, chunk: int = 2048
) -> TriangulationBatch:
    """Best view pair as the seed (lines >= MIN_RAY_SEPARATION_DEG apart), two gated refits, then robust reweighting over
    every view; kept where >= 2 views end within ``threshold`` and their lines meet at >= MIN_RAY_SEPARATION_DEG.
    Confidence is reprojection_confidence over every view that detected the joint. For every frame and joint at once.

    float64 on the GPU when there is one: each DLT is the smallest eigenvector of its 4x4 normal matrix, which agrees
    with the per-frame SVD it replaced to ~1e-13 m (cc808233 ep001: 7 s -> ~0.2 s per episode).
    """
    frames, views, joints = scores.shape
    xyz, counts, confidence = (
        np.full((frames, joints, 3), np.nan, np.float32),
        np.zeros((frames, joints), np.uint8),
        np.zeros((frames, joints), np.float32),
    )
    masks: Bool[ndarray, "n v j"] = np.zeros((frames, views, joints), dtype=bool)
    max_errors: Float32[ndarray, "n j"] = np.full((frames, joints), np.nan, dtype=np.float32)
    if views < 2 or not frames:
        return TriangulationBatch(xyz, counts, confidence, masks, max_errors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor(array: ndarray) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64, device=device)

    inverse = tensor(np.stack([np.linalg.inv(c.world_T_cam) for c in cameras]))
    rotation = tensor(np.stack([c.world_T_cam[:3, :3] for c in cameras]))
    lens = tensor(np.stack([c.K[:2, :2] for c in cameras]))
    centre = tensor(np.stack([c.K[:2, 2] for c in cameras]))
    radial = tensor(np.array([c.radial for c in cameras]))
    world_centre = tensor(np.stack([c.world_T_cam[:3, 3] for c in cameras]))
    cos_min = float(np.cos(np.deg2rad(MIN_RAY_SEPARATION_DEG)))
    focal = np.array([c.K[0, 0] for c in cameras])

    def solve(observed_np: Float64[ndarray, "c v j 2"], scores_np: Float64[ndarray, "c v j"]) -> TriangulationBatch:
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
        valid = (score >= MIN_DETECTION_SCORE) & torch.isfinite(score) & torch.isfinite(observed).all(-1) & torch.isfinite(norm).all(-1)
        norm = torch.where(torch.isfinite(norm), norm, 0.0)
        clean = torch.where(valid, score, 0.0)
        rows = torch.stack(
            [
                norm[..., 0, None] * inverse[None, :, None, 2] - inverse[None, :, None, 0],
                norm[..., 1, None] * inverse[None, :, None, 2] - inverse[None, :, None, 1],
            ],
            dim=3,
        )  # f v j 2 4

        def fit_weighted(weights: Tensor) -> Tensor:
            weighted = rows * torch.sqrt(weights)[..., None, None]
            gram = torch.einsum("fvjri,fvjrk->fjik", weighted, weighted).reshape(-1, 4, 4)
            vectors = torch.cat([torch.linalg.eigh(part)[1][..., 0] for part in gram.split(32768)])
            vectors = vectors.reshape(len(observed_np), joints, 4)
            return vectors[..., :3] / vectors[..., 3:4]

        def fit(mask: Tensor) -> Tensor:
            return fit_weighted(torch.where(mask, clean, 0.0))

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
            pair = valid[:, a] & valid[:, b] & ((rays[:, a] * rays[:, b]).sum(-1).abs() < cos_min)  # the lines' angle: facing phones too
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
        seeded = (inliers.sum(1) >= 2) & torch.isfinite(points).all(-1)
        # From the consensus seed, every phone with the point counts: score x Cauchy(threshold) weights, phones beyond
        # ROBUST_OUTER_GATE x threshold left out. Phones that guess a hidden joint a little differently (f28dc7af: RTMW-x
        # puts the ears ~15 px lower from the front than from the side) then meet in between instead of one pair winning.
        for _ in range(ROBUST_ROUNDS):
            error = residual(points)
            weights = torch.where(valid & (error <= ROBUST_OUTER_GATE * threshold), clean / (1.0 + (error / threshold) ** 2), 0.0)
            refined = fit_weighted(weights)
            points = torch.where((seeded & torch.isfinite(refined).all(-1))[..., None], refined, points)
        errors: Tensor = residual(points)
        inliers = valid & (errors <= threshold)
        count = inliers.sum(1)
        to_point = points[:, None] - world_centre[None, :, None]
        to_point = to_point / torch.linalg.norm(to_point, dim=-1, keepdim=True).clamp(min=1e-12)
        lines = torch.einsum("fajk,fbjk->fabj", to_point, to_point).abs()
        pairs = inliers[:, :, None] & inliers[:, None] & ~torch.eye(views, dtype=torch.bool, device=device)[None, :, :, None]
        widest = torch.where(pairs, lines, 1.0).amin(dim=(1, 2))
        good = seeded & (count >= 2) & (widest < cos_min) & torch.isfinite(points).all(-1)
        return TriangulationBatch(
            torch.where(good[..., None], points, torch.nan).float().cpu().numpy(),
            torch.where(good, count, 0).to(torch.uint8).cpu().numpy(),
            np.where(good.cpu().numpy(), reprojection_confidence(errors.cpu().numpy(), clean.cpu().numpy(), focal, valid.cpu().numpy()), 0.0),
            (inliers & good[:, None, :]).cpu().numpy(),
            torch.where(good, torch.where(inliers, errors, 0.0).amax(1), torch.nan).float().cpu().numpy(),
        )

    for begin in range(0, frames, chunk):
        stop = min(begin + chunk, frames)
        result = solve(uv[begin:stop], scores[begin:stop])
        xyz[begin:stop] = result.xyz
        counts[begin:stop] = result.n_views
        confidence[begin:stop] = result.confidence
        masks[begin:stop] = result.inliers
        max_errors[begin:stop] = result.max_reprojection_px
    return TriangulationBatch(xyz, counts, confidence, masks, max_errors)
