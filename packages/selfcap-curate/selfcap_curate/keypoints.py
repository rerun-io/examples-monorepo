"""COCO-133 wearer selection, confidence-weighted triangulation and layer writing."""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias, overload

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from dataforge.hands import log_keypoints2d, log_keypoints3d, log_projections
from dataforge.logging_toolkit import time_column
from dataforge.schema import coco133_uv_projected_path, coco133_xyz_path, pinhole_path
from dataforge.writing import atomic_write
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from simplecv.catalog_rig_layout import catalog_components
from simplecv.data.skeleton.coco_133 import COCO_133_LINKS
from simplecv.rrd_query_utils import first_valid_value
from torch import Tensor

from selfcap_curate.catalog_io import Segment, read_catalog_hands, scalar
from selfcap_curate.detection_io import COCO133_CLASS
from selfcap_curate.display_style import DISPLAY_JOINTS, log_display_skeleton
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import HAND_BASE, HAND_LATENCY_MS, TimeSync, oak_quest_offset_ms

PoseModel: TypeAlias = Literal["rtmw", "sapiens2"]

INLIER_PX: float = 8.0

DISPLAY_CONFIDENCE: float = 0.7
"""Lower bound of the display colour scale; never a visibility threshold."""
MIN_DETECTION_SCORE: float = 0.3
"""Permissive detection acceptance; triangulation still requires two 8 px inliers."""
ONE_EURO_MIN_CUTOFF_HZ: float = 1.0
"""One Euro cutoff at rest; with ONE_EURO_BETA, forward + backward cut cc808233 ep001 jitter 6x (median 11.8 -> 1.9 mm)."""
ONE_EURO_BETA: float = 2.0
"""Cutoff increase per m/s of joint speed, so fast moves are followed closely (12 mm median pull in the fastest 10%)."""
SMOOTH_MIN_RATE_HZ: float = 10.0
"""Plans sampled slower than this stay raw: a 1 Hz cutoff needs samples well above it."""
HEADSET_COVERED: Int64[ndarray, "k"] = np.r_[0:3, 23:91]
"""Body nose/eyes and the 68 face landmarks: every wearer has a Quest on, and RTMW and Sapiens2 both place these on its
front plate with high confidence (cc808233 ep001 head crops), so they are written as missing."""


@dataclass(frozen=True, slots=True)
class Camera:
    K: Float64[ndarray, "3 3"]  # Encoded-image pinhole matrix.
    world_T_cam: Float64[ndarray, "4 4"]  # RDF camera to Quest world, metres.
    radial: tuple[float, float]  # OpenCV k1,k2; refined iOS pinholes are rectified (zero).
    entity: str  # Camera node for the per-view overlay.
    size: tuple[int, int] = (1920, 1080)  # Calibrated image width and height.


@dataclass(frozen=True, slots=True)
class Triangulated:
    xyz: Float32[ndarray, "j 3"]  # World metres; NaN for missing/degenerate joints.
    n_views: UInt8[ndarray, "j"]  # Positive-depth inlier count, zero when unusable.
    confidence: Float32[ndarray, "j"]  # Mean detector score of the supporting views, not geometric certainty.
    inliers: Bool[ndarray, "v j"] | None = None
    max_reprojection_px: Float32[ndarray, "j"] | None = None


@dataclass(frozen=True, slots=True)
class TriangulationBatch:
    xyz: Float32[ndarray, "n j 3"]
    n_views: UInt8[ndarray, "n j"]
    confidence: Float32[ndarray, "n j"]
    inliers: Bool[ndarray, "n v j"]
    max_reprojection_px: Float32[ndarray, "n j"]


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


EGO_POSE_MAX_GAP_S: float = 0.1
"""Rig poses further apart than this are not interpolated across (the overlay is left missing)."""


def camera_transforms(
    times_s: Float64[ndarray, "n"], stamps: Float64[ndarray, "p"],
    world_T_rig: Float64[ndarray, "p 4 4"], rig_T_cam: Float64[ndarray, "4 4"],
) -> Float64[ndarray, "n 4 4"]:
    """Interpolate rig translation/rotation, then apply the camera lever arm; gaps over 100 ms stay missing."""
    output: Float64[ndarray, "n 4 4"] = np.full((len(times_s), 4, 4), np.nan)
    if len(stamps) < 2:
        return output
    from scipy.spatial.transform import Rotation, Slerp

    right = np.clip(np.searchsorted(stamps, times_s), 1, len(stamps) - 1)
    left = right - 1
    gap = stamps[right] - stamps[left]
    usable = (times_s >= stamps[0]) & (times_s <= stamps[-1]) & (gap <= EGO_POSE_MAX_GAP_S)
    clipped = np.clip(times_s, stamps[0], stamps[-1])
    alpha = ((clipped - stamps[left]) / np.maximum(gap, 1e-12))[:, None]
    translation = (1.0 - alpha) * world_T_rig[left, :3, 3] + alpha * world_T_rig[right, :3, 3]
    rotation = Slerp(stamps, Rotation.from_matrix(world_T_rig[:, :3, :3]))(clipped).as_matrix()
    world_T_cam_r = rotation @ rig_T_cam[:3, :3]
    world_T_cam_t = translation + np.einsum("nij,j->ni", rotation, rig_T_cam[:3, 3])
    output[:] = np.eye(4)
    output[:, :3, :3], output[:, :3, 3] = world_T_cam_r, world_T_cam_t
    output[~usable] = np.nan
    return output


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


def read_ego_cameras(calibration: DatasetEntry, base: DatasetEntry, segment: str, oak_rig: int) -> list[EgoCamera]:
    """The OAK cameras whose pose the calibration layer gives: one query for the rig pose track and stereo, one for the lenses."""
    rig = f"/world/rig_{oak_rig:02d}"
    cams = {cam: f"{rig}/cam_{cam:02d}" for cam in (0, 1, 2)}
    view = calibration.filter_segments(segment)
    if not any(item.entity_path == rig and item.field == "mat3x3" and not item.is_static for item in catalog_components(view.schema())):
        return []
    poses = (
        view
        .filter_contents([rig, *cams.values()])
        .reader(index="video_time")
        .select("video_time", f"{rig}:Transform3D:mat3x3", f"{rig}:Transform3D:translation")
        .to_arrow_table()
    )
    poses = poses.filter(poses[f"{rig}:Transform3D:mat3x3"].is_valid())
    if not len(poses):
        return []  # no calibrated OAK mount for this episode
    times_s: Float64[ndarray, "p"] = poses["video_time"].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64) * 1e-9
    world_T_rig: Float64[ndarray, "p 4 4"] = np.tile(np.eye(4), (len(poses), 1, 1))
    world_T_rig[:, :3, :3] = np.asarray(poses[f"{rig}:Transform3D:mat3x3"].to_pylist(), dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
    world_T_rig[:, :3, 3] = np.asarray(poses[f"{rig}:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    static = calibration.filter_segments(segment).filter_contents(list(cams.values())).reader(index=None).to_arrow_table()
    lenses = base.filter_segments(segment).filter_contents([f"{path}/pinhole" for path in cams.values()]).reader(index=None).to_arrow_table()

    def numeric(table, name: str) -> Float64[ndarray, "k"] | None:  # noqa: ANN001
        return np.asarray(first_valid_value(table[name]), dtype=np.float64).reshape(-1) if name in table.column_names and table[name].null_count < len(table) else None

    result: list[EgoCamera] = []
    for cam, path in cams.items():
        matrix, size = numeric(lenses, f"{path}/pinhole:Pinhole:image_from_camera"), numeric(lenses, f"{path}/pinhole:Pinhole:resolution")
        distortion = numeric(lenses, f"{path}/pinhole:simplecv.components.DistortionCoefficients")
        if matrix is None or size is None:
            continue
        rig_T_cam = np.eye(4)
        if cam != 0:
            rotation, translation = numeric(static, f"{path}:Transform3D:mat3x3"), numeric(static, f"{path}:Transform3D:translation")
            if rotation is None or translation is None:
                continue
            rig_T_cam[:3, :3], rig_T_cam[:3, 3] = rotation.reshape(3, 3).T, translation
        result.append(
            EgoCamera(oak_rig, cam, matrix.reshape(3, 3).T, distortion if distortion is not None else np.zeros(5), (int(size[0]), int(size[1])), rig_T_cam, times_s, world_T_rig)
        )
    return result


QUEST_HANDS_UV: str = "quest_coco133_uv_projected"
"""Quest hand joints projected into an OAK camera, beneath its pinhole (the viewer cannot apply the OAK lens model)."""
QUEST_HANDS_CLASS: rr.ClassDescription = rr.ClassDescription(
    info=rr.AnnotationInfo(id=0, label="Quest hands (projected)", color=(0, 0, 255)), keypoint_connections=COCO_133_LINKS
)
"""Blue like the base's /world/gt Quest hands; the points carry no colours of their own, so joints and bones take it."""


def ego_quest_hands(segment: Segment, sync: TimeSync) -> tuple[Float64[ndarray, "q"], Float32[ndarray, "q 133 3"]] | None:
    """Quest hand joints on video_time as the OAK sees them: capture time + OAK offset - hand latency (solve_oak's mapping)."""
    qp = oak_quest_offset_ms(sync)
    if qp is None:
        return None
    source = read_catalog_hands(segment)
    positions = source.positions.astype(np.float32)
    positions[(positions == 0.0).all(-1)] = np.nan
    return source.times_s + (qp - HAND_LATENCY_MS) / 1000.0, positions


@dataclass(frozen=True, slots=True)
class DisplaySkeleton:
    positions: Float32[ndarray, "n 133 3"]
    """Exo ears and upper body, plus Quest hands with triangulated fallback."""
    confidence: Float32[ndarray, "n 133"]
    """Geometric-support adjusted scores; Quest tracking is assigned 1.0 because no joint score exists."""
    quest_tracked: Bool[ndarray, "n 133"]
    """Hand joints supplied by valid Quest tracking at this timestamp."""


def display_skeleton(
    times: Float64[ndarray, "n"],
    xyz: Float32[ndarray, "n 133 3"],
    confidence: Float32[ndarray, "n 133"],
    quest_hands: tuple[Float64[ndarray, "q"], Float32[ndarray, "q 133 3"]] | None,
) -> DisplaySkeleton:
    """Exo ears/upper body plus clock-corrected Quest hands; confident exo hands fill tracking gaps."""
    selected: Bool[ndarray, "133"] = np.isin(np.arange(133), DISPLAY_JOINTS)
    output: Float32[ndarray, "n 133 3"] = np.where(selected[None, :, None], xyz, np.float32(np.nan))
    tracked: Bool[ndarray, "n 133"] = np.zeros(xyz.shape[:2], dtype=bool)
    if quest_hands is not None:
        stamps: Float64[ndarray, "q"] = quest_hands[0]
        hands: Float32[ndarray, "q 133 3"] = quest_hands[1]
        for start in (91, 112):
            valid: Bool[ndarray, "q"] = np.isfinite(hands[:, start : start + 21]).all(axis=(1, 2))
            for joint in range(start, start + 21):
                points: Float32[ndarray, "n 3"] = interpolate(times, stamps, hands[:, joint].astype(np.float64), valid, 0.04).astype(np.float32)
                available: Bool[ndarray, "n"] = np.isfinite(points).all(axis=1)
                output[available, joint] = points[available]
                tracked[available, joint] = True
    return DisplaySkeleton(output, np.where(tracked, 1.0, confidence).astype(np.float32), tracked)


def select_wearers(
    camera: Camera, anchors: Float64[ndarray, "n 3 3"], boxes: Float64[ndarray, "n p 4"], scores: Float64[ndarray, "n p"]
) -> Int64[ndarray, "n"]:
    """Per frame: nearest head/wrist rectangle distance, head weight two, reject cost >=0.5; -1 when no box qualifies."""
    frames, people = boxes.shape[:2]
    if not people:
        return np.full(frames, -1, np.int64)
    projected = project(camera, anchors.reshape(-1, 3)).reshape(frames, 3, 3)
    good: Bool[ndarray, "n 3"] = np.isfinite(projected[..., :2]).all(-1) & (projected[..., 2] > 0.0)
    uv = projected[..., :2]
    weights = np.where(good, np.array([2.0, 1.0, 1.0]), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        size = boxes[..., 2:] - boxes[..., :2]
        usable = (scores >= 0.3) & np.isfinite(boxes).all(-1) & ~np.any(size <= 0.0, axis=-1)
        diagonal = np.maximum(np.linalg.norm(size, axis=-1), 1.0)[..., None]
        outside = np.maximum(np.maximum(boxes[:, :, None, :2] - uv[:, None], uv[:, None] - boxes[:, :, None, 2:]), 0.0)
        centre = np.linalg.norm(uv[:, None] - (boxes[:, :, None, :2] + boxes[:, :, None, 2:]) / 2.0, axis=-1) / diagonal
        term = np.linalg.norm(outside, axis=-1) / diagonal + 0.05 * centre
        cost = np.where(good[:, None], term * weights[:, None], 0.0).sum(-1) / weights.sum(-1)[:, None] - 0.01 * scores
    cost = np.where(usable & good.any(-1)[:, None], cost, np.inf)
    best = np.argmin(cost, axis=1)
    return np.where(np.take_along_axis(cost, best[:, None], 1)[:, 0] < 0.5, best, -1)


def select_wearer(camera: Camera, anchors: Float64[ndarray, "3 3"], boxes: Float64[ndarray, "p 4"], scores: Float64[ndarray, "p"]) -> int:
    """One frame of select_wearers."""
    return int(select_wearers(camera, anchors[None], boxes[None], scores[None])[0])


def triangulate_frames(
    cameras: list[Camera], uv: Float64[ndarray, "n v j 2"], scores: Float64[ndarray, "n v j"], threshold: float = INLIER_PX, chunk: int = 2048
) -> TriangulationBatch:
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
    cos_min = float(np.cos(np.deg2rad(0.5)))

    def solve(observed_np: Float64[ndarray, "c v j 2"], scores_np: Float64[ndarray, "c v j"]) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
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
            (inliers & good[:, None, :]).cpu().numpy(),
            torch.where(good, torch.where(inliers, residual(points), 0.0).amax(1), torch.nan).float().cpu().numpy(),
        )

    for begin in range(0, frames, chunk):
        stop = min(begin + chunk, frames)
        xyz[begin:stop], counts[begin:stop], confidence[begin:stop], masks[begin:stop], max_errors[begin:stop] = solve(uv[begin:stop], scores[begin:stop])
    return TriangulationBatch(xyz, counts, confidence, masks, max_errors)


def triangulate(cameras: list[Camera], uv: Float64[ndarray, "v j 2"], scores: Float64[ndarray, "v j"], threshold: float = INLIER_PX) -> Triangulated:
    """One frame of triangulate_frames."""
    batch: TriangulationBatch = triangulate_frames(cameras, uv[None], scores[None], threshold)
    return Triangulated(batch.xyz[0], batch.n_views[0], batch.confidence[0], batch.inliers[0], batch.max_reprojection_px[0])


@dataclass(frozen=True, slots=True)
class ViewDetections:
    boxes: Float64[ndarray, "n p 4"]  # Candidate rectangles, in encoded-image pixels.
    box_scores: Float64[ndarray, "n p"]  # Person detection confidence.
    xy: Float64[ndarray, "n p 133 2"]  # RTMW COCO-133 points, image coordinates.
    scores: Float64[ndarray, "n p 133"]  # Joint confidence, joint confidence.

    def __post_init__(self) -> None:
        if self.boxes.shape[:2] != self.box_scores.shape or self.xy.shape[:2] != self.box_scores.shape or self.scores.shape != self.xy.shape[:3]:
            raise ValueError("candidate/frame dimensions must agree")


@dataclass(frozen=True, slots=True)
class Keypoints2D:
    times: Float64[ndarray, "n"]  # Aligned video_time sample grid, seconds.
    views: dict[str, ViewDetections]  # Full phone identity to candidate detections.
    rate: float  # Requested sample rate, in Hz.

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.times).all()
            or not np.all(np.diff(self.times) > 0.0)
            or not np.isfinite(self.rate)
            or self.rate <= 0.0
            or any(len(v.xy) != len(self.times) for v in self.views.values())
        ):
            raise ValueError("keypoint streams need matching frames, ordered finite times and positive rate")


@dataclass(frozen=True, slots=True)
class SelectedView:
    xy: Float32[ndarray, "n 133 2"]  # Chosen wearer's per-frame overlay, NaN when absent.
    scores: Float64[ndarray, "n 133"]  # Unmodified joint confidence.


def exo_display_observed(
    pixels: Float32[ndarray, "n 133 2"],
    selected: SelectedView,
    inliers: Bool[ndarray, "n 133"],
    quest_tracked: Bool[ndarray, "n 133"],
) -> Bool[ndarray, "n 133"]:
    """Quest joints use independent raw-camera evidence within 15 px; exo joints retain the 8 px inlier rule."""
    error: Float32[ndarray, "n 133"] = np.linalg.norm(pixels - selected.xy, axis=-1)
    quest_observed: Bool[ndarray, "n 133"] = (selected.scores >= DISPLAY_CONFIDENCE) & (error <= 15.0)
    return np.where(quest_tracked, quest_observed, inliers & (error <= INLIER_PX))


@dataclass(frozen=True, slots=True)
class Keypoints3D:
    times: Float64[ndarray, "n"]  # Aligned sample grid in seconds, with empty rows preserved.
    xyz: Float32[ndarray, "n 133 3"]  # Quest-world metres; missing joints are NaN.
    n_views: UInt8[ndarray, "n 133"]  # Per-joint inlier counts.
    confidence: Float32[ndarray, "n 133"]  # Mean supporting detector score, not geometric certainty; zero when missing.
    selected: dict[str, SelectedView]  # Wearer overlays indexed by phone.
    cameras: dict[str, Camera]  # Usable catalog calibration and overlay entity identities.
    rate: float  # Sample rate for provenance.
    inliers: Bool[ndarray, "n v 133"] | None = None  # Camera axis follows cameras insertion order.
    max_reprojection_px: Float32[ndarray, "n 133"] | None = None  # Maximum error over supporting cameras, NaN when missing.
    bridged: Bool[ndarray, "n 133"] | None = None  # Temporal estimate replacing a missing or rejected measurement.


def triangulate_episode(
    detections: Keypoints2D, cameras: dict[str, Camera], anchors: Float64[ndarray, "n 3 3"], threshold: float = INLIER_PX
) -> Keypoints3D:
    """Select one wearer/view/frame, then apply the served joint-wise pixel gates."""
    cameras = {p: cameras[p] for p in detections.views if p in cameras}
    phones, frames = list(cameras), len(detections.times)
    camera_list = list(cameras.values())
    if len(anchors) != frames:
        raise ValueError("anchors must match the sample grid")
    uv = np.full((frames, len(phones), 133, 2), np.nan)
    scores = np.zeros((frames, len(phones), 133))
    selected: dict[str, SelectedView] = {}
    for view, phone in enumerate(phones):
        observed = detections.views[phone]
        person = select_wearers(cameras[phone], anchors, observed.boxes, observed.box_scores)
        rows = np.flatnonzero(person >= 0)
        uv[rows, view], scores[rows, view] = observed.xy[rows, person[rows]], observed.scores[rows, person[rows]]
        selected[phone] = SelectedView(uv[:, view].astype(np.float32), scores[:, view].copy())
    batch: TriangulationBatch = triangulate_frames(camera_list, uv, scores, threshold)
    return Keypoints3D(detections.times, batch.xyz, batch.n_views, batch.confidence, selected, cameras, detections.rate, batch.inliers, batch.max_reprojection_px)


def one_euro(xyz: Float32[ndarray, "n j 3"], rate: float, *, beta: float = ONE_EURO_BETA) -> Float32[ndarray, "n j 3"]:
    """Causal One Euro filter (Casiez et al. 2012) per joint on the 3D vector; every gap restarts it."""

    @overload
    def alpha(cutoff: float) -> float: ...

    @overload
    def alpha(cutoff: Float64[ndarray, "k"]) -> Float64[ndarray, "k"]: ...

    def alpha(cutoff: Float64[ndarray, "k"] | float) -> Float64[ndarray, "k"] | float:
        return 1.0 / (1.0 + rate / (2.0 * np.pi * cutoff))

    out: Float32[ndarray, "n j 3"] = np.full_like(xyz, np.nan)
    state: Float64[ndarray, "j 3"] = np.zeros(xyz.shape[1:])
    velocity: Float64[ndarray, "j 3"] = np.zeros(xyz.shape[1:])
    alive: Bool[ndarray, "j"] = np.zeros(xyz.shape[1], bool)
    for t, sample in enumerate(xyz.astype(np.float64)):
        valid: Bool[ndarray, "j"] = np.isfinite(sample).all(1)
        start = valid & ~alive
        state[start], velocity[start] = sample[start], 0.0
        step = valid & alive
        velocity[step] += alpha(1.0) * ((sample[step] - state[step]) * rate - velocity[step])
        state[step] += alpha(ONE_EURO_MIN_CUTOFF_HZ + beta * np.linalg.norm(velocity[step], axis=1))[:, None] * (sample[step] - state[step])
        out[t, valid] = state[valid]
        alive = valid
    return out


def support_for_points(result: Keypoints3D, xyz: Float32[ndarray, "n 133 3"]) -> TriangulationBatch:
    """Measure support for the actual stored coordinates, including after smoothing."""
    projections: Float64[ndarray, "n v 133 3"] = np.stack(
        [project(camera, xyz.reshape(-1, 3).astype(np.float64)).reshape(len(xyz), 133, 3) for camera in result.cameras.values()], axis=1
    )
    detections: Float32[ndarray, "n v 133 2"] = np.stack([result.selected[phone].xy for phone in result.cameras], axis=1)
    errors: Float64[ndarray, "n v 133"] = np.linalg.norm(projections[..., :2] - detections, axis=-1)
    scores: Float64[ndarray, "n v 133"] = np.stack([result.selected[phone].scores for phone in result.cameras], axis=1)
    inliers: Bool[ndarray, "n v 133"] = (scores >= MIN_DETECTION_SCORE) & np.isfinite(errors) & (errors <= INLIER_PX) & (projections[..., 2] > 1e-5)
    counts: UInt8[ndarray, "n 133"] = inliers.sum(1).astype(np.uint8)
    good: Bool[ndarray, "n 133"] = counts >= 2
    inliers &= good[:, None]
    return TriangulationBatch(
        np.where(good[..., None], xyz, np.float32(np.nan)), np.where(good, counts, 0).astype(np.uint8),
        np.where(good, np.where(inliers, scores, 0.0).sum(1) / np.maximum(counts, 1), 0.0).astype(np.float32),
        inliers, np.where(good, np.where(inliers, errors, 0.0).max(1), np.nan).astype(np.float32),
    )


def smooth_keypoints(result: Keypoints3D) -> Keypoints3D:
    """Smooth measured points within their geometric support, then bridge short gaps with reduced confidence."""
    if result.rate < SMOOTH_MIN_RATE_HZ:
        return result
    smooth: Float32[ndarray, "n 133 3"] = ((one_euro(result.xyz, result.rate) + one_euro(result.xyz[::-1], result.rate)[::-1]) / 2.0).astype(np.float32)
    if result.inliers is None or not result.cameras:
        return replace(result, xyz=smooth)  # Synthetic callers without camera evidence.
    measured: TriangulationBatch = support_for_points(result, smooth)
    xyz: Float32[ndarray, "n 133 3"] = np.where((measured.n_views >= 2)[..., None], smooth, result.xyz)
    from selfcap_curate.temporal_keypoints import bridge_short_gaps, stable_body

    xyz = stable_body(result, xyz)
    final: TriangulationBatch = support_for_points(result, xyz)
    # Temporal estimates retain their coordinates even without two views at that
    # instant; support components still describe only the actual 8 px inliers.
    _, source_confidence, gaps = bridge_short_gaps(result.xyz, result.confidence, result.rate)
    finite = np.isfinite(xyz).all(-1)
    inferred = finite & (final.n_views < 2)
    confidence = np.where(inferred, source_confidence * 0.75, final.confidence).astype(np.float32)
    confidence *= np.where(final.n_views == 2, 0.9, 1.0).astype(np.float32)
    confidence[~finite] = 0.0
    bridged = gaps | inferred
    return replace(result, xyz=xyz, n_views=final.n_views, confidence=confidence, inliers=final.inliers, max_reprojection_px=final.max_reprojection_px, bridged=bridged)



def write_keypoints(
    calib: Path,
    output: Path,
    result: Keypoints3D,
    ego: tuple[EgoCamera, ...] = (),
    quest_hands: tuple[Float64[ndarray, "q"], Float32[ndarray, "q 133 3"]] | None = None,
    camera_center_chunks: tuple[Chunk, ...] = (),
    *,
    model: PoseModel = "rtmw",
) -> None:
    """Write RTMW with calibration, or an isolated Sapiens2 layer. The input file stays intact.

    Keypoint rows go through dataforge's shared writer: 133 joints per row, NaN for missing joints, and a confidence
    per joint (the mean inlier RTMW score in 3D, the raw RTMW score in 2D), on video_time only. All finite joints remain visible, and the shown 3D joints are projected into every calibrated phone
    through its lens model. Colours stretch from red at DISPLAY_CONFIDENCE to green at 1.0; scores stay raw. ``ego`` OAK cameras get the same projections plus the Quest hands (``quest_hands``),
    through the full OAK lens model: the viewer's Pinhole has no distortion, which put overlays up to 220 px off.
    Camera-centre chunks retain their native exo frame clocks when merged into the layer.
    """
    reader = RrdReader(calib)
    stores = reader.recordings()
    if len(stores) != 1:
        raise ValueError("calibration input must contain one recording")
    stamps: Int64[ndarray, "n"] = np.rint(result.times * 1e9).astype(np.int64)
    visible: Bool[ndarray, "133"] = ~np.isin(np.arange(133), HEADSET_COVERED)
    shown: Float32[ndarray, "n 133 3"] = np.where(visible[None, :, None], result.xyz, np.float32(np.nan))

    def projected(camera: Camera, points: Float32[ndarray, "n j 3"] = shown) -> Float32[ndarray, "n j 2"]:
        uvd: Float64[ndarray, "m 3"] = project(camera, points.reshape(-1, 3).astype(np.float64))
        uv: Float64[ndarray, "m 2"] = np.where((uvd[:, 2:] > 1e-5), uvd[:, :2], np.nan)
        return uv.reshape(len(points), points.shape[1], 2).astype(np.float32)

    def rig_cam(entity: str) -> tuple[int, int]:
        match = re.fullmatch(r"/world/rig_(\d+)/cam_(\d+)", entity)
        if match is None:
            raise ValueError(f"camera entity {entity} is not a rig camera")
        return int(match[1]), int(match[2])

    xyz_path: str = "/world/gt/sapiens2/coco133_xyz" if model == "sapiens2" else coco133_xyz_path(pred="rtmw_triangulated")
    uv_suffix: str = "sapiens2/coco133_uv" if model == "sapiens2" else "pred/rtmw/coco133_uv"
    projection_name: str = "sapiens2/coco133_uv_projected" if model == "sapiens2" else "coco133_uv_projected"
    parents: list[str] = [
        "/world/pred",
        *(f"{result.cameras[phone].entity}/pinhole/pred/rtmw" for phone in result.selected),
        *(coco133_uv_projected_path(*rig_cam(camera.entity)) for camera in result.cameras.values()),
        *(coco133_uv_projected_path(camera.rig, camera.cam) for camera in ego),
    ]
    if model == "sapiens2":
        parents = [
            xyz_path,
            *(f"{camera.entity}/pinhole/sapiens2" for camera in result.cameras.values()),
            *(f"{pinhole_path(camera.rig, camera.cam)}/sapiens2" for camera in ego),
        ]
    quest_paths: list[str] = [f"{pinhole_path(camera.rig, camera.cam)}/{QUEST_HANDS_UV}" for camera in ego] if quest_hands is not None else []

    def additions() -> Iterator[Chunk]:
        yield Chunk.from_property(
            "keypoints_sapiens2" if model == "sapiens2" else "calib",
            rr.AnyValues(
                kp3d_method="confidence-weighted DLT; exhaustive pair RANSAC",
                kp3d_model="YOLOX-m-humanart + " + ("Sapiens2 0.4B COCO-133" if model == "sapiens2" else "RTMW-x COCO-133"),
                kp3d_rate_hz=result.rate,
                kp3d_samples=len(result.times),
                kp3d_valid_joints=int(np.isfinite(result.xyz).all(-1).sum()),
                kp3d_possible_joints=int(result.n_views.size),
                kp3d_reprojection_threshold_px=INLIER_PX,
                kp3d_calibration_sha256=hashlib.sha256(calib.read_bytes()).hexdigest(),
                kp3d_display_confidence=DISPLAY_CONFIDENCE,
                kp3d_min_detection_score=MIN_DETECTION_SCORE,
                kp3d_support_camera_paths=[camera.entity for camera in result.cameras.values()],
                kp3d_missing_under_headset="body 0-2 and face 23-90: both models place them on the Quest front plate",
                kp3d_smoothing=f"Body: centred median-5 jump replacement and Gaussian sigma 2 frames; above 0.5 m/s follow jump-cleaned positions; gaps up to 0.5 s bridged; other joints: bidirectional One Euro beta {ONE_EURO_BETA}; restart at gaps",
            ),
        )
        for parent in parents:
            yield Chunk.from_columns(parent, indexes=[], columns=rr.AnnotationContext.columns(context=[COCO133_CLASS]))
        for path in quest_paths if model == "rtmw" else []:
            yield Chunk.from_columns(path, indexes=[], columns=rr.AnnotationContext.columns(context=[QUEST_HANDS_CLASS]))

    with TemporaryDirectory(prefix="selfcap-kp3d-") as work, atomic_write(output) as temporary:
        rows: Path = Path(work) / "keypoints.rrd"
        with rr.RecordingStream(stores[0].application_id, recording_id=stores[0].recording_id, send_properties=False) as recording:
            recording.save(rows)
            if model == "rtmw":
                display: DisplaySkeleton = display_skeleton(result.times, result.xyz, result.confidence, quest_hands)
                log_display_skeleton(recording, "/world/skeleton", display.positions, display.confidence, stamps)
                for camera in ego:
                    log_display_skeleton(recording, f"{pinhole_path(camera.rig, camera.cam)}/skeleton",
                                         project_ego(camera, result.times, display.positions), display.confidence, stamps)
            log_keypoints3d(
                recording,
                times_ns=stamps,
                frame_indices=None,
                positions=shown,
                confidence=result.confidence,
                path=xyz_path,
                color_range=(DISPLAY_CONFIDENCE, 1.0),
            )
            if result.inliers is not None and result.max_reprojection_px is not None:
                recording.log(xyz_path, rr.AnyValues(support_camera_paths=[camera.entity for camera in result.cameras.values()],
                              confidence_semantics="Mean inlier detector score; multiplied by 0.9 for two-view support, and by 0.75 at temporally bridged frames; not visibility probability"), static=True)
                mask: Bool[ndarray, "n 133 v"] = result.inliers.transpose(0, 2, 1)
                rr.send_columns(xyz_path, indexes=[time_column(stamps)], columns=rr.AnyValues.columns(
                    support_view_count=result.n_views.reshape(-1),
                    temporal_bridged=(np.zeros(result.xyz.shape[:2], dtype=bool) if result.bridged is None else result.bridged).reshape(-1),
                    support_max_reprojection_px=result.max_reprojection_px.reshape(-1),
                    support_inlier_mask=pa.FixedSizeListArray.from_arrays(pa.array(mask.reshape(-1)), mask.shape[-1]),
                ).partition([133] * len(stamps)), recording=recording)
            for phone, selected in result.selected.items():
                scores: Float32[ndarray, "n 133"] = selected.scores.astype(np.float32)
                log_keypoints2d(
                    recording,
                    f"{result.cameras[phone].entity}/pinhole/{uv_suffix}",
                    times_ns=stamps,
                    frame_indices=None,
                    positions=np.where(((scores >= DISPLAY_CONFIDENCE) & visible)[..., None], selected.xy, np.float32(np.nan)),
                    confidence=scores,
                    color_range=(DISPLAY_CONFIDENCE, 1.0),
                )
            log_projections(
                recording,
                (
                    *((rig_cam(camera.entity), projected(camera)) for camera in result.cameras.values()),
                    *(((camera.rig, camera.cam), project_ego(camera, result.times, shown)) for camera in ego),
                ),
                times_ns=stamps,
                frame_indices=None,
                confidence=result.confidence,
                color_range=(DISPLAY_CONFIDENCE, 1.0),
                entity_name=projection_name,
                camera_model="phones: pinhole with radial k1, k2 (selfcap-curate exo calibration)"
                + ("; OAK: OpenCV Brown-Conrady, 14 coefficients (factory)" if ego else ""),
            )
            if quest_hands is not None and model == "rtmw":
                hand_times, hand_xyz = quest_hands
                hand_stamps: Int64[ndarray, "q"] = np.rint(hand_times * 1e9).astype(np.int64)
                for camera, path in zip(ego, quest_paths, strict=True):
                    pixels: Float32[ndarray, "q 133 2"] = project_ego(camera, hand_times, hand_xyz)
                    rr.log(
                        path,
                        rr.Points2D.from_fields(class_ids=0, keypoint_ids=np.arange(133, dtype=np.uint16), radii=3.0, show_labels=False),
                        static=True,
                        recording=recording,
                    )
                    rr.send_columns(
                        path,
                        indexes=[time_column(hand_stamps)],
                        columns=rr.Points2D.columns(positions=pixels.reshape(-1, 2)).partition([133] * len(pixels)),
                        recording=recording,
                    )
            recording.flush()
        LazyChunkStream.merge(
            *([reader.stream(), LazyChunkStream.from_iter(camera_center_chunks)] if model == "rtmw" else []),
            LazyChunkStream.from_iter(additions()), RrdReader(rows).stream()
        ).write_rrd(
            temporary, application_id=stores[0].application_id, recording_id=stores[0].recording_id
        )


def read_cameras(dataset: DatasetEntry, segment: str, base_phones: tuple[str, ...] | None = None) -> dict[str, Camera]:
    """Read refined pinholes/poses from one source, retaining calib_status=ok."""
    view = dataset.filter_segments(segment)
    entities = sorted({item.entity_path for item in catalog_components(view.schema()) if item.field == "calib_status"})
    table = view.filter_contents([path for entity in entities for path in (entity, f"{entity}/pinhole")]).reader(index=None).to_arrow_table()
    names = {
        entity: str(scalar(table, f"{entity}:calib_phone"))
        for entity in entities
        if f"{entity}:calib_phone" in table.column_names and table[f"{entity}:calib_phone"].null_count < len(table)
    }
    if base_phones is not None and (unexpected := set(names.values()) - set(base_phones)):
        raise ValueError("local calib phones absent from base: " + ", ".join(sorted(unexpected)))

    def numeric(name: str) -> Float64[ndarray, "n"]:
        return np.asarray(first_valid_value(table[name]), dtype=np.float64).reshape(-1)

    result: dict[str, Camera] = {}
    for entity in entities:
        status = f"{entity}:calib_status"
        if status not in table.column_names or scalar(table, status, allow_none=True) != "ok":
            continue
        phone = names[entity]
        pose: Float64[ndarray, "4 4"] = np.eye(4)
        pose[:3, :3], pose[:3, 3] = numeric(f"{entity}:Transform3D:mat3x3").reshape(3, 3).T, numeric(f"{entity}:Transform3D:translation")
        size: Float64[ndarray, "n"] = numeric(f"{entity}/pinhole:Pinhole:resolution")
        result[phone] = Camera(numeric(f"{entity}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T, pose, (0.0, 0.0), entity, (int(size[0]), int(size[1])))
    return result


def interpolate(
    times: Float64[ndarray, "m"], stamps: Float64[ndarray, "n"], values: Float64[ndarray, "n 3"], valid: Bool[ndarray, "n"], max_gap: float
) -> Float64[ndarray, "m 3"]:
    """Linear interpolation only between valid neighbours within the source gap gate."""
    result = np.full((len(times), 3), np.nan)
    if len(stamps) < 2:
        return result
    right = np.clip(np.searchsorted(stamps, times), 1, len(stamps) - 1)
    left, gap = right - 1, stamps[right] - stamps[right - 1]
    good = (times >= stamps[0]) & (times <= stamps[-1]) & (gap > 0.0) & (gap <= max_gap) & valid[left] & valid[right]
    alpha = ((times - stamps[left]) / np.maximum(gap, 1e-12))[:, None]
    result[good] = ((1.0 - alpha) * values[left] + alpha * values[right])[good]
    return result


def read_anchors(segment: Segment, times: Float64[ndarray, "n"], sync: TimeSync) -> Float64[ndarray, "n 3 3"]:
    """Use the base's time_sync, left-eye head pose and filtered raw hand wrists."""
    # Phone offsets embed this same Quest offset, including flagged as-cut zero.
    qp = (sync.offsets_ms or {}).get("quest_pose")
    if qp is None:
        return np.full((len(times), 3, 3), np.nan)
    source = read_catalog_hands(segment)
    valid = np.isfinite(source.head_positions).all(1) & (np.linalg.norm(source.head_quaternions, axis=1) > 0.5)
    head = interpolate(times, source.head_times_s + qp / 1000.0, source.head_positions, valid, 0.1)
    hands = load_quest_hands(source)
    wrists = [
        interpolate(times, hands[side].times_s - HAND_LATENCY_MS / 1000.0 + qp / 1000.0, hands[side].joints[:, 0], hands[side].valid, 0.04)
        for side in HAND_BASE
    ]
    return np.stack([head, *wrists], axis=1)
