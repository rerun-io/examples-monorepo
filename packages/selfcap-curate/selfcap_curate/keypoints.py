"""COCO-133 wearer selection, confidence-weighted triangulation and layer writing."""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias, overload

import numpy as np
import pyarrow as pa
import rerun as rr
from dataforge.hands import log_keypoints2d, log_keypoints3d, log_projections
from dataforge.logging_toolkit import time_column
from dataforge.oak_mount import StaticCamera, read_oak_track, read_static_cameras
from dataforge.schema import SHOWN_MIN_CONFIDENCE, coco133_uv_projected_path, coco133_xyz_path, pinhole_path, rig_path
from dataforge.triangulation import (
    HEADSET_COVERED,
    INLIER_PX,
    MIN_DETECTION_SCORE,
    ROBUST_OUTER_GATE,
    ROBUST_ROUNDS,
    Camera,
    EgoCamera,
    TriangulationBatch,
    project,
    project_ego,
    reprojection_confidence,
    triangulate_frames,
)
from dataforge.writing import atomic_write
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from simplecv.catalog_rig_layout import catalog_components
from simplecv.data.skeleton.coco_133 import COCO_133_LINKS
from simplecv.rrd_query_utils import first_valid_value

from selfcap_curate.catalog_io import Segment, read_catalog_hands, scalar
from selfcap_curate.detection_io import COCO133_CLASS
from selfcap_curate.display_style import CONFIDENCE_3D_COLOR_FLOOR, DISPLAY_JOINTS, log_display_skeleton
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import HAND_BASE, HAND_LATENCY_MS, TimeSync, oak_quest_offset_ms

PoseModel: TypeAlias = Literal["rtmw", "sapiens2"]
QuestHandTrack: TypeAlias = tuple[Float64[ndarray, "q"], Float32[ndarray, "q 133 3"]]
"""Quest hand joints on video_time."""

MAX_HEAD_DISTANCE_M: float = 2.2
"""A triangulated joint farther than this from the wearer's head (Quest track) is not the wearer's: written missing."""

ONE_EURO_MIN_CUTOFF_HZ: float = 1.0
"""One Euro cutoff at rest; with ONE_EURO_BETA, forward + backward cut cc808233 ep001 jitter 6x (median 11.8 -> 1.9 mm)."""
ONE_EURO_BETA: float = 2.0
"""Cutoff increase per m/s of joint speed, so fast moves are followed closely (12 mm median pull in the fastest 10%)."""
SMOOTH_MIN_RATE_HZ: float = 10.0
"""Plans sampled slower than this stay raw: a 1 Hz cutoff needs samples well above it."""


@dataclass(frozen=True, slots=True)
class Triangulated:
    xyz: Float32[ndarray, "j 3"]  # World metres; NaN for missing/degenerate joints.
    n_views: UInt8[ndarray, "j"]  # Positive-depth inlier count, zero when unusable.
    confidence: Float32[ndarray, "j"]  # Kineo pairwise reprojection score over the views that saw the joint (0-1).
    inliers: Bool[ndarray, "v j"] | None = None
    max_reprojection_px: Float32[ndarray, "j"] | None = None



def read_ego_cameras(calibration: DatasetEntry, base: DatasetEntry, segment: str, oak_rig: int, quest_rig: int) -> list[EgoCamera]:
    """The OAK cameras of a fitted OAK mount: world poses are the Quest track times the mount; stereo and lenses are factory
    facts, which only the base layer writes (write_calib_layer holds the mount alone)."""
    rig = rig_path(oak_rig)
    cams = {cam: f"{rig}/cam_{cam:02d}" for cam in (0, 1, 2)}
    track = read_oak_track(calibration.filter_segments(segment), rig, rig_path(quest_rig), poses=base.filter_segments(segment))
    if track is None:
        return []  # no fitted OAK mount for this episode
    times_s, world_T_rig = track
    static: dict[str, StaticCamera] = read_static_cameras(calibration.filter_segments(segment), base.filter_segments(segment), list(cams.values()))
    result: list[EgoCamera] = []
    for cam, path in cams.items():
        camera: StaticCamera = static[path]
        rig_T_cam: Float64[ndarray, "4 4"] | None = np.eye(4) if cam == 0 else camera.rig_T_cam
        if camera.K is None or camera.resolution is None or rig_T_cam is None:
            continue
        result.append(EgoCamera(oak_rig, cam, camera.K, camera.distortion if camera.distortion is not None else np.zeros(5), camera.resolution, rig_T_cam, times_s, world_T_rig))
    return result


QUEST_HANDS_UV: str = "quest_coco133_uv_projected"
"""Quest hand joints projected into an OAK camera, beneath its pinhole (the viewer cannot apply the OAK lens model)."""
QUEST_HANDS_CLASS: rr.ClassDescription = rr.ClassDescription(
    info=rr.AnnotationInfo(id=0, label="Quest hands (projected)", color=(0, 0, 255)), keypoint_connections=COCO_133_LINKS
)
"""Blue like the base's /world/gt Quest hands; the points carry no colours of their own, so joints and bones take it."""


def ego_quest_hands(segment: Segment, sync: TimeSync) -> QuestHandTrack | None:
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
    quest_hands: QuestHandTrack | None,
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


@dataclass(frozen=True, slots=True)
class Keypoints3D:
    times: Float64[ndarray, "n"]  # Aligned sample grid in seconds, with empty rows preserved.
    xyz: Float32[ndarray, "n 133 3"]  # Quest-world metres; missing joints are NaN.
    n_views: UInt8[ndarray, "n 133"]  # Per-joint inlier counts.
    confidence: Float32[ndarray, "n 133"]  # Kineo pairwise reprojection score over the views that saw the joint; zero when missing.
    selected: dict[str, SelectedView]  # Wearer overlays indexed by phone.
    cameras: dict[str, Camera]  # Usable catalog calibration and overlay entity identities.
    rate: float  # Sample rate for provenance.
    inliers: Bool[ndarray, "n v 133"] | None = None  # Camera axis follows cameras insertion order.
    max_reprojection_px: Float32[ndarray, "n 133"] | None = None  # Maximum error over supporting cameras, NaN when missing.
    bridged: Bool[ndarray, "n 133"] | None = None  # Temporal estimate replacing a missing or rejected measurement.


BONES: tuple[tuple[int, int], ...] = (
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),  # arms and legs
    (5, 6), (11, 12), (5, 11), (6, 12), (3, 4), (3, 5), (4, 6),  # torso and ears
)
"""COCO body bones whose 3D length is fixed for one person."""
BONE_RATIO: float = 1.4
"""A bone longer than 1.4x (or shorter than 1/1.4x) its session median marks a wrong joint."""
MIN_BONE_FRAMES: int = 50
"""Frames a bone needs before its median length is trusted."""


def drop_bone_outliers(xyz: Float32[ndarray, "n 133 3"], confidence: Float32[ndarray, "n 133"], rounds: int = 3) -> Float32[ndarray, "n 133 3"]:
    """Drop body joints that stretch or shrink their bones far from the session's median lengths.

    Two views can agree within the pixel gate on a wrong point (left/right swaps, near-parallel rays); a person's bones
    do not change length. Per frame and round, the joint touching the most bad bones goes (the less confident on a tie).
    """
    out = xyz.copy()
    lengths = np.stack([np.linalg.norm(out[:, a] - out[:, b], axis=-1) for a, b in BONES], axis=1)
    finite = np.isfinite(lengths)
    median = np.array([np.median(lengths[finite[:, k], k]) if finite[:, k].sum() >= MIN_BONE_FRAMES else np.nan for k in range(len(BONES))])
    for _ in range(rounds):
        lengths = np.stack([np.linalg.norm(out[:, a] - out[:, b], axis=-1) for a, b in BONES], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = lengths / median
        bad = np.isfinite(ratio) & ((ratio > BONE_RATIO) | (ratio < 1.0 / BONE_RATIO))
        if not bad.any():
            break
        score, checked = np.zeros(out.shape[:2]), np.zeros(out.shape[:2])
        for k, (a, b) in enumerate(BONES):
            score[:, [a, b]] += bad[:, k, None]
            checked[:, [a, b]] += np.isfinite(ratio[:, k, None])
        # The share of a joint's checked bones that are bad first (an elbow with one good bone is not the culprit), then
        # the count, then the lower detector confidence.
        share = np.divide(score, checked, out=np.zeros_like(score), where=checked > 0)
        key = np.where(score > 0, share * 100.0 + score * 10.0 - np.nan_to_num(confidence, nan=0.0), -np.inf)
        frames = np.flatnonzero(np.isfinite(key.max(axis=1)))
        out[frames, key[frames].argmax(axis=1)] = np.nan
    return out


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
    xyz, n_views, confidence, inliers, reprojection = batch.xyz, batch.n_views, batch.confidence, batch.inliers, batch.max_reprojection_px
    with np.errstate(invalid="ignore"):
        far = np.linalg.norm(xyz - anchors[:, None, 0, :], axis=-1) > MAX_HEAD_DISTANCE_M
    xyz[far], n_views[far], confidence[far] = np.nan, 0, 0.0
    far |= np.isfinite(xyz).all(-1) & ~np.isfinite(drop_bone_outliers(xyz, confidence)).all(-1)  # wrong-length bones
    xyz[far], n_views[far], confidence[far] = np.nan, 0, 0.0
    inliers[np.broadcast_to(far[:, None, :], inliers.shape)] = False
    reprojection[far] = np.nan
    return Keypoints3D(detections.times, xyz, n_views, confidence, selected, cameras, detections.rate, inliers, reprojection)


def one_euro(xyz: Float32[ndarray, "n j 3"], rate: float) -> Float32[ndarray, "n j 3"]:
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
        state[step] += alpha(ONE_EURO_MIN_CUTOFF_HZ + ONE_EURO_BETA * np.linalg.norm(velocity[step], axis=1))[:, None] * (sample[step] - state[step])
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
    seen: Bool[ndarray, "n v 133"] = (scores >= MIN_DETECTION_SCORE) & np.isfinite(detections).all(-1)
    errors = np.where(projections[..., 2] > 1e-5, errors, np.inf)
    inliers: Bool[ndarray, "n v 133"] = seen & np.isfinite(errors) & (errors <= INLIER_PX)
    counts: UInt8[ndarray, "n 133"] = inliers.sum(1).astype(np.uint8)
    good: Bool[ndarray, "n 133"] = counts >= 2
    inliers &= good[:, None]
    return TriangulationBatch(
        np.where(good[..., None], xyz, np.float32(np.nan)), np.where(good, counts, 0).astype(np.uint8),
        np.where(good, reprojection_confidence(errors, scores, np.array([c.K[0, 0] for c in result.cameras.values()]), seen), 0.0),
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

    # Jump replacement and gap bridging come after the triangulation's bone check: check the bones again, so a bridge
    # to a wrong edge point is not written (a044c36b: thighs up to 1.5 m, 80% of them bridged).
    xyz = drop_bone_outliers(stable_body(result, xyz), result.confidence)
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
    quest_hands: QuestHandTrack | None = None,
    camera_center_chunks: tuple[Chunk, ...] = (),
    *,
    model: PoseModel = "rtmw",
) -> None:
    """Write the keypoints layer: the calibration it was fitted with, the display skeleton and the model's keypoints.
    The input file stays intact.

    Keypoint rows go through dataforge's shared writer: 133 joints per row, NaN for missing joints, and a confidence
    per joint (reprojection_confidence in 3D, the detector score in 2D), on video_time only. The 2D entity stores every detection; its display
    copy (``schema.shown_path``) holds those at or above SHOWN_MIN_CONFIDENCE outside HEADSET_COVERED, and the layout draws only the copy. The 3D joints outside
    HEADSET_COVERED are projected into every calibrated phone through its lens model. Colours stretch to green at 1.0 from red at CONFIDENCE_3D_COLOR_FLOOR for 3D confidence, at SHOWN_MIN_CONFIDENCE for 2D scores. ``ego`` OAK cameras get the same projections plus the Quest hands (``quest_hands``),
    through the full OAK lens model: the viewer's Pinhole has no distortion, which put overlays up to 220 px off.
    Camera-centre chunks retain their native exo frame clocks when merged into the layer.
    """
    if len(result.cameras) > 4:
        raise ValueError("support masks support at most four exo cameras; refusing to truncate")
    if result.inliers is not None and result.inliers.shape[1] != len(result.cameras):
        raise ValueError("support mask camera count does not match support_camera_paths")
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
                kp3d_method=(
                    f"confidence-weighted DLT: best view-pair seed, {ROBUST_ROUNDS} rounds of robust reweighting over every view "
                    f"(Cauchy at {INLIER_PX:g} px, no weight beyond {ROBUST_OUTER_GATE * INLIER_PX:g} px); "
                    "confidence: Kineo pairwise reprojection score (arXiv 2510.24464)"
                ),
                kp3d_model="YOLOX-m-humanart + " + ("Sapiens2 0.4B COCO-133" if model == "sapiens2" else "RTMW-x COCO-133"),
                kp3d_rate_hz=result.rate,
                kp3d_samples=len(result.times),
                kp3d_valid_joints=int(np.isfinite(result.xyz).all(-1).sum()),
                kp3d_possible_joints=int(result.n_views.size),
                kp3d_reprojection_threshold_px=INLIER_PX,
                kp3d_calibration_sha256=hashlib.sha256(calib.read_bytes()).hexdigest(),
                kp3d_color_floor=CONFIDENCE_3D_COLOR_FLOOR,
                kp2d_show_score=SHOWN_MIN_CONFIDENCE,
                kp3d_min_detection_score=MIN_DETECTION_SCORE,
                kp3d_support_camera_paths=[camera.entity for camera in result.cameras.values()],
                kp3d_missing_under_headset="body 0-2 and face 23-90: the detector places them on the Quest front plate",
                kp3d_smoothing=f"Body: centred median-5 jump replacement and Gaussian sigma 2 frames; above 0.5 m/s follow jump-cleaned positions; gaps up to 0.5 s bridged; other joints: bidirectional One Euro beta {ONE_EURO_BETA}; restart at gaps",
            ),
        )
        for parent in parents:
            yield Chunk.from_columns(parent, indexes=[], columns=rr.AnnotationContext.columns(context=[COCO133_CLASS]))
        for path in quest_paths:
            yield Chunk.from_columns(path, indexes=[], columns=rr.AnnotationContext.columns(context=[QUEST_HANDS_CLASS]))

    with TemporaryDirectory(prefix="selfcap-kp3d-") as work, atomic_write(output) as temporary:
        rows: Path = Path(work) / "keypoints.rrd"
        with rr.RecordingStream(stores[0].application_id, recording_id=stores[0].recording_id, send_properties=False) as recording:
            recording.save(rows)
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
                color_range=(CONFIDENCE_3D_COLOR_FLOOR, 1.0),
            )
            if result.inliers is not None and result.max_reprojection_px is not None:
                recording.log(xyz_path, rr.AnyValues(support_camera_paths=[camera.entity for camera in result.cameras.values()],
                              confidence_semantics="Kineo pairwise reprojection score (exp(-40 x error / focal)) over the views that detected the joint; multiplied by 0.9 for two-view support, and by 0.75 at temporally bridged frames; not visibility probability"), static=True)
                # Catalog schema is shared across three- and four-phone sessions. The
                # leading slots follow support_camera_paths; absent trailing slots are false.
                mask: Bool[ndarray, "n 133 4"] = np.zeros((len(stamps), 133, 4), dtype=bool)
                mask[:, :, :len(result.cameras)] = result.inliers.transpose(0, 2, 1)
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
                    positions=selected.xy,
                    confidence=scores,
                    color_range=(SHOWN_MIN_CONFIDENCE, 1.0),
                    shown=(scores >= SHOWN_MIN_CONFIDENCE) & visible,
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
                color_range=(CONFIDENCE_3D_COLOR_FLOOR, 1.0),
                entity_name=projection_name,
                camera_model="phones: pinhole with radial k1, k2 (selfcap-curate exo calibration)"
                + ("; OAK: OpenCV Brown-Conrady, 14 coefficients (factory)" if ego else ""),
            )
            if quest_hands is not None:
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
            reader.stream(), LazyChunkStream.from_iter(camera_center_chunks), LazyChunkStream.from_iter(additions()), RrdReader(rows).stream()
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
