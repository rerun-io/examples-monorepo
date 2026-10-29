"""COCO-133 wearer selection, confidence-weighted triangulation and layer writing."""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import rerun as rr
from dataforge.hands import log_keypoints2d, log_keypoints3d, log_projections
from dataforge.schema import coco133_xyz_path
from dataforge.writing import atomic_write
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from simplecv.catalog_rig_layout import catalog_components
from simplecv.rrd_query_utils import first_valid_value

from selfcap_curate.catalog_io import Segment, read_catalog_hands, scalar
from selfcap_curate.detection_io import COCO133_CLASS
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import HAND_BASE, HAND_LATENCY_MS, TimeSync

INLIER_PX: float = 8.0

DISPLAY_CONFIDENCE: float = 0.7
"""Joints scoring below this are not written (mv-api's display threshold); triangulation itself gates views at 0.3."""
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


@dataclass(frozen=True, slots=True)
class Triangulated:
    xyz: Float32[ndarray, "j 3"]  # World metres; NaN for missing/degenerate joints.
    n_views: UInt8[ndarray, "j"]  # Positive-depth inlier count, zero when unusable.
    confidence: Float32[ndarray, "j"]  # Mean RTMW score of the inlier views, zero when unusable.


def project(camera: Camera, xyz: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    """Project world points to u,v,depth with the camera's radial distortion."""
    inverse = np.linalg.inv(camera.world_T_cam)
    q = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[:, :2] / q[:, 2:3]
        radius = np.sum(xy * xy, axis=1)
        uv = (xy * (1.0 + camera.radial[0] * radius + camera.radial[1] * radius**2)[:, None]) @ camera.K[:2, :2].T + camera.K[:2, 2]
    return np.c_[uv, q[:, 2]]


def select_wearer(camera: Camera, anchors: Float64[ndarray, "3 3"], boxes: Float64[ndarray, "p 4"], scores: Float64[ndarray, "p"]) -> int:
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


def triangulate(cameras: list[Camera], uv: Float64[ndarray, "v j 2"], scores: Float64[ndarray, "v j"], threshold: float = INLIER_PX) -> Triangulated:
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
    confidence: Float32[ndarray, "n 133"]  # Mean inlier RTMW score; zero for missing joints.
    selected: dict[str, SelectedView]  # Wearer overlays indexed by phone.
    cameras: dict[str, Camera]  # Usable catalog calibration and overlay entity identities.
    rate: float  # Sample rate for provenance.


def triangulate_episode(
    detections: Keypoints2D, cameras: dict[str, Camera], anchors: Float64[ndarray, "n 3 3"], threshold: float = INLIER_PX
) -> Keypoints3D:
    """Select one wearer/view/frame, then apply the served joint-wise pixel gates."""
    cameras = {p: cameras[p] for p in detections.views if p in cameras}
    phones, frames = list(cameras), len(detections.times)
    camera_list = list(cameras.values())
    if len(anchors) != frames:
        raise ValueError("anchors must match the sample grid")
    xyz = np.full((frames, 133, 3), np.nan, np.float32)
    counts = np.zeros((frames, 133), np.uint8)
    confidence = np.zeros((frames, 133), np.float32)
    selected = {p: SelectedView(np.full((frames, 133, 2), np.nan, np.float32), np.zeros((frames, 133))) for p in phones}
    for i in range(frames):
        uv, scores = np.full((len(phones), 133, 2), np.nan), np.zeros((len(phones), 133))
        for view, phone in enumerate(phones):
            observed = detections.views[phone]
            person = select_wearer(cameras[phone], anchors[i], observed.boxes[i], observed.box_scores[i])
            if person >= 0:
                uv[view], scores[view] = observed.xy[i, person], observed.scores[i, person]
                selected[phone].xy[i], selected[phone].scores[i] = uv[view], scores[view]
        result = triangulate(camera_list, uv, scores, threshold)
        xyz[i], counts[i], confidence[i] = result.xyz, result.n_views, result.confidence
    return Keypoints3D(detections.times, xyz, counts, confidence, selected, cameras, detections.rate)


def one_euro(xyz: Float32[ndarray, "n j 3"], rate: float) -> Float32[ndarray, "n j 3"]:
    """Causal One Euro filter (Casiez et al. 2012) per joint on the 3D vector; every gap restarts it."""

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


def smooth_keypoints(result: Keypoints3D) -> Keypoints3D:
    """One Euro run forward and backward in time and averaged, so its lags cancel; gaps stay gaps, slow plans stay raw."""
    if result.rate < SMOOTH_MIN_RATE_HZ:
        return result
    smooth = (one_euro(result.xyz, result.rate) + one_euro(result.xyz[::-1], result.rate)[::-1]) / 2.0
    return replace(result, xyz=smooth.astype(np.float32))


def write_keypoints(calib: Path, output: Path, result: Keypoints3D) -> None:
    """One output file and calib property group; input calibration chunks stay intact.

    Keypoint rows go through dataforge's shared writer: 133 joints per row, NaN for missing joints, and a confidence
    per joint (the mean inlier RTMW score in 3D, the raw RTMW score in 2D), on video_time only. Joints under
    DISPLAY_CONFIDENCE are written as missing, and the shown 3D joints are projected into every calibrated phone
    through its lens model.
    """
    reader = RrdReader(calib)
    stores = reader.recordings()
    if len(stores) != 1:
        raise ValueError("calibration input must contain one recording")
    stamps: Int64[ndarray, "n"] = np.rint(result.times * 1e9).astype(np.int64)
    visible: Bool[ndarray, "133"] = ~np.isin(np.arange(133), HEADSET_COVERED)
    shown: Float32[ndarray, "n 133 3"] = np.where(((result.confidence >= DISPLAY_CONFIDENCE) & visible)[..., None], result.xyz, np.float32(np.nan))

    def projected(camera: Camera) -> Float32[ndarray, "n 133 2"]:
        uvd: Float64[ndarray, "m 3"] = project(camera, shown.reshape(-1, 3).astype(np.float64))
        uv: Float64[ndarray, "m 2"] = np.where((uvd[:, 2:] > 1e-5), uvd[:, :2], np.nan)
        return uv.reshape(len(shown), 133, 2).astype(np.float32)

    def rig_cam(entity: str) -> tuple[int, int]:
        match = re.fullmatch(r"/world/rig_(\d+)/cam_(\d+)", entity)
        if match is None:
            raise ValueError(f"camera entity {entity} is not a rig camera")
        return int(match[1]), int(match[2])

    parents: list[str] = ["/world/pred", *(f"{result.cameras[phone].entity}/pinhole/pred/rtmw" for phone in result.selected)]

    def additions() -> Iterator[Chunk]:
        yield Chunk.from_property(
            "calib",
            rr.AnyValues(
                kp3d_method="confidence-weighted DLT; exhaustive pair RANSAC",
                kp3d_model="YOLOX-m-humanart + RTMW-x COCO-133",
                kp3d_rate_hz=result.rate,
                kp3d_samples=len(result.times),
                kp3d_valid_joints=int((result.n_views >= 2).sum()),
                kp3d_possible_joints=int(result.n_views.size),
                kp3d_reprojection_threshold_px=INLIER_PX,
                kp3d_calibration_sha256=hashlib.sha256(calib.read_bytes()).hexdigest(),
                kp3d_display_confidence=DISPLAY_CONFIDENCE,
                kp3d_missing_under_headset="body 0-2 and face 23-90: both models place them on the Quest front plate",
                kp3d_smoothing=f"One Euro forward + backward, min cutoff {ONE_EURO_MIN_CUTOFF_HZ} Hz, beta {ONE_EURO_BETA}, restarts at gaps",
            ),
        )
        for parent in parents:
            yield Chunk.from_columns(parent, indexes=[], columns=rr.AnnotationContext.columns(context=[COCO133_CLASS]))

    with TemporaryDirectory(prefix="selfcap-kp3d-") as work, atomic_write(output) as temporary:
        rows: Path = Path(work) / "keypoints.rrd"
        with rr.RecordingStream(stores[0].application_id, recording_id=stores[0].recording_id, send_properties=False) as recording:
            recording.save(rows)
            log_keypoints3d(
                recording,
                times_ns=stamps,
                frame_indices=None,
                positions=shown,
                confidence=result.confidence,
                path=coco133_xyz_path(pred="rtmw_triangulated"),
            )
            for phone, selected in result.selected.items():
                scores: Float32[ndarray, "n 133"] = selected.scores.astype(np.float32)
                log_keypoints2d(
                    recording,
                    f"{result.cameras[phone].entity}/pinhole/pred/rtmw/coco133_uv",
                    times_ns=stamps,
                    frame_indices=None,
                    positions=np.where(((scores >= DISPLAY_CONFIDENCE) & visible)[..., None], selected.xy, np.float32(np.nan)),
                    confidence=scores,
                )
            log_projections(
                recording,
                ((rig_cam(camera.entity), projected(camera)) for camera in result.cameras.values()),
                times_ns=stamps,
                frame_indices=None,
                confidence=result.confidence,
                camera_model="pinhole with radial k1, k2 (selfcap-curate exo calibration)",
            )
            recording.flush()
        LazyChunkStream.merge(reader.stream(), LazyChunkStream.from_iter(additions()), RrdReader(rows).stream()).write_rrd(
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
        result[phone] = Camera(numeric(f"{entity}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T, pose, (0.0, 0.0), entity)
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
