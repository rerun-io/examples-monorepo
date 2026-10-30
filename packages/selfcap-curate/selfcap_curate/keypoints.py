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
import torch
from dataforge.hands import log_keypoints2d, log_keypoints3d, log_projections
from dataforge.schema import coco133_xyz_path
from dataforge.writing import atomic_write
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from simplecv.catalog_rig_layout import catalog_components
from simplecv.rrd_query_utils import first_valid_value
from torch import Tensor

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


def triangulate(cameras: list[Camera], uv: Float64[ndarray, "v j 2"], scores: Float64[ndarray, "v j"], threshold: float = INLIER_PX) -> Triangulated:
    """One frame of triangulate_frames."""
    xyz, counts, confidence = triangulate_frames(cameras, uv[None], scores[None], threshold)
    return Triangulated(xyz[0], counts[0], confidence[0])


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
    uv = np.full((frames, len(phones), 133, 2), np.nan)
    scores = np.zeros((frames, len(phones), 133))
    selected: dict[str, SelectedView] = {}
    for view, phone in enumerate(phones):
        observed = detections.views[phone]
        person = select_wearers(cameras[phone], anchors, observed.boxes, observed.box_scores)
        rows = np.flatnonzero(person >= 0)
        uv[rows, view], scores[rows, view] = observed.xy[rows, person[rows]], observed.scores[rows, person[rows]]
        selected[phone] = SelectedView(uv[:, view].astype(np.float32), scores[:, view].copy())
    xyz, counts, confidence = triangulate_frames(camera_list, uv, scores, threshold)
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
