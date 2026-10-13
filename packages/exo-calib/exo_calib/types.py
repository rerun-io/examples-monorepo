"""exo-calib edge values and their Rerun mapping (the rig itself is ``simplecv.types.rig.RigCameras``).

Entity layout: rigs at ``/world/exocalib/{init,refined}/<rig>/<cam>``; the Stage B record at ``/world/exocalib/kp2d/<cam>``
with its boxes and the gated skeleton drawn under each camera's ``pinhole/exocalib_kp2d*``; 3D keypoints at
``/world/exocalib/kp3d_{init,refined}``; the GT alignment on the rig roots plus error lines at ``/world/exocalib_error``.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.time import TimeNs
from dagkit.types import Log, row_times
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import serde
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence
from simplecv.types.rig import RigCameras

BoxSource: TypeAlias = Literal["yolox", "tracked", "none"]
"""Per-frame person-box provenance: detected, projected from the tracked skeleton, or absent."""
VARIANTS: tuple[str, str] = ("init", "refined")
VARIANT_COLOR: dict[str, tuple[int, int, int]] = {"init": (255, 214, 0), "refined": (0, 200, 255)}
"""One colour per rig variant for everything drawn about it: frusta, 3D keypoints, error lines."""
OVERLAY_COLOR: tuple[int, int, int] = (240, 140, 60)
RAW_COLOR: tuple[int, int, int] = (110, 110, 110)


def pinhole_leaf(root: str, view: str) -> str:
    """Overlay entity of a keypoint root under a camera: ``/world/<view>/pinhole/exocalib_<last root part>``."""
    return f"/world/{view}/pinhole/exocalib_{root.rsplit('/', 1)[1]}"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RigLayout:
    """One segment's cameras: the exo ones to calibrate, the worn ego ones, and those with dataset GT calibration."""

    exo_camera_names: tuple[str, ...]
    ego_camera_names: tuple[str, ...]
    calibrated_camera_names: tuple[str, ...]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MultiViewKeypoints:
    """Stage B record of one frameset: raw RTMW-x COCO-133 keypoints per camera with the box and crop they came from."""

    frame_index: int
    """Decoder sample index of the frameset."""
    views: tuple[str, ...]
    cam_t_ns: Int64[ndarray, " v"]
    kp_xy: Float32[ndarray, "v 133 2"]
    """NaN where no detection."""
    conf: Float32[ndarray, "v 133"]
    """Raw pose-network confidences; 0 where no detection."""
    bbox_xyxy: Float32[ndarray, "v 4"]
    """NaN rows where no box."""
    box_source: tuple[BoxSource, ...]
    crop_origin_xy: Float32[ndarray, "v 2"]
    crop_size_wh: Float32[ndarray, "v 2"]
    crop_input_wh: tuple[int, int]
    """Pose model crop input size; the AssemblyHands-X margin rule works in this pixel frame."""
    video_wh: tuple[int, int]

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = []
        for i, view in enumerate(self.views):
            record = rr.AnyValues(
                view_index=i, frame_index=self.frame_index, cam_t_ns=int(self.cam_t_ns[i]), kp_xy=self.kp_xy[i].reshape(-1), conf=self.conf[i],
                bbox_xyxy=self.bbox_xyxy[i], box_source=self.box_source[i], crop_origin_xy=self.crop_origin_xy[i], crop_size_wh=self.crop_size_wh[i],
                crop_input_wh=np.asarray(self.crop_input_wh), video_wh=np.asarray(self.video_wh),
            )  # fmt: skip
            box: Float32[ndarray, "k 4"] = self.bbox_xyxy[i : i + 1] if np.isfinite(self.bbox_xyxy[i]).all() else np.zeros((0, 4), np.float32)
            crop: Float32[ndarray, "k 4"] = np.concatenate((self.crop_origin_xy[i], self.crop_size_wh[i]))[None][: len(box)]
            color: tuple[int, int, int] = (0, 200, 255) if self.box_source[i] == "tracked" else OVERLAY_COLOR
            logs += [
                Log(f"{root}/{view}", record),
                Log(f"{pinhole_leaf(root, view)}_bbox", rr.Boxes2D(array=box, array_format=rr.Box2DFormat.XYXY, colors=color, labels=[self.box_source[i]] * len(box))),
                Log(f"{pinhole_leaf(root, view)}_crop", rr.Boxes2D(array=crop, array_format=rr.Box2DFormat.XYWH, colors=RAW_COLOR)),
            ]
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "MultiViewKeypoints"]]:
        views: list[str] = [c.removeprefix(f"{root}/").removesuffix(":view_index") for c in table.column_names if c.endswith(":view_index")]
        columns: dict[str, list] = {c: table.column(c).to_pylist() for c in table.column_names if c.startswith(f"{root}/")}
        for row, t_ns in enumerate(row_times(table, timeline)):

            def field(name: str, dtype: type, views: list[str] = views, row: int = row) -> ndarray:
                return np.stack([np.asarray(columns[f"{root}/{view}:{name}"][row], dtype=dtype) for view in views])

            order: Int64[ndarray, " v"] = np.argsort(field("view_index", np.int64)[:, 0])
            ordered: list[str] = [views[i] for i in order]
            yield t_ns, cls(
                int(field("frame_index", np.int64, ordered)[0, 0]), tuple(ordered), field("cam_t_ns", np.int64, ordered)[:, 0],
                field("kp_xy", np.float32, ordered).reshape(-1, 133, 2), field("conf", np.float32, ordered), field("bbox_xyxy", np.float32, ordered),
                tuple(field("box_source", str, ordered)[:, 0].tolist()), field("crop_origin_xy", np.float32, ordered),
                field("crop_size_wh", np.float32, ordered), tuple(field("crop_input_wh", np.int64, ordered)[0].tolist()),
                tuple(field("video_wh", np.int64, ordered)[0].tolist()),
            )  # fmt: skip


@dataclass(frozen=True, slots=True)
class Kp2dOverlay:
    """The AssemblyHands-X-gated skeleton of one frameset (visualization): kept joints, and the raw ones the gate rejected."""

    views: tuple[str, ...]
    gated_xy: Float64[ndarray, "v 133 2"]
    gated_conf: Float64[ndarray, "v 133"]
    """0 where the gate rejected the joint."""
    raw_xy: Float32[ndarray, "v 133 2"]
    raw_conf: Float32[ndarray, "v 133"]

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = []
        for i, view in enumerate(self.views):
            if not np.isfinite(self.raw_xy[i]).any():
                continue
            keep = self.gated_conf[i] > 0.0
            rejected = ~keep & np.isfinite(self.raw_xy[i]).all(axis=1)
            gated = Points2DWithConfidence(
                positions=np.where(keep[:, None], self.gated_xy[i], np.nan), confidences=self.gated_conf[i], class_ids=0, keypoint_ids=np.arange(133), radii=2.0
            )  # NaN joints also drop their skeleton edges
            raw = Points2DWithConfidence(
                positions=self.raw_xy[i][rejected], confidences=self.raw_conf[i][rejected], class_ids=0, keypoint_ids=np.flatnonzero(rejected), radii=1.5
            )
            logs += [Log(pinhole_leaf(root, view), gated), Log(f"{pinhole_leaf(root, view)}_rejected", raw)]
        return logs


@dataclass(frozen=True, slots=True)
class Keypoints3dTracks:
    """One frame of triangulated keypoints under both rigs (rows of ``xyz`` are ``VARIANTS``; NaN = not triangulated)."""

    keypoint_ids: Int64[ndarray, " n"]
    xyz: Float64[ndarray, "2 n 3"]
    conf: Float64[ndarray, "2 n"]
    """Kineo pairwise reprojection-consensus confidence."""

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = []
        for k, variant in enumerate(VARIANTS):
            keep = np.isfinite(self.xyz[k]).all(axis=1)
            if keep.any():
                points = Points3DWithConfidence(
                    positions=self.xyz[k][keep], confidences=self.conf[k][keep], class_ids=k + 1, keypoint_ids=self.keypoint_ids[keep], radii=0.008
                )
                logs.append(Log(f"{root}_{variant}", points))
        return logs


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RefinementDiagnostics:
    """What the refinement reports about its own run; static ``AnyValues`` on the refined rig entity, copied into ``eval.json``."""

    init_reprojection_px: float
    """Mean reprojection error of the robustly triangulated points against the Stage A cameras."""
    ba_reprojection_px: list[float]
    """After each bundle-adjustment round, then after the focal stage (and the rotation guard, when it fired)."""
    observation_count_per_view: list[int]
    metric_rescale_fix: float
    """Global scale applied after BA from MoGe-2 depth; ``1.0`` when the estimate was skipped."""
    focal_scale: list[float]

    def to_rerun(self, root: str) -> list[Log]:
        values = rr.AnyValues(
            init_reprojection_px=np.array([self.init_reprojection_px]), ba_reprojection_px=np.asarray(self.ba_reprojection_px),
            observation_count_per_view=np.asarray(self.observation_count_per_view, dtype=np.int64),
            metric_rescale_fix=np.array([self.metric_rescale_fix]), focal_scale=np.asarray(self.focal_scale),
        )  # fmt: skip
        return [Log(root, values, static=True)]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BaseCalibration:
    """Refined cameras written onto base camera nodes that carry no dataset GT; the marker keeps them out of GT discovery."""

    rig: RigCameras

    def to_rerun(self, root: str) -> list[Log]:
        return [*self.rig.to_rerun(root), *(Log(f"{root}/{name}", rr.AnyValues(exocalib_written=True), static=True) for name in self.rig.names)]


@dataclass(frozen=True, slots=True)
class RigAlignment:
    """SE(3) alignment of each rig variant onto GT (pred world -> GT world) and the aligned camera centres."""

    names: tuple[str, ...]
    gt_centers: Float64[ndarray, "v 3"]
    gt_R_pred: Float64[ndarray, "2 3 3"]
    gt_t_pred: Float64[ndarray, "2 3"]
    aligned_centers: Float64[ndarray, "2 v 3"]

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = []
        for k, variant in enumerate(VARIANTS):
            transform = rr.Transform3D(translation=self.gt_t_pred[k], mat3x3=self.gt_R_pred[k])
            errors_cm: Float64[ndarray, " v"] = np.linalg.norm(self.aligned_centers[k] - self.gt_centers, axis=1) * 100.0
            lines = rr.LineStrips3D(
                strips=np.stack((self.aligned_centers[k], self.gt_centers), axis=1), labels=[f"{n}: {e:.1f} cm" for n, e in zip(self.names, errors_cm, strict=True)],
                colors=VARIANT_COLOR[variant], radii=0.004, show_labels=True,
            )  # fmt: skip
            logs += [Log(f"{root}/exocalib/{variant}", transform, True), Log(f"{root}/exocalib/kp3d_{variant}", transform, True)]
            logs.append(Log(f"{root}/exocalib_error/{variant}", lines, static=True))
        return logs


@serde
@dataclass(frozen=True, slots=True)
class ErrorStats:
    per_camera: list[float]
    mean: float
    median: float
    max: float


@serde
@dataclass(frozen=True, slots=True)
class AlignedErrors:
    rotation_deg: ErrorStats
    translation_cm: ErrorStats
    scale: float
    """Scale the alignment applied (1.0 under SE(3))."""


@serde
@dataclass(frozen=True, slots=True)
class FocalErrors:
    per_camera: list[float]
    mean_abs: float
    max_abs: float


@serde
@dataclass(frozen=True, slots=True)
class VariantMetrics:
    se3: AlignedErrors
    sim3: AlignedErrors
    focal_error_pct: FocalErrors


@serde
@dataclass(frozen=True, slots=True)
class EvalReport:
    """``eval.json``: per-variant GT metrics (``None`` without GT, written as null) and the GT-free diagnostics."""

    segment_id: str
    dataset_name: str
    camera_names: list[str]
    init: VariantMetrics | None
    refined: VariantMetrics | None
    refinement_diagnostics: RefinementDiagnostics
    camera_centers_m: list[list[float]]
    """Refined rig in its own floor frame (z = 0 is the fitted floor)."""
    pairwise_distance_m: list[list[float]]

