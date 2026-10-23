"""exo-calib edge values and their Rerun mapping (the rig itself is ``simplecv.types.rig.RigCameras``).

Entity layout: rigs at ``/world/exocalib/{init,refined}/<rig>/<cam>`` (the Stage A scale on the init root); the Stage B
record at ``/world/exocalib/kp2d/<cam>`` with its boxes, raw joints and the gated skeleton drawn under each camera's
``pinhole/exocalib_kp2d*``; 3D keypoints at ``/world/exocalib/kp3d_{init,refined}``; the SMPL-X body at
``/world/exocalib/body``; the GT alignment on every root of a variant's world (``VARIANT_WORLD``) plus error lines at
``/world/exocalib_error``, and whether there was GT at ``/world/exocalib_report``.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.time import STATIC, TimeNs
from dagkit.types import Log, column, flat, row_times
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
VARIANT_WORLD: dict[str, tuple[str, ...]] = {"init": ("init", "kp3d_init"), "refined": ("refined", "kp3d_refined", "body")}
"""Roots under ``/world/exocalib`` that live in each rig variant's world frame (the body is fit through the refined rig)."""
OVERLAY_COLOR: tuple[int, int, int] = (240, 140, 60)
RAW_COLOR: tuple[int, int, int] = (110, 110, 110)


def pinhole_leaf(root: str, view: str) -> str:
    """Overlay entity of a keypoint root under a camera: ``/world/<view>/pinhole/exocalib_<last root part>``."""
    return f"/world/{view}/pinhole/exocalib_{root.rsplit('/', 1)[1]}"


class Drawing:
    """Base of edge values that are drawn, never resumed (SPEC 2.6: only Types with ``from_layer`` resume).

    Each one is derived from values that do resume (the kp2d record, the rigs), so a resume re-derives it, or stays in the
    layer a resume reads from (the Stage A scale); these two methods complete dagkit's ``Type`` protocol for the sinks and
    refuse a read-back.
    """

    __slots__ = ()

    @classmethod
    def contents(cls, root: str) -> list[str]:
        raise NotImplementedError(f"{cls.__name__} is drawn, not resumed: re-derive it from the values it was drawn from")

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Drawing"]]:
        raise NotImplementedError(f"{cls.__name__} is drawn, not resumed: re-derive it from the values it was drawn from")


LensModel: TypeAlias = Literal["pinhole", "kannala_brandt", "brown_conrady"]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Lens:
    """A camera's lens model and its terms in the catalog's order (``simplecv.rerun_custom_types.CameraDistortion``):
    Kannala-Brandt ``k1..k6, p1, p2``, Brown-Conrady OpenCV's ``k1, k2, p1, p2, k3..k6, s1..s4, tau_x, tau_y``."""

    model: LensModel
    coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.model == "kannala_brandt" and any(self.coefficients[4:]):
            raise ValueError(f"a Kannala-Brandt lens here models k1..k4 only (OpenCV fisheye); got {self.coefficients}")


PINHOLE = Lens("pinhole")
FISHEYE = Lens("kannala_brandt", (0.0, 0.0, 0.0, 0.0))
"""Equidistant: the ``fisheye`` fallback for a camera whose segment stores no lens."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RigLayout:
    """One segment's cameras: the exo ones to calibrate, the worn ego ones, and those with dataset GT calibration."""

    exo_camera_names: tuple[str, ...]
    ego_camera_names: tuple[str, ...]
    calibrated_camera_names: tuple[str, ...]
    lenses: dict[str, Lens] = field(default_factory=dict)
    """Each exo camera's lens: the one its segment stores, else the Config's."""

    def lens(self, camera: str) -> Lens:
        """The camera's lens; a camera missing from ``lenses`` is a pinhole."""
        return self.lenses.get(camera, PINHOLE)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MetricScale(Drawing):
    """Stage A's global metric scale: static ``AnyValues`` on the init rig entity."""

    metric_scale: float
    """Median over the views of MoGe-2 metric depth over G3T depth."""

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(root, rr.AnyValues(metric_scale=np.array([self.metric_scale])), static=True)]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ReportStatus(Drawing):
    """Whether Stage E had GT for every exo camera (without it, nothing is scored or aligned): static ``AnyValues``."""

    ground_truth_available: bool
    """False: an exo camera has no GT calibration in the segment (main's report logged only this case)."""

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(root, rr.AnyValues(ground_truth_available=self.ground_truth_available), static=True)]


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
        """The queryable record, one entity per camera at the frameset time (``cam_t_ns`` keeps each camera's own;
        ``Kp2dOverlay`` draws the boxes and skeletons at those times)."""
        return [
            Log(f"{root}/{view}", rr.AnyValues(
                view_index=i, frame_index=self.frame_index, cam_t_ns=int(self.cam_t_ns[i]), kp_xy=self.kp_xy[i].reshape(-1), conf=self.conf[i],
                bbox_xyxy=self.bbox_xyxy[i], box_source=self.box_source[i], crop_origin_xy=self.crop_origin_xy[i], crop_size_wh=self.crop_size_wh[i],
                crop_input_wh=np.asarray(self.crop_input_wh), video_wh=np.asarray(self.video_wh),
            ))
            for i, view in enumerate(self.views)
        ]  # fmt: skip

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "MultiViewKeypoints"]]:
        views: list[str] = [c.removeprefix(f"{root}/").removesuffix(":view_index") for c in table.column_names if c.endswith(":view_index")]
        dtypes: dict[str, type] = dict.fromkeys(("view_index", "frame_index", "cam_t_ns", "crop_input_wh", "video_wh"), np.int64) | {"box_source": str}
        dtypes |= dict.fromkeys(("kp_xy", "conf", "bbox_xyxy", "crop_origin_xy", "crop_size_wh"), np.float32)
        fields: dict[str, ndarray] = {  # each field stacked once: (rows, views, values)
            name: np.stack([np.asarray(table.column(f"{root}/{view}:{name}").to_pylist(), dtype=dtype) for view in views], axis=1) for name, dtype in dtypes.items()
        }  # fmt: skip
        for row, t_ns in enumerate(row_times(table, timeline)):
            order: Int64[ndarray, " v"] = np.argsort(fields["view_index"][row, :, 0])
            cells: dict[str, ndarray] = {name: values[row][order] for name, values in fields.items()}
            yield t_ns, cls(
                int(cells["frame_index"][0, 0]), tuple(views[i] for i in order), cells["cam_t_ns"][:, 0], cells["kp_xy"].reshape(-1, 133, 2), cells["conf"],
                cells["bbox_xyxy"], tuple(cells["box_source"][:, 0].tolist()), cells["crop_origin_xy"], cells["crop_size_wh"],
                tuple(cells["crop_input_wh"][0].tolist()), tuple(cells["video_wh"][0].tolist()),
            )  # fmt: skip


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Kp2dOverlay(Drawing):
    """One camera's Stage B drawing at that camera's own sample time: the person box (and its ``box_source``), the crop rectangle and the raw
    joints (``_raw``, grey, on every frame), plus (on the frames that carry ``gated_*``) the AssemblyHands-X-gated
    skeleton with the raw joints the gate rejected beside it; a frame without them clears the last ones. With no box and no
    joint (``nothing``: the camera's next video frame) it clears them all. Every value writes all five entities, so the viewer
    never shows one sample's skeleton beside another sample's box."""

    view: str
    bbox_xyxy: Float32[ndarray, "4"]
    """NaN when the frame had no box."""
    box_source: BoxSource
    crop_xywh: Float32[ndarray, "4"]
    raw_xy: Float32[ndarray, "133 2"]
    raw_conf: Float32[ndarray, "133"]
    gated_xy: Float64[ndarray, "133 2"] | None = None
    gated_conf: Float64[ndarray, "133"] | None = None
    """0 where the gate rejected the joint."""

    @classmethod
    def nothing(cls, view: str) -> "Kp2dOverlay":
        nan4: Float32[ndarray, "4"] = np.full(4, np.nan, np.float32)
        return cls(view, nan4, "none", nan4, np.full((133, 2), np.nan, np.float32), np.zeros(133, np.float32))

    def to_rerun(self, root: str) -> list[Log]:
        leaf: str = pinhole_leaf(root, self.view)
        if not np.isfinite(self.bbox_xyxy).all() and not np.isfinite(self.raw_xy).any():
            return [Log(f"{leaf}{suffix}", rr.Clear(recursive=False)) for suffix in ("", "_raw", "_rejected", "_bbox", "_crop")]
        box: Float32[ndarray, "k 4"] = self.bbox_xyxy[None] if np.isfinite(self.bbox_xyxy).all() else np.zeros((0, 4), np.float32)
        color: tuple[int, int, int] = (0, 200, 255) if self.box_source == "tracked" else OVERLAY_COLOR
        logs: list[Log] = [
            Log(f"{leaf}_bbox", rr.Boxes2D(array=box, array_format=rr.Box2DFormat.XYXY, colors=color, labels=[self.box_source] * len(box))),
            Log(f"{leaf}_bbox", rr.AnyValues(box_source=self.box_source)),
            Log(f"{leaf}_crop", rr.Boxes2D(array=self.crop_xywh[None][: len(box)], array_format=rr.Box2DFormat.XYWH, colors=RAW_COLOR)),
            Log(f"{leaf}_raw", Points2DWithConfidence(positions=self.raw_xy, confidences=self.raw_conf, class_ids=0, keypoint_ids=np.arange(133), radii=1.0)),
        ]
        if self.gated_xy is None or self.gated_conf is None or not np.isfinite(self.raw_xy).any():
            return [*logs, Log(leaf, rr.Clear(recursive=False)), Log(f"{leaf}_rejected", rr.Clear(recursive=False))]
        keep = self.gated_conf > 0.0
        rejected = ~keep & np.isfinite(self.raw_xy).all(axis=1)
        gated = Points2DWithConfidence(
            positions=np.where(keep[:, None], self.gated_xy, np.nan), confidences=self.gated_conf, class_ids=0, keypoint_ids=np.arange(133), radii=2.0
        )  # NaN joints also drop their skeleton edges
        raw = Points2DWithConfidence(
            positions=self.raw_xy[rejected], confidences=self.raw_conf[rejected], class_ids=0, keypoint_ids=np.flatnonzero(rejected), radii=1.5
        )
        return [*logs, Log(leaf, gated), Log(f"{leaf}_rejected", raw)]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Keypoints3dTracks(Drawing):
    """One frame of triangulated keypoints under both rigs (rows of ``xyz`` are ``VARIANTS``; NaN = not triangulated). A rig
    with no joint clears its drawing (``empty``: the video frame after the frameset)."""

    keypoint_ids: Int64[ndarray, " n"]
    xyz: Float64[ndarray, "2 n 3"]
    conf: Float64[ndarray, "2 n"]
    """Kineo pairwise reprojection-consensus confidence."""

    @classmethod
    def empty(cls) -> "Keypoints3dTracks":
        return cls(np.zeros(0, np.int64), np.zeros((2, 0, 3)), np.zeros((2, 0)))

    def to_rerun(self, root: str) -> list[Log]:
        logs: list[Log] = []
        for k, variant in enumerate(VARIANTS):
            keep = np.isfinite(self.xyz[k]).all(axis=1)
            if not keep.any():
                logs.append(Log(f"{root}_{variant}", rr.Clear(recursive=False)))
                continue
            points = Points3DWithConfidence(
                positions=self.xyz[k][keep], confidences=self.conf[k][keep], class_ids=k + 1, keypoint_ids=self.keypoint_ids[keep], radii=0.008
            )
            logs.append(Log(f"{root}_{variant}", points))
        return logs


BODY_COLOR: tuple[int, int, int] = (150, 210, 150)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyShape:
    """The fitted SMPL-X body's shape for the whole segment (STATIC): its shape coefficients and the mesh's triangles."""

    betas: Float32[ndarray, " b"]
    faces: Int64[ndarray, "f 3"]

    def to_rerun(self, root: str) -> list[Log]:
        mesh = rr.Mesh3D.from_fields(triangle_indices=self.faces.astype(np.uint32), albedo_factor=[*BODY_COLOR, 160])
        return [Log(root, mesh, static=True), Log(root, rr.AnyValues(betas=self.betas), static=True)]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "BodyShape"]]:
        if table.num_rows and column(table, f"{root}:betas")[0] is not None:
            faces: Int64[ndarray, "f 3"] = flat(table[f"{root}:Mesh3D:triangle_indices"][0], np.int64).reshape(-1, 3)
            yield STATIC, cls(flat(table[f"{root}:betas"][0], np.float32), faces)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyPose:
    """One frameset's fitted SMPL-X body in the refined (metric) rig's world: the axis-angle pose (smplx order: global, body,
    jaw, eyes, hands), the root translation and the mesh it gives. An empty one (no vertices) clears the drawing."""

    pose: Float32[ndarray, " 165"]
    trans: Float32[ndarray, " 3"]
    vertices: Float32[ndarray, "n 3"]

    @classmethod
    def empty(cls) -> "BodyPose":
        return cls(np.zeros(165, np.float32), np.zeros(3, np.float32), np.zeros((0, 3), np.float32))

    def to_rerun(self, root: str) -> list[Log]:
        if not len(self.vertices):
            return [Log(root, rr.Clear(recursive=False))]
        return [Log(root, rr.Mesh3D.from_fields(vertex_positions=self.vertices)), Log(root, rr.AnyValues(pose=self.pose, trans=self.trans))]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "BodyPose"]]:
        for row, t_ns in enumerate(row_times(table, timeline)):
            pose: Float32[ndarray, " p"] = flat(table[f"{root}:pose"][row], np.float32)
            if not len(pose):  # a Clear writes empty components
                yield t_ns, cls.empty()
                continue
            vertices: Float32[ndarray, "n 3"] = flat(table[f"{root}:Mesh3D:vertex_positions"][row], np.float32).reshape(-1, 3)
            yield t_ns, cls(pose, flat(table[f"{root}:trans"][row], np.float32), vertices)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RefinementDiagnostics:
    """What the refinement reports about its own run; static ``AnyValues`` on the refined rig entity, copied into ``eval.json``."""

    init_reprojection_px: float
    """Mean reprojection error of the robustly triangulated points against the Stage A poses, each camera model at one focal."""
    ba_reprojection_px: list[float]
    """After each refinement round (one, or ``LENS_ROUNDS`` when a camera has a lens)."""
    observation_count_per_view: list[int]
    metric_rescale_fix: float
    """Global scale applied after the refinement (the depth and body scales fused); ``1.0`` when neither was measured."""
    focal_scale: list[float]
    """Each camera's refined focal over its Stage A focal: refine starts every camera of one model at one focal (their median) and
    scales them together, so cameras of one model share the refined focal, not this ratio."""
    depth_scale: float | None = None
    """The scale MoGe-2's scene depth gave (None: skipped, or a layer written before it was kept)."""
    body_scale: float | None = None
    """The scale the person's arm bones gave (None: too few arm bones)."""

    def to_rerun(self, root: str) -> list[Log]:
        measured: dict[str, Any] = {name: np.array([v]) for name, v in (("depth_scale", self.depth_scale), ("body_scale", self.body_scale)) if v is not None}
        values = rr.AnyValues(
            init_reprojection_px=np.array([self.init_reprojection_px]), ba_reprojection_px=np.asarray(self.ba_reprojection_px),
            observation_count_per_view=np.asarray(self.observation_count_per_view, dtype=np.int64),
            metric_rescale_fix=np.array([self.metric_rescale_fix]), focal_scale=np.asarray(self.focal_scale), **measured,
        )  # fmt: skip
        return [Log(root, values, static=True)]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "RefinementDiagnostics"]]:
        def value(name: str) -> Float64[ndarray, " k"]:
            return flat(table[f"{root}:{name}"][0], np.float64)

        def measured(name: str) -> float | None:
            return None if column(table, f"{root}:{name}")[0] is None else float(value(name)[0])

        if table.num_rows and column(table, f"{root}:focal_scale")[0] is not None:
            counts: list[int] = value("observation_count_per_view").astype(np.int64).tolist()
            yield STATIC, cls(
                float(value("init_reprojection_px")[0]), value("ba_reprojection_px").tolist(), counts, float(value("metric_rescale_fix")[0]), value("focal_scale").tolist(),
                measured("depth_scale"), measured("body_scale"),
            )  # fmt: skip


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BaseCalibration:
    """Refined cameras written onto base camera nodes that carry no dataset GT; the marker keeps them out of GT discovery."""

    rig: RigCameras

    def to_rerun(self, root: str) -> list[Log]:
        return [*self.rig.to_rerun(root), *(Log(f"{root}/{name}", rr.AnyValues(exocalib_written=True), static=True) for name in self.rig.names)]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "BaseCalibration"]]:
        """Only the cameras this Type wrote (the ``exocalib_written`` marker), not the dataset's GT cameras beside them."""
        marker: str = ":exocalib_written"
        written: set[str] = {c.removeprefix(f"{root}/").removesuffix(marker) for c in table.column_names if c.endswith(marker) and table.column(c)[0].as_py() is not None}
        for t_ns, rig in RigCameras.from_layer(table, root, timeline):
            if written:
                yield t_ns, cls(rig.select([name for name in rig.names if name in written]))


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RigAlignment(Drawing):
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
            logs += [Log(f"{root}/exocalib/{world}", transform, True) for world in VARIANT_WORLD[variant]]
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
class RigGeometryReport:
    """The refined rig in its own floor frame (z = 0 is the fitted floor), metres rounded to 6 digits."""

    camera_centers_m: list[list[float]]
    pairwise_distance_m: list[list[float]]
    height_above_floor_m: list[float]


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
    refined_rig_geometry: RigGeometryReport

