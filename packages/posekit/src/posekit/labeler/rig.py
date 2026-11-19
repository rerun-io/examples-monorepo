"""RoboCap rig geometry for the MEgATrack propagator: the KB4 fisheye lens, robust N-view triangulation of hand keypoints,
the monocular lift of one view's keypoints, and MEgATrack's box around projected keypoints.

The rig comes from the segment's statics (the ``base`` layer): ``/world/rig_00/cam_NN`` holds the camera-from-rig
``Transform3D`` (relation ``ChildFromParent``) and ``.../pinhole`` the ``Pinhole`` K plus ``simplecv.components.DistortionModel``
``kannala_brandt`` with 8 ``DistortionCoefficients`` ``[k1..k4, 0, 0, 0, 0]`` (Fisheye62 with k5 = k6 = p1 = p2 = 0, which is
exactly Kannala-Brandt KB4). Points are in metres; pixels are native image pixels (pixel centres at integers).

KB4: a camera-frame point at angle ``theta`` off the optical axis lands at normalised radius
``theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)`` along its azimuth; ``unproject`` inverts it with
Newton steps on ``theta``, so pixels become exact unit rays (no OpenCV k1/k2 undistortion, no pinhole limit at 90 degrees).

Triangulation (``fuse_views``) is view-robust, because a whole view can be wrong (its crop held the other hand): every pair of
views proposes the 21 points (least squares on rays), every view scores the proposal by its median keypoint reprojection error,
the pair with the most inlier views wins, then each joint is re-solved from the inlier views (confidence weights) and a view
whose error on that joint exceeds ``JOINT_OUTLIER`` gates is dropped for that joint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.rrd_query_utils import first_valid_value

from posekit.labeler.paths import pinhole_path
from posekit.labeler.schema import Box

NUM_KEYPOINTS: int = 21
"""UmeTrack landmarks (KeyNet's order: five fingertips, wrist, finger joints, palm centre)."""
NEWTON_STEPS: int = 12
"""Newton steps of the KB4 inverse (converges to 1e-12 rad in ~5 on RoboCap lenses)."""
MIN_KEYPOINT_CONFIDENCE: float = 0.2
"""KeyNet heatmap peak from which a keypoint takes part in triangulation."""
MIN_POINTS_PER_VIEW: int = 8
"""Keypoints a view must contribute to score or support a hypothesis."""
INLIER_FRACTION: float = 0.15
"""A view is an inlier when its median reprojection error is at most this fraction of the hand's extent in that view."""
MIN_GATE_PX: float = 8.0
"""Floor of the inlier gate, px (small, far hands)."""
JOINT_OUTLIER: float = 3.0
"""In the final solve, a view's ray is dropped for one joint whose error exceeds this many of the view's gates."""
MIN_RAY_ANGLE_DEG: float = 1.0
"""Two rays closer than this leave depth free: a pair hypothesis needs the median angle of its rays above it."""
BONES: tuple[tuple[int, int, float], ...] = (
    (5, 6, 81.88), (6, 7, 33.79), (7, 0, 24.62),
    (5, 8, 99.11), (8, 9, 37.93), (9, 10, 24.30), (10, 1, 22.39),
    (5, 11, 95.70), (11, 12, 42.93), (12, 13, 27.55), (13, 2, 24.99),
    (5, 14, 90.63), (14, 15, 39.00), (15, 16, 26.57), (16, 3, 24.38),
    (5, 17, 86.51), (17, 18, 30.72), (18, 19, 20.31), (19, 4, 21.96),
    (8, 17, 61.67), (5, 20, 84.41),
)  # fmt: skip
"""Rigid landmark distances of the generic UmeTrack hand (handtrack ``assets/generic_hand_model.json`` rest landmarks), mm:
the 19 finger bones, the index-to-pinky knuckle span and wrist-to-palm-centre. The monocular lift scales a view's rays to them."""
LIFT_RANGE_M: tuple[float, float] = (0.08, 2.0)
"""Mean keypoint distances the monocular lift searches, m."""
MIN_INSIDE_KEYPOINTS: int = 11
"""A view gets a box when at least this many of the 21 projected keypoints lie in front of it and inside its image."""
MIN_BOX_SIDE_PX: float = 4.0
BOX_MARGIN: float = 1.2
"""The box's circle is the keypoints' minimum enclosing circle grown by this factor: keypoints are joint centres, the hand's
skin reaches past them (KeyNet's crops use the same 1.2 margin around the keypoints)."""
"""Smallest side of a box after clipping, px."""


@dataclass(frozen=True, slots=True)
class Kb4Rig:
    """The calibrated cameras of one RoboCap segment, rig order."""

    names: tuple[str, ...]
    """Camera names (``left_front`` ...)."""
    size_wh: Float64[ndarray, "c 2"]
    """Image width and height, px."""
    focal: Float64[ndarray, "c 2"]
    """fx, fy, px."""
    principal: Float64[ndarray, "c 2"]
    """cx, cy, px."""
    kb4: Float64[ndarray, "c 4"]
    """k1..k4."""
    cam_from_rig: Float64[ndarray, "c 4 4"]
    """Rigid transform from the rig frame to each camera frame (OpenCV RDF), metres."""

    def __post_init__(self) -> None:
        count: int = len(self.names)
        shapes: list[tuple[int, ...]] = [self.size_wh.shape, self.focal.shape, self.principal.shape, self.kb4.shape, self.cam_from_rig.shape]
        if shapes != [(count, 2), (count, 2), (count, 2), (count, 4), (count, 4, 4)]:
            raise ValueError(f"Kb4Rig arrays do not match {count} cameras: {shapes}")

    @property
    def cams(self) -> list[int]:
        """Camera indices."""
        return list(range(len(self.names)))

    def to_cam(self, cam: int, points_rig: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
        """Rig-frame points in camera ``cam``'s frame."""
        transform: Float64[ndarray, "4 4"] = self.cam_from_rig[cam]
        return points_rig @ transform[:3, :3].T + transform[:3, 3]

    def centre(self, cam: int) -> Float64[ndarray, "3"]:
        """Camera ``cam``'s optical centre in the rig frame."""
        transform: Float64[ndarray, "4 4"] = self.cam_from_rig[cam]
        return -transform[:3, :3].T @ transform[:3, 3]

    def rays_to_rig(self, cam: int, rays_cam: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
        """Camera-frame directions as rig-frame directions."""
        return rays_cam @ self.cam_from_rig[cam][:3, :3]

    def project(self, cam: int, points_cam: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
        """KB4 pixels of camera-frame points (unclipped; points behind the camera still project, check ``z > 0``)."""
        x: Float64[ndarray, "n"] = points_cam[:, 0]
        y: Float64[ndarray, "n"] = points_cam[:, 1]
        radius: Float64[ndarray, "n"] = np.sqrt(x * x + y * y)
        theta: Float64[ndarray, "n"] = np.arctan2(radius, points_cam[:, 2])
        theta2: Float64[ndarray, "n"] = theta * theta
        k1, k2, k3, k4 = (float(value) for value in self.kb4[cam])
        theta_d: Float64[ndarray, "n"] = theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
        # On the optical axis theta_d / radius -> 1 / z.
        scale: Float64[ndarray, "n"] = np.where(radius > 1e-12, theta_d / np.maximum(radius, 1e-12), 1.0 / np.where(np.abs(points_cam[:, 2]) > 1e-12, points_cam[:, 2], 1e-12))
        normalised: Float64[ndarray, "n 2"] = np.stack([x * scale, y * scale], axis=-1)
        return normalised * self.focal[cam] + self.principal[cam]

    def unproject(self, cam: int, pixels: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]:
        """Unit camera-frame rays through native pixels (the exact KB4 inverse by Newton steps on ``theta``)."""
        normalised: Float64[ndarray, "n 2"] = (pixels - self.principal[cam]) / self.focal[cam]
        theta_d: Float64[ndarray, "n"] = np.linalg.norm(normalised, axis=-1)
        k1, k2, k3, k4 = (float(value) for value in self.kb4[cam])
        theta: Float64[ndarray, "n"] = theta_d.copy()
        for _ in range(NEWTON_STEPS):
            theta2: Float64[ndarray, "n"] = theta * theta
            value: Float64[ndarray, "n"] = theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))) - theta_d
            slope: Float64[ndarray, "n"] = 1.0 + theta2 * (3.0 * k1 + theta2 * (5.0 * k2 + theta2 * (7.0 * k3 + theta2 * 9.0 * k4)))
            theta = np.clip(theta - value / np.where(np.abs(slope) > 1e-9, slope, 1e-9), 0.0, np.pi * 0.999)
        azimuth: Float64[ndarray, "n 2"] = np.where(theta_d[:, None] > 1e-12, normalised / np.maximum(theta_d, 1e-12)[:, None], 0.0)
        return np.concatenate([azimuth * np.sin(theta)[:, None], np.cos(theta)[:, None]], axis=-1)


def read_rig(dataset: DatasetEntry, segment_id: str) -> Kb4Rig:
    """The rig-0 cameras of ``segment_id`` from the catalog statics.

    Raises:
        ValueError: If a camera lacks its calibration, its extrinsic is not camera-from-rig, or its lens is not KB4.
    """
    cams: list[int] = sorted(
        int(column.entity_path.split("/")[3].removeprefix("cam_"))
        for column in dataset.filter_segments(segment_id).schema().component_columns()
        if column.component == "Pinhole:image_from_camera" and column.entity_path.startswith("/world/rig_00/cam_")
    )
    if cams != list(range(len(cams))) or not cams:
        raise ValueError(f"{segment_id}: expected rig-0 cameras cam_00..cam_NN with a Pinhole, found {cams}")
    entities: list[str] = [pinhole_path(cam).removesuffix("/pinhole") for cam in cams]
    statics: pa.Table = dataset.filter_segments(segment_id).filter_contents(entities + [pinhole_path(cam) for cam in cams]).reader(index=None).to_arrow_table()

    def floats(column: str) -> Float64[ndarray, "k"]:
        if column not in statics.column_names:
            raise ValueError(f"{segment_id}: no static {column}")
        return np.asarray(first_valid_value(statics[column], component_name=column), dtype=np.float64).ravel()

    def text(column: str) -> str:
        if column not in statics.column_names:
            raise ValueError(f"{segment_id}: no static {column}")
        value: object = first_valid_value(statics[column], component_name=column)
        return str(value[0]) if isinstance(value, list) and value else str(value)

    names: list[str] = []
    size_wh: Float64[ndarray, "c 2"] = np.zeros((len(cams), 2))
    focal: Float64[ndarray, "c 2"] = np.zeros((len(cams), 2))
    principal: Float64[ndarray, "c 2"] = np.zeros((len(cams), 2))
    kb4: Float64[ndarray, "c 4"] = np.zeros((len(cams), 4))
    cam_from_rig: Float64[ndarray, "c 4 4"] = np.tile(np.eye(4), (len(cams), 1, 1))
    for cam, entity in zip(cams, entities, strict=True):
        pinhole: str = pinhole_path(cam)
        names.append(text(f"{entity}:name"))
        if int(floats(f"{entity}:Transform3D:relation")[0]) != rr.components.TransformRelation.ChildFromParent.value:
            raise ValueError(f"{segment_id}: {entity}'s extrinsic is not ChildFromParent (camera from rig)")
        cam_from_rig[cam, :3, :3] = floats(f"{entity}:Transform3D:mat3x3").reshape(3, 3, order="F")
        cam_from_rig[cam, :3, 3] = floats(f"{entity}:Transform3D:translation")
        intrinsics: Float64[ndarray, "3 3"] = floats(f"{pinhole}:Pinhole:image_from_camera").reshape(3, 3, order="F")
        focal[cam] = (intrinsics[0, 0], intrinsics[1, 1])
        principal[cam] = (intrinsics[0, 2], intrinsics[1, 2])
        size_wh[cam] = floats(f"{pinhole}:Pinhole:resolution")[:2]
        model: str = text(f"{pinhole}:simplecv.components.DistortionModel")
        coefficients: Float64[ndarray, "k"] = floats(f"{pinhole}:simplecv.components.DistortionCoefficients")
        if model != "kannala_brandt" or len(coefficients) != 8 or np.any(coefficients[4:] != 0.0):
            raise ValueError(f"{segment_id}: {pinhole} has {model} {coefficients.tolist()}; expected KB4 (kannala_brandt [k1..k4, 0, 0, 0, 0])")
        kb4[cam] = coefficients[:4]
    return Kb4Rig(tuple(names), size_wh, focal, principal, kb4, cam_from_rig)


@dataclass(frozen=True, slots=True)
class ViewKeypoints:
    """One view's 2D hand keypoints (a KeyNet estimate)."""

    cam: int
    """Camera index."""
    uv: Float64[ndarray, "21 2"]
    """Keypoints in native px."""
    confidence: Float64[ndarray, "21"]
    """Per-keypoint confidence (KeyNet heatmap peak)."""
    d_rel_mm: Float64[ndarray, "21"]
    """Distance of each keypoint to the camera centre minus their mean, mm (KeyNet's 1D heatmaps, hand scale 1)."""
    presence: float
    """Probability that the crop holds the hand."""


@dataclass(frozen=True, slots=True)
class Fusion:
    """The 3D hand keypoints the views agree on."""

    points_rig: Float64[ndarray, "21 3"]
    """Rig frame, metres; NaN for a joint no two inlier views solved."""
    inliers: tuple[int, ...]
    """Cameras whose keypoints support the points."""
    errors_px: dict[int, float]
    """Median keypoint reprojection error per observed camera, px."""
    gates_px: dict[int, float]
    """Inlier gate per observed camera, px."""

    @property
    def outliers(self) -> tuple[int, ...]:
        """Observed cameras that do not support the points."""
        return tuple(cam for cam in self.errors_px if cam not in self.inliers)


def hand_extent_px(uv: Float64[ndarray, "k 2"]) -> float:
    """Diameter of the minimum enclosing circle of keypoints, px."""
    radius: float = float(cv2.minEnclosingCircle(uv.astype(np.float32).reshape(-1, 1, 2))[1])
    return 2.0 * radius


def solve_rays(
    centres: Float64[ndarray, "v 3"], rays: Float64[ndarray, "v k 3"], weights: Float64[ndarray, "v k"]
) -> tuple[Float64[ndarray, "k 3"], Bool[ndarray, "k"]]:
    """Per joint, the point closest to its weighted rays (least squares on perpendicular distances).

    Args:
        centres: Ray origins (camera centres), rig frame.
        rays: Unit ray directions, rig frame.
        weights: Per (view, joint) weight; 0 leaves a ray out.

    Returns:
        Points (NaN where unsolved) and which joints were solved (two weighted rays that are not parallel).
    """
    eye: Float64[ndarray, "3 3"] = np.eye(3)
    projector: Float64[ndarray, "v k 3 3"] = eye - rays[..., :, None] * rays[..., None, :]
    lhs: Float64[ndarray, "k 3 3"] = np.einsum("vk,vkij->kij", weights, projector)
    rhs: Float64[ndarray, "k 3"] = np.einsum("vk,vkij,vj->ki", weights, projector, centres)
    counts: Int64[ndarray, "k"] = (weights > 0).sum(axis=0).astype(np.int64)
    conditioned: Bool[ndarray, "k"] = (counts >= 2) & (np.linalg.cond(lhs) < 1e8)
    safe: Float64[ndarray, "k 3 3"] = np.where(conditioned[:, None, None], lhs, eye)
    points: Float64[ndarray, "k 3"] = np.linalg.solve(safe, rhs[..., None])[..., 0]
    return np.where(conditioned[:, None], points, np.nan), conditioned


def reprojection_errors(rig: Kb4Rig, view: ViewKeypoints, points_rig: Float64[ndarray, "21 3"]) -> Float64[ndarray, "21"]:
    """Pixel distance between each solved point's projection and the view's keypoint; NaN for unsolved or behind-camera points."""
    points_cam: Float64[ndarray, "21 3"] = rig.to_cam(view.cam, np.nan_to_num(points_rig))
    pixels: Float64[ndarray, "21 2"] = rig.project(view.cam, points_cam)
    usable: Bool[ndarray, "21"] = np.isfinite(points_rig).all(axis=-1) & (points_cam[:, 2] > 0)
    return np.where(usable, np.linalg.norm(pixels - view.uv, axis=-1), np.nan)


def fuse_views(rig: Kb4Rig, views: Sequence[ViewKeypoints]) -> Fusion | None:
    """The 21 points most views agree on (see the module docstring); None when no two views agree.

    Args:
        rig: The segment's cameras.
        views: One estimate per camera that sees the hand (presence already checked by the caller).
    """
    if len(views) < 2:
        return None
    centres: Float64[ndarray, "v 3"] = np.stack([rig.centre(view.cam) for view in views])
    rays: Float64[ndarray, "v 21 3"] = np.stack([rig.rays_to_rig(view.cam, rig.unproject(view.cam, view.uv)) for view in views])
    usable: Bool[ndarray, "v 21"] = np.stack([view.confidence >= MIN_KEYPOINT_CONFIDENCE for view in views])
    gates: Float64[ndarray, "v"] = np.asarray([max(MIN_GATE_PX, INLIER_FRACTION * hand_extent_px(view.uv)) for view in views])

    def view_errors(points: Float64[ndarray, "21 3"]) -> Float64[ndarray, "v"]:
        medians: list[float] = []
        for index, view in enumerate(views):
            errors: Float64[ndarray, "m"] = reprojection_errors(rig, view, points)[usable[index]]
            finite: Float64[ndarray, "f"] = errors[np.isfinite(errors)]
            medians.append(float(np.median(finite)) if len(finite) >= MIN_POINTS_PER_VIEW else np.inf)
        return np.asarray(medians, dtype=np.float64)

    best: tuple[int, float, Float64[ndarray, "21 3"]] | None = None
    for first, second in combinations(range(len(views)), 2):
        pair: list[int] = [first, second]
        weights: Float64[ndarray, "2 21"] = usable[pair].astype(np.float64)
        angles: Float64[ndarray, "21"] = np.degrees(np.arccos(np.clip((rays[first] * rays[second]).sum(-1), -1.0, 1.0)))
        if (weights.min(axis=0) > 0).sum() < MIN_POINTS_PER_VIEW or np.median(angles[(weights.min(axis=0) > 0)]) < MIN_RAY_ANGLE_DEG:
            continue
        points, solved = solve_rays(centres[pair], rays[pair], weights)
        if solved.sum() < MIN_POINTS_PER_VIEW:
            continue
        errors: Float64[ndarray, "v"] = view_errors(points)
        inlier: Bool[ndarray, "v"] = errors <= gates
        if not (inlier[first] and inlier[second]):
            continue
        score: tuple[int, float] = (int(inlier.sum()), -float(errors[inlier].sum()))
        if best is None or score > (best[0], -best[1]):
            best = (score[0], -score[1], points)
    if best is None:
        return None
    errors = view_errors(best[2])
    members: Int64[ndarray, "m"] = np.flatnonzero(errors <= gates).astype(np.int64)
    weights_all: Float64[ndarray, "v 21"] = np.zeros((len(views), NUM_KEYPOINTS))
    weights_all[members] = np.where(usable[members], np.stack([views[i].confidence for i in members]), 0.0)
    points, _ = solve_rays(centres, rays, weights_all)
    per_joint: Float64[ndarray, "v 21"] = np.stack([reprojection_errors(rig, view, points) for view in views])
    keep: Bool[ndarray, "v 21"] = ~(per_joint > JOINT_OUTLIER * gates[:, None])
    points, _ = solve_rays(centres, rays, np.where(keep, weights_all, 0.0))
    errors = view_errors(points)
    inliers: tuple[int, ...] = tuple(views[i].cam for i in range(len(views)) if errors[i] <= gates[i])
    if len(inliers) < 2:
        return None
    return Fusion(points, inliers, {view.cam: float(errors[i]) for i, view in enumerate(views)}, {view.cam: float(gates[i]) for i, view in enumerate(views)})


def lift_view(rig: Kb4Rig, view: ViewKeypoints) -> Float64[ndarray, "21 3"]:
    """One view's keypoints in 3D (rig frame): each joint on its ray at the mean distance plus KeyNet's relative distance, the
    mean distance chosen so the hand's rigid landmark distances (``BONES``) best match the generic hand (golden-section search)."""
    rays: Float64[ndarray, "21 3"] = rig.unproject(view.cam, view.uv)
    offsets: Float64[ndarray, "21"] = view.d_rel_mm / 1000.0
    first: Int64[ndarray, "b"] = np.asarray([bone[0] for bone in BONES], dtype=np.int64)
    second: Int64[ndarray, "b"] = np.asarray([bone[1] for bone in BONES], dtype=np.int64)
    lengths: Float64[ndarray, "b"] = np.asarray([bone[2] for bone in BONES]) / 1000.0

    def cost(distance: float) -> float:
        points: Float64[ndarray, "21 3"] = rays * np.maximum(distance + offsets, 1e-3)[:, None]
        return float(np.square(np.linalg.norm(points[first] - points[second], axis=-1) - lengths).sum())

    low, high = LIFT_RANGE_M
    ratio: float = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        a: float = high - ratio * (high - low)
        b: float = low + ratio * (high - low)
        if cost(a) < cost(b):
            high = b
        else:
            low = a
    distance: float = 0.5 * (low + high)
    points_cam: Float64[ndarray, "21 3"] = rays * np.maximum(distance + offsets, 1e-3)[:, None]
    transform: Float64[ndarray, "4 4"] = rig.cam_from_rig[view.cam]
    return (points_cam - transform[:3, 3]) @ transform[:3, :3]


def keypoint_box(rig: Kb4Rig, cam: int, points_rig: Float64[ndarray, "21 3"]) -> tuple[Box | None, Float64[ndarray, "21 2"]]:
    """MEgATrack's box of a hand in one view, and the projected keypoints.

    The rule (DetNet's label, MEgATrack section 3.3/4.2): project the 21 keypoints; when at least ``MIN_INSIDE_KEYPOINTS`` lie
    in front of the camera and inside the image, take the minimum enclosing circle of the in-front projections, grow its radius
    by ``BOX_MARGIN`` and return the square circumscribing it (side = 2 x 1.2 r, same centre), clipped to the image. Otherwise None
    (the hand is not in this view).
    """
    points_cam: Float64[ndarray, "21 3"] = rig.to_cam(cam, np.nan_to_num(points_rig))
    pixels: Float64[ndarray, "21 2"] = rig.project(cam, points_cam)
    width, height = (float(value) for value in rig.size_wh[cam])
    front: Bool[ndarray, "21"] = np.isfinite(points_rig).all(axis=-1) & (points_cam[:, 2] > 0) & np.isfinite(pixels).all(axis=-1)
    inside: Bool[ndarray, "21"] = front & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    if inside.sum() < MIN_INSIDE_KEYPOINTS:
        return None, pixels
    centre, radius = cv2.minEnclosingCircle(pixels[front].astype(np.float32).reshape(-1, 1, 2))
    cx: float = float(centre[0])
    cy: float = float(centre[1])
    radius = float(radius) * BOX_MARGIN
    x0, y0 = max(cx - radius, 0.0), max(cy - radius, 0.0)
    x1, y1 = min(cx + radius, width), min(cy + radius, height)
    if x1 - x0 < MIN_BOX_SIDE_PX or y1 - y0 < MIN_BOX_SIDE_PX:
        return None, pixels
    return Box(float(x0), float(y0), float(x1), float(y1)), pixels
