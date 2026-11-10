"""Ground-truth hands of five dataforge datasets, read from a Rerun catalog.

Every per-dataset read is one catalog round trip across all requested segments
(grouped locally by ``rerun_segment_id``); only the video packets are read per segment.
Ground truth is dataforge's ``/world/gt/coco133_xyz`` (COCO-WholeBody, 21 hand slots per
side in MediaPipe order) in world metres, moved into the camera with the rig pose and the
camera's extrinsic. The camera's 2D label entity holds dataforge's full-lens projection
(SHOW3D: its annotated 2D joints). Lenses are read natively (pinhole, Fisheye624, double
sphere): a camera without distortion components is a pinhole, and an unknown distortion model fails.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetSelection
from scipy.spatial.transform import Rotation
from serde import serde
from simplecv.catalog_video import CatalogVideo, read_catalog_videos

from posekit.cameras import CalibratedCamera
from posekit.predictions import HandSide

GtDataset: TypeAlias = Literal["dataforge-umetrack", "dataforge-hot3d-quest3", "dataforge-show3d", "gsi-six-view", "dataforge-gen-humanego-subset"]
"""Catalog datasets the evaluation reads."""
GT_XYZ: str = "/world/gt/coco133_xyz"
SIDE_SLOTS: dict[HandSide, int] = {"left": 91, "right": 112}
"""First COCO-WholeBody slot of each hand; 21 slots in MediaPipe order follow."""
HAND_SLOTS: slice = slice(SIDE_SLOTS["left"], SIDE_SLOTS["right"] + 21)
"""Both hands' slots, left then right."""
SIDES: tuple[HandSide, HandSide] = ("left", "right")
DERIVED_JOINTS: tuple[int, ...] = (1,)
"""dataforge derives hand joint 1 (thumb CMC) as the wrist/CMC midpoint; it is not scored."""
TIME_TOLERANCE_NS: int = 2_000_000
"""Largest gap between a video sample and its pose/label rows."""
MIN_JOINTS: int = 15
"""A hand is visible when its wrist and at least this many joints are scoreable."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class HandGtSubset:
    """One dataset's fixed evaluation segments and camera."""

    dataset: GtDataset
    """Catalog dataset name."""
    camera: str
    """Camera entity, e.g. ``/world/rig_00/cam_00``; its parent is the rig."""
    labels: str
    """2D label entity under the camera's pinhole."""
    segments: tuple[str, ...]
    """Segment ids."""


EVAL_SUBSETS: tuple[HandGtSubset, ...] = (
    HandGtSubset(
        "dataforge-umetrack",
        "/world/rig_00/cam_01",
        "coco133_uv_projected",
        (
            "umetrack__real__hand_hand__testing__user_05__recording_00",
            "umetrack__real__separate_hand__testing__user_19__recording_10",
            "umetrack__real__separate_hand__testing__user_29__recording_18",
        ),
    ),
    HandGtSubset(
        "dataforge-hot3d-quest3",
        "/world/rig_00/cam_00",
        "coco133_uv_projected",
        ("hot3d-quest3__P0002_1464cbdc", "hot3d-quest3__P0010_d534ad6b", "hot3d-quest3__P0017_1a07e9e7"),
    ),
    HandGtSubset(
        "dataforge-show3d",
        "/world/rig_01/cam_00",
        "coco133_uv",
        ("show3d__ASC023__aria_inspecting_f5e9", "show3d__LWA828__mustard_cleaning-the-outside_ca2a", "show3d__SLI325__milk_rotate-inspect_798e"),
    ),
    HandGtSubset(
        "gsi-six-view",
        "/world/rig_00/cam_00",
        "coco133_uv_projected",
        (
            "gsi-six-view__ep_v1_000906914d401d7c03c5e5c1d69f7a88",
            "gsi-six-view__ep_v1_4ccaa1b7aeeee0069bece3bba51e0bd2",
            "gsi-six-view__ep_v1_9f8c8809a7ef0ef1e5177120541b834f",
        ),
    ),
    HandGtSubset(
        "dataforge-gen-humanego-subset",
        "/world/rig_00/cam_02",
        "coco133_uv_projected",
        (
            "gen-humanego__0003946848af465d9544e451b165fddc",
            "gen-humanego__04203576a9ec441aabd47a3541c17065",
            "gen-humanego__1923790d81a340759d94ac582b449e62",
        ),
    ),
)
"""Three segments per dataset, disjoint from the EgoHand calibration segments (``assets/egohand_*catalog.rrd``).
HOT3D: SLAM-left camera of the train split (the test split has no labels); UmeTrack: front camera of real test
recordings; SHOW3D: the Aria RGB camera; GSI: the first of six views; Gen-HumanEgo: the double-sphere head camera."""


@dataclass(frozen=True, slots=True)
class HandGtSegment:
    """Evaluation frames of one segment: video samples, camera and both hands' ground truth."""

    segment_id: str
    """Catalog segment id."""
    camera: CalibratedCamera
    """Calibration of the evaluated camera, in its native lens model."""
    video: CatalogVideo
    """The camera's compressed video, all samples."""
    sample_indices: Int64[ndarray, "f"]
    """Video sample of each evaluation frame."""
    video_time_ns: Int64[ndarray, "f"]
    """``video_time`` of each evaluation frame."""
    xyz_camera: Float32[ndarray, "f 2 21 3"]
    """Ground-truth joints (left, right) in the camera frame, metres; NaN where missing."""
    uv: Float32[ndarray, "f 2 21 2"]
    """The 2D labels of those joints, source pixels; NaN where missing."""
    valid: Bool[ndarray, "f 2 21"]
    """Scoreable joints: finite 3D and 2D, inside the image, not a derived joint."""


def calibrated_camera(statics: pa.Table, camera: str) -> tuple[CalibratedCamera, Float64[ndarray, "4 4"]]:
    """The camera's native lens and ``cam_from_rig`` from one segment's static rows (Rerun matrices are column-major)."""

    def value(component: str) -> list | None:
        name: str = f"{camera}{component}"
        return next((v for v in statics[name].to_pylist() if v is not None), None) if name in statics.column_names else None

    image_from_camera: list | None = value("/pinhole:Pinhole:image_from_camera")
    resolution: list | None = value("/pinhole:Pinhole:resolution")
    if image_from_camera is None or resolution is None:
        raise ValueError(f"{camera}: no pinhole")
    intrinsics: Float64[ndarray, "3 3"] = np.asarray(image_from_camera[0], np.float64).reshape(3, 3).T
    width, height = (int(v) for v in resolution[0])
    double_sphere: list | None = value(":ds_coefficients")
    coefficients: list | None = value("/pinhole:simplecv.components.DistortionCoefficients")
    if double_sphere is not None:
        ds: Float64[ndarray, "6"] = np.asarray(double_sphere[0], np.float64)
        lens: CalibratedCamera = CalibratedCamera(np.array([[ds[0], 0.0, ds[2]], [0.0, ds[1], ds[3]], [0.0, 0.0, 1.0]]), ds[4:], width, height, lens="DOUBLE_SPHERE")
    elif coefficients is not None:
        distortion: Float64[ndarray, "d"] = np.asarray(coefficients[0], np.float64)
        model: list | None = value("/pinhole:simplecv.components.DistortionModel")
        if distortion.shape != (8,) or model is None or model[0] != "kannala_brandt":
            raise ValueError(f"{camera}: expected dataforge's 8-term kannala_brandt (Fisheye62) lens")
        lens = CalibratedCamera(intrinsics, np.concatenate([distortion, np.zeros(4)]), width, height, lens="FISHEYE624")
    else:
        lens = CalibratedCamera(intrinsics, np.zeros(0), width, height, lens="PINHOLE")
    mat3x3: list | None = value(":Transform3D:mat3x3")
    translation: list | None = value(":Transform3D:translation")
    if mat3x3 is None or translation is None:
        raise ValueError(f"{camera}: no extrinsic")
    transform: Float64[ndarray, "4 4"] = np.eye(4)
    transform[:3, :3] = np.asarray(mat3x3[0], np.float64).reshape(3, 3).T
    transform[:3, 3] = translation[0]
    relation: list | None = value(":Transform3D:relation")
    # ChildFromParent (2) stores cam_from_rig; Rerun's default (ParentFromChild) stores rig_from_cam.
    return lens, transform if relation is not None and relation[0] == 2 else np.linalg.inv(transform)


def _nearest(times: Int64[ndarray, "n"], targets: Int64[ndarray, "m"]) -> Int64[ndarray, "m"]:
    """Index of the nearest ``times`` row per target, -1 beyond the tolerance."""
    if not len(times):
        return np.full(len(targets), -1, dtype=np.int64)
    right: Int64[ndarray, "m"] = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left: Int64[ndarray, "m"] = np.clip(right - 1, 0, len(times) - 1)
    best: Int64[ndarray, "m"] = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    return np.where(np.abs(times[best] - targets) <= TIME_TOLERANCE_NS, best, -1).astype(np.int64)


def select_frames(valid: Bool[ndarray, "f 2 21"], *, max_frames: int) -> Int64[ndarray, "n"]:
    """Up to ``max_frames`` frames spread evenly over those with a visible hand; ground truth only, no repeats."""
    available: Int64[ndarray, "a"] = np.flatnonzero(((valid.sum(-1) >= MIN_JOINTS) & valid[:, :, 0]).any(-1))
    if not len(available):
        raise ValueError("No visible hands in this segment.")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    return available[np.linspace(0, len(available) - 1, min(max_frames, len(available)), dtype=np.int64)]


def read_hand_gt(dataset: DatasetEntry, subset: HandGtSubset, *, max_frames: int) -> list[HandGtSegment]:
    """Ground truth of the subset's segments on its camera, ``max_frames`` per segment (see ``select_frames``).

    Returns:
        One ``HandGtSegment`` per segment, in ``subset.segments`` order.
    """
    rig: str = subset.camera.rsplit("/", 1)[0]
    pinhole: str = f"{subset.camera}/pinhole"
    labels: str = f"{pinhole}/{subset.labels}"
    qa: str = f"/world/gt/quality/{rig.rsplit('/', 1)[-1]}/{subset.camera.rsplit('/', 1)[-1]}/qa_pass"
    available: set[str] = {column.name for column in dataset.schema().component_columns()}
    columns: list[str] = [
        name
        for name in (
            f"{rig}:Transform3D:translation", f"{rig}:Transform3D:quaternion", f"{rig}:Transform3D:mat3x3", f"{GT_XYZ}:Points3D:positions",
            f"{GT_XYZ}:simplecv.KeypointConfidence3D:confidences", f"{labels}:Points2D:positions", f"{qa}:Scalars:scalars",
        )
        if name in available
    ]
    scoped: DatasetSelection = dataset.filter_segments(list(subset.segments))
    statics: pa.Table = scoped.filter_contents([subset.camera, pinhole]).reader(index=None).to_arrow_table()
    temporal: pa.Table = (
        scoped.filter_contents([rig, GT_XYZ, labels, qa]).reader(index="video_time").select("rerun_segment_id", "video_time", *columns).to_arrow_table()
    )
    segments: list[HandGtSegment] = []
    for segment_id in subset.segments:
        camera, cam_from_rig = calibrated_camera(statics.filter(pc.field("rerun_segment_id") == segment_id), subset.camera)
        rows: pa.Table = temporal.filter(pc.field("rerun_segment_id") == segment_id)
        row_times: Int64[ndarray, "r"] = rows["video_time"].cast(pa.int64()).to_numpy()
        video: CatalogVideo = read_catalog_videos(dataset, segment_id, [f"{pinhole}/video"], "video_time")[0]
        times: Int64[ndarray, "f"] = video.t_ns.astype(np.int64)

        def aligned(
            name: str, segment_id: str = segment_id, rows: pa.Table = rows, row_times: Int64[ndarray, "r"] = row_times, times: Int64[ndarray, "f"] = times
        ) -> tuple[Float64[ndarray, "f ..."], Bool[ndarray, "f"]]:
            """The column's nearest non-null row per video sample, and whether one lies within the tolerance."""
            values: list = rows[name].to_pylist()
            present: Int64[ndarray, "k"] = np.asarray([i for i, v in enumerate(values) if v is not None], dtype=np.int64)
            if not len(present):
                raise ValueError(f"{segment_id}: no rows for {name}")
            present = present[np.argsort(row_times[present])]
            nearest: Int64[ndarray, "f"] = _nearest(row_times[present], times)
            return np.asarray([values[present[max(i, 0)]] for i in nearest], np.float64), nearest >= 0

        points, points_ok = aligned(f"{GT_XYZ}:Points3D:positions")
        confidence, confidence_ok = aligned(f"{GT_XYZ}:simplecv.KeypointConfidence3D:confidences")
        pixels, pixels_ok = aligned(f"{labels}:Points2D:positions")
        translation, pose_ok = aligned(f"{rig}:Transform3D:translation")
        if f"{rig}:Transform3D:quaternion" in columns:
            quaternion, rotation_ok = aligned(f"{rig}:Transform3D:quaternion")
            quaternion = quaternion.reshape(-1, 4)
            rotation_ok &= np.isfinite(quaternion).all(-1) & (np.linalg.norm(quaternion, axis=-1) > 0.0)
            quaternion[~rotation_ok] = [0.0, 0.0, 0.0, 1.0]
            rotations: Float64[ndarray, "f 3 3"] = Rotation.from_quat(quaternion).as_matrix()
        else:
            matrices, rotation_ok = aligned(f"{rig}:Transform3D:mat3x3")
            rotations = matrices.reshape(-1, 3, 3).transpose(0, 2, 1)
        world_from_rig: Float64[ndarray, "f 4 4"] = np.tile(np.eye(4), (len(times), 1, 1))
        world_from_rig[:, :3, :3] = rotations
        world_from_rig[:, :3, 3] = translation.reshape(-1, 3)
        cam_from_world: Float64[ndarray, "f 4 4"] = cam_from_rig @ np.linalg.inv(world_from_rig)
        hand_points: Float64[ndarray, "f 2 21 3"] = points.reshape(-1, 133, 3)[:, HAND_SLOTS].reshape(-1, 2, 21, 3)
        xyz: Float32[ndarray, "f 2 21 3"] = (
            np.einsum("fij,fskj->fski", cam_from_world[:, :3, :3], hand_points) + cam_from_world[:, None, None, :3, 3]
        ).astype(np.float32)
        uv: Float32[ndarray, "f 2 21 2"] = pixels.reshape(-1, 133, 2)[:, HAND_SLOTS].reshape(-1, 2, 21, 2).astype(np.float32)
        valid: Bool[ndarray, "f 2 21"] = (
            (confidence.reshape(-1, 133)[:, HAND_SLOTS].reshape(-1, 2, 21) > 0) & np.isfinite(xyz).all(-1) & np.isfinite(uv).all(-1) & (xyz[..., 2] > 0)
        )
        valid &= (uv[..., 0] >= 0) & (uv[..., 0] < camera.width) & (uv[..., 1] >= 0) & (uv[..., 1] < camera.height)
        frame_ok: Bool[ndarray, "f"] = points_ok & confidence_ok & pixels_ok & pose_ok & rotation_ok
        if f"{qa}:Scalars:scalars" in columns:
            qa_pass, qa_ok = aligned(f"{qa}:Scalars:scalars")
            frame_ok &= qa_ok & (qa_pass.reshape(-1) > 0.5)
        valid &= frame_ok[:, None, None]
        valid[:, :, list(DERIVED_JOINTS)] = False
        chosen: Int64[ndarray, "n"] = select_frames(valid, max_frames=max_frames)
        segments.append(HandGtSegment(segment_id, camera, video, chosen, times[chosen], xyz[chosen], uv[chosen], valid[chosen]))
    return segments
