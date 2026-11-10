"""KeyNet-F from the Rerun 0.42 catalog: NVDEC decode in the training process, perspective crops cut on the GPU, per-dataset echo pools.

Each dataset's producer streams its cameras' videos through ``simplecv.rerun_dataloader.KeyedNvdecDecoder`` (BlockShuffle over the
segments), turns each frame to native-resolution luma on the device and cuts ``stream.perspective_keynet_samples`` crops from it: the
crop cameras, presence negatives and tracker priors of the old catalog stream, at the camera's own resolution through its own lens.
The truth per segment comes from one of two places:

* ``hand_model`` (UmeTrack, SHOW3D, HOT3D): the hand-model timeline (``segment_labels`` and ``keypoint_priors``, as before).
* ``coco3d`` (GSI six-view, Gen-HumanEgo): the world COCO-133 keypoints (``/world/gt/coco133_xyz``), the rig pose per frame and each
  camera's extrinsic (``Transform3D:relation`` decides the direction) and lens (pinhole, or the double sphere when the camera has
  ``ds_coefficients``). Projecting them reproduces the stored ``coco133_uv_projected`` exactly (2026-11-09: 0.000 px on both
  datasets). COCO has no palm centre, so landmark 20 is built from the palm's rigid points (``palm_centre``). Presence counts follow
  ``labels.coco2d`` (16 of the 20 COCO points inside the image, confidence above zero, the dataset's trust columns).

The crops travel as ``KeyNetSamples`` (uint8 96x96 plus targets) into ``echo.EchoStream`` pools; ``keynet_finish`` augments every draw
anew on the device (intensity, gamma, blur, sensor noise, border occlusion, coherent whole-hand prior errors, zeroed priors), so a crop
that is drawn several times is seen differently each time.
"""

import dataclasses
import itertools
import math
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from serde import serde
from simplecv.umetrack_temp.generic_hand_model_torch import LANDMARK
from torch import Tensor

from handtrack.data.batches import CropKind, KeyNetBatch
from handtrack.data.catalog import camera_angles, layout_for, list_segments, read_hand_timeline, read_rig, read_statics
from handtrack.data.echo import Producer
from handtrack.data.luma import grey_from_rgb
from handtrack.data.rerun_source import SIDES, dense_rows
from handtrack.data.segment_labels import HandProjection, keypoint_priors, project_hands, segment_labels
from handtrack.data.stream import ImageHands, KeyNetAugment, KeyNetSamples, perspective_keynet_samples, sample_count
from handtrack.geometry.camera import CameraRig
from handtrack.geometry.letterbox import Letterbox, letterbox_for
from handtrack.labels.coco2d import COCO_SLOTS, RIGHT_HAND_SLOT_OFFSET, coco133_hands
from handtrack.labels.crops import boundary_occlusion, scale_intensity
from handtrack.labels.heatmaps import render_distance, render_heatmaps
from handtrack.labels.validity import HandLabel

LabelKind = Literal["hand_model", "coco3d"]
HANDS: str = "/world/gt/hands"
COCO_XYZ: str = "/world/gt/coco133_xyz"
NVDEC_SLICE: int = 16
"""Frames per NVDEC call: the luma transform briefly holds them as float RGB (16 x 1080p = 400 MB) before it keeps one byte per pixel."""
TRUTH_CACHE: int = 12
"""Segments whose truth a producer keeps (a block covers part of a segment, and BlockShuffle interleaves segments)."""
NOMINAL_FPS: int = 30
STALE_STEPS: int = 20
"""The stale prior is the pose this many tracker steps earlier (the paper's 10 % alternative input; ``segment_labels``)."""
PASS_STRIDE: int = 1_000_000


@dataclass(frozen=True, slots=True)
class KeyNetDataset:
    """One catalog dataset as the KeyNet producers read it."""

    cameras: tuple[str, ...]
    """Camera entities ``/world/rig_NN/cam_MM``; the video is at ``<camera>/pinhole/video``."""
    codec: Literal["h264", "h265", "av1"]
    labels: LabelKind
    rolls_deg: tuple[float, ...] | None
    """Each camera's crop roll in degrees (UmeTrack's ``camera_angle``); None reads the catalog's ``camera_angles`` (hand-model
    datasets). 0 for GSI and Gen-HumanEgo: robocap-live cuts RoboCap's crops without a roll (``hands/estimator.rs``), and GSI's eye
    views, like RoboCap's cameras 2-3, show the hands fingers-up as stored (99 % of hand frames), although world-up points down there."""
    min_score: float | None = None
    """coco3d only: a hand is trusted when its ``scores:pred_score_quality`` reaches this (None: no score column)."""
    tracker_step: int = 1
    """Frames per tracker step (the priors' extrapolation interval): 1 at 30 fps, 2 at 60 fps."""
    fetch_block_size: int = 1024
    """Frames per dataloader block. The dataloader decodes every camera of a block before yielding it, at native resolution here
    (KeyNet crops need the detail), so a block of six 1080p cameras costs 12.4 MB per frame: keep each block near 1.5 GB."""


KEYNET_DATASETS: dict[str, KeyNetDataset] = {
    "dataforge-umetrack": KeyNetDataset(tuple(f"/world/rig_00/cam_0{i}" for i in range(4)), "av1", "hand_model", None),
    "dataforge-show3d": KeyNetDataset(("/world/rig_01/cam_00", "/world/rig_01/cam_01"), "av1", "hand_model", None, tracker_step=2, fetch_block_size=512),
    "dataforge-hot3d-quest3": KeyNetDataset(("/world/rig_00/cam_00", "/world/rig_00/cam_01"), "av1", "hand_model", None),
    "gsi-six-view": KeyNetDataset(tuple(f"/world/rig_00/cam_0{i}" for i in range(6)), "h265", "coco3d", (0.0,) * 6, fetch_block_size=128),
    "dataforge-gen-humanego-subset": KeyNetDataset(tuple(f"/world/rig_00/cam_0{i}" for i in range(6)), "h264", "coco3d", (0.0,) * 6, min_score=0.8,
                                                   fetch_block_size=128),
}
"""The datasets KeyNet trains and is scored on. Camera ids across datasets follow ``camera_offsets``."""


def camera_offsets(datasets: Sequence[str]) -> dict[str, int]:
    """First camera id of each dataset, in list order (ids unique across the datasets)."""
    offsets: dict[str, int] = {}
    total: int = 0
    for name in datasets:
        offsets[name] = total
        total += len(KEYNET_DATASETS[name].cameras)
    return offsets


# --- the palm centre ----------------------------------------------------------------------------------------------------------------

_PALM_POINTS: tuple[int, int, int, int, int] = (int(LANDMARK.WRIST_JOINT), int(LANDMARK.INDEX_PROXIMAL_FRAME), int(LANDMARK.MIDDLE_PROXIMAL_FRAME),
                                                int(LANDMARK.RING_PROXIMAL_FRAME), int(LANDMARK.PINKY_PROXIMAL_FRAME))


def _palm_frame(points: Float32[Tensor, "*b 21 3"]) -> tuple[Float32[Tensor, "*b 3"], Float32[Tensor, "*b 3 3"], Float32[Tensor, "*b"]]:
    """Origin (wrist), orthonormal axes (rows: wrist->middle MCP, index->pinky MCP made orthogonal, their normal) and the
    wrist-middle MCP length of each hand."""
    wrist, index, middle, _, pinky = (points[..., i, :] for i in _PALM_POINTS)
    forward: Float32[Tensor, "*b 3"] = middle - wrist
    length: Float32[Tensor, "*b"] = torch.linalg.vector_norm(forward, dim=-1)
    e1: Float32[Tensor, "*b 3"] = forward / length.clamp_min(1e-9)[..., None]
    across: Float32[Tensor, "*b 3"] = pinky - index
    across = across - (across * e1).sum(-1, keepdim=True) * e1
    e2: Float32[Tensor, "*b 3"] = across / torch.linalg.vector_norm(across, dim=-1, keepdim=True).clamp_min(1e-9)
    e3: Float32[Tensor, "*b 3"] = torch.linalg.cross(e1, e2)
    return wrist, torch.stack([e1, e2, e3], dim=-2), length


def palm_coefficients(rest: Float32[Tensor, "21 3"]) -> Float32[Tensor, "3"]:
    """The palm centre of a hand model's rest landmarks in its palm frame, in units of the wrist-middle MCP length."""
    origin, axes, length = _palm_frame(rest)
    return axes @ (rest[int(LANDMARK.PALM_CENTER)] - origin) / length


def palm_centre(points: Float32[Tensor, "*b 21 3"], coefficients: Float32[Tensor, "3"]) -> Float32[Tensor, "*b 3"]:
    """Landmark 20 from the wrist and the four MCPs (rigid on the palm): the model's rest offset in each hand's palm frame."""
    origin, axes, length = _palm_frame(points)
    return origin + length[..., None] * torch.einsum("...ij,i->...j", axes, coefficients.to(points.device, points.dtype))


# --- truth per segment --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentTruth:
    """Every camera's KeyNet truth for the rows of one segment; ``row_of_frame`` maps a ``frame_index`` to its row (-1: none)."""

    rig: CameraRig
    letterboxes: tuple[Letterbox, ...]
    """The legacy letterboxes (net-frame validation metadata, as the old stream)."""
    rolls_rad: tuple[float, ...]
    phi: float
    row_of_frame: Int64[ndarray, "F"]
    image_valid: Bool[Tensor, "k c"]
    hand_label: Int64[Tensor, "k c 2"]
    truth: HandProjection
    valid: Bool[Tensor, "k 2"]
    extrapolated: HandProjection
    extrapolated_valid: Bool[Tensor, "k 2"]
    stale: HandProjection
    stale_valid: Bool[Tensor, "k 2"]

    def image_hands(self, which: Literal["truth", "extrapolated", "stale"], rows: Int64[Tensor, "m"], camera: int, device: torch.device) -> ImageHands:
        projection: HandProjection = getattr(self, which)
        valid: Bool[Tensor, "k 2"] = {"truth": self.valid, "extrapolated": self.extrapolated_valid, "stale": self.stale_valid}[which]
        return ImageHands(net_xy=projection.net_xy[rows, camera].to(device), points_cam=projection.points_cam[rows, camera].to(device),
                          in_front=projection.in_front[rows, camera].to(device), valid=valid[rows].to(device))


def _frame_rows(frames: Int64[ndarray, "k"]) -> Int64[ndarray, "F"]:
    out: Int64[ndarray, "F"] = np.full(int(frames.max()) + 1 if len(frames) else 0, -1, dtype=np.int64)
    out[frames] = np.arange(len(frames))
    return out


def hand_model_truth(dataset, name: str, segment: str, infos: dict) -> SegmentTruth:
    """UmeTrack / SHOW3D / HOT3D: the hand-model labels and ground-truth priors of the old stream, for every timeline row."""
    info = infos[segment]
    layout = layout_for(name)
    statics: pa.Table = read_statics(dataset, info)
    rig, letterboxes = read_rig(statics, info, None)
    timeline = read_hand_timeline(dataset, info, statics)
    entity: str = f"{layout.cameras[0]}/pinhole/video"
    column: str = f"{entity}:VideoStream:sample"
    table: pa.Table = dataset.filter_segments([segment]).filter_contents([entity]).reader(index="frame_index").select("frame_index", "video_time", column).to_arrow_table()
    present: Bool[ndarray, "k"] = np.asarray(table[column].is_valid())
    frames: Int64[ndarray, "k"] = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)[present]
    times: Int64[ndarray, "k"] = table["video_time"].cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)[present]
    rows: Int64[ndarray, "k"] = np.searchsorted(timeline.video_time_ns, times).clip(0, len(timeline.video_time_ns) - 1)
    matched: Bool[ndarray, "k"] = timeline.video_time_ns[rows] == times
    labels = segment_labels(timeline, rig, letterboxes, rows, layout.pose_gated, mesh_labels=False)
    priors = keypoint_priors(timeline, rig, letterboxes, rows, layout.tracker_step)
    return SegmentTruth(
        rig=rig, letterboxes=letterboxes, rolls_rad=tuple(math.radians(a) for a in camera_angles(statics, info)), phi=float(timeline.hand_scale),
        row_of_frame=_frame_rows(frames), image_valid=labels.image_valid & torch.from_numpy(matched)[:, None], hand_label=labels.hand_label,
        truth=labels.projection, valid=torch.isfinite(labels.landmarks).all(dim=-1).all(dim=-1),
        extrapolated=priors.extrapolated, extrapolated_valid=priors.extrapolated_valid, stale=priors.stale, stale_valid=priors.stale_valid,
    )


def _column_major(values: Sequence[float]) -> Float32[ndarray, "3 3"]:
    return np.asarray(values, dtype=np.float64).reshape(3, 3).T.astype(np.float32)


def _static(statics: pa.Table, column: str) -> list:
    value = statics[column][0].as_py()
    return value[0] if isinstance(value, list) and value and isinstance(value[0], list) else value


def coco3d_rig(statics: pa.Table, cameras: Sequence[str]) -> tuple[CameraRig, tuple[Letterbox, ...]]:
    """cam_from_rig, K and lens of each camera from the static row (``Transform3D:relation`` 2 = ChildFromParent, else ParentFromChild)."""
    cam_from_rig: list[Float32[ndarray, "4 4"]] = []
    focal: list[list[float]] = []
    principal: list[list[float]] = []
    sizes: list[list[float]] = []
    spheres: list[list[float]] = []
    for camera in cameras:
        rotation: Float32[ndarray, "3 3"] = _column_major(_static(statics, f"{camera}:Transform3D:mat3x3"))
        translation: Float32[ndarray, "3"] = np.asarray(_static(statics, f"{camera}:Transform3D:translation"), dtype=np.float32)
        relation_column: str = f"{camera}:Transform3D:relation"
        relation = statics[relation_column][0].as_py() if relation_column in statics.column_names else None
        relation = relation[0] if isinstance(relation, list) else relation
        transform: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
        if relation == 2:  # the stored transform maps rig points into the camera
            transform[:3, :3], transform[:3, 3] = rotation, translation
        else:  # rig_from_cam: invert it
            transform[:3, :3], transform[:3, 3] = rotation.T, -rotation.T @ translation
        cam_from_rig.append(transform)
        intrinsics: Float32[ndarray, "3 3"] = _column_major(_static(statics, f"{camera}/pinhole:Pinhole:image_from_camera"))
        focal.append([float(intrinsics[0, 0]), float(intrinsics[1, 1])])
        principal.append([float(intrinsics[0, 2]), float(intrinsics[1, 2])])
        sizes.append([float(v) for v in _static(statics, f"{camera}/pinhole:Pinhole:resolution")])
        ds_column: str = f"{camera}:ds_coefficients"
        if ds_column in statics.column_names and statics[ds_column][0].is_valid:
            fx, fy, cx, cy, xi, alpha = (float(v) for v in _static(statics, ds_column))
            focal[-1], principal[-1] = [fx, fy], [cx, cy]
            spheres.append([xi, alpha])
    if spheres and len(spheres) != len(cameras):
        raise ValueError("a rig mixes double-sphere and pinhole cameras")
    rig = CameraRig(names=tuple(cameras), image_size=torch.tensor(sizes), cam_from_rig=torch.from_numpy(np.stack(cam_from_rig)), focal=torch.tensor(focal),
                    principal=torch.tensor(principal), fisheye62=None, double_sphere=torch.tensor(spheres) if spheres else None)
    return rig, tuple(letterbox_for(int(w), int(h), None) for w, h in sizes)


def _world_from_rig(table: pa.Table, rig_entity: str = "/world/rig_00") -> Float32[ndarray, "k 4 4"]:
    """The rig pose per row (quaternion xyzw or column-major mat3x3, plus translation); NaN rows where the pose is missing."""
    translation, has_translation = dense_rows(table[f"{rig_entity}:Transform3D:translation"], 1, inner=3)
    out: Float32[ndarray, "k 4 4"] = np.full((table.num_rows, 4, 4), np.nan, dtype=np.float32)
    quaternion_column: str = f"{rig_entity}:Transform3D:quaternion"
    if quaternion_column in table.column_names and table[quaternion_column].null_count < table.num_rows:
        quaternion, has_rotation = dense_rows(table[quaternion_column], 1, inner=4)
        x, y, z, w = (quaternion[:, 0, i].astype(np.float64) for i in range(4))
        rotation: ndarray = np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ], -2).astype(np.float32)
    else:
        matrix, has_rotation = dense_rows(table[f"{rig_entity}:Transform3D:mat3x3"], 1, inner=9)
        rotation = matrix[:, 0].reshape(-1, 3, 3).transpose(0, 2, 1)
    ok: Bool[ndarray, "k"] = has_translation & has_rotation
    out[ok, :3, :3] = rotation[ok]
    out[ok, :3, 3] = translation[ok, 0]
    out[ok, 3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return out


def coco_landmarks(xyz: Float32[Tensor, "k 133 3"], confidence: Float32[Tensor, "k 133"], coefficients: Float32[Tensor, "3"]) -> tuple[Float32[Tensor, "k 2 21 3"], Bool[Tensor, "k 2 21"]]:
    """Both hands' 21 landmarks from COCO-133 points (``labels.coco2d.COCO_SLOTS``), the palm centre built from the palm (finite only
    when its five points are), and which landmarks carry a confident point."""
    slots: Int64[Tensor, "20"] = torch.tensor(list(COCO_SLOTS))
    landmarks: Int64[Tensor, "20"] = torch.tensor(list(COCO_SLOTS.values()))
    points: Float32[Tensor, "k 2 21 3"] = torch.full((xyz.shape[0], 2, 21, 3), torch.nan)
    known: Bool[Tensor, "k 2 21"] = torch.zeros((xyz.shape[0], 2, 21), dtype=torch.bool)
    for side in (0, 1):
        chosen: Float32[Tensor, "k 20 3"] = xyz[:, slots + side * RIGHT_HAND_SLOT_OFFSET]
        good: Bool[Tensor, "k 20"] = torch.isfinite(chosen).all(dim=-1) & (confidence[:, slots + side * RIGHT_HAND_SLOT_OFFSET] > 0)
        points[:, side, landmarks] = torch.where(good[..., None], chosen, torch.full_like(chosen, torch.nan))
        known[:, side, landmarks] = good
    palm: Float32[Tensor, "k 2 3"] = palm_centre(points, coefficients)
    palm_known: Bool[Tensor, "k 2"] = known[..., list(_PALM_POINTS)].all(dim=-1) & torch.isfinite(palm).all(dim=-1)
    points[..., int(LANDMARK.PALM_CENTER), :] = torch.where(palm_known[..., None], palm, torch.full_like(palm, torch.nan))
    known[..., int(LANDMARK.PALM_CENTER)] = palm_known
    return points, known


def coco3d_truth(dataset, name: str, spec: KeyNetDataset, segment: str, coefficients: Float32[Tensor, "3"], generic_rest: Float32[Tensor, "21 3"]) -> SegmentTruth:
    """GSI / Gen-HumanEgo: truth from the world COCO-133 keypoints, the rig poses and the lenses; priors extrapolated from them."""
    statics: pa.Table = dataset.filter_segments([segment]).filter_contents([*spec.cameras, *(f"{c}/pinhole" for c in spec.cameras)]).reader(index=None).to_arrow_table()
    rig, letterboxes = coco3d_rig(statics, spec.cameras)
    entities: list[str] = ["/world/rig_00", COCO_XYZ, *(f"{c}/pinhole/coco133_uv_projected" for c in spec.cameras),
                           *(f"{HANDS}/{s}/{p}" for s in SIDES for p in ("confidence", "scores"))]
    table: pa.Table = dataset.filter_segments([segment]).filter_contents(entities).reader(index="frame_index").to_arrow_table()
    frames: Int64[ndarray, "k"] = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    count: int = len(frames)
    world_from_rig: Float32[Tensor, "k 4 4"] = torch.from_numpy(_world_from_rig(table))
    xyz, has_xyz = dense_rows(table[f"{COCO_XYZ}:Points3D:positions"], 133, inner=3)
    weight, has_weight = dense_rows(table[f"{COCO_XYZ}:simplecv.KeypointConfidence3D:confidences"], 133)
    usable: Bool[ndarray, "k"] = has_xyz & has_weight & np.isfinite(world_from_rig.numpy()).all(axis=(1, 2))
    xyz[~usable] = np.nan
    weight = np.where(usable[:, None], weight, np.float32(0.0))
    trusted: Bool[ndarray, "k 2"] = np.repeat(usable[:, None], 2, axis=1)
    for side, hand in enumerate(SIDES):
        confidence_column: str = f"{HANDS}/{hand}/confidence:Scalars:scalars"
        if confidence_column in table.column_names:
            values: Float32[ndarray, "k"] = np.asarray([v[0] if v else np.nan for v in table[confidence_column].to_pylist()], dtype=np.float32)
            trusted[:, side] &= np.nan_to_num(values) > 0
        score_column: str = f"{HANDS}/{hand}/scores:pred_score_quality"
        if spec.min_score is not None and score_column in table.column_names:
            scores: Float32[ndarray, "k"] = np.asarray([v[0] if isinstance(v, list) and v else (v if v is not None else np.nan) for v in table[score_column].to_pylist()], dtype=np.float32)
            trusted[:, side] &= np.nan_to_num(scores) >= spec.min_score
    points_world, known = coco_landmarks(torch.from_numpy(xyz), torch.from_numpy(weight), coefficients)
    valid: Bool[Tensor, "k 2"] = torch.from_numpy(trusted) & known.all(dim=-1)
    points_world = torch.where(valid[..., None, None], points_world, torch.full_like(points_world, torch.nan))
    truth: HandProjection = project_hands(rig, letterboxes, world_from_rig, points_world)
    # Presence and image validity follow labels.coco2d on the stored projections (the dataset's own visibility statement).
    image_valid: Bool[Tensor, "k c"] = torch.zeros((count, len(spec.cameras)), dtype=torch.bool)
    hand_label: Int64[Tensor, "k c 2"] = torch.zeros((count, len(spec.cameras), 2), dtype=torch.int64)
    for c, camera in enumerate(spec.cameras):
        uv, has_points = dense_rows(table[f"{camera}/pinhole/coco133_uv_projected:Points2D:positions"], 133, inner=2)
        uv_weight, has_uv_weight = dense_rows(table[f"{camera}/pinhole/coco133_uv_projected:simplecv.KeypointConfidence2D:confidences"], 133)
        seen: Bool[ndarray, "k"] = has_points & has_uv_weight
        uv[~seen] = np.nan
        uv_weight = np.where(seen[:, None], uv_weight, np.float32(0.0))
        hands = coco133_hands(torch.from_numpy(uv), torch.from_numpy(uv_weight), torch.from_numpy(trusted), letterboxes[c])
        # Unlike DetNet's rule (drop the image when either hand is untrusted), an untrusted hand only stops being a statement: it
        # becomes PARTIAL (no positive, no negative of its own), and the image still trains the trusted hand.
        image_valid[:, c] = torch.from_numpy(usable & trusted.any(axis=1))
        hand_label[:, c] = torch.where(torch.from_numpy(trusted), hands.hand_label, torch.full_like(hands.hand_label, int(HandLabel.PARTIAL)))
    # A keypoint counts for crop fitting and in-crop counts when it is in front and confidently placed.
    truth = dataclasses.replace(truth, in_front=truth.in_front & known[:, None])
    extrapolated_world, extrapolated_valid, stale_world, stale_valid = _point_priors(points_world, valid, frames, spec.tracker_step)
    phi: float = _coco_phi(points_world, valid, generic_rest)
    return SegmentTruth(
        rig=rig, letterboxes=letterboxes, rolls_rad=tuple(math.radians(r) for r in spec.rolls_deg or (0.0,) * len(spec.cameras)), phi=phi,
        row_of_frame=_frame_rows(frames), image_valid=image_valid, hand_label=hand_label, truth=truth, valid=valid,
        extrapolated=project_hands(rig, letterboxes, world_from_rig, extrapolated_world), extrapolated_valid=extrapolated_valid,
        stale=project_hands(rig, letterboxes, world_from_rig, stale_world), stale_valid=stale_valid,
    )


def _point_priors(points: Float32[Tensor, "k 2 21 3"], valid: Bool[Tensor, "k 2"], frames: Int64[ndarray, "k"], step: int
                  ) -> tuple[Float32[Tensor, "k 2 21 3"], Bool[Tensor, "k 2"], Float32[Tensor, "k 2 21 3"], Bool[Tensor, "k 2"]]:
    """World-point analogues of ``keypoint_priors``: 2 p(t-s) - p(t-2s), or p(t-s) when p(t-2s) is missing; stale p(t - 20 s)."""
    rows: Int64[ndarray, "F"] = _frame_rows(frames)

    def lookup(offset: int) -> tuple[Float32[Tensor, "k 2 21 3"], Bool[Tensor, "k 2"]]:
        source: Int64[ndarray, "k"] = frames - offset
        inside: Bool[ndarray, "k"] = (source >= 0) & (source < len(rows))
        row: Int64[ndarray, "k"] = np.where(inside, rows[np.clip(source, 0, max(len(rows) - 1, 0))], -1)
        present: Bool[Tensor, "k"] = torch.from_numpy(row >= 0)
        index: Int64[Tensor, "k"] = torch.from_numpy(np.clip(row, 0, None))
        return points[index], valid[index] & present[:, None]

    previous, previous_valid = lookup(step)
    before, before_valid = lookup(2 * step)
    stale, stale_valid = lookup(STALE_STEPS * step)
    extrapolated: Float32[Tensor, "k 2 21 3"] = torch.where(before_valid[..., None, None], 2 * previous - before, previous)
    nan: Float32[Tensor, "k 2 21 3"] = torch.full_like(points, torch.nan)
    return torch.where(previous_valid[..., None, None], extrapolated, nan), previous_valid, torch.where(stale_valid[..., None, None], stale, nan), stale_valid


def _coco_phi(points: Float32[Tensor, "k 2 21 3"], valid: Bool[Tensor, "k 2"], generic_rest: Float32[Tensor, "21 3"]) -> float:
    """The subject's hand size against the generic hand (``hand_scale``'s phi): median ratio of the rigid wrist-to-MCP lengths."""
    wrist: int = int(LANDMARK.WRIST_JOINT)
    mcps: list[int] = list(_PALM_POINTS[1:])
    subject: Float32[Tensor, "k 2"] = torch.linalg.vector_norm(points[..., mcps, :] - points[..., wrist, None, :], dim=-1).sum(-1)
    generic: float = float(torch.linalg.vector_norm(generic_rest[mcps] - generic_rest[wrist], dim=-1).sum())
    ratios: Float32[Tensor, "n"] = subject[valid] * 1000.0 / generic  # metres against the model's millimetres
    return float(ratios.median()) if ratios.numel() else 1.0


# --- the producer -------------------------------------------------------------------------------------------------------------------


def _video_field(camera: str) -> str:
    return f"{camera}/pinhole/video:VideoStream:sample"


CutStream = Callable[[int], Iterator[tuple[KeyNetSamples, int, int]]]
"""An epoch of (crops, camera index within the dataset, augmentation variant index) from one dataset."""


def stream_cuts(name: str, dataset_index: int, segments: Sequence[str], *, catalog_url: str, device: torch.device, augments: Sequence[KeyNetAugment],
                seed: int, endless: bool = False, fetch_block_size: int | None = None, chunk: int = 32) -> CutStream:
    """NVDEC decode in this process, luma at native resolution, and per (segment, camera) chunk of ``chunk`` frames one
    ``perspective_keynet_samples`` cut per entry of ``augments`` (a cut is a cheap grid sample; a decode is not). ``fetch_block_size``
    None takes the dataset's own (``KeyNetDataset.fetch_block_size``)."""
    from simplecv.rerun_dataloader import KeyedFrame, KeyedNvdecDecoder

    from handtrack.data.catalog import _generic_model

    spec: KeyNetDataset = KEYNET_DATASETS[name]
    generic_rest: Float32[Tensor, "21 3"] = _generic_model().landmark_rest_positions.float()
    coefficients: Float32[Tensor, "3"] = palm_coefficients(generic_rest)

    def produce(epoch: int) -> Iterator[tuple[KeyNetSamples, int, int]]:
        import rerun as rr
        from rerun.experimental.dataloader import BlockShuffle, DataSource, Field, RerunIterableDataset

        dataset = rr.catalog.CatalogClient(catalog_url).get_dataset(name)
        fields = {f"cam{c}": Field(_video_field(camera), decode=KeyedNvdecDecoder(dataset, f"{camera}/pinhole/video".lstrip("/"), "frame_index", device, NOMINAL_FPS,
                                                                                   max_frames_per_call=NVDEC_SLICE, transform=grey_from_rgb))
                  for c, camera in enumerate(spec.cameras)}
        stream = RerunIterableDataset(DataSource(dataset=dataset, segments=list(segments)), index="frame_index", fields=fields,
                                      fetch_block_size=fetch_block_size or spec.fetch_block_size, shuffle_strategy=BlockShuffle(), decode_threads=1)
        infos: dict = {i.segment_id: i for i in list_segments(dataset, name)} if spec.labels == "hand_model" else {}
        generator: torch.Generator = torch.Generator(device=device).manual_seed(seed * 7919 + epoch)
        current: tuple[str, SegmentTruth] | None = None
        truths: OrderedDict[str, SegmentTruth] = OrderedDict()
        pending: dict[int, list[tuple[Tensor, int]]] = {}

        def cut_all(truth: SegmentTruth, camera: int, items: list[tuple[Tensor, int]]) -> Iterator[tuple[KeyNetSamples, int, int]]:
            for variant, augment in enumerate(augments):
                samples: KeyNetSamples = cut_samples(truth, camera, items, dataset_index, augment, generator, device)
                if sample_count(samples):
                    yield samples, camera, variant

        for pass_seed in itertools.count(epoch * PASS_STRIDE) if endless else (epoch,):
            stream.set_epoch(pass_seed)
            for sample in stream:
                for c in range(len(spec.cameras)):
                    keyed = sample[f"cam{c}"]
                    if not isinstance(keyed, KeyedFrame):
                        raise TypeError(f"{name}: camera {c} decoded to {type(keyed).__name__}, not a KeyedFrame")
                    if current is None or current[0] != keyed.segment_id:
                        if current is not None:
                            for camera, items in pending.items():
                                if items:
                                    yield from cut_all(current[1], camera, items)
                        pending.clear()
                        cached: SegmentTruth | None = truths.pop(keyed.segment_id, None)
                        truth: SegmentTruth = cached if cached is not None else (
                            hand_model_truth(dataset, name, keyed.segment_id, infos) if spec.labels == "hand_model"
                            else coco3d_truth(dataset, name, spec, keyed.segment_id, coefficients, generic_rest))
                        truths[keyed.segment_id] = truth
                        while len(truths) > TRUTH_CACHE:
                            truths.popitem(last=False)
                        current = (keyed.segment_id, truth)
                    truth = current[1]
                    row: int = int(truth.row_of_frame[keyed.index]) if keyed.index < len(truth.row_of_frame) else -1
                    if row < 0 or not bool(truth.image_valid[row, c]):
                        continue
                    pending.setdefault(c, []).append((keyed.image, row))
                    if len(pending[c]) >= chunk:
                        yield from cut_all(truth, c, pending[c])
                        pending[c] = []
            if current is not None:
                for camera, items in pending.items():
                    if items:
                        yield from cut_all(current[1], camera, items)
            pending.clear()

    return produce


def keynet_producer(name: str, dataset_index: int, segments: Sequence[str], *, catalog_url: str, device: torch.device, augment: KeyNetAugment, seed: int,
                    endless: bool = False, fetch_block_size: int | None = None, chunk: int = 32, cuts: int = 1) -> Producer:
    """A ``Producer`` of training crops for ``echo.EchoStream``: ``stream_cuts`` with ``cuts`` independent jitters of each chunk."""
    cut_stream: CutStream = stream_cuts(name, dataset_index, segments, catalog_url=catalog_url, device=device, augments=[augment] * cuts, seed=seed,
                                        endless=endless, fetch_block_size=fetch_block_size, chunk=chunk)

    def produce(epoch: int) -> Iterator[KeyNetSamples]:
        for samples, _, _ in cut_stream(epoch):
            yield samples

    return produce


def cut_samples(truth: SegmentTruth, camera: int, items: list[tuple[Tensor, int]], dataset_index: int, augment: KeyNetAugment, generator: torch.Generator,
                device: torch.device) -> KeyNetSamples:
    """``perspective_keynet_samples`` for one camera's frames of one segment (frames already on the device)."""
    frames: UInt8[Tensor, "m h w"] = torch.stack([image if image.dim() == 2 else image[0] for image, _ in items]).to(device)
    rows: Int64[Tensor, "m"] = torch.tensor([row for _, row in items], dtype=torch.int64)
    return perspective_keynet_samples(
        frames, truth.rig.select([camera]).to(device), truth.letterboxes[camera], truth.rolls_rad[camera],
        truth.image_hands("truth", rows, camera, device), truth.hand_label[rows, camera].to(device),
        extrapolated=truth.image_hands("extrapolated", rows, camera, device), stale=truth.image_hands("stale", rows, camera, device),
        phi=truth.phi, dataset=dataset_index, augment=augment, generator=generator,
    )


# --- per-draw augmentation ------------------------------------------------------------------------------------------------------------


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DrawAugment:
    """Augmentation applied every time a stored crop is drawn (our choices; the coherent prior error is the deployed net's recipe)."""

    intensity_range: tuple[float, float] = (0.6, 1.4)
    gamma_range: tuple[float, float] = (0.7, 1.4)
    """Intensity gamma drawn log-uniformly in this range (exposure curves, IR-like grey)."""
    blur_probability: float = 0.25
    blur_sigma_range: tuple[float, float] = (0.4, 1.6)
    """Gaussian blur sigma in crop pixels (defocus, motion, a camera with less detail than the crop)."""
    noise_probability: float = 0.3
    noise_std_range: tuple[float, float] = (0.005, 0.04)
    """Additive Gaussian sensor noise (intensity units)."""
    occlusion_probability: float = 0.2
    occlusion_max_fraction: float = 0.4
    zero_input_probability: float = 0.2
    """The keypoint input set to zeros (an untracked hand after a DetNet detection)."""
    prior_probability: float = 0.7
    """Share of tracked priors given one coherent whole-hand error (rotation, scale, shift about the centroid, one d offset)."""
    prior_shift_std: float = 0.10
    prior_rotation_std_deg: float = 12.0
    prior_scale_std: float = 0.12
    prior_d_shift_std: float = 0.10

    def __post_init__(self) -> None:
        probabilities: tuple[float, ...] = (self.blur_probability, self.noise_probability, self.occlusion_probability, self.zero_input_probability, self.prior_probability)
        if not all(0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError("augmentation probabilities must lie in [0, 1]")
        if not (0 < self.gamma_range[0] <= self.gamma_range[1] and 0 < self.blur_sigma_range[0] <= self.blur_sigma_range[1]):
            raise ValueError("gamma and blur ranges must be positive and ordered")


def coherent_prior_noise(keypoints: Float32[Tensor, "b 63"], generator: torch.Generator, augment: DrawAugment) -> Float32[Tensor, "b 63"]:
    """With ``prior_probability``, rotate and scale the 21 prior points about their centroid, shift them together and offset d for the
    whole hand (the tracker's own prior errors are coherent, not independent per keypoint). All-zero priors stay zero."""
    b: int = keypoints.shape[0]
    device: torch.device = keypoints.device
    points: Float32[Tensor, "b 21 3"] = keypoints.reshape(b, 21, 3)
    tracked: Bool[Tensor, "b"] = points.abs().sum(dim=(1, 2)) > 0
    draws: Float32[Tensor, "b 6"] = torch.randn((b, 6), generator=generator, device=device)
    chosen: Bool[Tensor, "b"] = tracked & (torch.rand(b, generator=generator, device=device) < augment.prior_probability)
    uv: Float32[Tensor, "b 21 2"] = points[..., :2]
    centre: Float32[Tensor, "b 1 2"] = uv.mean(dim=1, keepdim=True)
    angle: Float32[Tensor, "b"] = draws[:, 0] * math.radians(augment.prior_rotation_std_deg)
    scale: Float32[Tensor, "b"] = torch.exp(draws[:, 1] * augment.prior_scale_std)
    cosine, sine = torch.cos(angle) * scale, torch.sin(angle) * scale
    rotation: Float32[Tensor, "b 2 2"] = torch.stack((torch.stack((cosine, -sine), -1), torch.stack((sine, cosine), -1)), -2)
    moved: Float32[Tensor, "b 21 2"] = torch.einsum("bij,bkj->bki", rotation, uv - centre) + centre + draws[:, None, 2:4] * augment.prior_shift_std
    shifted: Float32[Tensor, "b 21 1"] = points[..., 2:] + draws[:, None, 4:5] * augment.prior_d_shift_std
    noisy: Float32[Tensor, "b 63"] = torch.cat((moved, shifted), dim=-1).reshape(b, 63)
    return torch.where(chosen[:, None], noisy, keypoints)


def _blur(images: Float32[Tensor, "b 1 h w"], sigma: Float32[Tensor, "b"]) -> Float32[Tensor, "b 1 h w"]:
    """Per-image separable Gaussian blur (radius 4 px, reflect padding) as one grouped convolution per axis."""
    b: int = images.shape[0]
    offsets: Float32[Tensor, "9"] = torch.arange(-4, 5, device=images.device, dtype=images.dtype)
    kernel: Float32[Tensor, "b 9"] = torch.exp(-offsets[None].square() / (2 * sigma[:, None].square()))
    kernel = kernel / kernel.sum(dim=-1, keepdim=True)
    flat: Float32[Tensor, "1 b h w"] = images.reshape(1, b, *images.shape[-2:])
    flat = F.conv2d(F.pad(flat, (4, 4, 0, 0), mode="reflect"), kernel[:, None, None, :], groups=b)
    flat = F.conv2d(F.pad(flat, (0, 0, 4, 4), mode="reflect"), kernel[:, None, :, None], groups=b)
    return flat.reshape(images.shape)


def keynet_finish(augment: DrawAugment | None):
    """``EchoStream`` finish: augment drawn crops (None: evaluation, no augmentation) and render the heatmap targets."""

    def finish(samples: KeyNetSamples, generator: torch.Generator) -> KeyNetBatch:
        crops: Float32[Tensor, "b 1 96 96"] = (samples.crops.float() / 255.0)[:, None]
        keypoints: Float32[Tensor, "b 63"] = samples.keypoints
        b: int = crops.shape[0]
        device: torch.device = crops.device
        if augment is not None:
            occluded: Bool[Tensor, "b 96 96"] = boundary_occlusion(b, generator, device, augment.occlusion_probability, augment.occlusion_max_fraction)
            crops = crops.masked_fill(occluded[:, None], 0.0)
            low, high = math.log(augment.gamma_range[0]), math.log(augment.gamma_range[1])
            gamma: Float32[Tensor, "b 1 1 1"] = torch.exp(low + (high - low) * torch.rand((b, 1, 1, 1), generator=generator, device=device))
            crops = crops.clamp_min(0.0).pow(gamma)
            crops = scale_intensity(crops, generator, *augment.intensity_range)
            draws: Float32[Tensor, "b 4"] = torch.rand((b, 4), generator=generator, device=device)
            sigma: Float32[Tensor, "b"] = augment.blur_sigma_range[0] + draws[:, 0] * (augment.blur_sigma_range[1] - augment.blur_sigma_range[0])
            crops = torch.where((draws[:, 1] < augment.blur_probability)[:, None, None, None], _blur(crops, sigma), crops)
            noise_std: Float32[Tensor, "b"] = augment.noise_std_range[0] + draws[:, 2] * (augment.noise_std_range[1] - augment.noise_std_range[0])
            noise: Float32[Tensor, "b 1 96 96"] = torch.randn(crops.shape, generator=generator, device=device) * noise_std[:, None, None, None]
            crops = torch.where((draws[:, 3] < augment.noise_probability)[:, None, None, None], (crops + noise).clamp(0.0, 1.0), crops)
            keypoints = coherent_prior_noise(keypoints, generator, augment)
            zero: Bool[Tensor, "b"] = torch.rand(b, generator=generator, device=device) < augment.zero_input_probability
            keypoints = torch.where(zero[:, None], torch.zeros_like(keypoints), keypoints)
        positive: Bool[Tensor, "b"] = samples.kind == int(CropKind.POSITIVE)
        return KeyNetBatch(
            crops=crops, keypoints=keypoints, heatmaps=render_heatmaps(samples.points_crop) * positive[:, None, None, None],
            distance=render_distance(samples.d_rel_mm) * positive[:, None, None], presence=samples.presence, positive=positive,
            presence_mask=torch.ones_like(positive), kind=samples.kind, dataset=samples.dataset, points_crop=samples.points_crop, d_rel_mm=samples.d_rel_mm,
        )

    return finish
