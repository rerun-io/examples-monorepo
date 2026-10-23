"""Gen-HumanWholeBody derived layers: ``pose``, ``actions``, ``projections`` and ``hand_mesh``, all from the sidecar.

The body track (``human_pose_fuse_smooth_concat``) is an SMPL-X-shaped skeleton, not SMPL-X parameters:
its 22 joint positions are good, its rotations are not (no forward kinematics of them keeps a constant
rest skeleton: bones drift 4 cm median, 15 cm worst), and it ships no betas or translation. The layers
therefore carry the positions only. The hands ship 21 world-frame bones and complete MANO parameters;
``hand_mesh`` evaluates those parameters and is written only for a hand whose MANO joints land on its
shipped bones (``MANO_GATE_M``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64, UInt32
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.ops.mano.mano_np import MANOLayerNP
from simplecv.ops.smplx.smplx_coco133 import smplx_joints_to_coco133_xyzc

from dataforge import actions, hands, meshes, paths, schema, writing
from dataforge.datasets.genrobot_source import (
    BODY_TOPIC,
    EEF_POSE_TOPIC,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    Annotation,
    CameraCalibration,
    Episode,
    EpisodeClock,
    EpisodeFile,
    Segment,
    Side,
    SubSegment,
    hand_topic,
    rig_T_cam,
    rotation_matrices,
    struct_field,
    struct_rows,
    topic_rows,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import frame_index_column, log_dense_pose_track, time_column

POSE_LAYER: str = "pose"
"""Body and hand keypoints share ``/world/gt/coco133_xyz``, so one layer writes both."""

SIDES: tuple[Side, Side] = ("left", "right")
"""Hand order of every ``[.., 2, ..]`` array, as ``hands.HAND_SIDES``."""

BODY_MODEL: str = "smplx22"
"""``/world/gt/body/smplx22``: the shipped SMPL-X body22 joint positions."""
BODY_CLASS: int = 1
"""Class of the body22 skeleton in its own annotation context; the root context's class 0 is COCO-133."""
BODY_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot",
    "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)  # fmt: skip
"""SMPL-X body joints 0-21 in the source's order."""
BODY_PARENTS: tuple[int, ...] = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
"""SMPL-X kinematic tree of those joints."""
BODY_COLOR: tuple[int, int, int] = (230, 90, 70)
"""Body skeleton tint."""

MANO_GATE_M: float = 0.002
"""Most a hand's median MANO-joint-to-shipped-bone distance may be for its mesh to be written."""
MANO_BATCH: int = 256
"""Frames skinned and sent per mesh chunk."""
MIRROR_X: Float64[ndarray, "3 3"] = np.diag([-1.0, 1.0, 1.0])
"""The left hand's shipped finger rotations are the right hand's convention: conjugating by this mirrors them."""


def body_context() -> rr.AnnotationContext:
    """Names and bones of the body22 skeleton, scoped to the entities that draw it."""
    return rr.AnnotationContext(
        [
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=BODY_CLASS, label="smplx_body22", color=BODY_COLOR),
                keypoint_annotations=[rr.AnnotationInfo(id=index, label=name) for index, name in enumerate(BODY_NAMES)],
                keypoint_connections=[(child, parent) for child, parent in enumerate(BODY_PARENTS) if parent >= 0],
            )
        ]
    )


@dataclass(frozen=True, slots=True)
class PoseTrack:
    """Body and hands on the episode's frames; NaN where a frame has no message or the message has no data."""

    times_ns: Int64[ndarray, "f"]
    """``video_time`` of each frame."""
    frame_indices: Int64[ndarray, "f"]
    """Shipped frame numbers (``frame_validity.frame_index``), ascending."""
    body: Float64[ndarray, "f 22 3"]
    """Body22 joint positions in world, metres."""
    bones: Float64[ndarray, "f 2 21 3"]
    """Hand bones in world (``to_global``), COCO hand order, metres."""
    wrists: Float64[ndarray, "f 2 4 4"]
    """``world_T_wrist`` of each hand (bone 0 ``to_global``)."""
    hand_present: Bool[ndarray, "f 2"]
    """Whether the frame has the hand's 21 tracked bones."""
    mano_present: Bool[ndarray, "f 2"]
    """Whether that message carries MANO parameters; a few frames ship bones without them."""
    betas: Float64[ndarray, "f 2 10"]
    """MANO shape, per message."""
    global_orient: Float64[ndarray, "f 2 3 3"]
    """MANO root rotation."""
    hand_pose: Float64[ndarray, "f 2 15 3 3"]
    """MANO finger rotations, as shipped (right-hand convention for both hands)."""
    global_transl: Float64[ndarray, "f 2 3"]
    """MANO translation; with the root rotation it puts the model in world."""
    is_valid: Bool[ndarray, "f 2"]
    """Shipped flag; False on every frame seen, so it is kept verbatim and not used as confidence."""
    reproj_loss: Float64[ndarray, "f 2"]
    """Shipped reprojection loss; 0 on every frame seen."""
    bbox_touches_edge: Bool[ndarray, "f 2"]
    """Shipped flag."""

    @property
    def coco133(self) -> Float32[ndarray, "f 133 3"]:
        """Body into its 12 COCO body slots and the two hand roots, then both hands' 21 bones over the hand slots."""
        stack: Float32[ndarray, "f 133 4"] = smplx_joints_to_coco133_xyzc(self.body)
        positions: Float32[ndarray, "f 133 3"] = np.where(stack[..., 3:] > 0.0, stack[..., :3], np.float32(np.nan))
        positions[:, hands.LEFT_HAND_IDX] = self.bones[:, 0].astype(np.float32)
        positions[:, hands.RIGHT_HAND_IDX] = self.bones[:, 1].astype(np.float32)
        return positions


def list_presence(values: pa.ListArray, width: int, *, what: str) -> Bool[ndarray, "n"]:
    """Rows of a list column that hold exactly ``width`` items; null and empty rows are absent.

    Raises:
        ValueError: A row holds some other count, which no reader here can interpret.
    """
    lengths: Int64[ndarray, "n"] = values.value_lengths().fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.all((lengths == width) | (lengths == 0)):
        raise ValueError(f"{what}: a row holds {sorted(set(lengths.tolist()) - {0, width})} items, not {width}")
    return lengths == width


def fixed_lists(values: pa.ListArray, width: int, *, what: str) -> tuple[Bool[ndarray, "n"], Float64[ndarray, "k width"]]:
    """The rows of a list column that hold ``width`` numbers, and those numbers; null or empty rows are absent."""
    present: Bool[ndarray, "n"] = list_presence(values, width, what=what)
    numbers: Float64[ndarray, "k width"] = values.filter(pa.array(present)).flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, width)
    return present, numbers


def row_positions(frames: Int64[ndarray, "f"], wanted: Int64[ndarray, "n"], *, what: str) -> tuple[Bool[ndarray, "n"], Int64[ndarray, "n"]]:
    """Where each wanted frame number sits in ``frames`` (ascending), and whether it is there at all.

    Raises:
        ValueError: Two messages of one topic fall on the same frame.
    """
    if np.unique(wanted).size != wanted.size:
        raise ValueError(f"{what}: two messages on one frame")
    positions: Int64[ndarray, "n"] = np.clip(np.searchsorted(frames, wanted), 0, max(frames.size - 1, 0))
    return frames[positions] == wanted, positions


def bools(values: pa.Array) -> Bool[ndarray, "n"]:
    """A shipped boolean column, null as False."""
    return values.fill_null(False).to_numpy(zero_copy_only=False).astype(bool)


def read_pose_track(file: EpisodeFile, clock: EpisodeClock) -> PoseTrack:
    """Body and hands of one episode on its frames: every reference frame, plus any frame a pose message falls on that the reference lacks."""
    body_stamps, body_values = topic_rows(file, BODY_TOPIC)
    hand_messages: list[tuple[Int64[ndarray, "n"], pa.Array]] = [topic_rows(file, hand_topic(side)) for side in SIDES]
    stamp_of: dict[int, int] = dict(zip(clock.frame_numbers.tolist(), clock.frame_times_ns.tolist(), strict=True))
    for stamps in (body_stamps, *(stamps for stamps, _ in hand_messages)):
        for number, stamp in zip(clock.frame_indices(stamps).tolist(), stamps.tolist(), strict=True):
            stamp_of.setdefault(number, stamp)
    frames: Int64[ndarray, "f"] = np.array(sorted(stamp_of), dtype=np.int64)
    count: int = frames.size
    body: Float64[ndarray, "f 22 3"] = np.full((count, 22, 3), np.nan)
    if len(body_values):
        rows: Int64[ndarray, "n"] = row_positions(frames, clock.frame_indices(body_stamps), what=BODY_TOPIC)[1]
        poses: pa.ListArray = body_values.field("poses")
        has_body: Bool[ndarray, "n"] = list_presence(poses, 22, what=f"{file.path}:{BODY_TOPIC} poses")
        body[rows[has_body]] = struct_rows(struct_field(poses.filter(pa.array(has_body)).flatten(), "position")).reshape(-1, 22, 3)
    bones: Float64[ndarray, "f 2 21 3"] = np.full((count, 2, 21, 3), np.nan)
    wrists: Float64[ndarray, "f 2 4 4"] = np.full((count, 2, 4, 4), np.nan)
    present: Bool[ndarray, "f 2"] = np.zeros((count, 2), dtype=bool)
    mano_present: Bool[ndarray, "f 2"] = np.zeros((count, 2), dtype=bool)
    betas: Float64[ndarray, "f 2 10"] = np.full((count, 2, 10), np.nan)
    orient: Float64[ndarray, "f 2 3 3"] = np.full((count, 2, 3, 3), np.nan)
    pose: Float64[ndarray, "f 2 15 3 3"] = np.full((count, 2, 15, 3, 3), np.nan)
    transl: Float64[ndarray, "f 2 3"] = np.full((count, 2, 3), np.nan)
    is_valid: Bool[ndarray, "f 2"] = np.zeros((count, 2), dtype=bool)
    reproj: Float64[ndarray, "f 2"] = np.full((count, 2), np.nan)
    edge: Bool[ndarray, "f 2"] = np.zeros((count, 2), dtype=bool)
    for side_index, (side, (stamps, values)) in enumerate(zip(SIDES, hand_messages, strict=True)):
        if not len(values):
            continue
        topic: str = f"{file.path}:{hand_topic(side)}"
        rows = row_positions(frames, clock.frame_indices(stamps), what=topic)[1]
        bone_lists: pa.ListArray = values.field("bone_data")
        has_bones: Bool[ndarray, "n"] = list_presence(bone_lists, 21, what=f"{topic} bone_data")
        to_global: pa.StructArray = struct_field(bone_lists.filter(pa.array(has_bones)).flatten(), "to_global")
        positions: Float64[ndarray, "k 21 3"] = struct_rows(struct_field(to_global, "position")).reshape(-1, 21, 3)
        orientations: Float64[ndarray, "k 21 4"] = struct_rows(struct_field(to_global, "orientation"), "xyzw").reshape(-1, 21, 4)
        wrist_norms: Float64[ndarray, "k"] = np.linalg.norm(orientations[:, 0], axis=1)
        tracked: Bool[ndarray, "k"] = np.isfinite(positions).all(axis=(1, 2)) & np.isfinite(wrist_norms) & (wrist_norms > 0.0)
        with_bones: Int64[ndarray, "k"] = rows[has_bones]
        bones[with_bones, side_index] = positions
        wrist: Float64[ndarray, "k 4 4"] = np.full((len(with_bones), 4, 4), np.nan)
        wrist[tracked] = np.eye(4)
        wrist[tracked, :3, :3] = rotation_matrices(orientations[tracked, 0])
        wrist[tracked, :3, 3] = positions[tracked, 0]
        wrists[with_bones, side_index] = wrist
        present[with_bones[tracked], side_index] = True
        mano: pa.StructArray = values.field("mano_params")
        fields: dict[str, tuple[Bool[ndarray, "n"], Float64[ndarray, "k w"]]] = {
            name: fixed_lists(mano.field(name), width, what=f"{topic} mano_params.{name}")
            for name, width in (("betas", 10), ("global_orient", 9), ("hand_pose", 135), ("global_transl", 3))
        }
        has_mano: Bool[ndarray, "n"] = np.all([has for has, _ in fields.values()], axis=0)
        if any(not np.array_equal(has, has_mano) for has, _ in fields.values()):
            raise ValueError(f"{topic}: MANO fields are present on different frames")
        with_mano: Int64[ndarray, "k"] = rows[has_mano]
        mano_present[with_mano, side_index] = True
        betas[with_mano, side_index] = fields["betas"][1]
        orient[with_mano, side_index] = fields["global_orient"][1].reshape(-1, 3, 3)
        pose[with_mano, side_index] = fields["hand_pose"][1].reshape(-1, 15, 3, 3)
        transl[with_mano, side_index] = fields["global_transl"][1]
        is_valid[rows, side_index] = bools(values.field("is_valid"))
        reproj[rows, side_index] = values.field("reproj_loss").fill_null(np.nan).to_numpy(zero_copy_only=False)
        edge[rows, side_index] = bools(values.field("bbox_touches_edge"))
    return PoseTrack(
        times_ns=clock.video_time_ns(np.array([stamp_of[number] for number in frames.tolist()], dtype=np.int64)),
        frame_indices=frames,
        body=body,
        bones=bones,
        wrists=wrists,
        hand_present=present,
        mano_present=mano_present,
        betas=betas,
        global_orient=orient,
        hand_pose=pose,
        global_transl=transl,
        is_valid=is_valid,
        reproj_loss=reproj,
        bbox_touches_edge=edge,
    )


def read_rig_poses(file: EpisodeFile, clock: EpisodeClock, frames: Int64[ndarray, "f"]) -> Float64[ndarray, "f 4 4"]:
    """``world_T_rig`` on the given frames; NaN where the VIO has no pose (often its first and last frames)."""
    poses: Float64[ndarray, "f 4 4"] = np.full((frames.size, 4, 4), np.nan)
    stamps, values = topic_rows(file, EEF_POSE_TOPIC)
    if len(values):
        found, rows = row_positions(frames, clock.frame_indices(stamps), what=f"{file.path}:{EEF_POSE_TOPIC}")
        pose: pa.StructArray = struct_field(values, "pose").filter(pa.array(found))
        quaternions: Float64[ndarray, "n 4"] = struct_rows(struct_field(pose, "orientation"), "xyzw")
        transform: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (len(quaternions), 1, 1))
        transform[:, :3, :3] = rotation_matrices(quaternions)
        transform[:, :3, 3] = struct_rows(struct_field(pose, "position"))
        poses[rows[found]] = transform
    return poses


def log_points(
    recording: rr.RecordingStream, path: str, *, times_ns: Int64[ndarray, "f"], frame_indices: Int64[ndarray, "f"], positions: Float64[ndarray, "f k d"]
) -> None:
    """Dense keypoint rows of one skeleton, NaN where absent, so the viewer never holds a stale skeleton."""
    rows: int = positions.shape[0]
    count: int = positions.shape[1]
    points: Float32[ndarray, "n d"] = positions.reshape(-1, positions.shape[2]).astype(np.float32)
    columns = (rr.Points3D if positions.shape[2] == 3 else rr.Points2D).columns(positions=points).partition([count] * rows)
    rr.send_columns(path, indexes=[time_column(times_ns), frame_index_column(frame_indices)], columns=columns, recording=recording)


def log_body(recording: rr.RecordingStream, track: PoseTrack) -> None:
    """``/world/gt/body/smplx22``: the 22 shipped joint positions, joined by the SMPL-X tree."""
    path: str = schema.body_path(BODY_MODEL)
    rr.log(path, body_context(), static=True, recording=recording)
    count: int = len(BODY_NAMES)
    rr.log(path, rr.Points3D.from_fields(keypoint_ids=range(count), class_ids=[BODY_CLASS] * count, radii=0.012, show_labels=False), static=True, recording=recording)
    log_points(recording, path, times_ns=track.times_ns, frame_indices=track.frame_indices, positions=track.body)


def write_pose_layer(identity: SequenceIdentity, track: PoseTrack, target: Path) -> bool:
    """Publish the ``pose`` layer: COCO-133, the body22 skeleton, both hands' wrists, MANO parameters and flags.

    Returns:
        Whether the layer was published: an episode with neither body nor hand data has nothing to carry.
    """
    if not (np.isfinite(track.body).all(axis=(1, 2)).any() or track.hand_present.any()):
        return False
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        hands.log_keypoints3d(recording, times_ns=track.times_ns, frame_indices=track.frame_indices, positions=track.coco133, confidence=None)
        log_body(recording, track)
        for side_index, side in enumerate(SIDES):
            hands.log_hand_confidence(
                recording, side, times_ns=track.times_ns, frame_indices=track.frame_indices, confidence=track.hand_present[:, side_index].astype(np.float64)
            )
            log_dense_pose_track(recording, schema.hand_wrist_path(side), times_ns=track.times_ns, frame_indices=track.frame_indices, transforms=track.wrists[:, side_index])
            indexes: list[rr.TimeColumn] = [time_column(track.times_ns), frame_index_column(track.frame_indices)]
            rr.send_columns(
                schema.hand_mano_path(side),
                indexes=indexes,
                columns=rr.AnyValues.columns(
                    betas=pa.FixedSizeListArray.from_arrays(pa.array(track.betas[:, side_index].reshape(-1)), 10),
                    global_orient_rotmat=pa.FixedSizeListArray.from_arrays(pa.array(track.global_orient[:, side_index].reshape(-1)), 9),
                    hand_pose_rotmat=pa.FixedSizeListArray.from_arrays(pa.array(track.hand_pose[:, side_index].reshape(-1)), 135),
                    global_transl=pa.FixedSizeListArray.from_arrays(pa.array(track.global_transl[:, side_index].reshape(-1)), 3),
                ),
                recording=recording,
            )
            rr.send_columns(
                f"{schema.hands_path(side)}/quality",
                indexes=indexes,
                columns=rr.AnyValues.columns(
                    is_valid=pa.array(track.is_valid[:, side_index]),
                    reproj_loss=pa.array(track.reproj_loss[:, side_index]),
                    bbox_touches_edge=pa.array(track.bbox_touches_edge[:, side_index]),
                ),
                recording=recording,
            )
        recording.send_property(
            POSE_LAYER,
            rr.AnyValues(
                body_source=pa.array([BODY_TOPIC], type=pa.string()),
                body_rotations=pa.array(["dropped: no constant rest skeleton under forward kinematics"], type=pa.string()),
                hand_confidence=pa.array(["1.0 where the hand message exists; the shipped is_valid is kept under quality/"], type=pa.string()),
                coverage_body=pa.array([float(np.isfinite(track.body).all(axis=(1, 2)).mean())], type=pa.float64()),
                coverage_left=pa.array([float(track.hand_present[:, 0].mean())], type=pa.float64()),
                coverage_right=pa.array([float(track.hand_present[:, 1].mean())], type=pa.float64()),
                mano_coverage_left=pa.array([float(track.mano_present[:, 0].mean())], type=pa.float64()),
                mano_coverage_right=pa.array([float(track.mano_present[:, 1].mean())], type=pa.float64()),
            ),
        )
    return True


def world_T_cams(rig_poses: Float64[ndarray, "f 4 4"], calibrations: Sequence[CameraCalibration]) -> Float64[ndarray, "c f 4 4"]:
    """Every camera's world pose on every reference frame; NaN where the rig has none."""
    return np.stack([rig_poses @ rig_T_cam(calibration.T_b_c) for calibration in calibrations])


def project(world_T_cam: Float64[ndarray, "f 4 4"], calibration: CameraCalibration, points: Float64[ndarray, "f k 3"]) -> Float64[ndarray, "f k 2"]:
    """World points into one camera through its double-sphere lens; NaN off the image or without a rig pose."""
    cam_T_world: Float64[ndarray, "f 4 4"] = np.full_like(world_T_cam, np.nan)
    posed: Bool[ndarray, "f"] = np.isfinite(world_T_cam).all(axis=(1, 2))
    cam_T_world[posed] = np.linalg.inv(world_T_cam[posed])
    in_cam: Float64[ndarray, "f k 3"] = np.einsum("fij,fkj->fki", cam_T_world[:, :3, :3], points) + cam_T_world[:, None, :3, 3]
    pixels: Float64[ndarray, "n 2"] = calibration.lens.project(in_cam.reshape(-1, 3))
    inside: Bool[ndarray, "n"] = (pixels[:, 0] >= 0) & (pixels[:, 0] < SOURCE_WIDTH) & (pixels[:, 1] >= 0) & (pixels[:, 1] < SOURCE_HEIGHT)
    return np.where(inside[:, None], pixels, np.nan).reshape(points.shape[0], points.shape[1], 2)


def smplx22_uv_projected_path(cam: int) -> str:
    """The body22 skeleton projected into one camera, beside its ``coco133_uv_projected``."""
    return f"{schema.pinhole_path(0, cam)}/smplx22_uv_projected"


def write_projections_layer(
    identity: SequenceIdentity, track: PoseTrack, rig_poses: Float64[ndarray, "f 4 4"], calibrations: Sequence[CameraCalibration], target: Path
) -> None:
    """Publish ``projections``: COCO-133 and the body22 skeleton through each camera's double-sphere lens.

    A Rerun ``Pinhole`` cannot draw 3D through this lens, so these 2D points are what the fisheye panes show.
    """
    poses: Float64[ndarray, "c f 4 4"] = world_T_cams(rig_poses, calibrations)
    coco: Float64[ndarray, "f 133 3"] = track.coco133.astype(np.float64)
    confidence: Float32[ndarray, "f 133"] = np.isfinite(coco).all(axis=-1).astype(np.float32)

    def pixels_by_camera() -> Iterator[tuple[tuple[int, int], Float32[ndarray, "f 133 2"]]]:
        for cam, calibration in enumerate(calibrations):
            yield (0, cam), project(poses[cam], calibration, coco).astype(np.float32)

    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        hands.log_projections(
            recording, pixels_by_camera(), times_ns=track.times_ns, frame_indices=track.frame_indices, confidence=confidence,
            camera_model="ds", calibration_source="camera_info (double sphere) per episode",
        )  # fmt: skip
        count: int = len(BODY_NAMES)
        for cam, calibration in enumerate(calibrations):
            path: str = smplx22_uv_projected_path(cam)
            rr.log(path, body_context(), static=True, recording=recording)
            rr.log(path, rr.Points2D.from_fields(keypoint_ids=range(count), class_ids=[BODY_CLASS] * count, radii=3.0, show_labels=False), static=True, recording=recording)
            log_points(recording, path, times_ns=track.times_ns, frame_indices=track.frame_indices, positions=project(poses[cam], calibration, track.body))


def mano_layer(side: Side, betas: Float32[ndarray, "10"]) -> MANOLayerNP:
    """A MANO layer that reproduces the source's hands.

    The left model gets the well-known MANO left-hand shape fix (the first shape direction's x is negated,
    as in ``smplx`` and ``manopth``). The array is copied first: the loaded model may be shared.
    """
    layer: MANOLayerNP = MANOLayerNP(side=side, betas=betas, use_pca=False)
    if side == "left":
        shapedirs: Float32[ndarray, "778 3 10"] = layer._mano_layer.th_shapedirs.copy()
        shapedirs[:, 0, :] *= -1.0
        layer._mano_layer.th_shapedirs = shapedirs
    return layer


def mano_parameters(track: PoseTrack, side_index: int, rows: Int64[ndarray, "n"]) -> tuple[Float32[ndarray, "n 48"], Float32[ndarray, "n 3"]]:
    """Axis-angle MANO pose (root, then 15 fingers) and translation of some frames of one hand."""
    fingers: Float64[ndarray, "n 15 3 3"] = track.hand_pose[rows, side_index]
    if SIDES[side_index] == "left":
        fingers = MIRROR_X @ fingers @ MIRROR_X
    rotations: Float64[ndarray, "n 16 3 3"] = np.concatenate([track.global_orient[rows, side_index][:, None], fingers], axis=1)
    pose: Float32[ndarray, "n 48"] = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 48).astype(np.float32)
    return pose, track.global_transl[rows, side_index].astype(np.float32)


def skin(track: PoseTrack, side_index: int, rows: Int64[ndarray, "n"]) -> tuple[Float32[ndarray, "n 778 3"], Float32[ndarray, "n 21 3"]]:
    """MANO vertices and joints of some frames of one hand, each frame with its own shipped shape (betas)."""
    vertices: Float32[ndarray, "n 778 3"] = np.empty((rows.size, 778, 3), dtype=np.float32)
    joints: Float32[ndarray, "n 21 3"] = np.empty((rows.size, 21, 3), dtype=np.float32)
    parameters: tuple[Float32[ndarray, "n 48"], Float32[ndarray, "n 3"]] = mano_parameters(track, side_index, rows)
    shapes: Float32[ndarray, "s 10"]
    shapes, shape_of_row = np.unique(track.betas[rows, side_index].astype(np.float32), axis=0, return_inverse=True)
    for shape_index, shape in enumerate(shapes):
        selected: Bool[ndarray, "n"] = shape_of_row.reshape(-1) == shape_index
        result: tuple[Float32[ndarray, "k 778 3"], Float32[ndarray, "k 21 3"]] = mano_layer(SIDES[side_index], shape)(parameters[0][selected], parameters[1][selected])
        vertices[selected] = result[0]
        joints[selected] = result[1]
    return vertices, joints


@dataclass(frozen=True, slots=True)
class ManoCheck:
    """How far one hand's MANO joints land from its shipped bones."""

    median_m: float
    """Median joint distance over every joint of every frame with both MANO parameters and bones; NaN without frames."""
    passed: bool
    """Whether the mesh may be written (``MANO_GATE_M``)."""


def check_mano(track: PoseTrack, side_index: int) -> ManoCheck:
    """Skin every frame of one hand that ships MANO parameters and bones, and compare its 21 joints with the bones."""
    rows: Int64[ndarray, "n"] = np.flatnonzero(track.mano_present[:, side_index] & track.hand_present[:, side_index])
    if rows.size == 0:
        return ManoCheck(median_m=float("nan"), passed=False)
    joints: Float32[ndarray, "n 21 3"] = skin(track, side_index, rows)[1]
    median: float = float(np.median(np.linalg.norm(joints - track.bones[rows, side_index], axis=-1)))
    return ManoCheck(median_m=median, passed=median < MANO_GATE_M)


def write_hand_mesh_layer(identity: SequenceIdentity, track: PoseTrack, target: Path) -> tuple[ManoCheck, ManoCheck]:
    """Publish ``hand_mesh`` for each hand that passes ``check_mano``; the checks become its properties.

    When neither hand passes, nothing is published: a layer without data would break the layer rule, and
    the next convert simply checks again.
    """
    checks: tuple[ManoCheck, ManoCheck] = (check_mano(track, 0), check_mano(track, 1))
    if not any(check.passed for check in checks):
        return checks
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        for side_index, side in enumerate(SIDES):
            if not checks[side_index].passed:
                continue
            faces: UInt32[ndarray, "t 3"] = np.asarray(mano_layer(side, np.zeros(10, dtype=np.float32)).f, dtype=np.uint32)
            present: Bool[ndarray, "f"] = track.mano_present[:, side_index]
            path: str = schema.hand_mesh_path(side)
            meshes.log_mesh_static(recording, path, albedo_factor=hands.HAND_ALBEDO[side])
            for start in range(0, present.size, MANO_BATCH):
                batch: slice = slice(start, start + MANO_BATCH)
                rows: Int64[ndarray, "n"] = start + np.flatnonzero(present[batch])
                meshes.log_mesh_batch(
                    recording, path, times_ns=track.times_ns[batch], frame_indices=track.frame_indices[batch], vertices=skin(track, side_index, rows)[0],
                    trusted=present[batch].tolist(), topology=faces,
                )  # fmt: skip
        recording.send_property(
            paths.HAND_MESH_LAYER,
            rr.AnyValues(
                model=pa.array(["MANO (full pose, no PCA; left shape fix; left finger rotations mirrored)"], type=pa.string()),
                gate_m=pa.array([MANO_GATE_M], type=pa.float64()),
                median_joint_error_left_m=pa.array([checks[0].median_m], type=pa.float64()),
                median_joint_error_right_m=pa.array([checks[1].median_m], type=pa.float64()),
                written_left=pa.array([checks[0].passed]),
                written_right=pa.array([checks[1].passed]),
            ),
        )
    return checks


def write_actions_layer(identity: SequenceIdentity, episode: Episode, target: Path) -> bool:
    """Publish ``actions``: coarse segments and fine sub-segments as active-label tracks, every field as properties.

    Labels are placed by their frame numbers (half-open: ``end_frame`` is the first frame after) on the
    episode clock; a bound off the reference frames takes the time the camera rate gives it.
    Sub-segments may cross segment bounds and run past the recording; they are kept as shipped.

    Returns:
        Whether the layer was published: an episode without annotated segments has no actions to carry.
    """
    annotation: Annotation | None = episode.annotation
    segments: list[Segment] = [] if annotation is None else annotation.segments_info or []
    if annotation is None or not segments:
        return False
    subs: list[SubSegment] = annotation.sub_segments
    levels: dict[str, list[actions.Segment]] = {
        "coarse": [actions.Segment(segment.start_frame, max(segment.end_frame, segment.start_frame), segment.fine_label) for segment in segments],
        "fine": [actions.Segment(sub.start_frame, max(sub.end_frame, sub.start_frame), sub.fine_label) for sub in subs],
    }
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        for level, spans in levels.items():
            if spans:
                boundaries, texts = actions.active_labels(spans)
                times: Int64[ndarray, "k"] = episode.clock.video_time_ns(episode.clock.stamps_ns(boundaries))
                actions.log_actions(recording, level, times_ns=times, frame_indices=boundaries, texts=texts)
        recording.send_property(
            paths.ACTIONS_LAYER,
            rr.AnyValues(
                segment_ids=pa.array([[segment.segment_id for segment in segments]], type=pa.list_(pa.string())),
                segment_labels=pa.array([[segment.fine_label for segment in segments]], type=pa.list_(pa.string())),
                segment_start_frames=pa.array([[segment.start_frame for segment in segments]], type=pa.list_(pa.int64())),
                segment_end_frames=pa.array([[segment.end_frame for segment in segments]], type=pa.list_(pa.int64())),
                sub_segment_ids=pa.array([[sub.segment_id for sub in subs]], type=pa.list_(pa.string())),
                sub_segment_labels=pa.array([[sub.fine_label for sub in subs]], type=pa.list_(pa.string())),
                sub_segment_start_frames=pa.array([[sub.start_frame for sub in subs]], type=pa.list_(pa.int64())),
                sub_segment_end_frames=pa.array([[sub.end_frame for sub in subs]], type=pa.list_(pa.int64())),
                sub_segment_is_success=pa.array([[sub.is_success for sub in subs]], type=pa.list_(pa.bool_())),
                sub_segment_details=pa.array([[sub.fine_label_detail for sub in subs]], type=pa.list_(pa.string())),
                environment_description=pa.array([annotation.environment_description], type=pa.string()),
            ),
        )
    return True
