"""The ``detnet_v1`` and ``handtrack_v1`` Rerun layers of a UmeTrack segment, written from the tracker's ``SegmentTrack``.

Each layer is one rrd per segment on the segment's own recording id, so the catalog stacks it onto the
base recording; everything is logged column-wise on the base layer's ``video_time`` and ``frame_index``.

Entity paths (rig 0, cameras ``cam_00``..``cam_03``, sides ``left``/``right``):

- ``detnet_v1``: DetNet alone on every frame and camera.
  - ``<pinhole>/boxes/{left,right}_hand``: Boxes2D, the schema's reserved box entity (§13), labelled with the presence;
    yellow at presence >= 0.5, grey below.
  - ``/detnet/presence/cam_NN/{side}``: Scalars, DetNet's presence per camera.
- ``handtrack_v1``: the full pipeline.
  - ``<pinhole>/handtrack/boxes/{side}_hand``: Boxes2D coloured by source (DetNet magenta, tracked green), labelled with it.
  - ``<pinhole>/handtrack/gt_boxes/{side}_hand``: Boxes2D, the ground-truth box (square of the smallest circle enclosing
    the finite projected ground-truth landmarks in front of the camera), for comparison.
  - ``<pinhole>/handtrack/keypoints/{side}``: Points2D, KeyNet's 21 keypoints with the UmeTrack skeleton.
  - ``<pinhole>/handtrack/gt_keypoints/{side}``: Points2D, the projected ground-truth landmarks with the same skeleton (white).
  - ``/world/runs/handtrack/hands/{side}/keypoints``: Points3D, the fitted landmarks with the skeleton.
  - ``/world/runs/handtrack/hands/{side}/mesh``: Mesh3D, the skinned hand (``dataforge.umetrack_hands.log_hand_meshes``).
  - ``/handtrack/error_mm/{side}``, ``/handtrack/presence/cam_NN/{side}``, ``/handtrack/tracked/{side}``,
    ``/handtrack/detnet_presence/{side}``, ``/handtrack/detnet_camera``, ``/handtrack/fit_energy/{side}``: Scalars.

The predicted 3D root follows the schema's ``/world/runs/<source>`` for derived outputs of one processing
source, beside the ground truth's ``/world/gt/hands``.
"""

from dataclasses import dataclass, replace

import numpy as np
import rerun as rr
import torch
from dataforge import hands, schema
from dataforge.logging_toolkit import frame_index_column, time_column
from dataforge.umetrack_hands import log_hand_meshes
from jaxtyping import Bool, Float32, Float64, Int64, Shaped, UInt8
from numpy import ndarray
from serde.json import from_json
from simplecv.umetrack_temp.generic_hand_model_numpy import LANDMARK, UME_HAND_CONNECTIONS, HandModelNumpy, skin_landmarks, wrist_for_hand
from torch import Tensor

from handtrack.data.catalog import is_show3d
from handtrack.geometry.camera import CameraRig, in_front, project, world_to_cameras
from handtrack.hand.pose import GENERIC_HAND_MODEL
from handtrack.labels.circles import enclosing_circles, square_boxes
from handtrack.results import BoxSource, SegmentTrack

RIG: int = 0
"""UmeTrack's headset is rig 0."""
NUM_CAMERAS: int = 4
SIDES: tuple[hands.Side, hands.Side] = ("left", "right")
"""Hand slots in ``SegmentTrack`` order (``hand.pose.Side``)."""
RUN_SOURCE: str = "handtrack"
"""Name of the pipeline under ``/world/runs``, under each camera's pinhole and at the root of its time series."""
SERIES_ROOT: str = "/handtrack"
DETNET_SERIES_ROOT: str = "/detnet"

SOURCE_COLORS: dict[int, tuple[int, int, int]] = {BoxSource.DETNET: (255, 60, 255), BoxSource.TRACKED: (60, 255, 60)}
SOURCE_NAMES: dict[int, str] = {BoxSource.DETNET: "detnet", BoxSource.TRACKED: "tracked"}
GT_BOX_COLOR: tuple[int, int, int] = (255, 255, 255)
PRED_COLORS: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 200, 255), (255, 150, 0))
"""Left and right prediction colours: keypoints, skeleton and series lines."""
DETNET_LAYER_COLOR: tuple[int, int, int] = (255, 220, 0)
"""``detnet_v1`` boxes: yellow, so DetNet-alone never reads as the tracker's own DetNet acquisitions (magenta)."""
DETNET_BELOW_THRESHOLD_COLOR: tuple[int, int, int] = (120, 120, 120)
PRESENCE_THRESHOLD: float = 0.5
KEYPOINT_RADIUS_UI: float = -2.5
"""2D keypoints: 2.5 UI points (``rr.components.Radius`` reads a negative radius as UI points, a positive one as scene units)."""
KEYPOINT_RADIUS_M: float = 0.004
LANDMARK_TOLERANCE_M: float = 0.001
"""The mesh's hand model must reproduce the track's landmarks to this, or the mesh is not the fitted hand."""


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """A segment's ground truth on its video frames: the rig, the headset path and both hands' poses."""

    segment: str
    rig: CameraRig
    video_time_ns: Int64[ndarray, "f"]
    frame_index: Int64[ndarray, "f"]
    world_from_rig: Float32[ndarray, "f 4 4"]
    """NaN where the headset is untracked."""
    rotation: Float32[ndarray, "f 2 3 3"]
    """World-from-wrist rotation; undefined where ``present`` is False."""
    translation: Float32[ndarray, "f 2 3"]
    """World-from-wrist translation in metres."""
    joint_angles: Float32[ndarray, "f 2 22"]
    present: Bool[ndarray, "f 2"]
    """The ground truth has this hand on this frame (confidence > 0)."""
    model: HandModelNumpy
    """The subject's hand model (the ``/world/gt/hands/profile`` document)."""

    def on_frames(self, video_time_ns: Int64[ndarray, "k"]) -> "GroundTruth":
        """The ground truth on exactly these video times; raises if one is missing."""
        index: Int64[ndarray, "k"] = np.searchsorted(self.video_time_ns, video_time_ns)
        if (index >= len(self.video_time_ns)).any() or not np.array_equal(self.video_time_ns[np.minimum(index, len(self.video_time_ns) - 1)], video_time_ns):
            raise ValueError(f"{self.segment}: track frames are not a subset of the ground-truth frames")
        return replace(
            self,
            video_time_ns=self.video_time_ns[index],
            frame_index=self.frame_index[index],
            world_from_rig=self.world_from_rig[index],
            rotation=self.rotation[index],
            translation=self.translation[index],
            joint_angles=self.joint_angles[index],
            present=self.present[index],
        )


def generic_numpy_model() -> HandModelNumpy:
    """UmeTrack's generic hand model as simplecv's numpy model."""
    return from_json(HandModelNumpy, GENERIC_HAND_MODEL.read_text())


def scaled_model(model: HandModelNumpy, scale: float) -> HandModelNumpy:
    """The model × ϕ, UmeTrack's ``scaled_hand_model``: rest joints, rest landmarks and mesh vertices scaled."""
    factor: np.float32 = np.float32(scale)
    return replace(
        model,
        joint_rest_positions=model.joint_rest_positions * factor,
        landmark_rest_positions=model.landmark_rest_positions * factor,
        mesh_vertices=model.mesh_vertices * factor,
    )


def prediction_model(track: SegmentTrack, truth: GroundTruth) -> HandModelNumpy:
    """The hand model the tracker fitted: the subject's for a known hand, the generic one × ϕ for an unknown hand."""
    return truth.model if track.meta.hand_mode == "known" else scaled_model(generic_numpy_model(), track.meta.hand_scale)


def wrists_mm(rotation: Float32[ndarray, "*batch 3 3"], translation: Float32[ndarray, "*batch 3"]) -> Float32[ndarray, "*batch 4 4"]:
    """World-from-wrist transforms in millimetres (the skinning unit), unmirrored; ``translation`` is in metres."""
    transform: Float32[ndarray, "*batch 4 4"] = np.zeros((*translation.shape[:-1], 4, 4), dtype=np.float32)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation * np.float32(1000.0)
    transform[..., 3, 3] = 1.0
    return transform


def skinned_landmarks(model: HandModelNumpy, rotation: Float32[ndarray, "f 2 3 3"], translation: Float32[ndarray, "f 2 3"], joint_angles: Float32[ndarray, "f 2 22"], keep: Bool[ndarray, "f 2"]) -> Float32[ndarray, "f 2 21 3"]:
    """World landmarks in metres of the kept hands, NaN elsewhere."""
    result: Float32[ndarray, "f 2 21 3"] = np.full((*keep.shape, 21, 3), np.nan, dtype=np.float32)
    wrist: Float32[ndarray, "f 2 4 4"] = wrists_mm(rotation, translation)
    for side in range(2):
        rows: Bool[ndarray, "f"] = keep[:, side]
        if rows.any():
            result[rows, side] = skin_landmarks(model, joint_angles[rows, side], wrist_for_hand(wrist[rows, side], side)) * np.float32(0.001)
    return result


def gt_landmarks(truth: GroundTruth) -> Float32[ndarray, "f 2 21 3"]:
    """The ground truth's 21 landmarks per hand in world metres, NaN where the hand is absent."""
    return skinned_landmarks(truth.model, truth.rotation, truth.translation, truth.joint_angles, truth.present)


def camera_pixels(rig: CameraRig, world_from_rig: Float32[ndarray, "f 4 4"], points: Float32[ndarray, "f h n 3"]) -> Float32[ndarray, "f c h n 2"]:
    """Project world points into every camera (unclipped lens model); NaN behind a camera or where an input is NaN."""
    num_frames, num_hands, num_points = points.shape[:3]
    flat: Float32[Tensor, "f m 3"] = torch.from_numpy(np.ascontiguousarray(points.reshape(num_frames, num_hands * num_points, 3)))
    cam_points: Float32[Tensor, "f c m 3"] = world_to_cameras(rig, torch.from_numpy(np.ascontiguousarray(world_from_rig)), flat)
    pixels: Float32[Tensor, "f c m 2"] = project(rig, cam_points)
    pixels[~in_front(cam_points)] = torch.nan
    return pixels.numpy().reshape(num_frames, len(rig.names), num_hands, num_points, 2)


def inside_images(rig: CameraRig, pixels: Float32[ndarray, "f c h n 2"]) -> Bool[ndarray, "f c h n"]:
    """Pixels inside [0, W) x [0, H) of their camera."""
    size: Float32[ndarray, "c 1 1 2"] = rig.image_size.numpy()[:, None, None, :]
    return np.isfinite(pixels).all(-1) & (pixels >= 0).all(-1) & (pixels < size).all(-1)


def enclosing_squares(points: Float32[ndarray, "*batch n 2"], valid: Bool[ndarray, "*batch n"], *, enlarge: float = 1.0) -> Float32[ndarray, "*batch 4"]:
    """(x0, y0, x1, y1) of the square around the smallest circle enclosing the valid points, half side = radius × enlarge; NaN without a valid point."""
    circles: Float32[ndarray, "*batch 3"] = enclosing_circles(points, valid)
    return square_boxes(torch.from_numpy(circles), enlarge=enlarge).numpy()


def keypoint_error_mm(predicted: Float32[ndarray, "f 2 21 3"], truth: Float32[ndarray, "f 2 21 3"], tracked: Bool[ndarray, "f 2"]) -> Float32[ndarray, "f 2"]:
    """Mean 3D landmark distance per frame and hand in millimetres, NaN unless the hand is tracked and in the ground truth."""
    distance: Float32[ndarray, "f 2"] = (np.linalg.norm(predicted - truth, axis=-1).mean(axis=-1) * 1000.0).astype(np.float32)
    return np.where(tracked & np.isfinite(distance), distance, np.float32(np.nan)).astype(np.float32)


def camera_root(camera: int, rig: int = RIG) -> str:
    return f"{schema.pinhole_path(rig, camera)}/{RUN_SOURCE}"


def pred_box_path(camera: int, side: hands.Side, rig: int = RIG) -> str:
    return f"{camera_root(camera, rig)}/boxes/{side}_hand"


def gt_box_path(camera: int, side: hands.Side, rig: int = RIG) -> str:
    return f"{camera_root(camera, rig)}/gt_boxes/{side}_hand"


def pred_keypoints2d_path(camera: int, side: hands.Side, rig: int = RIG) -> str:
    return f"{camera_root(camera, rig)}/keypoints/{side}"


def gt_keypoints2d_path(camera: int, side: hands.Side, rig: int = RIG) -> str:
    return f"{camera_root(camera, rig)}/gt_keypoints/{side}"


def pred_hand_path(side: hands.Side) -> str:
    return f"{schema.run_path(RUN_SOURCE)}/hands/{side}"


def pred_keypoints3d_path(side: hands.Side) -> str:
    return f"{pred_hand_path(side)}/keypoints"


def pred_mesh_path(side: hands.Side) -> str:
    return f"{pred_hand_path(side)}/mesh"


def error_path(side: hands.Side) -> str:
    return f"{SERIES_ROOT}/error_mm/{side}"


def presence_path(camera: int, side: hands.Side) -> str:
    return f"{SERIES_ROOT}/presence/cam_{camera:02}/{side}"


def tracked_path(side: hands.Side) -> str:
    return f"{SERIES_ROOT}/tracked/{side}"


def round_robin_presence_path(side: hands.Side) -> str:
    return f"{SERIES_ROOT}/detnet_presence/{side}"


def fit_energy_path(side: hands.Side) -> str:
    return f"{SERIES_ROOT}/fit_energy/{side}"


DETNET_CAMERA_PATH: str = f"{SERIES_ROOT}/detnet_camera"


def detnet_box_path(camera: int, side: hands.Side, rig: int = RIG) -> str:
    return schema.boxes_path(rig, camera, f"{side}_hand")


def detnet_presence_path(camera: int, side: hands.Side) -> str:
    return f"{DETNET_SERIES_ROOT}/presence/cam_{camera:02}/{side}"


GT_CLASS_OFFSET: int = 2
"""Class id of the ground-truth left hand; the right hand is the next one."""


def hand_annotation_context() -> rr.AnnotationContext:
    """Classes 0/1 = predicted left/right hand, 2/3 = ground-truth left/right hand, each with the 21 UmeTrack landmarks and their skeleton."""
    classes: list[tuple[int, str, tuple[int, int, int]]] = [
        *((index, f"{side} hand", PRED_COLORS[index]) for index, side in enumerate(SIDES)),
        *((GT_CLASS_OFFSET + index, f"{side} hand (ground truth)", GT_BOX_COLOR) for index, side in enumerate(SIDES)),
    ]
    return rr.AnnotationContext(
        [
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=class_id, label=label, color=color),
                keypoint_annotations=[rr.AnnotationInfo(id=int(landmark), label=landmark.name.lower()) for landmark in LANDMARK],
                keypoint_connections=sorted(UME_HAND_CONNECTIONS),
            )
            for class_id, label, color in classes
        ]
    )


@dataclass(frozen=True, slots=True)
class _Clock:
    """The two index columns every column of one layer is sent on."""

    video_time_ns: Int64[ndarray, "f"]
    frame_index: Int64[ndarray, "f"]

    def indexes(self) -> list[rr.TimeColumn]:
        return [time_column(self.video_time_ns), frame_index_column(self.frame_index)]


def _send_scalars(
    recording: rr.RecordingStream,
    path: str,
    clock: _Clock,
    values: Float32[ndarray, "f"] | Float64[ndarray, "f"],
    *,
    name: str,
    color: tuple[int, int, int],
    width: float = 1.5,
    interpolation: rr.components.InterpolationMode = rr.components.InterpolationMode.Linear,
) -> None:
    """One series on every frame; NaN where there is no value (a gap in the line)."""
    rr.log(path, rr.SeriesLines(names=name, colors=color, widths=width, interpolation_mode=interpolation), static=True, recording=recording)
    rr.send_columns(path, indexes=clock.indexes(), columns=rr.Scalars.columns(scalars=values.astype(np.float64)), recording=recording)


def _send_boxes(
    recording: rr.RecordingStream,
    path: str,
    clock: _Clock,
    boxes: Float32[ndarray, "f 4"],
    keep: Bool[ndarray, "f"],
    *,
    colors: UInt8[ndarray, "f 3"],
    labels: list[str] | None,
) -> None:
    """One box row per frame; a frame without a box writes an empty row, so the previous box never lingers."""
    kept: Float32[ndarray, "k 4"] = boxes[keep]
    rr.send_columns(
        path,
        indexes=clock.indexes(),
        columns=rr.Boxes2D.columns(
            centers=(kept[:, :2] + kept[:, 2:]) / 2,
            half_sizes=(kept[:, 2:] - kept[:, :2]) / 2,
            colors=colors[keep],
            labels=None if labels is None else [label for label, ok in zip(labels, keep, strict=True) if ok],
        ).partition(keep.astype(np.int64).tolist()),
        recording=recording,
    )


def _send_keypoints(recording: rr.RecordingStream, path: str, clock: _Clock, points: Float32[ndarray, "f 21 d"], *, class_id: int, radius: float) -> None:
    """Finite keypoints with their original temporal IDs; an empty row when none remain.

    ``radius`` is in scene units, or UI points when negative (``rr.components.Radius``).
    """
    keep: Bool[ndarray, "f 21"] = np.isfinite(points).all(axis=-1)
    archetype: type[rr.Points2D] | type[rr.Points3D] = rr.Points2D if points.shape[-1] == 2 else rr.Points3D
    rr.log(path, archetype.from_fields(class_ids=class_id, radii=radius, show_labels=False), static=True, recording=recording)
    rr.send_columns(
        path, indexes=clock.indexes(), columns=archetype.columns(positions=points[keep], keypoint_ids=np.broadcast_to(np.arange(21), keep.shape)[keep]).partition(keep.sum(axis=-1).tolist()), recording=recording
    )


def write_detnet_layer(recording: rr.RecordingStream, detections: SegmentTrack, truth: GroundTruth) -> None:
    """``detnet_v1``: DetNet's box per camera and hand (labelled with its presence) and the presence series.

    ``detections`` is the DetNet-alone record (``meta.kind == "detnet_alone"``): ``presence`` is DetNet's per camera
    and hand, ``box`` and ``box_source`` = DetNet where it reported a hand. A box below the presence threshold is grey.
    ``truth`` supplies the full base segment clock; a partial run clears at its next frame on both timelines.
    """
    rig: int = 1 if is_show3d(detections.meta.dataset) else 0
    detections = with_clearing_frame(detections, truth)
    clock: _Clock = _Clock(detections.video_time_ns, detections.frame_index)
    if detections.meta.kind != "detnet_alone":
        raise ValueError(f"{detections.meta.segment}: detnet_v1 needs a DetNet-alone record, got kind={detections.meta.kind!r}")
    recording.send_property("detnet", rr.AnyValues(detnet_sha256=detections.meta.detnet_sha256, detector=detections.meta.detector))
    for camera in range(len(truth.rig.names)):
        for index, side in enumerate(SIDES):
            presence: Float32[ndarray, "f"] = detections.presence[:, camera, index]
            keep: Bool[ndarray, "f"] = (detections.box_source[:, camera, index] == BoxSource.DETNET) & np.isfinite(detections.box[:, camera, index]).all(axis=-1)
            colors: UInt8[ndarray, "f 3"] = np.where(
                (presence >= PRESENCE_THRESHOLD)[:, None], np.array(DETNET_LAYER_COLOR, dtype=np.uint8), np.array(DETNET_BELOW_THRESHOLD_COLOR, dtype=np.uint8)
            ).astype(np.uint8)
            labels: list[str] = [f"{side} {value:.2f}" for value in presence.tolist()]
            _send_boxes(recording, detnet_box_path(camera, side, rig), clock, detections.box[:, camera, index], keep, colors=colors, labels=labels)
            _send_scalars(recording, detnet_presence_path(camera, side), clock, presence, name=f"cam_{camera:02} {side}", color=PRED_COLORS[index])


def with_clearing_frame(track: SegmentTrack, truth: GroundTruth) -> SegmentTrack:
    """The track plus one untracked frame at the segment's next frame after the run, so no overlay outlives a partial run."""
    later: Int64[ndarray, "k"] = np.flatnonzero(truth.video_time_ns > track.video_time_ns[-1])
    if not len(later):
        return track
    after: int = int(later[0])

    def extended(values: Shaped[ndarray, "f *rest"], fill: float | int | bool) -> Shaped[ndarray, "g *rest"]:
        return np.concatenate([values, np.full((1, *values.shape[1:]), fill, dtype=values.dtype)])

    return replace(
        track,
        video_time_ns=np.append(track.video_time_ns, truth.video_time_ns[after]),
        frame_index=np.append(track.frame_index, truth.frame_index[after]),
        tracked=extended(track.tracked, False),
        rotation=np.concatenate([track.rotation, np.tile(np.eye(3, dtype=np.float32), (1, 2, 1, 1))]),
        translation=extended(track.translation, 0.0),
        joint_angles=extended(track.joint_angles, 0.0),
        landmarks=extended(track.landmarks, np.nan),
        box=extended(track.box, np.nan),
        box_source=extended(track.box_source, BoxSource.NONE),
        keypoints_2d=extended(track.keypoints_2d, np.nan),
        presence=extended(track.presence, np.nan),
        detnet_camera=extended(track.detnet_camera, -1),
        detnet_presence=extended(track.detnet_presence, np.nan),
        fit_energy=extended(track.fit_energy, np.nan),
    )


def write_handtrack_layer(recording: rr.RecordingStream, track: SegmentTrack, truth: GroundTruth, model: HandModelNumpy) -> None:
    """``handtrack_v1``: per camera the boxes (predicted by source, and ground truth), KeyNet's and the ground truth's keypoints,
    and KeyNet's presence; per hand the fitted landmarks, the skinned mesh, the track state and the 3D error against the ground truth.

    The ground-truth overlays cover every frame of the segment; the predictions cover the run's frames and clear on the
    next one (``with_clearing_frame``); the series cover the run's frames.

    Args:
        recording: The layer's stream (the segment's recording id, no recording properties).
        track: The tracker's output.
        truth: The segment's ground truth on all its frames.
        model: The hand model the tracker fitted (``prediction_model``); it skins the mesh.

    Raises:
        ValueError: If ``model`` does not reproduce the track's landmarks (``LANDMARK_TOLERANCE_M``).
    """
    rig: int = 1 if is_show3d(track.meta.dataset) else 0
    skinned: Float32[ndarray, "f 2 21 3"] = skinned_landmarks(
        model, np.nan_to_num(track.rotation), np.nan_to_num(track.translation), np.nan_to_num(track.joint_angles), track.tracked
    )
    mismatch: Float32[ndarray, "k"] = np.linalg.norm(skinned - track.landmarks, axis=-1)[track.tracked].reshape(-1)
    if len(mismatch) and not float(np.max(mismatch)) <= LANDMARK_TOLERANCE_M:
        raise ValueError(f"{track.meta.segment}: the mesh's hand model misses the track's landmarks by up to {float(np.max(mismatch)) * 1000:.1f} mm")
    recording.send_property(
        RUN_SOURCE,
        rr.AnyValues(
            detnet_sha256=track.meta.detnet_sha256,
            keynet_sha256=track.meta.keynet_sha256,
            hand_mode=track.meta.hand_mode,
            hand_scale=track.meta.hand_scale,
            detector=track.meta.detector,
            keypoints=track.meta.keypoints,
        ),
    )
    shown: SegmentTrack = with_clearing_frame(track, truth)
    clock: _Clock = _Clock(track.video_time_ns, track.frame_index)
    shown_clock: _Clock = _Clock(shown.video_time_ns, shown.frame_index)
    truth_clock: _Clock = _Clock(truth.video_time_ns, truth.frame_index)
    gt: Float32[ndarray, "g 2 21 3"] = gt_landmarks(truth)
    gt_pixels: Float32[ndarray, "g c 2 21 2"] = camera_pixels(truth.rig, truth.world_from_rig, gt)
    gt_boxes: Float32[ndarray, "g c 2 4"] = enclosing_squares(gt_pixels, np.isfinite(gt_pixels).all(-1))
    error: Float32[ndarray, "f 2"] = keypoint_error_mm(track.landmarks, gt_landmarks(truth.on_frames(track.video_time_ns)), track.tracked)
    palette: UInt8[ndarray, "3 3"] = np.array([(0, 0, 0), SOURCE_COLORS[BoxSource.DETNET], SOURCE_COLORS[BoxSource.TRACKED]], dtype=np.uint8)
    context: rr.AnnotationContext = hand_annotation_context()
    for camera in range(len(truth.rig.names)):
        rr.log(camera_root(camera, rig), context, static=True, recording=recording)
        for index, side in enumerate(SIDES):
            source: Int64[ndarray, "s"] = shown.box_source[:, camera, index].astype(np.int64)
            keep: Bool[ndarray, "s"] = (source != BoxSource.NONE) & np.isfinite(shown.box[:, camera, index]).all(axis=-1)
            shown_presence: Float32[ndarray, "s"] = shown.presence[:, camera, index]
            labels: list[str] = [
                f"{side} {SOURCE_NAMES.get(kind, '')}" + ("" if np.isnan(value) else f" {value:.2f}") for kind, value in zip(source.tolist(), shown_presence.tolist(), strict=True)
            ]
            _send_boxes(recording, pred_box_path(camera, side, rig), shown_clock, shown.box[:, camera, index], keep, colors=palette[source], labels=labels)
            gt_keep: Bool[ndarray, "g"] = np.isfinite(gt_boxes[:, camera, index]).all(axis=-1)
            gt_colors: UInt8[ndarray, "g 3"] = np.tile(np.array(GT_BOX_COLOR, dtype=np.uint8), (len(gt_keep), 1))
            _send_boxes(recording, gt_box_path(camera, side, rig), truth_clock, gt_boxes[:, camera, index], gt_keep, colors=gt_colors, labels=None)
            _send_keypoints(recording, pred_keypoints2d_path(camera, side, rig), shown_clock, shown.keypoints_2d[:, camera, index], class_id=index, radius=KEYPOINT_RADIUS_UI)
            _send_keypoints(
                recording, gt_keypoints2d_path(camera, side, rig), truth_clock, gt_pixels[:, camera, index], class_id=GT_CLASS_OFFSET + index, radius=KEYPOINT_RADIUS_UI
            )
            _send_scalars(recording, presence_path(camera, side), clock, track.presence[:, camera, index], name=f"cam_{camera:02} {side}", color=PRED_COLORS[index])
    rr.log(schema.run_path(RUN_SOURCE), context, static=True, recording=recording)
    for index, side in enumerate(SIDES):
        _send_keypoints(recording, pred_keypoints3d_path(side), shown_clock, shown.landmarks[:, index], class_id=index, radius=KEYPOINT_RADIUS_M)
        log_hand_meshes(
            recording,
            hands.HAND_SIDES[index],
            model,
            shown.joint_angles[:, index],
            wrists_mm(shown.rotation[:, index], shown.translation[:, index]),
            shown.tracked[:, index],
            times_ns=shown.video_time_ns,
            frame_indices=shown.frame_index,
            path=pred_mesh_path(side),
        )
        _send_scalars(recording, error_path(side), clock, error[:, index], name=side, color=PRED_COLORS[index])
        _send_scalars(recording, tracked_path(side), clock, track.tracked[:, index].astype(np.float64), name=f"{side} tracked", color=PRED_COLORS[index], width=3.0, interpolation=rr.components.InterpolationMode.StepAfter)
        _send_scalars(recording, round_robin_presence_path(side), clock, track.detnet_presence[:, index], name=f"{side} DetNet presence", color=PRED_COLORS[index])
        _send_scalars(recording, fit_energy_path(side), clock, track.fit_energy[:, index], name=side, color=PRED_COLORS[index])
    _send_scalars(
        recording, DETNET_CAMERA_PATH, clock, np.where(track.detnet_camera >= 0, track.detnet_camera, np.nan).astype(np.float64), name="DetNet camera", color=(255, 60, 255)
    )
