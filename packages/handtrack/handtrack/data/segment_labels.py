"""Labels for a whole segment at once, vectorised over its frames: FK landmarks, projections into every camera and the
net frame, visibility counts, enclosing circles, the validity rules and the per-hand ``HandLabel``; plus the keypoint
priors KeyNet's input is built from (the pose extrapolated from the two previous tracker steps, and the pose 20 tracker
steps earlier).

Everything here runs on the device of its inputs; the stream computes it on CPU on the producer thread.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from torch import Tensor

from handtrack.data.catalog import CatalogDataError, HandTimeline
from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.geometry.letterbox import Letterbox
from handtrack.hand.pose import HandPose, Side, extrapolate, landmarks
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.validity import SHOW3D_CONFIDENCE_THRESHOLD, HandLabel, classify_visibility, show3d_hands, umetrack_hands

STALE_TRACKER_STEPS: int = 20
"""KeyNet's alternative keypoint input: the ground-truth pose this many tracker steps earlier (the paper's 20 frames)."""


@dataclass(frozen=True, slots=True)
class HandProjection:
    """Both hands' 21 keypoints in every camera of k frames (c cameras, slot 0 = left)."""

    points_cam: Float32[Tensor, "k c 2 21 3"]
    """Camera-space metres; NaN where the hand has no pose or the headset none."""
    pixels: Float32[Tensor, "k c 2 21 2"]
    """Unclipped pixel coordinates in the camera's own image."""
    net_xy: Float32[Tensor, "k c 2 21 2"]
    """The same pixels in the 640x480 net frame (letterboxed)."""
    in_front: Bool[Tensor, "k c 2 21"]
    visible: Int64[Tensor, "k c 2"]
    """Keypoints in front of the camera and inside its image [0, W) x [0, H)."""


def project_hands(rig: CameraRig, letterboxes: Sequence[Letterbox], world_from_rig: Float32[Tensor, "k 4 4"], points_world: Float32[Tensor, "k 2 21 3"]) -> HandProjection:
    """World landmarks of both hands into every camera, its image and the net frame."""
    frames: int = points_world.shape[0]
    flat_world: Float32[Tensor, "k 42 3"] = points_world.reshape(frames, 42, 3)
    points_cam: Float32[Tensor, "k c 42 3"] = world_to_cameras(rig, world_from_rig, flat_world)
    pixels: Float32[Tensor, "k c 42 2"] = project(rig, points_cam)
    net_xy: Float32[Tensor, "k c 42 2"] = torch.stack([letterbox.to_net(pixels[:, camera]) for camera, letterbox in enumerate(letterboxes)], dim=1)
    front: Bool[Tensor, "k c 42"] = in_front(points_cam)
    inside: Bool[Tensor, "k c 42"] = front & inside_image(rig, pixels)
    cameras: int = rig.image_size.shape[0]
    return HandProjection(
        points_cam=points_cam.reshape(frames, cameras, 2, 21, 3),
        pixels=pixels.reshape(frames, cameras, 2, 21, 2),
        net_xy=net_xy.reshape(frames, cameras, 2, 21, 2),
        in_front=front.reshape(frames, cameras, 2, 21),
        visible=inside.reshape(frames, cameras, 2, 21).sum(dim=-1),
    )


def select_pose(pose: HandPose, rows: Int64[Tensor, "k"]) -> HandPose:
    return HandPose(rotation=pose.rotation[rows], translation=pose.translation[rows], joint_angles=pose.joint_angles[rows])


def hand_landmarks(timeline: HandTimeline, poses: tuple[HandPose, HandPose], valid: Bool[Tensor, "k 2"]) -> Float32[Tensor, "k 2 21 3"]:
    """Skinned world landmarks of both hands; NaN where ``valid`` is False (the skinning never sees a NaN pose)."""
    points: list[Float32[Tensor, "k 21 3"]] = []
    for side in Side:
        pose: HandPose = poses[side]
        mask: Bool[Tensor, "k"] = valid[:, side]
        safe: HandPose = HandPose(
            rotation=torch.where(mask[:, None, None], pose.rotation, torch.eye(3, dtype=pose.rotation.dtype, device=pose.rotation.device).expand_as(pose.rotation)),
            translation=torch.where(mask[:, None], pose.translation, torch.zeros_like(pose.translation)),
            joint_angles=torch.where(mask[:, None], pose.joint_angles, torch.zeros_like(pose.joint_angles)),
        )
        skinned: Float32[Tensor, "k 21 3"] = landmarks(timeline.hand_model, safe, side)
        points.append(torch.where(mask[:, None, None], skinned, torch.full_like(skinned, torch.nan)))
    return torch.stack(points, dim=1)


@dataclass(frozen=True, slots=True)
class SegmentLabels:
    """Per kept frame k, camera c and hand slot h: the ground truth DetNet and KeyNet train on."""

    rows: Int64[ndarray, "k"]
    """Timeline rows these labels describe."""
    landmarks: Float32[Tensor, "k 2 21 3"]
    """World metres; NaN where the hand has no pose."""
    projection: HandProjection
    circles: Float32[Tensor, "k c 2 3"]
    """Smallest enclosing circle (cx, cy, r) of the finite in-front keypoints in the net frame; NaN where there are none."""
    image_valid: Bool[Tensor, "k c"]
    """The validity rules and the dataset's per-camera quality flags: False drops the whole image."""
    labelled: Bool[Tensor, "k c 2"]
    """The hand carries a usable label (UmeTrack confidence 1; SHOW3D and HOT3D confidence > 0.1 with a pose)."""
    hand_label: Int64[Tensor, "k c 2"]
    """``HandLabel`` in each image: PRESENT (>= 17 visible), PARTIAL (1-16) for labelled hands, ABSENT otherwise."""


def segment_labels(timeline: HandTimeline, rig: CameraRig, letterboxes: Sequence[Letterbox], rows: Int64[ndarray, "k"], pose_gated: bool) -> SegmentLabels:
    """Labels for the given timeline rows (all at once).

    ``pose_gated`` is ``DatasetLayout.pose_gated``: SHOW3D's rule (SHOW3D, HOT3D) when True, UmeTrack's binary confidence when False.
    """
    index: Int64[Tensor, "k"] = torch.from_numpy(rows)
    poses: tuple[HandPose, HandPose] = (select_pose(timeline.poses[0], index), select_pose(timeline.poses[1], index))
    has_pose: Bool[Tensor, "k 2"] = timeline.has_pose[index]
    points_world: Float32[Tensor, "k 2 21 3"] = hand_landmarks(timeline, poses, has_pose)
    projection: HandProjection = project_hands(rig, letterboxes, timeline.world_from_rig[index], points_world)
    cameras: int = len(letterboxes)
    confidence: Float32[Tensor, "k 2"] = timeline.confidence[index]
    headset_valid: Bool[Tensor, "k"] = timeline.headset_valid[index]
    image_valid: Bool[Tensor, "k c"]
    labelled: Bool[Tensor, "k c 2"]
    if pose_gated:
        validity: tuple[Bool[Tensor, "k c"], Bool[Tensor, "k c 2"]] = show3d_hands(confidence, has_pose, projection.visible, headset_valid)
        image_valid, labelled = validity[0], validity[1]
    else:
        confidence_available: Bool[Tensor, "k 2"] = torch.isfinite(confidence)
        try:
            frame_rules: tuple[Bool[Tensor, "k"], Bool[Tensor, "k 2"]] = umetrack_hands(
                torch.where(confidence_available, confidence, 0.0), headset_valid & confidence_available.all(dim=-1)
            )
        except ValueError as error:
            raise CatalogDataError(str(error)) from error
        # A labelled hand must have a pose; if the stored pose is missing the frame cannot be labelled.
        frame_valid: Bool[Tensor, "k"] = frame_rules[0] & (has_pose | ~frame_rules[1]).all(dim=-1)
        image_valid = frame_valid[:, None].expand(-1, cameras).clone()
        labelled = frame_rules[1][:, None, :].expand(-1, cameras, -1).clone()
    if timeline.camera_valid is not None:
        image_valid = image_valid & timeline.camera_valid[index]
    hand_label: Int64[Tensor, "k c 2"] = torch.where(labelled, classify_visibility(projection.visible), torch.full_like(projection.visible, int(HandLabel.ABSENT)))
    circles: Float32[ndarray, "k c 2 3"] = enclosing_circles(projection.net_xy.numpy(), projection.in_front.numpy())
    return SegmentLabels(
        rows=rows,
        landmarks=points_world,
        projection=projection,
        circles=torch.from_numpy(circles),
        image_valid=image_valid,
        labelled=labelled,
        hand_label=hand_label,
    )


@dataclass(frozen=True, slots=True)
class KeypointPriors:
    """What a tracker would know before frame t, projected with frame t's headset pose (KeyNet's keypoint input)."""

    extrapolated: HandProjection
    """θ̂ = 2θ(t−s) − θ(t−2s) (s = one tracker step), or θ(t−s) as it is when θ(t−2s) is missing."""
    extrapolated_valid: Bool[Tensor, "k 2"]
    stale: HandProjection
    """θ(t − 20 s): the paper's 10% alternative input."""
    stale_valid: Bool[Tensor, "k 2"]


def trusted_poses(timeline: HandTimeline) -> Bool[Tensor, "f 2"]:
    """Poses a tracker history may come from: a pose with confidence above SHOW3D's 0.1 (UmeTrack confidence is 0 or 1)."""
    return timeline.has_pose & (timeline.confidence > SHOW3D_CONFIDENCE_THRESHOLD)


def keypoint_priors(timeline: HandTimeline, rig: CameraRig, letterboxes: Sequence[Letterbox], rows: Int64[ndarray, "k"], tracker_step: int) -> KeypointPriors:
    """The extrapolated and the stale ground-truth priors for the given rows."""
    trusted: Bool[Tensor, "f 2"] = trusted_poses(timeline)
    frames: int = trusted.shape[0]
    index: Int64[Tensor, "k"] = torch.from_numpy(rows)

    def lookup(offset: int) -> tuple[Int64[Tensor, "k"], Bool[Tensor, "k 2"]]:
        source: Int64[Tensor, "k"] = index - offset
        inside: Bool[Tensor, "k"] = (source >= 0) & (source < frames)
        clamped: Int64[Tensor, "k"] = source.clamp(0, frames - 1)
        return clamped, trusted[clamped] & inside[:, None]

    previous_rows, previous_valid = lookup(tracker_step)
    before_rows, before_valid = lookup(2 * tracker_step)
    stale_rows, stale_valid = lookup(STALE_TRACKER_STEPS * tracker_step)
    extrapolated_poses: list[HandPose] = []
    stale_poses: list[HandPose] = []
    for side in Side:
        previous: HandPose = select_pose(timeline.poses[side], previous_rows)
        before: HandPose = select_pose(timeline.poses[side], before_rows)
        guess: HandPose = extrapolate(previous, before)
        both: Bool[Tensor, "k"] = before_valid[:, side]
        extrapolated_poses.append(
            HandPose(
                rotation=torch.where(both[:, None, None], guess.rotation, previous.rotation),
                translation=torch.where(both[:, None], guess.translation, previous.translation),
                joint_angles=torch.where(both[:, None], guess.joint_angles, previous.joint_angles),
            )
        )
        stale_poses.append(select_pose(timeline.poses[side], stale_rows))
    world_from_rig: Float32[Tensor, "k 4 4"] = timeline.world_from_rig[index]
    extrapolated_world: Float32[Tensor, "k 2 21 3"] = hand_landmarks(timeline, (extrapolated_poses[0], extrapolated_poses[1]), previous_valid)
    stale_world: Float32[Tensor, "k 2 21 3"] = hand_landmarks(timeline, (stale_poses[0], stale_poses[1]), stale_valid)
    return KeypointPriors(
        extrapolated=project_hands(rig, letterboxes, world_from_rig, extrapolated_world),
        extrapolated_valid=previous_valid & timeline.headset_valid[index][:, None],
        stale=project_hands(rig, letterboxes, world_from_rig, stale_world),
        stale_valid=stale_valid & timeline.headset_valid[index][:, None],
    )
