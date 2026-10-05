from dataclasses import replace

import numpy as np
import torch

from handtrack.data.catalog import HandTimeline
from handtrack.data.segment_labels import SegmentLabels, keypoint_priors, segment_labels
from handtrack.geometry.camera import CameraRig
from handtrack.geometry.letterbox import letterbox_for
from handtrack.hand.pose import HandPose, generic_hand_model
from handtrack.labels.validity import HandLabel

FRAMES: int = 6


def _rig() -> CameraRig:
    return CameraRig(
        names=("/world/rig_00/cam_00",),
        image_size=torch.tensor([[640.0, 480.0]]),
        cam_from_rig=torch.eye(4)[None],
        focal=torch.tensor([[300.0, 300.0]]),
        principal=torch.tensor([[320.0, 240.0]]),
        fisheye62=None,
    )


def _timeline(left_confidence: float = 1.0, right_translation: tuple[float, float, float] | None = None) -> HandTimeline:
    """A left hand sliding along x at 1 cm per frame, 0.4 m in front of a camera at the world origin."""
    translation: torch.Tensor = torch.stack([torch.tensor([0.01 * i - 0.03, 0.0, 0.4]) for i in range(FRAMES)])
    left: HandPose = HandPose(rotation=torch.eye(3).expand(FRAMES, 3, 3).clone(), translation=translation, joint_angles=torch.zeros(FRAMES, 22))
    if right_translation is None:
        right: HandPose = HandPose(
            rotation=torch.full((FRAMES, 3, 3), torch.nan), translation=torch.full((FRAMES, 3), torch.nan), joint_angles=torch.full((FRAMES, 22), torch.nan)
        )
    else:
        right = HandPose(rotation=torch.eye(3).expand(FRAMES, 3, 3).clone(), translation=torch.tensor(right_translation).expand(FRAMES, 3).clone(), joint_angles=torch.zeros(FRAMES, 22))
    has_right: bool = right_translation is not None
    return HandTimeline(
        video_time_ns=np.arange(FRAMES, dtype=np.int64) * 33_333_333,
        world_from_rig=torch.eye(4).expand(FRAMES, 4, 4).clone(),
        headset_valid=torch.ones(FRAMES, dtype=torch.bool),
        poses=(left, right),
        confidence=torch.tensor([[left_confidence, 1.0 if has_right else 0.0]]).expand(FRAMES, 2).clone(),
        has_pose=torch.tensor([[True, has_right]]).expand(FRAMES, 2).clone(),
        hand_model=generic_hand_model(),
        hand_scale=1.0,
    )


def test_umetrack_labels_present_and_absent() -> None:
    rows: np.ndarray = np.arange(FRAMES, dtype=np.int64)
    labels: SegmentLabels = segment_labels(_timeline(), _rig(), (letterbox_for(640, 480),), rows, pose_gated=False)
    assert labels.image_valid.all()
    assert labels.labelled[..., 0].all() and not labels.labelled[..., 1].any()
    assert (labels.projection.visible[:, 0, 0] == 21).all()
    assert (labels.hand_label[:, 0, 0] == HandLabel.PRESENT).all() and (labels.hand_label[:, 0, 1] == HandLabel.ABSENT).all()
    circles: torch.Tensor = labels.circles[:, 0, 0]
    reach: torch.Tensor = (labels.projection.net_xy[:, 0, 0] - circles[:, None, :2]).norm(dim=-1)
    assert (reach <= circles[:, None, 2] + 1e-3).all()
    assert torch.isnan(labels.circles[:, 0, 1]).all()


def test_show3d_low_confidence_visible_hand_drops_the_image() -> None:
    rows: np.ndarray = np.arange(FRAMES, dtype=np.int64)
    kept: SegmentLabels = segment_labels(_timeline(left_confidence=0.5, right_translation=(0.0, 0.0, -0.4)), _rig(), (letterbox_for(640, 480),), rows, pose_gated=True)
    # The right hand is behind the camera: known absent, so the image stays.
    assert kept.image_valid.all() and (kept.hand_label[:, 0, 1] == HandLabel.ABSENT).all()
    dropped: SegmentLabels = segment_labels(_timeline(left_confidence=0.05), _rig(), (letterbox_for(640, 480),), rows, pose_gated=True)
    assert not dropped.image_valid.any()


def test_pose_gated_missing_pose_drops_the_image_where_umetrack_calls_it_absent() -> None:
    # The right hand has no pose and confidence 0: UmeTrack reads that as absent, SHOW3D and HOT3D as unlabelled.
    rows: np.ndarray = np.arange(FRAMES, dtype=np.int64)
    umetrack: SegmentLabels = segment_labels(_timeline(), _rig(), (letterbox_for(640, 480),), rows, pose_gated=False)
    gated: SegmentLabels = segment_labels(_timeline(), _rig(), (letterbox_for(640, 480),), rows, pose_gated=True)
    assert umetrack.image_valid.all() and not gated.image_valid.any()


def test_camera_quality_flags_drop_single_images() -> None:
    rows: np.ndarray = np.arange(FRAMES, dtype=np.int64)
    flags: torch.Tensor = torch.tensor([[True], [False], [True], [True], [False], [True]])
    timeline: HandTimeline = replace(_timeline(right_translation=(0.0, 0.0, -0.4)), camera_valid=flags)
    labels: SegmentLabels = segment_labels(timeline, _rig(), (letterbox_for(640, 480),), rows, pose_gated=True)
    assert labels.image_valid.equal(flags)
    # The flag removes the image, not the hand's label.
    assert labels.labelled[:, 0, 0].all() and (labels.hand_label[:, 0, 0] == HandLabel.PRESENT).all()


def test_extrapolated_prior_recovers_constant_velocity() -> None:
    timeline: HandTimeline = _timeline()
    rows: np.ndarray = np.arange(FRAMES, dtype=np.int64)
    labels: SegmentLabels = segment_labels(timeline, _rig(), (letterbox_for(640, 480),), rows, pose_gated=False)
    priors = keypoint_priors(timeline, _rig(), (letterbox_for(640, 480),), rows, tracker_step=1)
    assert priors.extrapolated_valid[:, 0].tolist() == [False] + [True] * (FRAMES - 1)
    assert not priors.extrapolated_valid[:, 1].any() and not priors.stale_valid.any()
    # Rows >= 2 extrapolate from two poses: exact for constant velocity. Row 1 reuses θ(0), 1 cm behind.
    torch.testing.assert_close(priors.extrapolated.net_xy[2:, 0, 0], labels.projection.net_xy[2:, 0, 0], atol=1e-3, rtol=0.0)
    shift: torch.Tensor = (priors.extrapolated.net_xy[1, 0, 0] - labels.projection.net_xy[1, 0, 0])[:, 0]
    assert (shift < -1.0).all()
