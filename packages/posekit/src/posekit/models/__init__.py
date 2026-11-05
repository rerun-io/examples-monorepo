"""Model registry: tyro subcommand unions over detector and pose configs.

Follows the simplecv exoego pattern — a dict of defaults becomes a CLI union,
so ``--pose rtmpose --pose.backend tensorrt`` style selection works everywhere
without per-tool boilerplate.
"""

from typing import TYPE_CHECKING

import tyro

from posekit.models.base import (
    BodyPose3dPipeline,
    HandDetector,
    HandPose3dPipeline,
    IdentityEncoder,
    InstancePose2d,
    PersonDetector,
    Pose2dPipeline,
    PromptableSegmenter,
    SegmentationPrompts,
    TopDownBodyPose3d,
    TopDownDenseLandmarks2d,
    TopDownHandPose3d,
    TopDownPose2d,
    VideoSegmenter,
)
from posekit.models.clip_identity import ClipIdentityConfig
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig
from posekit.models.egohand_fast import EgoHandFastConfig, fast_egohand_configs
from posekit.models.rtdetr import RtDetrDetectorConfig
from posekit.models.rtmpose import RtmPoseConfig
from posekit.models.sam3_segmenter import Sam3SegmenterConfig
from posekit.models.sam3d_body import Sam3dBodyConfig
from posekit.models.sapiens import SapiensPoseConfig
from posekit.models.vitpose import VitPoseConfig
from posekit.models.wilor import WilorHandDetectorConfig, WilorHandPoseConfig
from posekit.models.yolox import YoloxDetectorConfig

__all__ = (
    "AnnotatedBodyPose3dConfig",
    "AnnotatedDetectorConfig",
    "AnnotatedHandDetectorConfig",
    "AnnotatedHandPose3dConfig",
    "AnnotatedPose2dConfig",
    "BodyPose3dConfig",
    "BodyPose3dPipeline",
    "ClipIdentityConfig",
    "DetectorConfig",
    "EgoHandConfig",
    "EgoHandDetectorConfig",
    "EgoHandFastConfig",
    "HandDetector",
    "HandDetectorConfig",
    "HandPose3dConfig",
    "HandPose3dPipeline",
    "IdentityEncoder",
    "InstancePose2d",
    "PersonDetector",
    "Pose2dConfig",
    "Pose2dPipeline",
    "PromptableSegmenter",
    "RtDetrDetectorConfig",
    "RtmPoseConfig",
    "Sam3SegmenterConfig",
    "Sam3dBodyConfig",
    "SapiensPoseConfig",
    "SegmentationPrompts",
    "TopDownBodyPose3d",
    "TopDownDenseLandmarks2d",
    "TopDownHandPose3d",
    "TopDownPose2d",
    "VideoSegmenter",
    "VitPoseConfig",
    "WilorHandDetectorConfig",
    "WilorHandPoseConfig",
    "YoloxDetectorConfig",
)

if TYPE_CHECKING:
    DetectorConfig = YoloxDetectorConfig | RtDetrDetectorConfig
    Pose2dConfig = RtmPoseConfig | SapiensPoseConfig | VitPoseConfig
    HandDetectorConfig = WilorHandDetectorConfig | EgoHandDetectorConfig
    HandPose3dConfig = WilorHandPoseConfig | EgoHandConfig | EgoHandFastConfig
    BodyPose3dConfig = Sam3dBodyConfig
else:
    DetectorConfig = tyro.extras.subcommand_type_from_defaults(
        {"yolox": YoloxDetectorConfig(), "rtdetr": RtDetrDetectorConfig()}, prefix_names=False
    )
    Pose2dConfig = tyro.extras.subcommand_type_from_defaults(
        {"rtmpose": RtmPoseConfig(), "sapiens": SapiensPoseConfig(), "vitpose": VitPoseConfig()}, prefix_names=False
    )

    _fast_hand_detector, _fast_hand_pose = fast_egohand_configs()
    HandDetectorConfig = tyro.extras.subcommand_type_from_defaults(
        {"wilor": WilorHandDetectorConfig(), "egohand": EgoHandDetectorConfig(), "egohand-fast": _fast_hand_detector}, prefix_names=False
    )
    HandPose3dConfig = tyro.extras.subcommand_type_from_defaults(
        {"wilor": WilorHandPoseConfig(), "egohand": EgoHandConfig(), "egohand-fast": _fast_hand_pose}, prefix_names=False
    )
    BodyPose3dConfig = tyro.extras.subcommand_type_from_defaults(
        {"sam3d-body": Sam3dBodyConfig(mode="full"), "sam3d-body-bodyonly": Sam3dBodyConfig(mode="body")}, prefix_names=False
    )

AnnotatedDetectorConfig = tyro.conf.OmitSubcommandPrefixes[DetectorConfig]
AnnotatedHandDetectorConfig = tyro.conf.OmitSubcommandPrefixes[HandDetectorConfig]
AnnotatedHandPose3dConfig = tyro.conf.OmitSubcommandPrefixes[HandPose3dConfig]
AnnotatedPose2dConfig = tyro.conf.OmitSubcommandPrefixes[Pose2dConfig]
AnnotatedBodyPose3dConfig = tyro.conf.OmitSubcommandPrefixes[BodyPose3dConfig]
