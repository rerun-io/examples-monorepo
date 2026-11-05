"""Model registry: tyro subcommand unions over detector and pose configs.

Follows the simplecv exoego pattern — a dict of defaults becomes a CLI union,
so ``--pose rtmpose --pose.backend tensorrt`` style selection works everywhere
without per-tool boilerplate.
"""

from typing import TYPE_CHECKING

import tyro

from posekit.models.base import (
    HandDetector,
    HandPose3dPipeline,
    IdentityEncoder,
    InstancePose2d,
    PersonDetector,
    Pose2dPipeline,
    PromptableSegmenter,
    SegmentationPrompts,
    TopDownDenseLandmarks2d,
    TopDownHandPose3d,
    TopDownPose2d,
    VideoSegmenter,
)
from posekit.models.clip_identity import ClipIdentityConfig
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig, fast_egohand_configs
from posekit.models.rtdetr import RtDetrDetectorConfig
from posekit.models.rtmpose import RtmPoseConfig
from posekit.models.sam3_segmenter import Sam3SegmenterConfig
from posekit.models.sapiens import SapiensPoseConfig
from posekit.models.vitpose import VitPoseConfig
from posekit.models.wilor import WilorHandDetectorConfig, WilorHandPoseConfig
from posekit.models.yolox import YoloxDetectorConfig

__all__ = (
    "AnnotatedDetectorConfig",
    "AnnotatedHandDetectorConfig",
    "AnnotatedHandPose3dConfig",
    "AnnotatedPose2dConfig",
    "ClipIdentityConfig",
    "DetectorConfig",
    "EgoHandConfig",
    "EgoHandDetectorConfig",
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
    "SapiensPoseConfig",
    "SegmentationPrompts",
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
    HandPose3dConfig = WilorHandPoseConfig | EgoHandConfig
else:
    DetectorConfig = tyro.extras.subcommand_type_from_defaults(
        {"yolox": YoloxDetectorConfig(), "rtdetr": RtDetrDetectorConfig()}, prefix_names=False
    )
    Pose2dConfig = tyro.extras.subcommand_type_from_defaults(
        {"rtmpose": RtmPoseConfig(), "sapiens": SapiensPoseConfig(), "vitpose": VitPoseConfig()}, prefix_names=False
    )

    HandDetectorConfig = tyro.extras.subcommand_type_from_defaults(
        {"wilor": WilorHandDetectorConfig(), "egohand": EgoHandDetectorConfig(), "egohand-fast": fast_egohand_configs()[0]}, prefix_names=False
    )
    HandPose3dConfig = tyro.extras.subcommand_type_from_defaults({"wilor": WilorHandPoseConfig(), "egohand": EgoHandConfig(), "egohand-fast": fast_egohand_configs()[1]}, prefix_names=False)

AnnotatedDetectorConfig = tyro.conf.OmitSubcommandPrefixes[DetectorConfig]
AnnotatedHandDetectorConfig = tyro.conf.OmitSubcommandPrefixes[HandDetectorConfig]
AnnotatedHandPose3dConfig = tyro.conf.OmitSubcommandPrefixes[HandPose3dConfig]
AnnotatedPose2dConfig = tyro.conf.OmitSubcommandPrefixes[Pose2dConfig]
