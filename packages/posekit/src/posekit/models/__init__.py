"""Model registry: tyro subcommand unions over detector and pose configs.

Follows the simplecv exoego pattern — a dict of defaults becomes a CLI union,
so ``--pose rtmpose --pose.backend tensorrt`` style selection works everywhere
without per-tool boilerplate.
"""

from typing import TYPE_CHECKING

import tyro

from posekit.models.base import (
    IdentityEncoder,
    InstancePose2d,
    PersonDetector,
    Pose2dPipeline,
    PromptableSegmenter,
    SegmentationPrompts,
    TopDownDenseLandmarks2d,
    TopDownPose2d,
    VideoSegmenter,
)
from posekit.models.clip_identity import ClipIdentityConfig
from posekit.models.rtdetr import RtDetrDetectorConfig
from posekit.models.rtmpose import RtmPoseConfig
from posekit.models.sam3_segmenter import Sam3SegmenterConfig
from posekit.models.sapiens import SapiensPoseConfig
from posekit.models.vitpose import VitPoseConfig
from posekit.models.yolox import YoloxDetectorConfig

__all__ = (
    "AnnotatedDetectorConfig",
    "AnnotatedPose2dConfig",
    "ClipIdentityConfig",
    "DetectorConfig",
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
    "TopDownPose2d",
    "VideoSegmenter",
    "VitPoseConfig",
    "YoloxDetectorConfig",
)

if TYPE_CHECKING:
    DetectorConfig = YoloxDetectorConfig | RtDetrDetectorConfig
    Pose2dConfig = RtmPoseConfig | SapiensPoseConfig | VitPoseConfig
else:
    DetectorConfig = tyro.extras.subcommand_type_from_defaults(
        {"yolox": YoloxDetectorConfig(), "rtdetr": RtDetrDetectorConfig()}, prefix_names=False
    )
    Pose2dConfig = tyro.extras.subcommand_type_from_defaults(
        {"rtmpose": RtmPoseConfig(), "sapiens": SapiensPoseConfig(), "vitpose": VitPoseConfig()}, prefix_names=False
    )

AnnotatedDetectorConfig = tyro.conf.OmitSubcommandPrefixes[DetectorConfig]
AnnotatedPose2dConfig = tyro.conf.OmitSubcommandPrefixes[Pose2dConfig]
