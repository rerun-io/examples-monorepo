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
from posekit.models.detnet import DetNetHandDetectorConfig
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig
from posekit.models.egohand_fast import EgoHandFastConfig, fast_egohand_configs
from posekit.models.rtdetr import RtDetrDetectorConfig
from posekit.models.rtmpose import RtmPoseConfig
from posekit.models.sam3_segmenter import Sam3SegmenterConfig
from posekit.models.sam3d_body import Sam3dBodyConfig
from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
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
    "DetNetHandDetectorConfig",
    "EgoHandDetectorConfig",
    "EgoHandFastConfig",
    "HAND_DETECTORS",
    "HAND_POSES",
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
    "Sam3dBodyFastConfig",
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

_fast_hand_detector, _fast_hand_pose = fast_egohand_configs()
_bf16_hand_detector, _bf16_hand_pose = fast_egohand_configs(backbone_precision="bf16")
HAND_DETECTORS: dict[str, WilorHandDetectorConfig | EgoHandDetectorConfig] = {
    "wilor": WilorHandDetectorConfig(), "egohand": EgoHandDetectorConfig(), "egohand-fast": _fast_hand_detector, "egohand-fast-bf16": _bf16_hand_detector,
}
"""Hand detector of every named hand stack; the CLI subcommands and the app's stack choices."""
HAND_POSES: dict[str, WilorHandPoseConfig | EgoHandConfig | EgoHandFastConfig] = {
    "wilor": WilorHandPoseConfig(), "egohand": EgoHandConfig(), "egohand-fast": _fast_hand_pose, "egohand-fast-bf16": _bf16_hand_pose,
}
"""3D hand model of every named hand stack; same names as ``HAND_DETECTORS``."""

if TYPE_CHECKING:
    DetectorConfig = YoloxDetectorConfig | RtDetrDetectorConfig
    Pose2dConfig = RtmPoseConfig | SapiensPoseConfig | VitPoseConfig
    HandDetectorConfig = WilorHandDetectorConfig | EgoHandDetectorConfig | DetNetHandDetectorConfig
    HandPose3dConfig = WilorHandPoseConfig | EgoHandConfig | EgoHandFastConfig
    BodyPose3dConfig = Sam3dBodyConfig | Sam3dBodyFastConfig
else:
    DetectorConfig = tyro.extras.subcommand_type_from_defaults(
        {"yolox": YoloxDetectorConfig(), "rtdetr": RtDetrDetectorConfig()}, prefix_names=False
    )
    Pose2dConfig = tyro.extras.subcommand_type_from_defaults(
        {"rtmpose": RtmPoseConfig(), "sapiens": SapiensPoseConfig(), "vitpose": VitPoseConfig()}, prefix_names=False
    )

    # DetNet-F is a detector only (no pose stage), so it is a detector subcommand and not a named hand stack.
    HandDetectorConfig = tyro.extras.subcommand_type_from_defaults({**HAND_DETECTORS, "detnet": DetNetHandDetectorConfig()}, prefix_names=False)
    HandPose3dConfig = tyro.extras.subcommand_type_from_defaults(HAND_POSES, prefix_names=False)
    BodyPose3dConfig = tyro.extras.subcommand_type_from_defaults(
        {
            "sam3d-body": Sam3dBodyConfig(mode="full"), "sam3d-body-bodyonly": Sam3dBodyConfig(mode="body"),
            "sam3d-body-fast": Sam3dBodyFastConfig(mode="full"), "sam3d-body-fast-bodyonly": Sam3dBodyFastConfig(mode="body"),
            "sam3d-body-fast-bf16": Sam3dBodyFastConfig(backbone_precision="bf16"),
            "sam3d-body-fast-torch": Sam3dBodyFastConfig(backbone_onnx=None, compile_torch_backbone=True),
            "sam3d-body-fast-nvfp4": Sam3dBodyFastConfig(backbone_precision="nvfp4"),
        }, prefix_names=False
    )

AnnotatedDetectorConfig = tyro.conf.OmitSubcommandPrefixes[DetectorConfig]
AnnotatedHandDetectorConfig = tyro.conf.OmitSubcommandPrefixes[HandDetectorConfig]
AnnotatedHandPose3dConfig = tyro.conf.OmitSubcommandPrefixes[HandPose3dConfig]
AnnotatedPose2dConfig = tyro.conf.OmitSubcommandPrefixes[Pose2dConfig]
AnnotatedBodyPose3dConfig = tyro.conf.OmitSubcommandPrefixes[BodyPose3dConfig]
