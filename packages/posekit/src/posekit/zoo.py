"""Typed model-zoo registry and solution presets.

This is rtmlib's zoo curation adopted *as data* (docs/design.md §4): every
entry records where the weights come from, which inference paradigm the model
uses, and whether posekit runs it today — so tools and consumers can enumerate
or validate model choices without touching model code. rtmlib's solution
classes (``Body``, ``Wholebody``, ...) become named presets pairing a detector
config with a pose config.
"""

from dataclasses import dataclass
from typing import Literal

from posekit.models import (
    DetectorConfig,
    Pose2dConfig,
    RtDetrDetectorConfig,
    RtmPoseConfig,
    SapiensPoseConfig,
    VitPoseConfig,
    YoloxDetectorConfig,
)
from posekit.models.rtmpose import RTMPOSE_ONNX_ZIP_URLS
from posekit.models.yolox import YOLOX_ONNX_ZIP_URLS

ZooRole = Literal["detector", "pose2d", "instance-pose2d", "pose3d", "segmenter", "identity"]
ZooParadigm = Literal["detector", "top-down", "one-stage", "query-based", "full-frame"]
ZooSource = Literal["openmmlab-onnx", "transformers", "local"]
ZooStatus = Literal["implemented", "planned"]


@dataclass(frozen=True, slots=True)
class ZooModel:
    """One model checkpoint the abstraction knows about."""

    key: str
    """Registry key, unique."""
    role: ZooRole
    """Pipeline role the model fills."""
    paradigm: ZooParadigm
    """Inference paradigm (docs/design.md §2)."""
    source: ZooSource
    """Where the weights come from."""
    location: str
    """ONNX zip URL, HF model id, or local checkpoint hint."""
    input_size: tuple[int, int]
    """Network input size as ``(width, height)``."""
    skeleton: str | None
    """Skeleton registry name, or ``None`` for detectors/segmenters."""
    status: ZooStatus
    """Whether posekit runs it today."""
    notes: str = ""
    """Constraints worth knowing before picking the model."""


ZOO: dict[str, ZooModel] = {
    model.key: model
    for model in (
        # Detectors.
        ZooModel(
            "yolox-m-humanart", "detector", "detector", "openmmlab-onnx",
            YOLOX_ONNX_ZIP_URLS["yolox-m-humanart"],
            (640, 640), None, "implemented", "mmdeploy NMS stripped at load; GPU torchvision NMS instead.",
        ),
        ZooModel(
            "yolox-x-humanart", "detector", "detector", "openmmlab-onnx",
            YOLOX_ONNX_ZIP_URLS["yolox-x-humanart"],
            (640, 640), None, "implemented", "",
        ),
        ZooModel(
            "rtdetr-v2-r50", "detector", "query-based", "transformers",
            "PekingU/rtdetr_v2_r50vd", (640, 640), None, "implemented",
            "NMS-free; person class read from config.id2label.",
        ),
        ZooModel(
            "rtmdet-nano-hand", "detector", "detector", "openmmlab-onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip",
            (320, 320), None, "planned", "Hand detector for the rtmlib Hand solution; NMS strip untested on RTMDet topology.",
        ),
        # Top-down 2D pose.
        ZooModel(
            "rtmpose-m-coco17", "pose2d", "top-down", "openmmlab-onnx",
            RTMPOSE_ONNX_ZIP_URLS["rtmpose-m-coco17"],
            (192, 256), "coco_17", "implemented", "body7 checkpoint.",
        ),
        ZooModel(
            "rtmpose-x-coco17", "pose2d", "top-down", "openmmlab-onnx",
            RTMPOSE_ONNX_ZIP_URLS["rtmpose-x-coco17"],
            (288, 384), "coco_17", "implemented", "body7 checkpoint.",
        ),
        ZooModel(
            "rtmw-x-coco133", "pose2d", "top-down", "openmmlab-onnx",
            RTMPOSE_ONNX_ZIP_URLS["rtmw-x-coco133"],
            (192, 256), "coco_133", "implemented", "cocktail14 wholebody checkpoint (DW-distilled).",
        ),
        ZooModel(
            "vitpose-base-coco17", "pose2d", "top-down", "transformers",
            "usyd-community/vitpose-base-simple", (192, 256), "coco_17", "implemented",
            "UDP crops + DARK-UDP decode; dataset_index pins the expert on ViTPose+ checkpoints.",
        ),
        ZooModel(
            "sapiens2-pose-coco133", "pose2d", "full-frame", "local",
            "facebook/sapiens2 via sapiens2-pose package", (768, 1024), "coco_133", "implemented",
            "308-kpt full-frame net projected to coco133; fp32 dynamo export + bf16 TRT only (fp16 overflows).",
        ),
        ZooModel(
            "rtmpose-m-hand21", "pose2d", "top-down", "openmmlab-onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
            (256, 256), "hand_21", "planned", "Pairs with rtmdet-nano-hand for the rtmlib Hand solution.",
        ),
        ZooModel(
            "rtmo-l-coco17", "instance-pose2d", "one-stage", "openmmlab-onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
            (640, 640), "coco_17", "planned", "One full-frame pass, boxes+keypoints jointly; needs GPU NMS port (Phase 4).",
        ),
        # 3D (Phase 4).
        ZooModel(
            "rtmw3d-l-coco133", "pose3d", "top-down", "local",
            "mmpose projects/rtmpose3d (no official ONNX; posekit exports the raw simcc_x/y/z heads)",
            (192, 256), "coco_133", "planned", "3D SimCC, root-relative z at kpts 11/12; upstream scores ignore z.",
        ),
    )
}


@dataclass(frozen=True, slots=True)
class PosePreset:
    """A named detector + top-down pose pairing (rtmlib solution equivalent)."""

    detector: DetectorConfig
    """Detector stage configuration."""
    pose: Pose2dConfig
    """Pose stage configuration."""
    description: str
    """What the preset is tuned for."""


PRESETS: dict[str, PosePreset] = {
    "body": PosePreset(
        YoloxDetectorConfig(variant="yolox-m-humanart"),
        RtmPoseConfig(variant="rtmpose-m-coco17"),
        "rtmlib Body(mode='balanced'): YOLOX-m + RTMPose-m body7, COCO-17.",
    ),
    "body-performance": PosePreset(
        YoloxDetectorConfig(variant="yolox-x-humanart"),
        RtmPoseConfig(variant="rtmpose-x-coco17"),
        "rtmlib Body(mode='performance'): YOLOX-x + RTMPose-x body7, COCO-17.",
    ),
    "wholebody": PosePreset(
        YoloxDetectorConfig(variant="yolox-m-humanart"),
        RtmPoseConfig(variant="rtmw-x-coco133"),
        "rtmlib Wholebody: YOLOX-m + RTMW cocktail14, COCO-133 (mv-api's pairing).",
    ),
    "body-nmsfree": PosePreset(
        RtDetrDetectorConfig(),
        VitPoseConfig(),
        "Transformers-sourced pairing: RT-DETRv2 (query-based, NMS-free) + ViTPose-base, COCO-17.",
    ),
    "wholebody-fullframe": PosePreset(
        YoloxDetectorConfig(variant="yolox-m-humanart"),
        SapiensPoseConfig(),
        "Sapiens2 full-frame 308->133 keypoints with YOLOX instancing.",
    ),
}


def preset(name: str) -> PosePreset:
    """Look up a named detector + pose preset.

    Args:
        name: Preset key from :data:`PRESETS`.

    Returns:
        The preset's configs, ready for ``.setup()``.

    Raises:
        KeyError: If the preset name is unknown.
    """
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; available: {sorted(PRESETS)}.")
    return PRESETS[name]


__all__ = ("PRESETS", "ZOO", "PosePreset", "ZooModel", "preset")
