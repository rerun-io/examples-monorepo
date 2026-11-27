"""Build this machine's TensorRT engines for the default YOLOX and RTMPose/RTMW configs before their first real run.

``auto`` (the default backend of those configs) runs TensorRT on an NVIDIA GPU, and a first run on a new machine builds
each engine (minutes). This sets up every variant with its default config, which resolves ``auto`` exactly as later runs
do, so the cached engines are the ones they load: the zoo presets and mv-api's modes. Pipelines that build their own
engines (other batch profiles, an fp16 trunk) and the DetNet / EgoHand detectors, whose weights are not on every
machine, are not covered. Where ``auto`` resolves to ONNX Runtime there is nothing to build.
"""

import time
from dataclasses import dataclass
from typing import TypeAlias, get_args

from posekit.models.rtmpose import RtmPoseConfig, RtmPoseVariant
from posekit.models.yolox import YoloxDetectorConfig, YoloxVariant
from posekit.runtimes import TensorRtBackendConfig, resolve_backend

WarmedConfig: TypeAlias = YoloxDetectorConfig | RtmPoseConfig


@dataclass(frozen=True, slots=True)
class WarmEnginesConfig:
    """Which engines to build; every variant by default."""

    detectors: tuple[YoloxVariant, ...] = get_args(YoloxVariant)
    """YOLOX variants."""
    poses: tuple[RtmPoseVariant, ...] = get_args(RtmPoseVariant)
    """RTMPose / RTMW variants."""


def warm_configs(config: WarmEnginesConfig) -> list[WarmedConfig]:
    """The default model configs to set up.

    Args:
        config: Which variants.

    Returns:
        One default config per variant whose backend resolves to TensorRT here; empty where ``auto`` runs ONNX Runtime.
    """
    defaults: list[WarmedConfig] = [YoloxDetectorConfig(variant=variant) for variant in config.detectors]
    defaults += [RtmPoseConfig(variant=variant) for variant in config.poses]
    return [model for model in defaults if isinstance(resolve_backend(model.backend), TensorRtBackendConfig)]


def main(config: WarmEnginesConfig) -> None:
    """Set up each model once, which builds and caches its engine.

    Args:
        config: Which variants.
    """
    configs: list[WarmedConfig] = warm_configs(config)
    if not configs:
        print("[posekit] auto runs ONNX Runtime on this machine (no CUDA device or no TensorRT bindings): nothing to build")
        return
    for model in configs:
        started: float = time.perf_counter()
        model.setup()
        print(f"[posekit] {model.variant}: engine ready in {time.perf_counter() - started:.1f} s")
