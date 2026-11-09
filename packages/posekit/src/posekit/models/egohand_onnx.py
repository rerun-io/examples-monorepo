"""EgoHand DINOv3 NVFP4 linears and FP8 native-attention ONNX export."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import onnx
from trtkit.onnx_qdq import (
    LINEAR_INPUT_HEADROOM,
    Fp8Linear,
    ModulePrecision,
    Nvfp4Linear,
    QdqPlan,
    fp8_attention_plan,
    quantize_qdq,
    save_with_external_data,
)

from posekit.models.dinov3_calibration import DINO_BLOCKS
from posekit.models.egohand_quantization import validate_fp8_tail_blocks


@runtime_checkable
class DinoActivationRanges(Protocol):
    """Measured inputs of the shared ViT-H+/16 architecture, using each model's own weights."""

    @property
    def input_amax(self) -> dict[str, float]: ...

    @property
    def attention_amax(self) -> dict[str, float]: ...


def backbone_qdq_plan(calibration: DinoActivationRanges, *, fp8_tail_blocks: int = 0) -> QdqPlan:
    """NVFP4 linears, with the last ``fp8_tail_blocks`` blocks in FP8, and FP8 attention inputs.

    Per-output-channel FP8 weight scales in the tail retain accuracy near the
    final representation consumed by the metric camera and shape heads.
    """
    validate_fp8_tail_blocks(fp8_tail_blocks)
    modules: dict[str, ModulePrecision] = {}
    for module, amax in calibration.input_amax.items():
        block: int = int(module.split(".")[2])
        input_amax: float = amax * LINEAR_INPUT_HEADROOM
        modules[module] = Fp8Linear(input_amax) if block >= DINO_BLOCKS - fp8_tail_blocks else Nvfp4Linear(input_amax)
    return QdqPlan(modules, fp8_attention_plan("encoder.blocks", DINO_BLOCKS, calibration.attention_amax), weight_device="cuda")


def quantize_backbone(input_path: Path, output_path: Path, calibration: DinoActivationRanges, *, fp8_tail_blocks: int = 0) -> None:
    """Quantize the released 160 linears and 32 native attention blocks."""
    plan: QdqPlan = backbone_qdq_plan(calibration, fp8_tail_blocks=fp8_tail_blocks)
    save_with_external_data(quantize_qdq(onnx.load(str(input_path)), plan), output_path)
