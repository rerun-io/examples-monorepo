"""EgoHand DINOv3 NVFP4 linears and FP8 native-attention ONNX export."""

from pathlib import Path

import onnx
from trtkit.onnx_qdq import Fp8Attention, Fp8Linear, ModulePrecision, Nvfp4Linear, QdqPlan, quantize_qdq, save_with_external_data

from posekit.models.egohand_quantization import EgoHandCalibration

DINO_BLOCKS: int = 32
"""Transformer blocks in EgoHand's DINOv3 encoder; each has five linears and one native Attention op."""
INPUT_RANGE_HEADROOM: float = 2.0
"""Calibrated linear input maxima are doubled before they set the static input scales."""


def backbone_qdq_plan(calibration: EgoHandCalibration, *, fp8_tail_blocks: int = 0) -> QdqPlan:
    """NVFP4 linears, with the last ``fp8_tail_blocks`` blocks in FP8, and FP8 attention inputs.

    Per-output-channel FP8 weight scales in the tail retain accuracy near the
    final representation consumed by the metric camera and shape heads.
    """
    if not 0 <= fp8_tail_blocks <= DINO_BLOCKS:
        raise ValueError(f"fp8_tail_blocks must be between 0 and {DINO_BLOCKS}.")
    modules: dict[str, ModulePrecision] = {}
    for module, amax in calibration.input_amax.items():
        block: int = int(module.split(".")[2])
        input_amax: float = amax * INPUT_RANGE_HEADROOM
        modules[module] = Fp8Linear(input_amax) if block >= DINO_BLOCKS - fp8_tail_blocks else Nvfp4Linear(input_amax)
    attention: tuple[Fp8Attention, ...] = tuple(
        Fp8Attention(
            f"encoder.blocks.{block}.attn",
            (
                calibration.attention_amax[f"encoder.blocks.{block}.attn.q"],
                calibration.attention_amax[f"encoder.blocks.{block}.attn.k"],
                calibration.attention_amax[f"encoder.blocks.{block}.attn.v"],
            ),
        )
        for block in range(DINO_BLOCKS)
    )
    return QdqPlan(modules, attention, weight_device="cuda")


def quantize_backbone(input_path: Path, output_path: Path, calibration: EgoHandCalibration, *, fp8_tail_blocks: int = 0) -> None:
    """Quantize the released 160 linears and 32 native attention blocks."""
    plan: QdqPlan = backbone_qdq_plan(calibration, fp8_tail_blocks=fp8_tail_blocks)
    save_with_external_data(quantize_qdq(onnx.load(str(input_path)), plan), output_path)
