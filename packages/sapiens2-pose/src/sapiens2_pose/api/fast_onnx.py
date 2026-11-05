"""Sapiens2-specific NVFP4 backbone and FP8 head/attention ONNX surgery."""

from pathlib import Path

import onnx
from trtkit.onnx_qdq import Fp8Attention, Fp8Conv, ModulePrecision, Nvfp4Linear, QdqPlan, quantize_qdq, save_with_external_data

from sapiens2_pose.api.fast_calibration import FastPoseCalibration

BACKBONE_BLOCKS: int = 24
"""Transformer blocks of the Sapiens2 0.4B backbone; each has one native Attention op."""
LINEAR_INPUT_HEADROOM: float = 2.0
"""Calibrated linear input maxima are doubled before they set the static NVFP4 input scales."""


def fast_pose_qdq_plan(calibration: FastPoseCalibration) -> QdqPlan:
    """NVFP4 backbone linears, FP8 head convolutions and FP8 attention inputs.

    The patch embedding and the final heatmap convolution keep BF16.
    """
    modules: dict[str, ModulePrecision] = {}
    for module, amax in calibration.input_amax.items():
        if module.startswith("backbone.blocks."):
            modules[module] = Nvfp4Linear(amax * LINEAR_INPUT_HEADROOM)
        elif module.startswith("decode_head.") and not module.endswith("conv_pose"):
            modules[module] = Fp8Conv(amax)
    attention: tuple[Fp8Attention, ...] = tuple(
        Fp8Attention(
            f"backbone.blocks.{block}.attn",
            (
                calibration.attention_amax[f"backbone.blocks.{block}.attn.q"],
                calibration.attention_amax[f"backbone.blocks.{block}.attn.k"],
                calibration.attention_amax[f"backbone.blocks.{block}.attn.v"],
            ),
        )
        for block in range(BACKBONE_BLOCKS)
    )
    return QdqPlan(modules, attention)


def quantize_fast_pose(input_path: Path, output_path: Path, calibration: FastPoseCalibration) -> None:
    """Quantize all backbone linears, head convolutions, and native attention inputs."""
    save_with_external_data(quantize_qdq(onnx.load(str(input_path)), fast_pose_qdq_plan(calibration)), output_path)
