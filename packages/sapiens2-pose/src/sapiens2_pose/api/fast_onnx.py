"""Sapiens2-specific NVFP4 backbone and FP8 head/attention ONNX surgery."""

from pathlib import Path

import onnx
from trtkit.onnx_qdq import (
    LINEAR_INPUT_HEADROOM,
    Fp8Conv,
    ModulePrecision,
    Nvfp4Linear,
    QdqPlan,
    fp8_attention_plan,
    quantize_qdq,
    save_with_external_data,
)

from sapiens2_pose.api.fast_calibration import FastPoseCalibration

BACKBONE_BLOCKS: int = 24
"""Transformer blocks of the Sapiens2 0.4B backbone; each has one native Attention op."""


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
    return QdqPlan(modules, fp8_attention_plan("backbone.blocks", BACKBONE_BLOCKS, calibration.attention_amax))


def quantize_fast_pose(input_path: Path, output_path: Path, calibration: FastPoseCalibration) -> None:
    """Quantize all backbone linears, head convolutions, and native attention inputs."""
    save_with_external_data(quantize_qdq(onnx.load(str(input_path)), fast_pose_qdq_plan(calibration)), output_path)
