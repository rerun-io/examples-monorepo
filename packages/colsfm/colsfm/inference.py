"""What runs colsfm's learned-feature graphs, and where their tensors live.

The runtime is chosen once per run (`colsfm.run_config.PipelineOptions.inference_runtime`) and carried to stages 2, 4
and 5; anything that needs the device asks `runtime_device`. A leaf module (torch only), so any layer may import it.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import torch

InferenceRuntime: TypeAlias = Literal["tensorrt", "onnxruntime"]
"""What runs the RaCo / LightGlue+ / FlexGlue graphs: TensorRT engines on the GPU (the default), or ONNX Runtime on the CPU
(`trtkit.OnnxDynamicRuntime`) on the same graphs, for machines and test lanes without an engine."""

ResolvedDevice: TypeAlias = Literal["cuda", "cpu"]
"""The device a stage really ran on."""


def runtime_device(runtime: InferenceRuntime) -> ResolvedDevice:
    """The device a runtime's tensors live on: the GPU for TensorRT engines, the CPU for ONNX Runtime."""
    return "cpu" if runtime == "onnxruntime" else "cuda"


def frames_device(runtime: InferenceRuntime) -> torch.device:
    """Where a runtime wants its frames decoded."""
    return torch.device(runtime_device(runtime))
