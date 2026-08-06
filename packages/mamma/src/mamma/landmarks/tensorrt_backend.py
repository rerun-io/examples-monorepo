"""TensorRT backend for MammaNet (ONNX export + trtkit engine build).

Inference runs directly on :class:`trtkit.TensorRtRuntime` with
``use_cuda_graph=True`` (inputs ``crops``/``masks``, outputs ``joints2d``/
``visibility``/``contact``/``floor_contact`` — see ``export_mammanet_onnx``).

Follows the shared trtkit pattern: the `tensorrt-cu13` python API (not
torch-tensorrt), static batch, a strongly-typed engine whose fp16 compute is
baked into the ONNX graph (fp32 I/O boundary casts in the export wrapper), and
one captured ``execute_async_v3`` launch replayed per call — which composes
cleanly with the fitter's manual CUDA graph.

Engines are machine-local artifacts (sm-specific): built once into
``.trt_cache/`` by ``tools/build_trt_engine.py``, never committed.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.landmarks.mammanet import MammaNet

ENGINE_BATCH: int = 4
"""Static batch baked into the engine (cameras x persons rounded up)."""


class _FlattenOutputs(torch.nn.Module):
    """Adapter shaping MammaNet's dict output into the flat ONNX output tuple."""

    def __init__(self, model: MammaNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, masks: torch.Tensor):
        out = self.model(x, masks)
        return out["joints2d"], out["visibility"], out["contact"], out["floor_contact"]


def export_mammanet_onnx(model: MammaNet, onnx_path: Path, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> None:
    """Export MammaNet to a static-batch, fp16-compute ONNX graph via trtkit."""
    from trtkit import export_onnx

    x = torch.randn(ENGINE_BATCH, 3, config.crop_height, config.crop_width, device="cuda")
    masks = torch.rand(ENGINE_BATCH, 1, config.crop_height, config.crop_width, device="cuda")
    export_onnx(
        _FlattenOutputs(model).eval().cuda(),
        (x, masks),
        onnx_path,
        input_names=["crops", "masks"],
        output_names=["joints2d", "visibility", "contact", "floor_contact"],
        compute_dtype=torch.float16,
    )


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    """Build the static-batch strongly-typed MammaNet engine via trtkit."""
    from trtkit import TrtBuildConfig
    from trtkit import build_engine as trtkit_build_engine

    trtkit_build_engine(onnx_path, engine_path, TrtBuildConfig(max_batch_size=ENGINE_BATCH, opt_batch_size=ENGINE_BATCH))
