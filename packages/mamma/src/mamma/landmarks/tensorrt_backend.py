"""TensorRT backend for MammaNet (ONNX export + trtkit engine build).

Inference runs directly on :class:`trtkit.TensorRtRuntime` with
``use_cuda_graph=True`` (inputs ``crops``/``masks``, outputs ``joints2d``/
``visibility``/``contact``/``floor_contact`` — see ``export_mammanet_onnx``).

Follows the proven sapiens2-pose pattern in this monorepo: the `tensorrt-cu13`
python API (not torch-tensorrt), static batch, FP16 builder flag, and one
captured ``execute_async_v3`` launch replayed per call — which composes cleanly
with the fitter's manual CUDA graph.

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


class _ExportWrapper(torch.nn.Module):
    """Tuple-returning wrapper (ONNX needs flat outputs, not a dict)."""

    def __init__(self, model: MammaNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, masks: torch.Tensor):
        out = self.model(x, masks)
        return out["joints2d"], out["visibility"], out["contact"], out["floor_contact"]


def export_mammanet_onnx(model: MammaNet, onnx_path: Path, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> None:
    """Export MammaNet to a static-batch ONNX graph.

    The dynamo exporter's output failed TRT 10.13's parser ("Failed to import
    initializer"), so this uses the legacy TorchScript exporter at opset 17 —
    the same combination sapiens2-pose ships.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _ExportWrapper(model).eval().cuda()
    x = torch.randn(ENGINE_BATCH, 3, config.crop_height, config.crop_width, device="cuda")
    masks = torch.rand(ENGINE_BATCH, 1, config.crop_height, config.crop_width, device="cuda")
    torch.onnx.export(
        wrapper,
        (x, masks),
        str(onnx_path),
        input_names=["crops", "masks"],
        output_names=["joints2d", "visibility", "contact", "floor_contact"],
        opset_version=17,
        dynamo=False,
    )


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    """Build the static-batch FP16 (TF32 off) MammaNet engine via trtkit."""
    from trtkit import TrtBuildConfig
    from trtkit import build_engine as trtkit_build_engine

    trtkit_build_engine(
        onnx_path,
        engine_path,
        TrtBuildConfig(max_batch_size=ENGINE_BATCH, opt_batch_size=ENGINE_BATCH, precision="fp16", allow_tf32=False),
    )
