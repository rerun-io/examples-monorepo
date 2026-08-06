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


class _ExportWrapper(torch.nn.Module):
    """Flat-output wrapper with fp32 I/O and fp16 autocast compute.

    TensorRT 11 builds are strongly typed: the graph's dtypes are the engine's
    dtypes. Tracing under autocast bakes fp16 casts around the matmul-heavy ops
    (fp32 islands stay where autocast keeps them) while the I/O contract and
    MammaNet's Float32 jaxtyping hints stay fp32 — the same mixed numerics the
    old weakly-typed FP16 builder flag produced, and the same recipe the eager
    fp16 path uses at inference.
    """

    def __init__(self, model: MammaNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, masks: torch.Tensor):
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(x, masks)
        return out["joints2d"].float(), out["visibility"].float(), out["contact"].float(), out["floor_contact"].float()


def export_mammanet_onnx(model: MammaNet, onnx_path: Path, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> None:
    """Export MammaNet to a static-batch, fp16-compute ONNX graph (dynamo exporter)."""
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
        dynamo=True,
    )


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    """Build the static-batch strongly-typed MammaNet engine via trtkit."""
    from trtkit import TrtBuildConfig
    from trtkit import build_engine as trtkit_build_engine

    trtkit_build_engine(onnx_path, engine_path, TrtBuildConfig(max_batch_size=ENGINE_BATCH, opt_batch_size=ENGINE_BATCH))
