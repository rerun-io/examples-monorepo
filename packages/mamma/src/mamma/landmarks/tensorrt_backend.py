"""TensorRT backend for MammaNet (ONNX export + trtkit engine build).

The static streaming path runs directly on :class:`trtkit.TensorRtRuntime`
with ``use_cuda_graph=True`` (inputs ``crops``/``masks``, outputs
``joints2d``/``visibility``/``contact``/``floor_contact`` — see
``export_mammanet_onnx``).

Follows the shared trtkit pattern: the `tensorrt-cu13` python API (not
torch-tensorrt) and a strongly-typed engine whose fp16 compute is baked into
the ONNX graph (fp32 I/O boundary casts in the export wrapper). The streaming
path retains its static-batch CUDA graph; posekit exports one cached dynamic
ONNX graph and lets trtkit build and cache engines by content hash.

Engines are machine-local artifacts and are never committed. The static
streaming engine is built into ``.trt_cache/`` by
``tools/build_trt_engine.py``; posekit's dynamic engine uses trtkit's cache.
"""

from __future__ import annotations

import hashlib
from os import stat_result
from pathlib import Path

import torch
from jaxtyping import Float32

from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.landmarks.mammanet import MammaNet

ENGINE_BATCH: int = 4
"""Static batch baked into the engine (cameras x persons rounded up)."""
INPUT_NAMES: tuple[str, str] = ("crops", "masks")
"""MammaNet ONNX input bindings in graph order."""
OUTPUT_NAMES: tuple[str, str, str, str] = ("joints2d", "visibility", "contact", "floor_contact")
"""MammaNet ONNX output bindings in graph order."""


class FlattenOutputs(torch.nn.Module):
    """Adapter shaping MammaNet's dict output into the flat ONNX output tuple."""

    def __init__(self, model: MammaNet) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, crops: Float32[torch.Tensor, "b 3 h w"], masks: Float32[torch.Tensor, "b 1 h w"]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        outputs: dict[str, torch.Tensor | None] = self.model(crops, masks)
        return outputs[OUTPUT_NAMES[0]], outputs[OUTPUT_NAMES[1]], outputs[OUTPUT_NAMES[2]], outputs[OUTPUT_NAMES[3]]


def export_mammanet_onnx(
    model: MammaNet,
    onnx_path: Path,
    *,
    batch: int = ENGINE_BATCH,
    dynamic_batch_max: int | None = None,
    config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG,
) -> None:
    """Export MammaNet to one static- or dynamic-batch fp16-compute ONNX graph via trtkit."""
    from trtkit import export_onnx

    crops: Float32[torch.Tensor, "b 3 h w"] = torch.randn(batch, 3, config.crop_height, config.crop_width, device="cuda")
    masks: Float32[torch.Tensor, "b 1 h w"] = torch.rand(batch, 1, config.crop_height, config.crop_width, device="cuda")
    export_onnx(
        FlattenOutputs(model).eval().cuda(),
        (crops, masks),
        onnx_path,
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        compute_dtype=torch.float16,
        dynamic_batch_max=dynamic_batch_max,
    )


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    """Build the static-batch strongly-typed MammaNet engine via trtkit."""
    from trtkit import TrtBuildConfig
    from trtkit import build_engine as trtkit_build_engine

    trtkit_build_engine(onnx_path, engine_path, TrtBuildConfig(max_batch_size=ENGINE_BATCH, opt_batch_size=ENGINE_BATCH))


def ensure_mammanet_onnx(model: MammaNet, weights_path: Path, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> Path:
    """Return MammaNet's cached dynamic-batch fp16 ONNX export."""
    from posekit.artifacts import DEFAULT_ONNX_CACHE_DIR

    checkpoint_stat: stat_result = weights_path.stat()
    # Metadata identity avoids hashing the 1.5 GB checkpoint on every setup.
    checkpoint_id: str = hashlib.sha1(
        f"{weights_path}:{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}".encode()
    ).hexdigest()[:10]
    onnx_path: Path = DEFAULT_ONNX_CACHE_DIR / f"mammanet_dense512_dynamic_b32_fp16_{checkpoint_id}.onnx"
    if onnx_path.exists():
        return onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mamma] exporting MammaNet to ONNX (one-time): {onnx_path.name}")
    export_mammanet_onnx(model, onnx_path, batch=8, dynamic_batch_max=32, config=config)
    return onnx_path
