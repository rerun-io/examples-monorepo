"""TensorRT EfficientTAM image encoder: a drop-in for ``predictor.encode_image(batch)``.

The eager encoder (bf16 autocast) costs ~10 ms of GPU and ~370 kernel
launches per 4-camera tick, and the launches dominate its main-thread wall.
This module exports ``image_encoder`` (trunk + neck) as one strongly-typed
fp16 ONNX graph with a static camera batch, builds a TensorRT engine via
trtkit, and replays it as a single CUDA graph.

Only the image-dependent part runs in TensorRT. Preprocessing (``/255``,
antialiased resize to 512x512, normalize) stays in the predictor's own torch
transforms, and the sine position encoding depends only on the feature-map
shape, so it is computed once by the eager encoder and reused.

Engines are machine-local artifacts built by ``tools/build_trt_encoder.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float32
from torch import Tensor, nn

ENCODER_BATCH: int = 4
"""Static batch baked into the engine (one image per camera)."""
INPUT_NAME: str = "images"
"""Engine input: normalized RGB at the model resolution, ``[b, 3, s, s]`` fp32."""
OUTPUT_NAME: str = "embeddings"
"""Engine output: the lowest-resolution FPN level, ``[b, c, s/16, s/16]`` fp32."""


class EncoderExport(nn.Module):
    """ONNX adapter: normalized images -> the single FPN level ``encode_image`` returns."""

    def __init__(self, predictor: Any) -> None:
        super().__init__()
        if predictor.num_feature_levels != 1 or predictor.use_high_res_features_in_sam:
            raise ValueError("TrtImageEncoder supports single-level encoders without high-res SAM features (EfficientTAM).")
        self.image_encoder: nn.Module = predictor.image_encoder

    def forward(self, images: Float32[Tensor, "b 3 s s"]) -> Float32[Tensor, "b c h w"]:
        return self.image_encoder(images)["backbone_fpn"][-1]


def export_encoder_onnx(predictor: Any, onnx_path: Path, *, batch: int = ENCODER_BATCH) -> None:
    """Export the image encoder to a static-batch fp16-compute ONNX graph via trtkit."""
    from trtkit import export_onnx

    size: int = int(predictor.image_size)
    images: Float32[Tensor, "b 3 s s"] = torch.randn(batch, 3, size, size, device="cuda")
    export_onnx(
        EncoderExport(predictor).eval().cuda(),
        (images,),
        onnx_path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        compute_dtype=torch.float16,
    )


def build_encoder_engine(onnx_path: Path, engine_path: Path, *, batch: int = ENCODER_BATCH) -> None:
    """Build the static-batch strongly-typed encoder engine via trtkit."""
    from trtkit import TrtBuildConfig, build_engine

    build_engine(onnx_path, engine_path, TrtBuildConfig(max_batch_size=batch, opt_batch_size=batch))


class TrtImageEncoder:
    """Callable with the exact contract of ``predictor.encode_image(batch)``.

    Returns ``([embeddings], [pos_embeddings])``: one-level lists of fresh
    fp32 ``[b, c, h, w]`` tensors, like the eager encoder. Batches larger than
    the engine's static batch fall back to the eager encoder.
    """

    def __init__(self, predictor: Any, engine_path: Path, *, use_cuda_graph: bool = True) -> None:
        """Load the engine and precompute the shape-only position encoding.

        Args:
            predictor: The SAM2-family predictor whose ``image_encoder`` was exported.
            engine_path: Engine built by ``tools/build_trt_encoder.py`` for this machine.
            use_cuda_graph: Replay the engine launch as a private CUDA graph. Pass
                ``False`` when the caller captures this call into its own graph.
        """
        from trtkit.tensorrt_runtime import TensorRtRuntime

        self._predictor: Any = predictor
        self._runtime: TensorRtRuntime = TensorRtRuntime(engine_path, use_cuda_graph=use_cuda_graph)
        self.batch: int = self._runtime.spec.max_batch_size
        size: int = int(predictor.image_size)
        with torch.inference_mode():
            probe: Tensor = torch.zeros(self.batch, 3, size, size, dtype=torch.uint8, device="cuda")
            pos_embeddings: list[Tensor] = list(predictor.encode_image(probe)[1])
        self._pos_embeddings: list[Tensor] = pos_embeddings

    @torch.inference_mode()
    def __call__(self, image: Tensor | list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Encode a camera batch (uint8 or float RGB, any size) like ``encode_image``."""
        images: Float32[Tensor, "b 3 s s"] = self._predictor._prepare_images(image)
        batch: int = int(images.shape[0])
        if batch > self.batch:
            return self._predictor.encode_image(image)
        # The runtime's output buffer is overwritten by the next call; callers
        # (memory bank, batched propagation) may hold embeddings across ticks.
        embeddings: Tensor = self._runtime({INPUT_NAME: images.float()})[OUTPUT_NAME].clone()
        return [embeddings], [pos[:batch].clone() for pos in self._pos_embeddings]
