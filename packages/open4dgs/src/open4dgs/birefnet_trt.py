"""BiRefNet mattes through TensorRT: the fp16 snapshot exported once to ONNX and built into a cached, machine-local engine.

The engine takes the exact input of the torch model in ``masks.predict_mattes`` (ImageNet-normalised fp16 crops at
``MODEL_SIZE``), batches of up to ``MAX_BATCH`` (a shorter batch is zero-padded: the engine is static), and returns the
last decoder logits, so the two are interchangeable there. ``load_matte_model`` takes the engine when this machine's cache already holds it and the torch model otherwise:
a run never waits for an engine build (``build_engine`` / the ``open4dgs-birefnet-engine`` task builds it).

The snapshot's deformable convolutions (``torchvision.ops.deform_conv2d``, with a modulation mask) export as ONNX
``DeformConv`` (opset 19+), which the TensorRT ONNX parser maps onto its ``ModulatedDeformConv2d`` plugin.
The engine is static in the batch on purpose: an engine built from a dynamic-batch export (profile 1..8) gave the right
mattes at batch 8 but wrong ones (|p| differences up to 1.0 over ~70 % of pixels) at every other batch size.
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float16
from torch import Tensor

from open4dgs.masks import MODEL_SIZE, MatteNet, load_birefnet

if TYPE_CHECKING:
    from trtkit import TrtBuildConfig

MAX_BATCH: int = 8
"""The engine's batch (``predict_mattes``' default batch)."""
ONNX_OPSET: int = 20
"""ONNX ``DeformConv`` needs opset 19 or later."""
DEFAULT_CACHE_DIR: Path = Path(os.environ.get("OPEN4DGS_CACHE_DIR", Path.home() / ".cache" / "open4dgs")) / "birefnet"
"""ONNX files here, engines under ``trt/`` (machine-local)."""


def snapshot_digest(model_dir: Path) -> str:
    """Identity of a BiRefNet snapshot: its code and config files, the weight file's size and its first and last MiB
    (hashing all 444 MB from the NAS on every run would cost seconds)."""
    digest = hashlib.sha256()
    for name in ("birefnet.py", "BiRefNet_config.py", "config.json"):
        digest.update((model_dir / name).read_bytes())
    weights: Path = model_dir / "model.safetensors"
    size: int = weights.stat().st_size
    digest.update(str(size).encode())
    with weights.open("rb") as file:
        digest.update(file.read(1 << 20))
        file.seek(max(0, size - (1 << 20)))
        digest.update(file.read(1 << 20))
    return digest.hexdigest()


def onnx_path(model_dir: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Where the ONNX export of this snapshot lives (with its ``.data`` weights sidecar)."""
    return cache_dir / f"birefnet-{snapshot_digest(model_dir.expanduser().resolve())[:12]}_fp16_{MODEL_SIZE}_b{MAX_BATCH}.onnx"


def engine_config() -> "TrtBuildConfig":
    """The TensorRT build settings (part of the engine cache key): the static batch at the static crop size."""
    from trtkit import TrtBuildConfig

    return TrtBuildConfig(max_batch_size=MAX_BATCH, opt_batch_size=MAX_BATCH, builder_optimization_level=3, workspace_gib=8.0)


def _deform_conv2d_onnx(
    input: object,
    weight: object,
    offset: object,
    mask: object,
    bias: object,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dil_h: int,
    dil_w: int,
    n_weight_grps: int,
    n_offset_grps: int,
    use_mask: bool,
) -> object:
    """``torchvision::deform_conv2d`` as ONNX ``DeformConv`` (same offset layout and modulation mask)."""
    from onnxscript import opset20 as op

    return op.DeformConv(
        input,
        weight,
        offset,
        bias,
        mask,
        dilations=[dil_h, dil_w],
        group=n_weight_grps,
        offset_group=n_offset_grps,
        pads=[pad_h, pad_w, pad_h, pad_w],
        strides=[stride_h, stride_w],
    )


class _Logits(torch.nn.Module):
    """The exported graph: BiRefNet's last prediction only."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: torch.nn.Module = model

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)[-1]


def export_onnx(model_dir: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Export the snapshot to ONNX once (on the CPU: the dynamo exporter traces with fake tensors); published atomically."""
    from trtkit.artifacts import build_once

    path: Path = onnx_path(model_dir, cache_dir)

    def build() -> None:
        model: torch.nn.Module = load_birefnet(model_dir, torch.device("cpu"))
        staging: Path = path.with_name(f"{path.name}.part{os.getpid()}")
        staging.mkdir(parents=True)
        try:
            with torch.inference_mode():
                torch.onnx.export(
                    _Logits(model).eval(),
                    (torch.zeros((MAX_BATCH, 3, MODEL_SIZE, MODEL_SIZE), dtype=torch.float16),),
                    str(staging / path.name),
                    input_names=["image"],
                    output_names=["logits"],
                    opset_version=ONNX_OPSET,
                    dynamo=True,
                    custom_translation_table={torch.ops.torchvision.deform_conv2d.default: _deform_conv2d_onnx},
                )
            # The sidecar first: the graph must never be visible while the weights it references are missing.
            sidecar: Path = staging / f"{path.name}.data"
            if sidecar.exists():
                os.replace(sidecar, path.with_name(sidecar.name))
            os.replace(staging / path.name, path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return build_once(path, build)


def cached_engine(model_dir: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path | None:
    """The engine of this snapshot when this machine has already built it; checks the caches only. None without trtkit."""
    try:
        from trtkit.trt_builder import cached_engine_path, onnx_content_hash
    except ImportError:
        return None
    path: Path = onnx_path(model_dir, cache_dir)
    if not path.exists():
        return None
    trt_dir: Path = cache_dir / "trt"
    engine: Path = cached_engine_path(path, engine_config(), cache_dir=trt_dir, onnx_sha256=onnx_content_hash(path, cache_dir=trt_dir))
    return engine if engine.exists() else None


def register_plugins() -> None:
    """Load TensorRT's plugin library: the parser maps ``DeformConv`` onto its ``ModulatedDeformConv2d`` plugin, and an
    engine holding that plugin deserialises only once the library is registered."""
    import tensorrt as trt

    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")


def build_engine(model_dir: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Export (once) and build (once per machine) the engine; returns its path."""
    from trtkit import ensure_engine

    register_plugins()
    return ensure_engine(export_onnx(model_dir, cache_dir), engine_config(), cache_dir=cache_dir / "trt")


class BiRefNetTrt:
    """The engine behind the torch model's call: ``net(x)[-1]`` are the logits of the fp16 crops ``x``.

    The logits are a view into the runtime's output buffer, valid until the next call (``predict_mattes`` reads each
    batch's before it sends the next)."""

    def __init__(self, engine: Path) -> None:
        from trtkit import TensorRtRuntime

        register_plugins()
        self.runtime = TensorRtRuntime(engine)

    def __call__(self, image: Float16[Tensor, "b 3 s s"]) -> list[Tensor]:
        return [self.runtime({"image": image.contiguous()})["logits"]]


def load_matte_model(model_dir: Path, device: torch.device, cache_dir: Path = DEFAULT_CACHE_DIR) -> MatteNet:
    """The cached TensorRT engine when this machine has one for the snapshot (on CUDA), else the torch model."""
    engine: Path | None = cached_engine(model_dir, cache_dir) if device.type == "cuda" else None
    return load_birefnet(model_dir, device) if engine is None else BiRefNetTrt(engine)
