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
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float32

from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.landmarks.mammanet import MammaNet

if TYPE_CHECKING:
    import onnx

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


FP8_LINEARS: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")
"""Linear names quantized by the FP8 build: ViT block ``qkv``/``proj``/``fc1``/``fc2``;
``linear1``/``linear2`` add the decoder feed-forward layers."""
CALIBRATED_PREFIXES: tuple[str, ...] = ("backbone.blocks.", "decoder.decoder_detr.layers.")
"""Module prefixes whose ``nn.Linear`` inputs are calibrated."""


def calibrate_backbone_amax(
    model: MammaNet, crops: Float32[torch.Tensor, "n b 3 h w"], masks: Float32[torch.Tensor, "n b 1 h w"]
) -> dict[str, float]:
    """Max absolute input of every ViT block and decoder-layer linear over real crop batches, under the eager fp16 autocast."""
    amax: dict[str, float] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith(CALIBRATED_PREFIXES):

            def record(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], name: str = name) -> None:
                amax[name] = max(amax.get(name, 0.0), args[0].detach().abs().amax().float().item())

            hooks.append(module.register_forward_pre_hook(record))
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for batch_crops, batch_masks in zip(crops, masks, strict=True):
                model(batch_crops.cuda().float(), batch_masks.cuda().float())
    finally:
        for hook in hooks:
            hook.remove()
    return amax


def _split_qkv_linears(model: onnx.ModelProto) -> int:
    """Rewrite each ViT ``attn.qkv`` linear into separate q/k/v linears, in place.

    TensorRT's Myelin splits the fused qkv GEMM per q/k/v to emit head
    layouts directly, and in doing so dequantizes an FP8 weight back to an
    fp16 GEMM. Three explicit linears (``attn.{q,k,v}.weight``) keep FP8.
    The dead fused nodes are left for ``quantize_qdq``'s pruning.

    Returns:
        Number of qkv linears rewritten.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    initializers: dict[str, onnx.TensorProto] = {tensor.name: tensor for tensor in graph.initializer}
    producers: dict[str, onnx.NodeProto] = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    index_of: dict[str, int] = {node.output[0]: index for index, node in enumerate(graph.node)}
    inserted: dict[int, list[onnx.NodeProto]] = {}
    removed: set[int] = set()
    for index, node in enumerate(graph.node):
        if node.op_type != "MatMul" or node.input[1] not in producers:
            continue
        transpose_w: onnx.NodeProto = producers[node.input[1]]
        cast_w: onnx.NodeProto | None = producers.get(transpose_w.input[0])
        if transpose_w.op_type != "Transpose" or cast_w is None or cast_w.op_type != "Cast" or not cast_w.input[0].endswith("attn.qkv.weight"):
            continue
        (add,) = consumers[node.output[0]]
        (reshape,) = consumers[add.output[0]]
        (permute,) = consumers[reshape.output[0]]
        gathers: list[onnx.NodeProto] = consumers[permute.output[0]]
        if add.op_type != "Add" or reshape.op_type != "Reshape" or list(permute.attribute[0].ints) != [2, 0, 3, 1, 4] or len(gathers) != 3:
            raise ValueError(f"unexpected qkv pattern after {node.name}")
        weight: np.ndarray = numpy_helper.to_array(initializers[cast_w.input[0]])
        bias: np.ndarray = numpy_helper.to_array(initializers[add.input[1]])
        batch, tokens, three, heads, head_dim = (int(v) for v in numpy_helper.to_array(initializers[reshape.input[1]]))
        width: int = heads * head_dim if head_dim > 0 else weight.shape[0] // three
        prefix: str = cast_w.input[0].removesuffix("qkv.weight")
        new_nodes: list[onnx.NodeProto] = []
        for gather in gathers:
            part: int = int(numpy_helper.to_array(initializers[gather.input[1]]))
            name: str = f"{prefix}{'qkv'[part]}"
            rows = slice(part * width, (part + 1) * width)
            graph.initializer.extend([
                numpy_helper.from_array(np.ascontiguousarray(weight[rows]), f"{name}.weight"),
                numpy_helper.from_array(np.ascontiguousarray(bias[rows]), f"{name}.bias"),
                numpy_helper.from_array(np.array([batch, tokens, heads, width // heads], dtype=np.int64), f"{name}.shape"),
            ])
            new_nodes.extend([
                helper.make_node("Cast", [f"{name}.weight"], [f"{name}.w_cast"], to=cast_w.attribute[0].i),
                helper.make_node("Transpose", [f"{name}.w_cast"], [f"{name}.w_t"], perm=[1, 0]),
                helper.make_node("MatMul", [node.input[0], f"{name}.w_t"], [f"{name}.mm"]),
                helper.make_node("Add", [f"{name}.mm", f"{name}.bias"], [f"{name}.out"]),
                helper.make_node("Reshape", [f"{name}.out", f"{name}.shape"], [f"{name}.heads"]),
                helper.make_node("Transpose", [f"{name}.heads"], [gather.output[0]], perm=[0, 2, 1, 3]),
            ])
            removed.add(index_of[gather.output[0]])
        inserted[index] = new_nodes
    ordered: list[onnx.NodeProto] = []
    for index, node in enumerate(graph.node):
        ordered.extend(inserted.get(index, []))
        if index not in removed:
            ordered.append(node)
    graph.ClearField("node")
    graph.node.extend(ordered)
    return len(inserted)


def quantize_mammanet_fp8(onnx_path: Path, output_path: Path, input_amax: dict[str, float], linears: tuple[str, ...] = FP8_LINEARS) -> None:
    """Rewrite the fp16 MammaNet export with FP8 Q/DQ on the chosen ViT block linears.

    Weights get per-output-channel FP8 scales; inputs get static per-tensor
    scales from ``input_amax`` (keyed like ``backbone.blocks.0.mlp.fc1``).
    Every other weight MatMul keeps fp16. The graph opset is raised to 21,
    the first that types FP8 QuantizeLinear with fp16 scales.
    """
    import onnx
    from trtkit.onnx_qdq import Fp8Linear, KeepPrecision, QdqPlan, quantize_qdq, save_with_external_data

    model: onnx.ModelProto = onnx.load(str(onnx_path))
    if "qkv" in linears:
        _split_qkv_linears(model)
        # The split linears read the fused qkv input, so they share its range.
        input_amax = {**input_amax, **{
            f"{name.removesuffix('qkv')}{part}": amax for name, amax in input_amax.items() if name.endswith(".qkv") for part in "qkv"
        }}
        linears = (*(name for name in linears if name != "qkv"), "q", "k", "v")
    initializers: set[str] = {tensor.name for tensor in model.graph.initializer}
    producers: dict[str, onnx.NodeProto] = {output: node for node in model.graph.node for output in node.output}
    modules: dict[str, Fp8Linear | KeepPrecision] = {}
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        weight: str = node.input[1]
        while weight not in initializers and weight in producers:
            weight = producers[weight].input[0]
        if weight not in initializers or not weight.endswith(".weight"):
            continue
        module: str = weight.removeprefix("inner.").removesuffix(".weight")
        key: str = module.removeprefix("model.")
        quantize: bool = key in input_amax and key.rsplit(".", 1)[-1] in linears
        modules[module] = Fp8Linear(input_amax[key]) if quantize else KeepPrecision()
    plan: QdqPlan = QdqPlan(modules, attention=(), weight_device="cuda", scale_dtype="float16")
    quantized: onnx.ModelProto = quantize_qdq(model, plan)
    for opset in quantized.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = max(int(opset.version), 21)
    save_with_external_data(quantized, output_path)
