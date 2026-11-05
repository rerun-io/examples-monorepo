"""Q/DQ precision rewrites of BF16 ONNX exports for strongly typed TensorRT builds.

A :class:`QdqPlan` names the precision of each module: NVFP4 or FP8 linears, FP8
convolutions, and FP8 inputs of the native Attention ops. Each entry carries the
activation maximum its static input scale is derived from. :func:`quantize_qdq`
rewrites the graph from that plan and :func:`save_with_external_data` publishes
the result. Tensor names and their order are part of the exported bytes, which
key the TensorRT engine caches.
"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias

import onnx
import torch
from jaxtyping import Float
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor

from trtkit.nvfp4 import BLOCK, bf16_round_up, quant_nvfp4_mse

FP8_MAX: float = 448.0
"""Largest finite float8_e4m3fn magnitude."""
FP4_MAX: float = 6.0
"""Largest E2M1 magnitude."""
NVFP4_GLOBAL_HEADROOM: float = 256.0
"""Per-tensor weight scale divisor that leaves E4M3 block scales room for the MSE scale search."""
LINEAR_WEIGHT_CHAIN: list[str] = ["Transpose", "Cast"]
"""Producers between a torch-exported MatMul and its ``[out, in]`` weight initializer."""


@dataclass(frozen=True, slots=True)
class Nvfp4Linear:
    """NVFP4 weights with FP8 per-16 block scales; inputs quantized to NVFP4 at run time."""

    input_amax: float
    """Input magnitude the static global input scale covers."""


@dataclass(frozen=True, slots=True)
class Fp8Linear:
    """FP8 weights with per-output-channel scales; statically scaled FP8 inputs."""

    input_amax: float
    """Input magnitude the static FP8 input scale covers."""


@dataclass(frozen=True, slots=True)
class Fp8Conv:
    """FP8 convolution weights with per-output-channel scales; statically scaled FP8 inputs."""

    input_amax: float
    """Input magnitude the static FP8 input scale covers."""


ModulePrecision: TypeAlias = Nvfp4Linear | Fp8Linear | Fp8Conv


@dataclass(frozen=True, slots=True)
class Fp8Attention:
    """Statically scaled FP8 query, key and value inputs of one native Attention op."""

    module: str
    """Name prefix of the new tensors, e.g. ``encoder.blocks.0.attn``."""
    amax: tuple[float, float, float]
    """Query, key and value magnitudes the static scales cover."""


@dataclass(frozen=True, slots=True)
class QdqPlan:
    """Precision of every quantized module in one graph."""

    modules: dict[str, ModulePrecision]
    """Precision by module: the weight initializer name without ``inner.`` and ``.weight``.
    Every weight MatMul needs an entry; a Conv without one keeps its precision."""
    attention: tuple[Fp8Attention, ...]
    """One entry per native Attention op, in graph order."""
    weight_device: Literal["cpu", "cuda"] = "cpu"
    """Device that quantizes the weights; it fixes the reduction order of the NVFP4 scale search."""


def _bf16_scalar(name: str, value: float) -> TensorProto:
    bits: Tensor = torch.tensor([value], dtype=torch.bfloat16).view(torch.uint16)
    return helper.make_tensor(name, TensorProto.BFLOAT16, [], vals=bits.numpy().tobytes(), raw=True)


def _raw_tensor(name: str, data_type: int, dims: list[int], data: Tensor) -> TensorProto:
    return helper.make_tensor(name, data_type, dims, vals=data.contiguous().cpu().view(torch.uint8).numpy().tobytes(), raw=True)


def _fp8_input(tensor: str, prefix: str, amax: float, zero_point: str) -> tuple[list[TensorProto], list[onnx.NodeProto], str]:
    """Static FP8 quantize/dequantize of ``tensor``; returns new initializers, nodes and the dequantized name."""
    scale: str = f"{prefix}.a_s8"
    quantized: str = f"{prefix}.a_q8"
    dequantized: str = f"{prefix}.a_dq8"
    initializers: list[TensorProto] = [
        _bf16_scalar(scale, amax / FP8_MAX),
        _raw_tensor(zero_point, TensorProto.FLOAT8E4M3FN, [], torch.zeros((), dtype=torch.float8_e4m3fn)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("QuantizeLinear", [tensor, scale, zero_point], [quantized], saturate=1),
        helper.make_node("DequantizeLinear", [quantized, scale], [dequantized]),
    ]
    return initializers, nodes, dequantized


def _fp8_weight(module: str, weight: Float[Tensor, "out *rest"]) -> tuple[list[TensorProto], onnx.NodeProto]:
    """FP8 weights with BF16 per-output-channel scales; the node writes ``{module}.w_dq``."""
    scales: Tensor = (weight.abs().flatten(1).amax(1) / FP8_MAX).clamp_min(1e-12).to(torch.bfloat16)
    codes: Tensor = (weight / scales.float().reshape(-1, *[1] * (weight.ndim - 1))).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    initializers: list[TensorProto] = [
        _raw_tensor(f"{module}.w_fp8", TensorProto.FLOAT8E4M3FN, list(weight.shape), codes),
        _raw_tensor(f"{module}.w_s8", TensorProto.BFLOAT16, [int(weight.shape[0])], scales),
    ]
    return initializers, helper.make_node("DequantizeLinear", [f"{module}.w_fp8", f"{module}.w_s8"], [f"{module}.w_dq"], axis=0)


def _nvfp4_weight(module: str, weight: Float[Tensor, "out k"]) -> tuple[list[TensorProto], list[onnx.NodeProto]]:
    """NVFP4 weights, FP8 block scales and a BF16 global scale; the nodes write ``{module}.w_dq``."""
    weight_amax: float = weight.abs().amax().item()
    global_scale: float = bf16_round_up(weight_amax / (FP4_MAX * NVFP4_GLOBAL_HEADROOM))
    quantized: tuple[Tensor, Tensor, Tensor] = quant_nvfp4_mse(weight, tensor_div=1.0 / global_scale * weight_amax)
    codes: Tensor = quantized[0]
    block_scales: Tensor = quantized[1]
    outputs: int = int(weight.shape[0])
    inputs: int = int(weight.shape[1])
    initializers: list[TensorProto] = [
        _raw_tensor(f"{module}.w_fp4", TensorProto.FLOAT4E2M1, [outputs, inputs], codes),
        _raw_tensor(f"{module}.w_sc", TensorProto.FLOAT8E4M3FN, [outputs, inputs // BLOCK], block_scales),
        _bf16_scalar(f"{module}.w_s2", global_scale),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("DequantizeLinear", [f"{module}.w_sc", f"{module}.w_s2"], [f"{module}.w_scdq"]),
        helper.make_node("DequantizeLinear", [f"{module}.w_fp4", f"{module}.w_scdq"], [f"{module}.w_dq"], axis=-1, block_size=BLOCK),
    ]
    return initializers, nodes


def _nvfp4_input(tensor: str, module: str, amax: float) -> tuple[list[TensorProto], list[onnx.NodeProto], str]:
    """TensorRT's dynamic NVFP4 input quantization; returns new initializers, nodes and the dequantized name."""
    scale: str = f"{module}.a_s"
    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "TRT_FP4DynamicQuantize", [tensor, scale], [f"{module}.a_q", f"{module}.a_sc"],
            domain="trt.plugins", axis=-1, block_size=BLOCK, scale_type=TensorProto.FLOAT8E4M3FN,
        ),
        helper.make_node("DequantizeLinear", [f"{module}.a_sc", scale], [f"{module}.a_scdq"]),
        helper.make_node("DequantizeLinear", [f"{module}.a_q", f"{module}.a_scdq"], [f"{module}.a_dq"], axis=-1, block_size=BLOCK),
    ]
    return [_bf16_scalar(scale, amax / (FP4_MAX * FP8_MAX))], nodes, f"{module}.a_dq"


def quantize_qdq(model: onnx.ModelProto, plan: QdqPlan) -> onnx.ModelProto:
    """Rewrite ``model`` in place from ``plan`` and drop the weights it replaces.

    Linears that read the same activation at the same precision share one input
    quantization (Q/K/V projections, gated MLP inputs).

    Raises:
        ValueError: If a weight MatMul has no plan entry, an entry does not fit its
            op or is absent from the graph, or the Attention count differs.
    """
    graph: onnx.GraphProto = model.graph
    producers: dict[str, onnx.NodeProto] = {output: node for node in graph.node for output in node.output}
    initializers: dict[str, TensorProto] = {tensor.name: tensor for tensor in graph.initializer}
    added: list[TensorProto] = []
    inserted: dict[int, list[onnx.NodeProto]] = {}
    quantized_inputs: dict[tuple[str, str], str] = {}
    applied: set[str] = set()

    for index, node in enumerate(list(graph.node)):
        if node.op_type not in ("MatMul", "Conv"):
            continue
        weight_name: str = node.input[1]
        chain: list[str] = []
        while weight_name not in initializers and weight_name in producers:
            chain.append(producers[weight_name].op_type)
            weight_name = producers[weight_name].input[0]
        if weight_name not in initializers or not weight_name.endswith(".weight"):
            continue
        module: str = weight_name.removeprefix("inner.").removesuffix(".weight")
        precision: ModulePrecision | None = plan.modules.get(module)
        if precision is None:
            if node.op_type == "MatMul":
                raise ValueError(f"The plan has no precision for linear {module}.")
            continue
        if (node.op_type == "Conv") != isinstance(precision, Fp8Conv):
            raise ValueError(f"{type(precision).__name__} does not apply to {node.op_type} {module}.")
        weight: Tensor = torch.from_numpy(numpy_helper.to_array(initializers[weight_name]).copy()).to(plan.weight_device)
        nodes: list[onnx.NodeProto] = []
        if isinstance(precision, Fp8Conv):
            weight_initializers, weight_node = _fp8_weight(module, weight)
            input_initializers, input_nodes, dequantized_input = _fp8_input(node.input[0], module, precision.input_amax, f"{module}.a_zp")
            added.extend([*weight_initializers, *input_initializers])
            inserted[index] = [weight_node, *input_nodes]
            node.input[0], node.input[1] = dequantized_input, f"{module}.w_dq"
            applied.add(module)
            continue
        if chain != LINEAR_WEIGHT_CHAIN:
            raise ValueError(f"Linear {module} reads its weight through {chain}, not {LINEAR_WEIGHT_CHAIN}.")
        activation: str = node.input[0]
        if isinstance(precision, Fp8Linear):
            weight_initializers, weight_node = _fp8_weight(module, weight)
            added.extend(weight_initializers)
            nodes.append(weight_node)
            key: tuple[str, str] = (activation, "fp8")
            if key not in quantized_inputs:
                # The zero point name is part of the released EgoHand export bytes.
                input_initializers, input_nodes, quantized_inputs[key] = _fp8_input(activation, module, precision.input_amax, f"{module}.a_zp8")
                added.extend(input_initializers)
                nodes.extend(input_nodes)
        else:
            weight_initializers, weight_nodes = _nvfp4_weight(module, weight)
            added.extend(weight_initializers)
            nodes.extend(weight_nodes)
            key = (activation, "fp4")
            if key not in quantized_inputs:
                input_initializers, input_nodes, quantized_inputs[key] = _nvfp4_input(activation, module, precision.input_amax)
                added.extend(input_initializers)
                nodes.extend(input_nodes)
        nodes.append(helper.make_node("Transpose", [f"{module}.w_dq"], [f"{module}.w_t"], perm=[1, 0]))
        node.input[0], node.input[1] = quantized_inputs[key], f"{module}.w_t"
        inserted[index] = nodes
        applied.add(module)

    missing: set[str] = set(plan.modules) - applied
    if missing:
        raise ValueError(f"Planned modules are not weight MatMul/Conv nodes of the graph: {sorted(missing)}.")
    attention_nodes: list[tuple[int, onnx.NodeProto]] = [(index, node) for index, node in enumerate(graph.node) if node.op_type == "Attention"]
    if len(attention_nodes) != len(plan.attention):
        raise ValueError(f"Expected {len(plan.attention)} native Attention ops, got {len(attention_nodes)}.")
    for (index, node), attention in zip(attention_nodes, plan.attention, strict=True):
        for slot, (tag, amax) in enumerate(zip("qkv", attention.amax, strict=True)):
            prefix: str = f"{attention.module}.{tag}"
            input_initializers, input_nodes, node.input[slot] = _fp8_input(node.input[slot], prefix, amax, f"{prefix}.a_zp")
            added.extend(input_initializers)
            inserted.setdefault(index, []).extend(input_nodes)

    ordered: list[onnx.NodeProto] = [new for index, node in enumerate(graph.node) for new in [*inserted.get(index, []), node]]
    graph.ClearField("node")
    graph.node.extend(ordered)
    graph.initializer.extend(added)
    # Walk back from the outputs; the replaced weights and their Cast/Transpose chains fall away.
    live: set[str] = {output.name for output in graph.output}
    kept: list[onnx.NodeProto] = []
    for node in reversed(list(graph.node)):
        if any(output in live for output in node.output):
            kept.append(node)
            live.update(name for name in node.input if name)
    kept.reverse()
    graph.ClearField("node")
    graph.node.extend(kept)
    kept_initializers: list[TensorProto] = [tensor for tensor in graph.initializer if tensor.name in live]
    graph.ClearField("initializer")
    graph.initializer.extend(kept_initializers)
    if not any(opset.domain == "trt.plugins" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("trt.plugins", 1))
    return model


def save_with_external_data(model: onnx.ModelProto, output_path: Path) -> None:
    """Write ``output_path`` and ``<name>.data`` beside it; the model is published last, so its presence means both are complete."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_name: str = output_path.name + ".data"
    with TemporaryDirectory(prefix=".qdq-", dir=output_path.parent) as directory:
        staged: Path = Path(directory) / output_path.name
        onnx.save(model, str(staged), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
        # onnx writes no data file when every tensor is under the size threshold.
        if (staged.parent / data_name).is_file():
            (staged.parent / data_name).replace(output_path.parent / data_name)
        else:
            (output_path.parent / data_name).unlink(missing_ok=True)
        staged.replace(output_path)
