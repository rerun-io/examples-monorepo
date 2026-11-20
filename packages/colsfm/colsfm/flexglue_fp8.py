"""FlexGlue's matcher in FP8: E4M3 Q/DQ on its transformer, with static scales calibrated once and baked into the graph.

The matcher's transformer attends over the descriptor's dense 64x64 token grid (4,096 tokens per image, whatever the
keypoint count): about 2 TFLOP per pair, which the fp16/bf16 engine runs near the 5090's bf16 peak. FP8 halves the
cost of every tensor-core op it reaches. Quantized here, on the graph `colsfm.flexglue_trt.prepare_matcher_graph`
prepares:

* every transformer linear (`matcher.transformers.*`): FP8 weights with per-output-channel scales, quantized once per
  weight (both images' MatMuls share it), FP8 inputs with one static scale each (trtkit's `fp8_weight_dq` / `fp8_input_qdq`);
* both attention-core MatMuls of every attention (q.k^T and p.v): FP8 operands with static bf16 scales; the softmax
  probabilities use amax 1.

The input projection, the positional encoding and the match head keep their precision. An input scale is the
largest magnitude the calibration pairs produced, times `trtkit.onnx_qdq.LINEAR_INPUT_HEADROOM`. The scales live in
the graph (`colsfm.matching.FlexGlueOptions.fp8_file`, one per set of thresholds), which `colsfm.apis.flexglue_fp8`
writes once per machine and model; a run refuses to start without it (`colsfm.model_assets`), and matching refuses
one calibrated from another prepared matcher (`check_fp8_graph`).

Measured on the 5090 against the fp16/bf16 matcher (2026-11-19, calibration on the Galileo frames, every test segment
outside it): the matching stage takes 21-25 % less time (MSD MGO05 28.1 -> 21.7 s, EuRoC V1_01 19.2 -> 15.0 s, KITTI
06 44.2 -> 34.8 s, TartanAir V2 P004 cube faces 60.8 -> 45.8 s) and the whole run 4-12 % less, with ground-truth ATE
equal within its run-to-run spread (MSD 12.1 -> 11.3 mm, EuRoC 35.3 -> 35.3 mm, KITTI 1,270 -> 1,255 mm, TartanAir
14.6 -> 14.6 mm; the cube's repeated runs of one arm differ by up to 2.2 mm).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from trtkit.onnx_qdq import LINEAR_INPUT_HEADROOM, fp8_input_qdq, fp8_weight_dq, prune_dead

from colsfm.matching import FLEXGLUE_FP8_COMMAND, FlexGlueOptions

TRANSFORMER_WEIGHTS: Final[str] = "matcher.transformers."
"""Weight-name prefix of the linears that go to FP8; the head, the input projection and the posenc keep their precision."""

FP8_OPSET: Final[int] = 21
"""ONNX opset of the quantized graph: FP8 QuantizeLinear with `saturate` needs 19 or later."""

CALIBRATION_PAIRS: Final[int] = 200
"""Image pairs the activation maxima are taken over."""

PROBABILITY_AMAX: Final[float] = 1.0
"""The softmax probabilities' range, fixed rather than calibrated."""

SOURCE_PROPERTY: Final[str] = "colsfm_fp8_source"
"""Metadata key of the FP8 graph naming the prepared matcher it was calibrated from."""


@dataclass(frozen=True, slots=True)
class Fp8Targets:
    """The MatMuls of a prepared matcher graph that go to FP8, by node name."""

    linears: tuple[tuple[str, str, str], ...]
    """`(node, activation, weight initializer)` per transformer linear; one weight may serve several nodes."""
    cores: tuple[tuple[str, str, str], ...]
    """`(node, left operand, right operand)` per attention-core MatMul (bf16 operands)."""
    probabilities: frozenset[str]
    """Left operands that are softmax outputs."""

    def calibrated(self) -> list[str]:
        """The activations whose maxima calibration measures, in graph order, each once."""
        names: list[str] = [activation for _, activation, _ in self.linears]
        names += [operand for _, left, right in self.cores for operand in (left, right) if operand not in self.probabilities]
        return list(dict.fromkeys(names))


def fp8_targets(graph: onnx.GraphProto) -> Fp8Targets:
    """Find the transformer linears (`MatMul(x, Transpose(weight))`) and the bf16 attention-core MatMuls.

    Args:
        graph: A matcher graph from `prepare_matcher_graph` (contrib ops rewritten, no conditionals).

    Returns:
        The FP8 targets.

    Raises:
        ValueError: When the graph has no transformer linear or no attention core (not the FlexGlue matcher).
    """
    producers: dict[str, onnx.NodeProto] = {output: node for node in graph.node for output in node.output}
    initializers: set[str] = {tensor.name for tensor in graph.initializer}
    linears: list[tuple[str, str, str]] = []
    cores: list[tuple[str, str, str]] = []
    probabilities: set[str] = set()
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        left, right = node.input
        right_producer: onnx.NodeProto | None = producers.get(right)
        if right_producer is None:
            continue
        if (
            right_producer.op_type == "Transpose"
            and right_producer.input[0] in initializers
            and right_producer.input[0].startswith(TRANSFORMER_WEIGHTS)
        ):
            linears.append((node.name, left, right_producer.input[0]))
        elif right_producer.op_type == "Cast" and helper.get_attribute_value(right_producer.attribute[0]) == TensorProto.BFLOAT16:
            cores.append((node.name, left, right))
            left_producer: onnx.NodeProto | None = producers.get(left)
            if left_producer is not None and left_producer.op_type == "Softmax":
                probabilities.add(left)
    if not linears or not cores:
        raise ValueError(f"found {len(linears)} transformer linears and {len(cores)} attention cores; not a prepared FlexGlue matcher")
    return Fp8Targets(linears=tuple(linears), cores=tuple(cores), probabilities=frozenset(probabilities))


def calibration_graph(model: onnx.ModelProto, targets: Fp8Targets) -> onnx.ModelProto:
    """The matcher with every calibrated activation as an extra output (fp16 linear inputs, bf16 attention operands).

    Args:
        model: The prepared matcher; not modified.
        targets: Its FP8 targets.

    Returns:
        A copy with the extra outputs.
    """
    calibration: onnx.ModelProto = onnx.ModelProto()
    calibration.CopyFrom(model)
    core_operands: set[str] = {operand for _, left, right in targets.cores for operand in (left, right)}
    calibration.graph.output.extend(
        helper.make_tensor_value_info(name, TensorProto.BFLOAT16 if name in core_operands else TensorProto.FLOAT16, None)
        for name in targets.calibrated()
    )
    return calibration


def quantize(model: onnx.ModelProto, targets: Fp8Targets, amax: Mapping[str, float]) -> onnx.ModelProto:
    """Insert the FP8 Q/DQ nodes, with static scales from the calibrated maxima, into a copy of the matcher, and drop the
    fp16 transformer weights they replace.

    Args:
        model: The prepared matcher; not modified.
        targets: Its FP8 targets.
        amax: Largest magnitude of every calibrated activation (`Fp8Targets.calibrated`).

    Returns:
        The FP8 graph (opset `FP8_OPSET`).

    Raises:
        ValueError: When a calibrated maximum is missing, zero or not finite.
    """
    bad: list[str] = [name for name in targets.calibrated() if not (0.0 < amax.get(name, float("nan")) < float("inf"))]
    if bad:
        raise ValueError(f"{len(bad)} activations have no usable maximum, e.g. {bad[:3]}")
    quantized: onnx.ModelProto = onnx.ModelProto()
    quantized.CopyFrom(model)
    graph: onnx.GraphProto = quantized.graph
    weights: dict[str, onnx.TensorProto] = {tensor.name: tensor for tensor in graph.initializer}
    linears: dict[str, tuple[str, str]] = {node: (activation, weight) for node, activation, weight in targets.linears}
    cores: dict[str, tuple[str, str]] = {node: (left, right) for node, left, right in targets.cores}
    nodes: list[onnx.NodeProto] = []
    added: list[onnx.TensorProto] = []
    inputs: dict[str, str] = {}
    """Dequantized name per quantized linear input: the q, k and v projections of one activation share it."""
    transposed: dict[str, str] = {}
    """Transposed FP8 weight per weight initializer: the MatMuls of both images share it."""
    for node in graph.node:
        if node.name in linears:
            activation, weight = linears[node.name]
            if weight not in transposed:
                module: str = weight.removesuffix(".weight")
                weight_initializers, weight_node = fp8_weight_dq(
                    module, torch.from_numpy(numpy_helper.to_array(weights[weight]).astype(np.float32)), "float16"
                )
                added.extend(weight_initializers)
                nodes.extend([weight_node, helper.make_node("Transpose", [f"{module}.w_dq"], [f"{module}.w_t"], perm=[1, 0])])
                transposed[weight] = f"{module}.w_t"
            if activation not in inputs:
                input_initializers, input_nodes, inputs[activation] = fp8_input_qdq(
                    activation, f"{weight.removesuffix('.weight')}@{node.name}.x", amax[activation] * LINEAR_INPUT_HEADROOM, "float16"
                )
                added.extend(input_initializers)
                nodes.extend(input_nodes)
            nodes.append(helper.make_node("MatMul", [inputs[activation], transposed[weight]], list(node.output), name=node.name))
        elif node.name in cores:
            dequantized: list[str] = []
            for slot, operand in enumerate(cores[node.name]):
                limit: float = PROBABILITY_AMAX if operand in targets.probabilities else amax[operand] * LINEAR_INPUT_HEADROOM
                operand_initializers, operand_nodes, name = fp8_input_qdq(operand, f"{node.name}.in{slot}", limit, "bfloat16")
                added.extend(operand_initializers)
                nodes.extend(operand_nodes)
                dequantized.append(name)
            nodes.append(helper.make_node("MatMul", dequantized, list(node.output), name=node.name))
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(added)
    prune_dead(graph)
    for opset in quantized.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = FP8_OPSET
    return quantized


def fp8_source(fp8_graph: Path) -> str | None:
    """The prepared matcher an FP8 graph was calibrated from (`SOURCE_PROPERTY`), or None when the graph is missing.

    Args:
        fp8_graph: The calibrated FP8 matcher.

    Returns:
        The prepared matcher's file name, or None.
    """
    if not fp8_graph.is_file():
        return None
    return next((p.value for p in onnx.load(fp8_graph, load_external_data=False).metadata_props if p.key == SOURCE_PROPERTY), None)


def check_fp8_graph(model_dir: Path, matcher_graph: Path, options: FlexGlueOptions) -> Path:
    """Find the FP8 graph for these thresholds; refuse one that is missing or was calibrated from another prepared matcher.

    Args:
        model_dir: FlexGlue's graphs.
        matcher_graph: The prepared matcher this run would quantize (its name carries the thresholds and the rewrite).
        options: The thresholds folded into `matcher_graph`.

    Returns:
        The FP8 graph.

    Raises:
        FileNotFoundError: When the FP8 graph is missing or stale, naming the calibration command for these thresholds.
    """
    fp8_graph: Path = model_dir / options.fp8_file()
    command: str = f"{FLEXGLUE_FP8_COMMAND} --min-covisibility {options.min_covisibility:g}"
    if not fp8_graph.is_file():
        raise FileNotFoundError(f"No FP8 FlexGlue matcher at {fp8_graph}: calibrate it once with `{command}`")
    source: str | None = fp8_source(fp8_graph)
    if source != matcher_graph.name:
        raise FileNotFoundError(f"{fp8_graph} was calibrated from {source!r}, not {matcher_graph.name}: recalibrate with `{command}`")
    return fp8_graph


def calibration_pairs(image_dirs: Sequence[Sequence[Path]], count: int = CALIBRATION_PAIRS, seed: int = 0) -> list[tuple[Path, Path]]:
    """Pairs that span what the matcher meets: neighbours, frames ten apart, and unrelated frames across sequences.

    Args:
        image_dirs: Each sequence's frames, in capture order (at least two sequences, at least 12 frames each).
        count: Pairs to draw, in turn: neighbours, ten apart, then an unrelated frame of another sequence.
        seed: Seed of the draw.

    Returns:
        `count` image pairs.

    Raises:
        ValueError: With fewer than two sequences or a sequence shorter than 12 frames.
    """
    if len(image_dirs) < 2 or any(len(frames) < 12 for frames in image_dirs):
        raise ValueError(f"calibration needs at least two sequences of at least 12 frames, got lengths {[len(f) for f in image_dirs]}")
    draw: random.Random = random.Random(seed)
    pairs: list[tuple[Path, Path]] = []
    for index in range(count):
        sequence: int = index % len(image_dirs)
        frames: Sequence[Path] = image_dirs[sequence]
        first: int = draw.randrange(len(frames) - 11)
        if index % 3 == 0:
            pairs.append((frames[first], frames[first + 1]))
        elif index % 3 == 1:
            pairs.append((frames[first], frames[first + 10]))
        else:
            other: Sequence[Path] = image_dirs[(sequence + 1 + (index // len(image_dirs)) % (len(image_dirs) - 1)) % len(image_dirs)]
            pairs.append((frames[first], other[draw.randrange(len(other))]))
    return pairs
