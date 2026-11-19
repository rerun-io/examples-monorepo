"""FlexGlue on TensorRT: COLMAP's FlexGlue graphs (cvg/FlexGlue v0.1) run from colsfm's own matching loop.

COLMAP runs FlexGlue through ONNX Runtime (`colsfm.matching`'s `flexglue_ort` backend, kept as the parity reference).
This module builds the same two graphs as strongly typed TensorRT engines through trtkit and drives them pair by pair,
the way `colsfm.matching_trt` drives LightGlue+:

* **Descriptor** (`flexglue_descriptor.onnx`): image `[1, 3, 512, 512]` in [0, 1] -> dense features `[1, 256, 64, 64]`.
  Each image runs once; its features stay in an LRU on the GPU (`FlexGlueTensorRT.features`), like COLMAP's cache.
* **Matcher** (`flexglue_matcher_early_stop_fp16.onnx`): two feature maps and two keypoint sets -> `matches0`
  (mutual nearest neighbours on the regressed correspondences, already gated by confidence, distance and pair
  covisibility inside the graph) and `covisibility_score`. Its transformer runs in FP8 (`colsfm.flexglue_fp8`),
  from a graph calibrated once per machine (`colsfm.apis.flexglue_fp8`).

TensorRT cannot parse the matcher as shipped: it uses ONNX Runtime's contrib `com.microsoft.MultiHeadAttention` and
`com.microsoft.Gelu`. `prepare_matcher_graph` rewrites both into standard ONNX ops (attention as reshape, matmul,
softmax, matmul, the pattern TensorRT fuses; Gelu as its erf form), inside every early-stop `If` branch, and folds the four scalar options
into constants, so an engine is one fixed set of options. The early-stop `If`s become `Where` selects (`_lower_early_stop`).

Pre- and post-processing mirror `src/colmap/feature/flexglue.cc` at the recipe's revision: the RGB image is squashed to
512x512 with torch's bilinear resize (align_corners=False, no antialias) and divided by 255; keypoints are scaled by
512/width and 512/height; both `image_size` inputs are the descriptor size; every non-negative `matches0` entry is a match.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import numpy as np
import onnx
import onnxscript.optimizer
import pycolmap
import torch
from numpy import ndarray
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor
from trtkit.artifacts import build_once
from trtkit.tensorrt_runtime import TensorRtDynamicRuntime
from trtkit.trt_builder import onnx_content_hash

from colsfm.database import ImagePair, KeypointsXY
from colsfm.flexglue_fp8 import check_fp8_graph
from colsfm.frames import FrameSource, open_frame_source
from colsfm.matching import (
    FLEXGLUE_MODEL_FILES,
    FlexGlueOptions,
    MatchingOptions,
    MatchReport,
    PairMatchStats,
    require_no_geometries,
    spatial_cap,
    verification_options,
    verifier_inputs,
    write_and_count,
)
from colsfm.tensorrt_runtime import ShapeProfile, resolve_engine
from colsfm.verification import ChunkedVerifier, PairToVerify

DESCRIPTOR_SIZE: Final[int] = 512
"""FlexGlue's descriptor input side; the graph is static at 512x512 (COLMAP's default)."""

MATCHER_MAX_KEYPOINTS: Final[int] = 4096
"""Upper end of the matcher's keypoint profile; RaCo-ALIKED stores at most 2048 per image."""

FEATURE_CACHE_IMAGES: Final[int] = 512
"""Dense feature maps kept on the GPU (4 MB each, 2 GB): a few hundred rig frames of sequential pairs stay resident."""

DESCRIPTOR_BATCH: Final[int] = 8
"""Images per descriptor call: the next uncached images in the order the pair loop needs them."""

MATCHER_STREAMS: Final[int] = 2
"""Matcher runtimes, one per CUDA stream, used in turn."""

GRAPH_REWRITE_VERSION: Final[int] = 3
"""Bumped whenever `prepare_matcher_graph` / `prepare_descriptor_graph` change what they write; part of the cached file name."""

ATTENTION_HEADS_ATTRIBUTE: Final[str] = "num_heads"
"""The contrib `MultiHeadAttention` attribute that carries the head count."""


def matcher_profile(max_keypoints: int = MATCHER_MAX_KEYPOINTS) -> dict[str, ShapeProfile]:
    """The matcher's optimisation profile over both keypoint counts.

    Args:
        max_keypoints: Largest keypoint count either image may carry.

    Returns:
        One `(minimum, optimal, maximum)` shape per keypoint binding.
    """
    return {name: ((1, 1, 2), (1, 2048, 2), (1, max_keypoints, 2)) for name in ("kpts0", "kpts1")}


def _constant(name: str, value: ndarray) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value, name=name))


def _attention_nodes(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    """Standard-op replacement for one `com.microsoft.MultiHeadAttention(query, key, value)` with 3-D inputs.

    `[batch, sequence, hidden]` inputs, `num_heads` heads of `hidden / num_heads`, scale `1 / sqrt(head_size)`, no mask
    and no bias: the only form the FlexGlue graph uses (checked by the caller). MatMul -> Softmax -> MatMul in one dtype,
    with no cast between, is the pattern TensorRT fuses into one attention kernel; a cast to fp32 around the softmax
    broke the fusion and made the matcher 2x slower.

    That core runs in bf16, the fp16 matcher's other tensors stay fp16. TensorRT 11.2's fused fp16 attention overflows
    on some pairs: the covisibility comes back NaN, which fails every gate, so the pair keeps no match where ONNX
    Runtime's contrib kernel keeps up to 137 (74 of 688 random MGO07 pairs that run every layer). bf16 has fp32's range;
    it heals all of them for 12.1 against 10.0 ms per 2048x2048 pair, where an fp32 core costs 60 ms. Breaking single
    fusions with extra engine outputs only moved the overflow to another layer.
    """
    heads: int = next(helper.get_attribute_value(a) for a in node.attribute if a.name == ATTENTION_HEADS_ATTRIBUTE)
    query, key, value = node.input
    (output,) = node.output
    p = f"{output}__mha"
    return [
        _constant(f"{p}_split", np.array([0, 0, heads, -1], dtype=np.int64)),
        _constant(f"{p}_merge", np.array([0, 0, -1], dtype=np.int64)),
        helper.make_node("Reshape", [query, f"{p}_split"], [f"{p}_q4"]),
        helper.make_node("Reshape", [key, f"{p}_split"], [f"{p}_k4"]),
        helper.make_node("Reshape", [value, f"{p}_split"], [f"{p}_v4"]),
        helper.make_node("Transpose", [f"{p}_q4"], [f"{p}_q"], perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", [f"{p}_k4"], [f"{p}_kt"], perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", [f"{p}_v4"], [f"{p}_v"], perm=[0, 2, 1, 3]),
        # scale = 1 / sqrt(head_size), from the runtime shape so the rewrite needs no shape inference
        helper.make_node("Shape", [f"{p}_q4"], [f"{p}_q4_shape"], start=3, end=4),
        helper.make_node("Cast", [f"{p}_q4_shape"], [f"{p}_head_size"], to=TensorProto.FLOAT),
        helper.make_node("Sqrt", [f"{p}_head_size"], [f"{p}_sqrt"]),
        helper.make_node("Reciprocal", [f"{p}_sqrt"], [f"{p}_scale_f"]),
        helper.make_node("CastLike", [f"{p}_scale_f", query], [f"{p}_scale"]),
        helper.make_node("Mul", [f"{p}_q", f"{p}_scale"], [f"{p}_qs"]),
        helper.make_node("Cast", [f"{p}_qs"], [f"{p}_qs_bf16"], to=TensorProto.BFLOAT16),
        helper.make_node("Cast", [f"{p}_kt"], [f"{p}_kt_bf16"], to=TensorProto.BFLOAT16),
        helper.make_node("Cast", [f"{p}_v"], [f"{p}_v_bf16"], to=TensorProto.BFLOAT16),
        helper.make_node("MatMul", [f"{p}_qs_bf16", f"{p}_kt_bf16"], [f"{p}_logits"]),
        helper.make_node("Softmax", [f"{p}_logits"], [f"{p}_weights"], axis=-1),
        helper.make_node("MatMul", [f"{p}_weights", f"{p}_v_bf16"], [f"{p}_context4_bf16"]),
        helper.make_node("CastLike", [f"{p}_context4_bf16", query], [f"{p}_context4"]),
        helper.make_node("Transpose", [f"{p}_context4"], [f"{p}_context"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", [f"{p}_context", f"{p}_merge"], [output]),
    ]


def _gelu_nodes(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    """`com.microsoft.Gelu(x)` as `0.5 * x * (1 + erf(x / sqrt(2)))`, its exact (erf) definition."""
    (x,) = node.input
    (output,) = node.output
    p = f"{output}__gelu"
    return [
        _constant(f"{p}_rsqrt2_f", np.array(1.0 / np.sqrt(2.0), dtype=np.float32)),
        _constant(f"{p}_half_f", np.array(0.5, dtype=np.float32)),
        _constant(f"{p}_one_f", np.array(1.0, dtype=np.float32)),
        helper.make_node("CastLike", [f"{p}_rsqrt2_f", x], [f"{p}_rsqrt2"]),
        helper.make_node("CastLike", [f"{p}_half_f", x], [f"{p}_half"]),
        helper.make_node("CastLike", [f"{p}_one_f", x], [f"{p}_one"]),
        helper.make_node("Mul", [x, f"{p}_rsqrt2"], [f"{p}_scaled"]),
        helper.make_node("Erf", [f"{p}_scaled"], [f"{p}_erf"]),
        helper.make_node("Add", [f"{p}_erf", f"{p}_one"], [f"{p}_onep"]),
        helper.make_node("Mul", [x, f"{p}_half"], [f"{p}_halfx"]),
        helper.make_node("Mul", [f"{p}_halfx", f"{p}_onep"], [output]),
    ]


def _rewrite(graph: onnx.GraphProto) -> tuple[int, int]:
    """Replace every contrib attention and Gelu node in a graph and, recursively, in its `If` branches.

    Args:
        graph: Edited in place.

    Returns:
        How many attention and Gelu nodes were replaced.

    Raises:
        ValueError: On a contrib node of another kind, or an attention node with a mask or bias input.
    """
    replaced: list[onnx.NodeProto] = []
    attention = gelu = 0
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                a, g = _rewrite(attribute.g)
                attention, gelu = attention + a, gelu + g
        if node.domain != "com.microsoft":
            replaced.append(node)
        elif node.op_type == "MultiHeadAttention":
            if len(node.input) != 3:
                raise ValueError(f"{node.name}: MultiHeadAttention with {len(node.input)} inputs; only query, key, value are rewritten")
            replaced.extend(_attention_nodes(node))
            attention += 1
        elif node.op_type == "Gelu":
            replaced.extend(_gelu_nodes(node))
            gelu += 1
        else:
            raise ValueError(f"{node.name}: no rewrite for com.microsoft.{node.op_type}")
    del graph.node[:]
    graph.node.extend(replaced)
    return attention, gelu


EARLY_STOP_LAYERS: Final[tuple[int, ...]] = (2, 4, 6, 8)
"""The pinned matcher's early-stop checks, outermost first: `early_stop_layer<k>_if`, each nested in the previous one's else."""

EARLY_STOP_THEN_OPS: Final[list[str]] = ["Shape", "ConstantOfShape", "ConstantOfShape", "Identity"]
"""What every early-stop `then` branch computes: no matches, zero scores, and the covisibility score so far."""


def _lower_early_stop(graph: onnx.GraphProto, depth: int = 0) -> int:
    """Turn the pinned graph's early-stop conditionals into tensor selects that keep their decisions.

    TensorRT 11.2 parses these `If`s but its Myelin compiler cannot build them with dynamic keypoint counts ("Could not
    find any implementation for node {ForeignNode[...early_stop_layer2_if...]}"). Each one becomes both branches inlined
    plus `Where(stop, then_output, else_output)` per output: the original stop predicate (with its `early_stop_decay`)
    still decides, and a stopped pair still returns the stop branch's no-match outputs, so the matches equal the
    conditional graph's. The later layers now always execute, so a stopped pair costs the full matcher.

    Only the known structure is accepted: one `early_stop_layer<k>_if` per nesting level, in `EARLY_STOP_LAYERS` order,
    each with the `EARLY_STOP_THEN_OPS` stop branch. Any other conditional is refused rather than lowered.

    Args:
        graph: Edited in place.
        depth: Nesting level, i.e. the index into `EARLY_STOP_LAYERS` this graph's conditional must have.

    Returns:
        How many conditionals were lowered.

    Raises:
        ValueError: On a conditional that is not the expected early stop.
    """
    lowered: int = 0
    body: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type != "If":
            body.append(node)
            continue
        expected: str | None = f"early_stop_layer{EARLY_STOP_LAYERS[depth]}_if" if depth < len(EARLY_STOP_LAYERS) else None
        then_branch: onnx.GraphProto = next(a.g for a in node.attribute if a.name == "then_branch")
        else_branch: onnx.GraphProto = next(a.g for a in node.attribute if a.name == "else_branch")
        if node.name != expected or [inner.op_type for inner in then_branch.node] != EARLY_STOP_THEN_OPS:
            raise ValueError(f"unexpected conditional {node.name!r} at depth {depth}: only the pinned early-stop structure is lowered")
        lowered += 1 + _lower_early_stop(else_branch, depth + 1)
        body.extend(then_branch.node)
        body.extend(else_branch.node)
        # Branch-local initializers (the later layers' weights live in the else branches).
        graph.initializer.extend(then_branch.initializer)
        graph.initializer.extend(else_branch.initializer)
        body.extend(
            helper.make_node("Where", [node.input[0], stopped.name, continued.name], [output], name=f"{node.name}_select_{index}")
            for index, (stopped, continued, output) in enumerate(zip(then_branch.output, else_branch.output, node.output, strict=True))
        )
    del graph.node[:]
    graph.node.extend(body)
    return lowered


def _cached_graph(source: Path, cache_dir: Path, label: str, write: Callable[[onnx.ModelProto], onnx.ModelProto]) -> Path:
    """Rewrite a graph once per source content, rewrite version and label, publishing it atomically across processes.

    The cached name carries the source's content hash and `GRAPH_REWRITE_VERSION`, so a re-downloaded source or a
    changed rewrite cannot reuse a stale graph (whose engine trtkit would then build faithfully); `build_once` and a
    staging directory keep concurrent runs from interleaving writes. The pattern of `trtkit.onnx_precision.fp16_trunk`.

    Args:
        source: The graph to rewrite.
        cache_dir: Directory the rewritten graphs live in.
        label: What distinguishes this rewrite of the source (e.g. the folded options).
        write: The rewrite, from the loaded source to the model to save.

    Returns:
        The rewritten graph.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination: Path = cache_dir / f"{source.stem}_{onnx_content_hash(source, cache_dir=cache_dir)[:16]}_{label}_v{GRAPH_REWRITE_VERSION}.onnx"

    def build() -> None:
        with TemporaryDirectory(prefix=".flexglue-", dir=cache_dir) as work:
            staged: Path = Path(work) / destination.name
            onnx.save(write(onnx.load(source)), staged)
            staged.replace(destination)

    return build_once(destination, build)


def prepare_matcher_graph(source: Path, cache_dir: Path, options: FlexGlueOptions) -> Path:
    """The TensorRT-parsable matcher: contrib ops rewritten, early stop lowered to selects, scalar options folded into constants.

    Args:
        source: COLMAP's matcher graph (the fp16 variant).
        cache_dir: Directory the rewritten graph is cached in.
        options: The thresholds to fold.

    Returns:
        The rewritten graph.

    Raises:
        ValueError: When the graph has a contrib node this rewrite does not know.
    """

    def write(model: onnx.ModelProto) -> onnx.ModelProto:
        attention, gelu = _rewrite(model.graph)
        lowered: int = _lower_early_stop(model.graph)
        folded: dict[str, float] = {
            "filter_threshold": options.filter_threshold,
            "max_distance_px": options.max_distance_px if options.max_distance_px > 0 else float("inf"),
            "min_covisibility": options.min_covisibility,
            "early_stop_decay": options.early_stop_decay,
        }
        kept_inputs = [value for value in model.graph.input if value.name not in folded]
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)
        model.graph.initializer.extend(numpy_helper.from_array(np.array(v, dtype=np.float32), name=k) for k, v in folded.items())
        del model.opset_import[:]
        model.opset_import.append(helper.make_opsetid("", 18))
        # TensorRT folds an operand such as ReduceMean's axes only when it is already a constant; the exporter left some
        # as Reshape(Constant, Constant). onnxscript's optimizer folds those, in the If branches too.
        model = onnxscript.optimizer.optimize(model)
        onnx.checker.check_model(model)
        print(
            f"[colsfm] flexglue matcher graph: {attention} attention and {gelu} Gelu nodes rewritten, {lowered} early-stop conditionals lowered to selects, {options.tag()}"
        )
        return model

    return _cached_graph(source, cache_dir, f"fp16_{options.tag()}", write)


def prepare_descriptor_graph(source: Path, cache_dir: Path) -> Path:
    """The descriptor with a symbolic batch dimension, so one engine call encodes several images.

    The shipped graph is static at batch 1; nothing inside it depends on that (a batch of 8 matches eight batch-1 runs
    to TF32 rounding), so only the input and output declarations change.

    Args:
        source: COLMAP's descriptor graph.
        cache_dir: Directory the rewritten graph is cached in.

    Returns:
        The batch-dynamic graph.
    """

    def write(model: onnx.ModelProto) -> onnx.ModelProto:
        for value in [*model.graph.input, *model.graph.output]:
            value.type.tensor_type.shape.dim[0].dim_param = "batch"
        return model

    return _cached_graph(source, cache_dir, "batch", write)


def descriptor_profile(max_batch: int = DESCRIPTOR_BATCH) -> dict[str, ShapeProfile]:
    """The descriptor's batch profile at the fixed 512x512 input.

    Args:
        max_batch: Largest batch.

    Returns:
        One `(minimum, optimal, maximum)` shape for the `image` binding.
    """
    image: tuple[int, int, int] = (3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
    return {"image": ((1, *image), (max_batch, *image), (max_batch, *image))}


@dataclass(frozen=True, slots=True)
class DenseImage:
    """What the matcher needs of one image, resident on the GPU."""

    features: Tensor
    """Dense features `[1, 256, 64, 64]`."""
    keypoints: Tensor
    """Keypoints `[1, n, 2]` scaled into the 512x512 descriptor frame; `n` may be 0."""
    keypoints_px: KeypointsXY
    """The same keypoints in image pixels, on the host: what verification needs, kept so it reads nothing again."""


class FlexGlueTensorRT:
    """FlexGlue's engines with everything between them on the GPU.

    * Images: GPU frames from the run's `colsfm.frames.FrameSource` (NVDEC for a catalog input, nvJPEG for a folder),
      resized on the GPU with torch's bilinear interpolation (the reference COLMAP's CPU resize mirrors) and encoded
      `DESCRIPTOR_BATCH` at a time by the fp16 descriptor engine.
    * Features and scaled keypoints stay in a GPU LRU (`DenseImage`); one upload per image.
    * The matcher runs on two CUDA streams in turn (a pair's tail overlaps the next pair's head: 9.7 vs 11.3 ms per
      pair on the 5090) and its matches stay on the GPU until `flush` downloads a whole chunk at once. Each chunk then
      goes to `colsfm.verification.ChunkedVerifier`, whose CPU threads verify it while the GPU matches the next one.
    """

    def __init__(self, descriptor_engine: Path, matcher_engine: Path, *, cache_images: int = FEATURE_CACHE_IMAGES) -> None:
        """Load the descriptor engine and two matcher runtimes on the same matcher engine.

        Args:
            descriptor_engine: The batch-dynamic descriptor engine.
            matcher_engine: The matcher engine (keypoint-count profile).
            cache_images: Images kept on the GPU (about 4 MB each).
        """
        self.device: torch.device = torch.device("cuda")
        self.descriptor: TensorRtDynamicRuntime = TensorRtDynamicRuntime(descriptor_engine)
        self.matchers: tuple[TensorRtDynamicRuntime, ...] = tuple(TensorRtDynamicRuntime(matcher_engine) for _ in range(MATCHER_STREAMS))
        self.streams: tuple[torch.cuda.Stream, ...] = tuple(torch.cuda.Stream(self.device) for _ in range(MATCHER_STREAMS))
        self.cache_images: int = max(cache_images, 2 * DESCRIPTOR_BATCH)
        self.size: Tensor = torch.tensor([[DESCRIPTOR_SIZE, DESCRIPTOR_SIZE]], dtype=torch.float32, device=self.device)
        self.images: OrderedDict[int, DenseImage] = OrderedDict()
        self.pending: list[tuple[ImagePair, Tensor, Tensor, KeypointsXY, KeypointsXY]] = []
        self.nonfinite_pairs: int = 0
        """Pairs whose covisibility came back NaN or infinite: the engine miscomputed them, and they kept no match."""
        self.calls: int = 0
        self.descriptor_runs: int = 0
        self.descriptor_images: int = 0

    def encode(self, batch: Sequence[tuple[int, Tensor, KeypointsXY]]) -> None:
        """Run the descriptor over a batch of decoded images and cache features and scaled keypoints.

        Args:
            batch: `(image_id, RGB uint8 [3, h, w] on the GPU, keypoint pixels)` per image, at most `DESCRIPTOR_BATCH`.
        """
        resized: list[Tensor] = []
        scales: list[tuple[float, float]] = []
        for _, rgb, _ in batch:
            height, width = int(rgb.shape[1]), int(rgb.shape[2])
            resized.append(
                torch.nn.functional.interpolate(
                    rgb[None].float(), size=(DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), mode="bilinear", align_corners=False, antialias=False
                )
            )
            scales.append((DESCRIPTOR_SIZE / width, DESCRIPTOR_SIZE / height))
        features: Tensor = self.descriptor({"image": torch.cat(resized).div_(255.0)})["features"]
        self.descriptor_runs += 1
        self.descriptor_images += len(batch)
        for index, (image_id, _, keypoints) in enumerate(batch):
            scaled: Tensor = torch.from_numpy(keypoints * np.asarray(scales[index], dtype=np.float32)).to(self.device)[None]
            # One clone per image, so evicting an image frees it (a view would pin the whole batch).
            self.images[image_id] = DenseImage(features=features[index : index + 1].clone(), keypoints=scaled, keypoints_px=keypoints)
            self.images.move_to_end(image_id)
        while len(self.images) > self.cache_images:
            self.images.popitem(last=False)

    def enqueue(self, pair: ImagePair) -> None:
        """Launch the matcher for one pair whose images are cached; its matches stay on the GPU until `flush`.

        Args:
            pair: The image pair, first image first.
        """
        first: DenseImage = self.images[pair[0]]
        second: DenseImage = self.images[pair[1]]
        self.images.move_to_end(pair[0])
        self.images.move_to_end(pair[1])
        if first.keypoints.shape[1] == 0 or second.keypoints.shape[1] == 0:
            self.pending.append(
                (
                    pair,
                    torch.full((int(first.keypoints.shape[1]),), -1, dtype=torch.int64, device=self.device),
                    self.size.new_zeros(1),
                    first.keypoints_px,
                    second.keypoints_px,
                )
            )
            return
        slot: int = self.calls % MATCHER_STREAMS
        self.calls += 1
        stream: torch.cuda.Stream = self.streams[slot]
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for tensor in (first.features, first.keypoints, second.features, second.keypoints):
                # The cache may drop these while this stream still reads them; keep the allocator from reusing them early.
                tensor.record_stream(stream)
            outputs: dict[str, Tensor] = self.matchers[slot](
                {
                    "features0": first.features,
                    "features1": second.features,
                    "kpts0": first.keypoints,
                    "kpts1": second.keypoints,
                    "image_size0": self.size,
                    "image_size1": self.size,
                }
            )
            self.pending.append(
                (pair, outputs["matches0"][0].clone(), outputs["covisibility_score"].reshape(1).float(), first.keypoints_px, second.keypoints_px)
            )

    def flush(self) -> list[PairToVerify]:
        """Download every pending pair's matches in one transfer.

        Returns:
            Each pair with its int64 `[num_matches, 2]` keypoint index pairs and both images' keypoint pixels, in launch
            order: what the verifier needs, with nothing to read back from the database.
        """
        if not self.pending:
            return []
        current: torch.cuda.Stream = torch.cuda.current_stream(self.device)
        for stream in self.streams:
            current.wait_stream(stream)
        lengths: list[int] = [len(matches0) for _, matches0, _, _, _ in self.pending]
        everything: ndarray = torch.cat([matches0 for _, matches0, _, _, _ in self.pending]).cpu().numpy()
        self.nonfinite_pairs += int((~torch.isfinite(torch.cat([covisibility for _, _, covisibility, _, _ in self.pending]))).sum())
        results: list[PairToVerify] = []
        offset: int = 0
        for (pair, _, _, keypoints1, keypoints2), length in zip(self.pending, lengths, strict=True):
            matches0: ndarray = everything[offset : offset + length]
            offset += length
            rows: ndarray = np.flatnonzero(matches0 >= 0)
            results.append(
                PairToVerify(
                    pair=pair, matches=np.stack([rows, matches0[rows]], axis=1).astype(np.int64), keypoints1=keypoints1, keypoints2=keypoints2
                )
            )
        self.pending = []
        return results


def match_pairs_flexglue(database_path: Path, pairs: Sequence[ImagePair], options: MatchingOptions, frames: FrameSource | None = None) -> MatchReport:
    """Match and verify a pair list with FlexGlue on TensorRT, writing both tables into the database.

    The same contract as `colsfm.matching_trt.match_pairs_tensorrt` (both tables filled), but streamed: each downloaded
    chunk is verified on `colsfm.verification`'s worker threads while the GPU matches the next one, and the chunk's raw
    matches and geometries are written together.

    Args:
        database_path: An existing COLMAP database whose images carry keypoints.
        pairs: Image-id pairs, normalised and deduplicated.
        options: Matching settings; `image_dir`, `flexglue_model_dir`, `flexglue_min_covisibility` and the
            verification fields are read.

    Returns:
        Per-pair raw and inlier counts and the wall time.

    Raises:
        FileNotFoundError: When the database or a graph is missing.
        ValueError: When `options.image_dir` is unset.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if options.image_dir is None:
        raise ValueError("the flexglue matcher reads image pixels: MatchingOptions.image_dir is required")
    if not pairs:
        return MatchReport(device="cuda")
    flexglue_options: FlexGlueOptions = FlexGlueOptions(min_covisibility=options.flexglue_min_covisibility)
    models: Path = options.flexglue_model_dir
    matcher_graph: Path = prepare_matcher_graph(models / FLEXGLUE_MODEL_FILES["matcher_model_path_fp16"], models / "trt", flexglue_options)
    fp8_graph: Path = check_fp8_graph(models, matcher_graph, flexglue_options)
    descriptor_graph: Path = prepare_descriptor_graph(models / FLEXGLUE_MODEL_FILES["descriptor_model_path"], models / "trt")
    descriptor_engine: Path = resolve_engine(descriptor_graph, descriptor_profile(), precision="fp16-trunk")
    matcher_engine: Path = resolve_engine(fp8_graph, matcher_profile(), precision="fp32")
    owned: bool = frames is None
    source: FrameSource = open_frame_source(options.image_dir) if frames is None else frames
    try:
        return _match_pairs_flexglue(database_path, pairs, options, flexglue_options, descriptor_engine, matcher_engine, source)
    finally:
        if owned:
            source.close()


def _match_pairs_flexglue(
    database_path: Path,
    pairs: Sequence[ImagePair],
    options: MatchingOptions,
    flexglue_options: FlexGlueOptions,
    descriptor_engine: Path,
    matcher_engine: Path,
    frames: FrameSource,
) -> MatchReport:
    started: float = time.perf_counter()
    require_no_geometries(database_path, pairs)
    stats: dict[ImagePair, PairMatchStats] = {}
    num_capped: int = 0
    with pycolmap.Database.open(database_path) as database:
        names: dict[int, str] = {image.image_id: image.name for image in database.read_all_images()}
        cameras, sizes = verifier_inputs(database)
        flexglue: FlexGlueTensorRT = FlexGlueTensorRT(descriptor_engine, matcher_engine)
        # Images in the order the pair loop first needs them: the descriptor batches follow it.
        order: list[int] = list(dict.fromkeys(image_id for pair in pairs for image_id in pair))
        position: dict[int, int] = {image_id: index for index, image_id in enumerate(order)}

        def ensure(image_id: int) -> None:
            if image_id in flexglue.images:
                # Refresh it: encoding the pair's other image must not evict this one before the matcher reads it.
                flexglue.images.move_to_end(image_id)
                return
            start: int = position[image_id]
            batch_ids: list[int] = [image_id] + [
                candidate for candidate in order[start + 1 : start + 4 * DESCRIPTOR_BATCH] if candidate not in flexglue.images
            ][: DESCRIPTOR_BATCH - 1]
            decoded: list[tuple[str, Tensor]] = list(frames.frames([names[candidate] for candidate in batch_ids]))
            flexglue.encode(
                [
                    (candidate, rgb, np.ascontiguousarray(database.read_keypoints(candidate)[:, :2], dtype=np.float32))
                    for candidate, (_, rgb) in zip(batch_ids, decoded, strict=True)
                ]
            )

        def downloaded_chunks() -> Iterator[list[PairToVerify]]:
            for index, (image_id1, image_id2) in enumerate(pairs):
                ensure(image_id1)
                ensure(image_id2)
                flexglue.enqueue((image_id1, image_id2))
                if (index + 1) % options.verify.verify_chunk_pairs == 0 or index + 1 == len(pairs):
                    yield flexglue.flush()

        # The verifier pulls the next chunk from the GPU only while its window has room, so a CPU that falls behind holds
        # the matcher back; each finished chunk is written (matches and geometries, one transaction) and dropped.
        with ChunkedVerifier(cameras, verification_options(options), spatial_cap(options, sizes), options.verify) as verifier:
            for finished in verifier.verified(downloaded_chunks()):
                num_capped += write_and_count(database, finished, stats, write_matches=True)
    report: MatchReport = MatchReport(pair_stats=stats, device="cuda", elapsed_seconds=time.perf_counter() - started)
    print(
        f"colsfm.matching[flexglue-trt]: {report.summary()} | matching and verification {report.elapsed_seconds:.1f} s for {len(pairs)} pairs, "
        f"{flexglue.descriptor_images} images in {flexglue.descriptor_runs} descriptor batches, spatial cap dropped "
        f"{num_capped} inlier matches, {flexglue.nonfinite_pairs} pairs with a non-finite covisibility"
    )
    return report
