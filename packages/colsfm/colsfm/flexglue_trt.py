"""FlexGlue on TensorRT: COLMAP's FlexGlue graphs (cvg/FlexGlue v0.1) run from colsfm's own matching loop.

COLMAP runs FlexGlue through ONNX Runtime (`colsfm.matching`'s `flexglue_ort` backend, kept as the parity reference).
This module builds the same two graphs as strongly typed TensorRT engines through trtkit and drives them pair by pair,
the way `colsfm.matching_trt` drives LightGlue+:

* **Descriptor** (`flexglue_descriptor.onnx`): image `[1, 3, 512, 512]` in [0, 1] -> dense features `[1, 256, 64, 64]`.
  Each image runs once; its features stay in an LRU on the GPU (`FlexGlueTensorRT.features`), like COLMAP's cache.
* **Matcher** (`flexglue_matcher_early_stop_fp16.onnx`): two feature maps and two keypoint sets -> `matches0`
  (mutual nearest neighbours on the regressed correspondences, already gated by confidence, distance and pair
  covisibility inside the graph) and `covisibility_score`.

TensorRT cannot parse the matcher as shipped: it uses ONNX Runtime's contrib `com.microsoft.MultiHeadAttention` and
`com.microsoft.Gelu`. `prepare_matcher_graph` rewrites both into standard ONNX ops (attention as reshape, matmul,
softmax, matmul, the pattern TensorRT fuses; Gelu as its erf form), inside every early-stop `If` branch, and folds the four scalar options
into constants, so an engine is one fixed set of options. The early-stop `If`s are inlined (`_flatten_early_stop`).

Pre- and post-processing mirror `src/colmap/feature/flexglue.cc` at the recipe's revision: the RGB image is squashed to
512x512 with torch's bilinear resize (align_corners=False, no antialias) and divided by 255; keypoints are scaled by
512/width and 512/height; both `image_size` inputs are the descriptor size; every non-negative `matches0` entry is a match.
"""

from __future__ import annotations

import tempfile
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import onnx
import onnxscript.optimizer
import pycolmap
import torch
from numpy import ndarray
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor
from torchvision.io import ImageReadMode, decode_jpeg
from trtkit.tensorrt_runtime import TensorRtDynamicRuntime

from colsfm.database import ImagePair, KeypointsXY, MatchIndices, delete_two_view_geometries, pair_inlier_counts, raw_match_counts
from colsfm.matching import FLEXGLUE_MODEL_FILES, MatchingOptions, MatchReport, PairMatchStats, verification_options
from colsfm.matching_trt import FeatureCache
from colsfm.pairs import write_pair_list
from colsfm.tensorrt_runtime import ShapeProfile, resolve_engine

DESCRIPTOR_SIZE: Final[int] = 512
"""FlexGlue's descriptor input side; the graph is static at 512x512 (COLMAP's default)."""

FEATURE_SHAPE: Final[tuple[int, int, int, int]] = (1, 256, 64, 64)
"""Dense features per image: stride 8 at 512x512."""

MATCHER_MAX_KEYPOINTS: Final[int] = 4096
"""Upper end of the matcher's keypoint profile; RaCo-ALIKED stores at most 2048 per image."""

FEATURE_CACHE_IMAGES: Final[int] = 512
"""Dense feature maps kept on the GPU (4 MB each, 2 GB): a few hundred rig frames of sequential pairs stay resident."""

DECODE_WORKERS: Final[int] = 4
"""Threads reading image files ahead of the pair loop (decoding itself runs on the GPU)."""

DECODE_WINDOW: Final[int] = 32
"""Images read ahead of the pair loop."""

DESCRIPTOR_BATCH: Final[int] = 8
"""Images per descriptor call: the next uncached images in the order the pair loop needs them."""

MATCHER_STREAMS: Final[int] = 2
"""Matcher runtimes, one per CUDA stream, used in turn."""

DOWNLOAD_CHUNK: Final[int] = 256
"""Pairs whose matches are downloaded and written together."""

ATTENTION_HEADS_ATTRIBUTE: Final[str] = "num_heads"
"""The contrib `MultiHeadAttention` attribute that carries the head count."""


@dataclass(frozen=True, slots=True)
class FlexGlueOptions:
    """The four scalar inputs of FlexGlue's matcher, with COLMAP's defaults (`FlexGlueMatchingOptions`)."""

    filter_threshold: float = 0.1
    """Minimum match confidence (geometric mean of both directions)."""
    max_distance_px: float = 3.0
    """Keypoint-to-correspondence distance gate in the 512x512 frame; COLMAP passes infinity for a non-positive value."""
    min_covisibility: float = 0.5
    """A pair whose predicted covisibility is below this keeps no match."""
    early_stop_decay: float = 0.5
    """Early stop of non-covisible pairs after layers 2, 4, 6 and 8; 1 disables it."""

    def tag(self) -> str:
        """A file-name-safe key for the folded graph and its engine.

        Returns:
            E.g. `ft0.1_md3_mc0.5_es0.5`.
        """
        return f"ft{self.filter_threshold:g}_md{self.max_distance_px:g}_mc{self.min_covisibility:g}_es{self.early_stop_decay:g}"


DESCRIPTOR_PROFILE: Final[dict[str, ShapeProfile]] = {}
"""The descriptor graph is fully static, so it needs no optimisation profile."""


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
        helper.make_node("MatMul", [f"{p}_qs", f"{p}_kt"], [f"{p}_logits"]),
        helper.make_node("Softmax", [f"{p}_logits"], [f"{p}_weights"], axis=-1),
        helper.make_node("MatMul", [f"{p}_weights", f"{p}_v"], [f"{p}_context4"]),
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


def _flatten_early_stop(graph: onnx.GraphProto) -> int:
    """Inline the continue (`else`) branch of every early-stop `If`, recursively, so the graph runs every layer.

    TensorRT 11.2 parses the conditionals but its Myelin compiler cannot build them with dynamic keypoint counts
    ("Could not find any implementation for node {ForeignNode[...early_stop_layer2_if...]}"). Early stop only skips the
    remaining layers of a pair that is not covisible; the final layer's covisibility gate still rejects that pair, so the
    flattened graph returns the same matches and spends a little more time on rejected pairs.

    Args:
        graph: Edited in place.

    Returns:
        How many `If` nodes were inlined.
    """
    inlined: int = 0
    body: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type != "If":
            body.append(node)
            continue
        branch: onnx.GraphProto = next(a.g for a in node.attribute if a.name == "else_branch")
        inlined += 1 + _flatten_early_stop(branch)
        body.extend(branch.node)
        # The later layers' weights are the branch's own initializers.
        graph.initializer.extend(branch.initializer)
        body.extend(helper.make_node("Identity", [inner.name], [outer]) for inner, outer in zip(branch.output, node.output, strict=True))
    del graph.node[:]
    graph.node.extend(body)
    return inlined


def prepare_matcher_graph(source: Path, destination: Path, options: FlexGlueOptions) -> Path:
    """Write the TensorRT-parsable matcher: contrib ops rewritten, scalar options folded into constants.

    Kept when the destination already exists (its name carries `options.tag()`).

    Args:
        source: COLMAP's matcher graph (the fp16 variant).
        destination: Where the rewritten graph goes.
        options: The four scalar options to fold.

    Returns:
        The destination.

    Raises:
        ValueError: When the graph has a contrib node this rewrite does not know.
    """
    if destination.is_file():
        return destination
    model: onnx.ModelProto = onnx.load(source)
    attention, gelu = _rewrite(model.graph)
    inlined: int = _flatten_early_stop(model.graph)
    max_distance: float = options.max_distance_px if options.max_distance_px > 0 else float("inf")
    folded: dict[str, float] = {
        "filter_threshold": options.filter_threshold,
        "max_distance_px": max_distance,
        "min_covisibility": options.min_covisibility,
        "early_stop_decay": options.early_stop_decay,
    }
    kept_inputs = [value for value in model.graph.input if value.name not in folded]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)
    model.graph.initializer.extend(numpy_helper.from_array(np.array(v, dtype=np.float32), name=k) for k, v in folded.items())
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", 18))
    # TensorRT folds an operand such as ReduceMean's axes only when it is already a constant; the exporter left some as
    # Reshape(Constant, Constant). onnxscript's optimizer folds those, in the If branches too.
    model = onnxscript.optimizer.optimize(model)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = destination.with_suffix(".onnx.part")
    onnx.save(model, partial)
    partial.rename(destination)
    print(f"[colsfm] flexglue matcher graph: {attention} attention and {gelu} Gelu nodes rewritten, {inlined} early-stop branches inlined, options {options.tag()}")
    return destination


def prepare_descriptor_graph(source: Path, destination: Path) -> Path:
    """Write the descriptor with a symbolic batch dimension, so one engine call encodes several images.

    The shipped graph is static at batch 1; nothing inside it depends on that (a batch of 8 matches eight batch-1 runs
    to TF32 rounding), so only the input and output declarations change. Kept when the destination exists.

    Args:
        source: COLMAP's descriptor graph.
        destination: Where the batch-dynamic graph goes.

    Returns:
        The destination.
    """
    if destination.is_file():
        return destination
    model: onnx.ModelProto = onnx.load(source)
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = destination.with_suffix(".onnx.part")
    onnx.save(model, partial)
    partial.rename(destination)
    return destination


def descriptor_profile(max_batch: int = DESCRIPTOR_BATCH) -> dict[str, ShapeProfile]:
    """The descriptor's batch profile at the fixed 512x512 input.

    Args:
        max_batch: Largest batch.

    Returns:
        One `(minimum, optimal, maximum)` shape for the `image` binding.
    """
    image: tuple[int, int, int] = (3, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)
    return {"image": ((1, *image), (max_batch, *image), (max_batch, *image))}


def _decode_on_gpu(payload: bytes, path: Path, device: torch.device) -> Tensor:
    """Decode one image to a `[3, height, width]` uint8 RGB tensor on the GPU.

    JPEGs go through nvjpeg (`torchvision.io.decode_jpeg` on the device); anything else is decoded by OpenCV and
    uploaded once.

    Args:
        payload: The file's bytes (read by a prefetch thread).
        path: The file, for the error message and the fallback.
        device: The CUDA device.

    Returns:
        The RGB image on the device.

    Raises:
        FileNotFoundError: When the image cannot be decoded.
    """
    if payload[:2] == b"\xff\xd8":
        decoded: Tensor | list[Tensor] = decode_jpeg(torch.frombuffer(bytearray(payload), dtype=torch.uint8), mode=ImageReadMode.RGB, device=device)
        # One encoded tensor in, one image out; the list form is for batched input.
        return decoded if isinstance(decoded, Tensor) else decoded[0]
    bgr: ndarray | None = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"FlexGlue reads image pixels and could not decode {path}")
    return torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).to(device).permute(2, 0, 1)


@dataclass(frozen=True, slots=True)
class DenseImage:
    """What the matcher needs of one image, resident on the GPU."""

    features: Tensor
    """Dense features `[1, 256, 64, 64]`."""
    keypoints: Tensor
    """Keypoints `[1, n, 2]` scaled into the 512x512 descriptor frame."""
    num_keypoints: int
    """`n`; 0 when the image has none."""


class FlexGlueTensorRT:
    """FlexGlue's engines with everything between them on the GPU.

    * Images: bytes read ahead on a thread pool, JPEG-decoded by nvjpeg, resized on the GPU with torch's bilinear
      interpolation (the reference COLMAP's CPU resize mirrors) and encoded `DESCRIPTOR_BATCH` at a time by the fp16
      descriptor engine.
    * Features and scaled keypoints stay in a GPU LRU (`DenseImage`); one upload per image.
    * The matcher runs on two CUDA streams in turn (a pair's tail overlaps the next pair's head: 9.7 vs 11.3 ms per
      pair on the 5090) and its matches stay on the GPU until `flush` downloads a whole chunk at once.
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
        self.pending: list[tuple[ImagePair, Tensor]] = []
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
        features: Tensor = self.descriptor({"image": (torch.cat(resized) / 255.0).contiguous()})["features"].clone()
        self.descriptor_runs += 1
        self.descriptor_images += len(batch)
        for index, (image_id, _, keypoints) in enumerate(batch):
            scaled: Tensor = torch.from_numpy(keypoints * np.asarray(scales[index], dtype=np.float32)).to(self.device)[None]
            self.images[image_id] = DenseImage(features=features[index : index + 1], keypoints=scaled, num_keypoints=len(keypoints))
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
        if first.num_keypoints == 0 or second.num_keypoints == 0:
            self.pending.append((pair, torch.full((first.num_keypoints,), -1, dtype=torch.int64, device=self.device)))
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
            self.pending.append((pair, outputs["matches0"][0].clone()))

    def flush(self) -> list[tuple[ImagePair, MatchIndices]]:
        """Download every pending pair's matches in one transfer.

        Returns:
            Int64 `[num_matches, 2]` keypoint index pairs per pair, in launch order.
        """
        if not self.pending:
            return []
        current: torch.cuda.Stream = torch.cuda.current_stream(self.device)
        for stream in self.streams:
            current.wait_stream(stream)
        lengths: list[int] = [len(matches0) for _, matches0 in self.pending]
        everything: ndarray = torch.cat([matches0 for _, matches0 in self.pending]).cpu().numpy()
        results: list[tuple[ImagePair, MatchIndices]] = []
        offset: int = 0
        for (pair, _), length in zip(self.pending, lengths, strict=True):
            matches0: ndarray = everything[offset : offset + length]
            offset += length
            rows: ndarray = np.flatnonzero(matches0 >= 0)
            results.append((pair, np.stack([rows, matches0[rows]], axis=1).astype(np.int64)))
        self.pending = []
        return results


def first_use_order(pairs: Sequence[ImagePair]) -> list[int]:
    """Images in the order the pair loop first needs them, for the decode prefetch and the descriptor batches.

    Args:
        pairs: The pair list.

    Returns:
        Each image id once.
    """
    return list(dict.fromkeys(image_id for pair in pairs for image_id in pair))


def match_pairs_flexglue(database_path: Path, pairs: Sequence[ImagePair], options: MatchingOptions) -> MatchReport:
    """Match and verify a pair list with FlexGlue on TensorRT, writing both tables into the database.

    The same contract as `colsfm.matching_trt.match_pairs_tensorrt`: raw matches are written per pair, then COLMAP's
    batch `verify_matches` runs over the same pair file.

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
    descriptor_graph: Path = prepare_descriptor_graph(models / FLEXGLUE_MODEL_FILES["descriptor_model_path"], models / "trt" / "flexglue_descriptor_batch.onnx")
    descriptor_engine: Path = resolve_engine(descriptor_graph, descriptor_profile(), precision="fp16-trunk")
    matcher_graph: Path = prepare_matcher_graph(
        models / FLEXGLUE_MODEL_FILES["matcher_model_path_fp16"], models / "trt" / f"flexglue_matcher_fp16_{flexglue_options.tag()}.onnx", flexglue_options
    )
    matcher_engine: Path = resolve_engine(matcher_graph, matcher_profile(), precision="fp32")
    image_root: Path = options.image_dir
    started: float = time.perf_counter()
    with pycolmap.Database.open(database_path) as database:
        names: dict[int, str] = {image.image_id: image.name for image in database.read_all_images()}
        keypoints: FeatureCache = FeatureCache(database)
        flexglue: FlexGlueTensorRT = FlexGlueTensorRT(descriptor_engine, matcher_engine)
        order: list[int] = first_use_order(pairs)
        position: dict[int, int] = {image_id: index for index, image_id in enumerate(order)}
        with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
            reads: dict[int, Future[bytes]] = {}

            def read(image_id: int) -> bytes:
                future: Future[bytes] | None = reads.pop(image_id, None)
                return future.result() if future is not None else (image_root / names[image_id]).read_bytes()

            def prefetch(start: int) -> None:
                for image_id in order[start : start + DECODE_WINDOW]:
                    if image_id not in reads and image_id not in flexglue.images:
                        reads[image_id] = pool.submit((image_root / names[image_id]).read_bytes)

            def ensure(image_id: int) -> None:
                if image_id in flexglue.images:
                    return
                start: int = position[image_id]
                batch_ids: list[int] = [image_id] + [
                    candidate for candidate in order[start + 1 : start + 4 * DESCRIPTOR_BATCH] if candidate not in flexglue.images
                ][: DESCRIPTOR_BATCH - 1]
                prefetch(start + len(batch_ids))
                flexglue.encode(
                    [(candidate, _decode_on_gpu(read(candidate), image_root / names[candidate], flexglue.device), keypoints[candidate][0]) for candidate in batch_ids]
                )

            prefetch(0)
            for index, (image_id1, image_id2) in enumerate(pairs):
                ensure(image_id1)
                ensure(image_id2)
                flexglue.enqueue((image_id1, image_id2))
                if (index + 1) % DOWNLOAD_CHUNK == 0:
                    for (first, second), matches in flexglue.flush():
                        database.write_matches(first, second, matches.astype(np.uint32))
            for (first, second), matches in flexglue.flush():
                database.write_matches(first, second, matches.astype(np.uint32))
            for future in reads.values():
                future.cancel()
    matching_seconds: float = time.perf_counter() - started

    raw: dict[ImagePair, int] = raw_match_counts(database_path, pairs)
    delete_two_view_geometries(database_path, pairs)
    with tempfile.TemporaryDirectory(prefix="colsfm-pairs-") as scratch:
        pair_list_path: Path = Path(scratch) / "pairs.txt"
        write_pair_list(pair_list_path, pairs, database_path)
        pycolmap.verify_matches(database_path, pair_list_path, verification_options(options))
    elapsed_seconds: float = time.perf_counter() - started
    inliers: dict[ImagePair, int] = pair_inlier_counts(database_path, pairs)
    report: MatchReport = MatchReport(
        pair_stats={pair: PairMatchStats(raw_matches=raw[pair], inlier_matches=inliers[pair]) for pair in pairs},
        device="cuda",
        elapsed_seconds=elapsed_seconds,
    )
    print(
        f"colsfm.matching[flexglue-trt]: {report.summary()} | matching {matching_seconds:.1f} s for {len(pairs)} pairs, "
        f"{flexglue.descriptor_images} images in {flexglue.descriptor_runs} descriptor batches"
    )
    return report
