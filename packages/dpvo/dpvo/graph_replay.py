"""CUDA-graph replay of DPVO's fixed-shape inference work, bounded per run.

Inside :func:`graph_replay`, :class:`~dpvo.extractor.BasicEncoder4` and :class:`~dpvo.net.Update` replay a captured
CUDA graph per input shape and precision instead of dispatching thousands of small kernels; a consumer can route its own
feature extractor through the same cache (:func:`replay_features`; MetricTrail's DINOv3). Outside the context, or for
training, gradients or CPU tensors, every forward runs eagerly.

Every replayed output is a copy (a later replay cannot overwrite a tensor the tracker kept). Models are weak keys, so a
model released by FP32 recovery takes its graph pools with it. One :class:`GraphCache` bounds every graph of a run.

Only module-level functions cross these functions' ``Callable`` parameters, never per-call closures or bound methods:
beartype (the dev environments) memoises every callable it checks against a ``Callable`` hint in a strong-keyed dict,
which would pin a closure's model (and its graph pools) for the life of the process.
"""

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar
from weakref import WeakKeyDictionary

import torch
from jaxtyping import Float, Int, Int64
from torch import Tensor, nn

from dpvo import fastba

Captured = TypeVar("Captured")
Model = TypeVar("Model", bound=nn.Module)
UpdateOutput: TypeAlias = tuple[Float[Tensor, "1 edges dim"], tuple[Float[Tensor, "1 edges 2"], Float[Tensor, "1 edges 2"], None]]
"""Updated hidden state, displacement, confidence, and the unused flow output."""
MAX_GRAPHS: int = 16
"""Graphs one run may capture across all models, shapes and precisions; further shapes run eagerly."""
UPDATE_MIN_CALLS: int = 3
"""An update shape is captured on its third call: the tracker's initialisation sees many shapes once or twice."""


@dataclass(frozen=True, slots=True)
class UpdateTopology:
    """Neighbor and aggregation indices prepared before neural inference.

    Dynamic group discovery can synchronize CUDA. Keeping it outside the
    neural forward lets that fixed-shape work run in a CUDA graph.
    """

    previous: Int64[Tensor, "edges"]
    """Previous edge of each patch, or -1 when there is none."""
    following: Int64[Tensor, "edges"]
    """Next edge of each patch, or -1 when there is none."""
    patch_groups: tuple[Int64[Tensor, "edges"], int]
    """Contiguous patch-group index of each edge and the group count."""
    frame_groups: tuple[Int64[Tensor, "edges"], int]
    """Contiguous source/target frame-pair index of each edge and the group count."""

    @classmethod
    def from_edges(cls, ii: Int[Tensor, "edges"], jj: Int[Tensor, "edges"], kk: Int[Tensor, "edges"]) -> "UpdateTopology":
        """Prepare the same stable neighbors and sorted groups used by an ordinary update."""
        previous, following = fastba.neighbors(kk, jj)
        patches, patch_inverse = torch.unique(kk, return_inverse=True)
        pairs, pair_inverse = torch.unique(ii * 12345 + jj, return_inverse=True)
        return cls(previous, following, (patch_inverse, patches.numel()), (pair_inverse, pairs.numel()))


@dataclass(frozen=True, slots=True)
class PrecisionKey:
    """The precision policy a captured graph bakes in; a change (the FP32 recovery path) needs another graph."""

    autocast: bool
    """Whether CUDA autocast is active."""
    autocast_dtype: torch.dtype
    """Autocast target dtype."""
    matmul_tf32: bool
    """Matrix multiplication precision."""
    cudnn_tf32: bool
    """Convolution precision."""

    @staticmethod
    def current() -> "PrecisionKey":
        """The policy in force now."""
        return PrecisionKey(
            torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
            bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32),
        )


def capture(
    run: Callable[..., Captured], args: tuple[Any, ...], device: torch.device, precision: PrecisionKey
) -> tuple[torch.cuda.CUDAGraph, Captured]:
    """Warm ``run(*args)`` up three times on a side stream, then capture it; returns the graph and the outputs each
    replay overwrites. ``args`` must be persistent buffers (and the model)."""
    current: torch.cuda.Stream = torch.cuda.current_stream(device)
    warmup: torch.cuda.Stream = torch.cuda.Stream(device=device)
    warmup.wait_stream(current)
    # Autocast's temporary weight cache must not outlive allocations made outside the graph.
    with torch.autocast("cuda", enabled=precision.autocast, dtype=precision.autocast_dtype, cache_enabled=False):
        with torch.cuda.stream(warmup):
            for _ in range(3):
                run(*args)
        current.wait_stream(warmup)
        graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        # Another thread (a depth producer) may submit work during capture.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            outputs: Captured = run(*args)
    return graph, outputs


@dataclass
class GraphCache:
    """Captured graphs of one run, per model (weak keys) and key, with one bound across all of them."""

    max_graphs: int = MAX_GRAPHS
    """Graphs this cache may hold in total."""
    graphs: WeakKeyDictionary[nn.Module, dict[Hashable, Any]] = field(default_factory=WeakKeyDictionary)
    """Each model's graphs by key."""
    calls: WeakKeyDictionary[nn.Module, dict[Hashable, int]] = field(default_factory=WeakKeyDictionary)
    """Calls seen per model and key before its graph was captured."""

    def count(self) -> int:
        """Graphs held now."""
        return sum(len(graphs) for graphs in self.graphs.values())

    def get(self, model: nn.Module, key: Hashable) -> Any | None:
        """The graph of ``model`` for ``key``, or None."""
        return self.graphs.get(model, {}).get(key)

    def admit(self, model: nn.Module, key: Hashable, *, min_calls: int = 1) -> bool:
        """Count a call without a graph; True when this is the ``min_calls``-th and the cache has room to capture."""
        calls: dict[Hashable, int] = self.calls.setdefault(model, {})
        calls[key] = calls.get(key, 0) + 1
        return calls[key] >= min_calls and self.count() < self.max_graphs

    def put(self, model: nn.Module, key: Hashable, graph: object) -> None:
        """Keep ``graph`` for ``model`` and ``key``."""
        self.graphs.setdefault(model, {})[key] = graph


_ACTIVE: GraphCache | None = None


def active() -> GraphCache | None:
    """The cache of the :func:`graph_replay` context that is open, or None."""
    return _ACTIVE


@contextmanager
def graph_replay(max_graphs: int = MAX_GRAPHS) -> Iterator[GraphCache]:
    """Replay DPVO's encoders and recurrent update (and :func:`replay_features` callers) from CUDA graphs while open."""
    global _ACTIVE
    previous: GraphCache | None = _ACTIVE
    _ACTIVE = GraphCache(max_graphs)
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = previous


def replayable(model: nn.Module, tensor: Tensor) -> GraphCache | None:
    """The active cache when ``model`` may replay on ``tensor`` (CUDA, inference, no gradients), else None."""
    if _ACTIVE is None or not tensor.is_cuda or model.training or torch.is_grad_enabled():
        return None
    return _ACTIVE


@dataclass(frozen=True, slots=True)
class FeatureGraph:
    """A feature extractor's captured work for one input shape and precision."""

    image: Float[Tensor, "..."]
    """Persistent input copied before replay."""
    features: Float[Tensor, "..."]
    """Persistent output, copied out after replay."""
    graph: torch.cuda.CUDAGraph
    """The extractor's captured CUDA work."""


def replay_features(model: Model, images: Float[Tensor, "..."], eager: Callable[[Model, Tensor], Tensor]) -> Float[Tensor, "..."] | None:
    """``eager(model, images)`` from a captured graph of ``model`` (fixed weights; a graph per shape, dtype and
    precision), or None when this call must run eagerly. ``eager`` is the class's unbound method (module docstring)."""
    cache: GraphCache | None = replayable(model, images)
    if cache is None:
        return None
    precision: PrecisionKey = PrecisionKey.current()
    key: tuple[object, ...] = ("features", images.shape, images.dtype, images.device, precision)
    saved: FeatureGraph | None = cache.get(model, key)
    if saved is None:
        if not cache.admit(model, key):
            return None
        static: Float[Tensor, "..."] = images.clone()
        graph, features = capture(eager, (model, static), images.device, precision)
        saved = FeatureGraph(static, features, graph)
        cache.put(model, key, saved)
    saved.image.copy_(images)
    saved.graph.replay()
    return saved.features.clone()


@dataclass(frozen=True, slots=True)
class UpdateGraph:
    """The recurrent update's captured work for one edge count, group sizes and precision."""

    inputs: tuple[Tensor, ...]
    """Every input copied before replay, in :func:`update_inputs` order."""
    outputs: UpdateOutput
    """Outputs overwritten by each replay; callers receive copies."""
    graph: torch.cuda.CUDAGraph
    """Captured neural update kernels."""


def update_inputs(net: Tensor, inp: Tensor, corr: Tensor, ii: Tensor, jj: Tensor, kk: Tensor, topology: UpdateTopology) -> tuple[Tensor, ...]:
    """The tensors a captured update reads: the floating inputs, the edge indices, and the topology's neighbour and
    group index buffers (equal edge counts with other connectivity must not reuse stale neighbours)."""
    return (net, inp, corr, ii, jj, kk, topology.previous, topology.following, topology.patch_groups[0], topology.frame_groups[0])


def replay_update(
    model: nn.Module,
    net: Float[Tensor, "1 edges dim"],
    inp: Float[Tensor, "1 edges dim"],
    corr: Float[Tensor, "1 edges corr_dim"],
    ii: Int[Tensor, "edges"],
    jj: Int[Tensor, "edges"],
    kk: Int[Tensor, "edges"],
    topology: UpdateTopology,
    eager: Callable[..., UpdateOutput],
) -> UpdateOutput | None:
    """``eager(model, net, inp, corr, ii, jj, kk, topology)`` from a captured graph (group discovery stays outside it),
    or None when this call must run eagerly. ``eager`` is the class's unbound method (module docstring)."""
    cache: GraphCache | None = replayable(model, net)
    if cache is None:
        return None
    precision: PrecisionKey = PrecisionKey.current()
    key: tuple[object, ...] = (
        "update", (net.shape, inp.shape, corr.shape), (net.dtype, inp.dtype, corr.dtype), net.device,
        (topology.patch_groups[1], topology.frame_groups[1]), precision,
    )
    live: tuple[Tensor, ...] = update_inputs(net, inp, corr, ii, jj, kk, topology)
    saved: UpdateGraph | None = cache.get(model, key)
    if saved is None:
        if not cache.admit(model, key, min_calls=UPDATE_MIN_CALLS):
            return None
        static: tuple[Tensor, ...] = tuple(value.clone() for value in live)
        hidden, context, correlations, source, target, patch, previous, following, patch_index, frame_index = static
        static_topology = UpdateTopology(previous, following, (patch_index, topology.patch_groups[1]), (frame_index, topology.frame_groups[1]))
        graph, outputs = capture(eager, (model, hidden, context, correlations, source, target, patch, static_topology), net.device, precision)
        saved = UpdateGraph(static, outputs, graph)
        cache.put(model, key, saved)
    for buffer, value in zip(saved.inputs, live, strict=True):
        buffer.copy_(value)
    saved.graph.replay()
    hidden_out, (delta, weight, _) = saved.outputs
    return hidden_out.clone(), (delta.clone(), weight.clone(), None)
