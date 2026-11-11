"""Run DPVO's prepared recurrent update with bounded CUDA graph replay."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import TypeAlias
from weakref import WeakKeyDictionary

import torch
from dpvo.net import Update, UpdateTopology
from jaxtyping import Float, Int
from torch import Tensor

from metrictrail.feature_graphs import PrecisionKey, capture

UpdateOutput: TypeAlias = tuple[Float[Tensor, "1 edges dim"], tuple[Float[Tensor, "1 edges 2"], Float[Tensor, "1 edges 2"], None]]
"""Updated hidden state, displacement, confidence, and the unused flow output."""
MAX_UPDATE_GRAPHS: int = 8
"""Bound graph pools per run; uncommon shapes keep the eager path."""
MIN_REPLAY_CALLS: int = 3
"""Avoid capturing shapes seen only once or twice during tracker initialization."""


@dataclass(frozen=True, slots=True)
class UpdateGraphKey:
    """Tensor shapes, group sizes, and precision fixed by one model's captured update."""
    shapes: tuple[torch.Size, torch.Size, torch.Size]
    """Hidden-state, context, and correlation dimensions."""
    dtypes: tuple[torch.dtype, torch.dtype, torch.dtype]
    """Storage dtypes of the three floating inputs."""
    device: torch.device
    """Device that owns inputs and buffers."""
    groups: tuple[int, int]
    """Patch and frame-pair group counts."""
    precision: PrecisionKey
    """Precision policy captured by the graph."""


@dataclass(frozen=True, slots=True)
class UpdateGraph:
    """Persistent input buffers, prepared topology, and captured outputs."""

    inputs: tuple[Tensor, ...]
    """Every input copied before replay, in :func:`update_inputs` order: the floating inputs, the edge indices and the
    topology's neighbor and aggregation index buffers."""
    outputs: UpdateOutput
    """Outputs overwritten by each replay; callers receive copies."""
    graph: torch.cuda.CUDAGraph
    """Captured neural update kernels."""


def update_inputs(
    net: Tensor, inp: Tensor, corr: Tensor, ii: Tensor, jj: Tensor, kk: Tensor, topology: UpdateTopology
) -> tuple[Tensor, ...]:
    """The tensors a captured update reads (:attr:`UpdateGraph.inputs` order)."""
    return (net, inp, corr, ii, jj, kk, topology.previous, topology.following, topology.patch_groups[0], topology.frame_groups[0])


@contextmanager
def update_cuda_graphs() -> Iterator[None]:
    """Replay recurring update shapes during one inference run, with at most eight graph pools.

    Group discovery stays outside capture. Every topology tensor is copied before replay, so equal
    edge counts with different graph connectivity cannot reuse stale neighbors or group membership.
    Weak model keys let recovery release the old network and its graph pools before loading another.
    """
    original = Update.forward
    cache: WeakKeyDictionary[Update, dict[UpdateGraphKey, UpdateGraph]] = WeakKeyDictionary()
    seen: WeakKeyDictionary[Update, dict[UpdateGraphKey, int]] = WeakKeyDictionary()

    @wraps(original)
    def forward(
        self: Update,
        net: Float[Tensor, "1 edges dim"],
        inp: Float[Tensor, "1 edges dim"],
        corr: Float[Tensor, "1 edges corr_dim"],
        flow: None,
        ii: Int[Tensor, "edges"],
        jj: Int[Tensor, "edges"],
        kk: Int[Tensor, "edges"],
        *,
        topology: UpdateTopology | None = None,
    ) -> UpdateOutput:
        if not net.is_cuda or self.training or torch.is_grad_enabled():
            return original(self, net, inp, corr, flow, ii, jj, kk, topology=topology)
        if topology is None:
            topology = UpdateTopology.from_edges(ii, jj, kk)
        key: UpdateGraphKey = UpdateGraphKey(
            (net.shape, inp.shape, corr.shape), (net.dtype, inp.dtype, corr.dtype), net.device,
            (topology.patch_groups[1], topology.frame_groups[1]), PrecisionKey.current(),
        )
        live: tuple[Tensor, ...] = update_inputs(net, inp, corr, ii, jj, kk, topology)
        model_cache: dict[UpdateGraphKey, UpdateGraph] = cache.setdefault(self, {})
        saved: UpdateGraph | None = model_cache.get(key)
        if saved is None:
            model_seen: dict[UpdateGraphKey, int] = seen.setdefault(self, {})
            model_seen[key] = model_seen.get(key, 0) + 1
            if model_seen[key] < MIN_REPLAY_CALLS or sum(len(graphs) for graphs in cache.values()) >= MAX_UPDATE_GRAPHS:
                return original(self, net, inp, corr, flow, ii, jj, kk, topology=topology)
            static: tuple[Tensor, ...] = tuple(value.clone() for value in live)
            hidden, context, correlations, source, target, patch, previous, following, patch_index, frame_index = static
            static_topology: UpdateTopology = UpdateTopology(
                previous, following, (patch_index, topology.patch_groups[1]), (frame_index, topology.frame_groups[1])
            )
            graph, outputs = capture(
                lambda: original(self, hidden, context, correlations, None, source, target, patch, topology=static_topology),
                net.device, key.precision,
            )
            saved = UpdateGraph(static, outputs, graph)
            model_cache[key] = saved
        for buffer, value in zip(saved.inputs, live, strict=True):
            buffer.copy_(value)
        saved.graph.replay()
        hidden_out, (delta, weight, _) = saved.outputs
        return hidden_out.clone(), (delta.clone(), weight.clone(), None)

    Update.forward = forward
    try:
        yield
    finally:
        Update.forward = original
