"""Run DPVO's prepared recurrent update with bounded CUDA graph replay."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import TypeAlias

import torch
from dpvo.net import Update, UpdateTopology
from jaxtyping import Float, Int
from torch import Tensor

UpdateOutput: TypeAlias = tuple[Float[Tensor, "1 edges dim"], tuple[Float[Tensor, "1 edges 2"], Float[Tensor, "1 edges 2"], None]]
"""Updated hidden state, displacement, confidence, and the unused flow output."""
MAX_UPDATE_GRAPHS: int = 8
"""Bound graph pools per run; uncommon shapes keep the eager path."""
MIN_REPLAY_CALLS: int = 3
"""Avoid capturing shapes seen only once or twice during tracker initialization."""


@dataclass(frozen=True, slots=True)
class UpdateGraphKey:
    """Model, tensor shapes, group sizes, and precision fixed by a captured update."""

    model: Update
    """Model that owns the captured weights."""
    shapes: tuple[torch.Size, torch.Size, torch.Size]
    """Hidden-state, context, and correlation dimensions."""
    dtypes: tuple[torch.dtype, torch.dtype, torch.dtype]
    """Storage dtypes of the three floating inputs."""
    device: torch.device
    """Device that owns inputs and buffers."""
    groups: tuple[int, int]
    """Patch and frame-pair group counts."""
    autocast: bool
    """Whether CUDA autocast is enabled."""
    autocast_dtype: torch.dtype
    """Autocast target dtype."""
    matmul_tf32: bool
    """Matrix multiplication precision."""
    cudnn_tf32: bool
    """Convolution precision policy, kept separate during recovery."""


@dataclass(frozen=True, slots=True)
class UpdateGraph:
    """Persistent input buffers, prepared topology, and captured outputs."""

    hidden: Float[Tensor, "1 edges dim"]
    """Hidden state copied before replay."""
    context: Float[Tensor, "1 edges dim"]
    """Context features copied before replay."""
    correlations: Float[Tensor, "1 edges corr_dim"]
    """Correlation features copied before replay."""
    source: Int[Tensor, "edges"]
    """Source frame indices."""
    target: Int[Tensor, "edges"]
    """Target frame indices."""
    patch: Int[Tensor, "edges"]
    """Patch indices."""
    topology: UpdateTopology
    """Persistent neighbor and aggregation index buffers."""
    outputs: UpdateOutput
    """Outputs overwritten by each replay; callers receive copies."""
    graph: torch.cuda.CUDAGraph
    """Captured neural update kernels."""


@contextmanager
def update_cuda_graphs() -> Iterator[None]:
    """Replay recurring update shapes during one inference run, with at most eight graph pools.

    Group discovery stays outside capture. Every topology tensor is copied before replay, so equal
    edge counts with different graph connectivity cannot reuse stale neighbors or group membership.
    """
    original = Update.forward
    cache: dict[UpdateGraphKey, UpdateGraph] = {}
    seen: dict[UpdateGraphKey, int] = {}

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
            self, (net.shape, inp.shape, corr.shape), (net.dtype, inp.dtype, corr.dtype), net.device,
            (topology.patch_groups[1], topology.frame_groups[1]),
            torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
            bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32),
        )
        saved: UpdateGraph | None = cache.get(key)
        if saved is None:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] < MIN_REPLAY_CALLS or len(cache) >= MAX_UPDATE_GRAPHS:
                return original(self, net, inp, corr, flow, ii, jj, kk, topology=topology)
            hidden: Float[Tensor, "1 edges dim"] = net.clone()
            context: Float[Tensor, "1 edges dim"] = inp.clone()
            correlations: Float[Tensor, "1 edges corr_dim"] = corr.clone()
            source: Int[Tensor, "edges"] = ii.clone()
            target: Int[Tensor, "edges"] = jj.clone()
            patch: Int[Tensor, "edges"] = kk.clone()
            static_topology: UpdateTopology = UpdateTopology(
                topology.previous.clone(), topology.following.clone(),
                (topology.patch_groups[0].clone(), topology.patch_groups[1]),
                (topology.frame_groups[0].clone(), topology.frame_groups[1]),
            )
            current: torch.cuda.Stream = torch.cuda.current_stream(net.device)
            warmup: torch.cuda.Stream = torch.cuda.Stream(device=net.device)
            warmup.wait_stream(current)
            with torch.autocast("cuda", enabled=key.autocast, dtype=key.autocast_dtype, cache_enabled=False):
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        original(self, hidden, context, correlations, None, source, target, patch, topology=static_topology)
                current.wait_stream(warmup)
                graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    outputs: UpdateOutput = original(self, hidden, context, correlations, None, source, target, patch, topology=static_topology)
            saved = UpdateGraph(hidden, context, correlations, source, target, patch, static_topology, outputs, graph)
            cache[key] = saved
        saved.hidden.copy_(net)
        saved.context.copy_(inp)
        saved.correlations.copy_(corr)
        saved.source.copy_(ii)
        saved.target.copy_(jj)
        saved.patch.copy_(kk)
        saved.topology.previous.copy_(topology.previous)
        saved.topology.following.copy_(topology.following)
        saved.topology.patch_groups[0].copy_(topology.patch_groups[0])
        saved.topology.frame_groups[0].copy_(topology.frame_groups[0])
        saved.graph.replay()
        hidden_out, (delta, weight, _) = saved.outputs
        return hidden_out.clone(), (delta.clone(), weight.clone(), None)

    Update.forward = forward
    try:
        yield
    finally:
        Update.forward = original
