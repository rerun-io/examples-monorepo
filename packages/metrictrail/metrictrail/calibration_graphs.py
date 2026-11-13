"""Prepared DROID calibration updates, with one CUDA graph for projection through BA.

The edge policy is ViPE's DROID policy (Apache-2.0 / MIT), unchanged. Host mirrors hold
integer graph metadata; poses, disparities, neural state and calibration results are never cached.
"""

import copy
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import torch
from dpvo.graph_replay import PrecisionKey
from jaxtyping import Int64
from torch import Tensor
from vipe.ext.lietorch import SE3
from vipe.ext.scatter import scatter_mean
from vipe.slam.components import backend, frontend
from vipe.slam.components.buffer import GraphBuffer
from vipe.slam.components.factor_graph import FactorGraph
from vipe.slam.maths import geom
from vipe.slam.networks.droid_net import AltCorrBlock, CorrBlock, GraphAgg
from vipe.utils.cameras import CameraType
from vipe_mojo.slam_ext import BaTopology, ba_extended, prepare_topology

MAX_CALIBRATION_BUCKETS: int = 2
"""Captured shape buckets kept per graph (least recently used dropped first)."""
GROUPS_ATTRIBUTE: str = "_metrictrail_groups"
"""Attribute on a grouped index tensor that carries its group count to :func:`grouped_aggregate` (no host sync)."""


class CalibrationAccelerationError(BaseException):
    """The calibration's CUDA graphs or TensorRT GRU failed. Not an ``Exception``, so ViPE's calibration cannot turn it
    into its MoGe-2 focal fallback: the run fails instead of tracking with a K the failure changed."""


def grouped_inverse(sources: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    """The sorted source groups of ``sources`` (host) and each edge's group index on ``device``, tagged with the group
    count for :func:`grouped_aggregate`."""
    groups, inverse = torch.unique(sources, sorted=True, return_inverse=True)
    inverse = inverse.to(device)
    setattr(inverse, GROUPS_ATTRIBUTE, len(groups))
    return groups, inverse


@dataclass(frozen=True)
class StepState:
    """Tensor addresses retained by a shape bucket; connectivity and values are copied on reuse."""

    sources: Tensor
    targets: Tensor
    hidden: Tensor
    context: Tensor
    target: Tensor
    weight: Tensor
    inactive_target: Tensor
    inactive_weight: Tensor
    correlations: CorrBlock


@dataclass
class PreparedUpdate:
    """Fixed edge groups and solve tables for the current frontend topology."""

    groups: Int64[Tensor, "groups"]
    inverse: Int64[Tensor, "edges"]
    sources: Int64[Tensor, "all_edges"]
    targets: Int64[Tensor, "all_edges"]
    topology: BaTopology
    state: StepState
    t0: int
    t1: int
    iterations: int
    motion_only: bool
    limited_disp: bool
    use_inactive: bool
    graph: torch.cuda.CUDAGraph | None = None
    warmed: bool = False

    def tensors(self) -> tuple[Tensor, ...]:
        """All captured inputs, including every topology-dependent BA assembly table."""
        topology, state = self.topology, self.state
        assert topology.pose_system.device_tables is not None
        return (
            self.groups, self.inverse, self.sources, self.targets,
            topology.kx, topology.kk_exp, topology.ii_exp, topology.pair_idx, topology.rhs_rows, topology.target_rows,
            *(value for accum in (topology.accum_ii_kx, topology.accum_ii_ts, topology.accum_exp_kx)
              for value in (accum.order, accum.starts, accum.ends)),
            *topology.pose_system.device_tables,
            state.sources, state.targets, state.hidden, state.context, state.target, state.weight,
            state.inactive_target, state.inactive_weight, *state.correlations.corr_pyramid,
        )

    def key(self) -> tuple[object, ...]:
        """Shapes, pose bounds, iteration count and precision determine the captured launch parameters."""
        return (self.t0, self.t1, self.iterations, self.motion_only, self.limited_disp,
                tuple((tuple(value.shape), value.dtype, value.device) for value in self.tensors()),
                PrecisionKey.current())


def grouped_aggregate(self: GraphAgg, net: Tensor, ix: Tensor) -> tuple[Tensor, Tensor]:
    """The original aggregation, with its output size supplied by the host topology."""
    groups: int | None = getattr(ix, GROUPS_ATTRIBUTE, None)
    batch, count, channels, height, width = net.shape
    net = self.relu(self.conv1(net.view(batch * count, channels, height, width)))
    net = scatter_mean(net.view(batch, count, 128, height, width), ix, dim=1, dim_size=groups)
    net = self.relu(self.conv2(net.view(-1, 128, height, width)))
    eta = self.eta(net).view(batch, -1, height, width)
    upmask = self.upmask(net).view(batch, -1, 8 * 8 * 9, height, width)
    return 0.01 * eta, upmask


class CalibrationGraph(FactorGraph):
    """Single-view fused-BA graph with CPU topology and device-only recurrent steps."""

    corr: CorrBlock | None
    inp: Tensor | None
    f_net: Tensor | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sources = torch.empty(0, dtype=torch.int64)
        self.targets = torch.empty(0, dtype=torch.int64)
        self.inactive_sources = torch.empty(0, dtype=torch.int64)
        self.inactive_targets = torch.empty(0, dtype=torch.int64)
        # Ages are bookkeeping: all decisions about them happen on the host.
        self.age = torch.empty(0, dtype=torch.int64)
        self.prepared: PreparedUpdate | None = None
        self.buckets: OrderedDict[tuple[object, ...], PreparedUpdate] = OrderedDict()
        self.bucket_reuses = 0
        self.capture_count = 0
        self.replay_count = 0
        if self.buffer.n_views != 1 or self.buffer.camera_type != CameraType.PINHOLE:
            raise CalibrationAccelerationError("calibration graphs require a single pinhole view")
        if not self.buffer.ba_config.fused or self.buffer.sparse_tracks.enabled:
            raise CalibrationAccelerationError("calibration graphs require fused dense BA")

    @torch.amp.autocast("cuda", enabled=True)
    def add_factors(self, ii: Tensor | list[int], jj: Tensor | list[int], remove: bool = False) -> None:
        """Apply the original duplicate/age policy before transferring new edge indices."""
        sources = torch.as_tensor(ii, dtype=torch.int64, device="cpu")
        targets = torch.as_tensor(jj, dtype=torch.int64, device="cpu")
        existing = set(zip(self.sources.tolist() + self.inactive_sources.tolist(), self.targets.tolist() + self.inactive_targets.tolist(), strict=True))
        keep = torch.tensor([(i, j) not in existing for i, j in zip(sources.tolist(), targets.tolist(), strict=True)], dtype=torch.bool)
        sources, targets = sources[keep], targets[keep]
        if not sources.numel():
            return
        if self.max_factors > 0 and len(self.sources) + len(sources) > self.max_factors and self.corr is not None and remove:
            # CUDA's unstable argsort orders ties differently from CPU argsort. Keep that
            # original decision at topology changes, outside the captured recurrent step.
            ix = torch.arange(len(self.age))[torch.argsort(self.age.to(self.device)).cpu()]
            self.rm_factors(ix >= self.max_factors - len(sources), store=True)
        ii, jj = sources.to(self.device), targets.to(self.device)
        if self.incremental:
            corr = CorrBlock(self.buffer.fmaps[ii, 0][None], self.buffer.fmaps[jj, 0][None])
            self.corr = corr if self.corr is None else self.corr.cat(corr)
            inp = self.buffer.inps[ii, 0][None]
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)
        with torch.autocast("cuda", enabled=False):
            target, _ = self.buffer.reproject_dense_disp(ii, jj)
            target = target[None]
        self.ii, self.jj = torch.cat([self.ii, ii]), torch.cat([self.jj, jj])
        self.sources, self.targets = torch.cat([self.sources, sources]), torch.cat([self.targets, targets])
        self.age = torch.cat([self.age, torch.zeros_like(sources)])
        net = self.buffer.nets[ii, 0][None]
        self.f_net = net if self.f_net is None else torch.cat([self.f_net, net], 1)
        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, torch.zeros_like(target)], 1)
        self.prepared = None

    def rm_factors(self, mask: Tensor, store: bool = False) -> None:
        """Select known-length integer indices, avoiding dynamic CUDA boolean indexing."""
        mask = mask.cpu()
        remove = torch.where(mask)[0].to(self.device)
        keep = torch.where(~mask)[0].to(self.device)
        if not remove.numel():
            return
        if store:
            self.inactive_sources = torch.cat([self.inactive_sources, self.sources[mask]])
            self.inactive_targets = torch.cat([self.inactive_targets, self.targets[mask]])
            self.ii_inac = torch.cat([self.ii_inac, self.ii[remove]])
            self.jj_inac = torch.cat([self.jj_inac, self.jj[remove]])
            self.target_inac = torch.cat([self.target_inac, self.target[:, remove]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:, remove]], 1)
        self.sources, self.targets, self.age = self.sources[~mask], self.targets[~mask], self.age[~mask]
        self.ii, self.jj = self.ii[keep], self.jj[keep]
        if self.corr is not None:
            self.corr = self.corr[keep]
        if self.f_net is not None:
            self.f_net = self.f_net[:, keep]
        if self.inp is not None:
            self.inp = self.inp[:, keep]
        self.target, self.weight = self.target[:, keep], self.weight[:, keep]
        self.prepared = None

    def rm_second_newest_keyframe(self, ix: int) -> None:
        """Mirror the original removal and renumbering without reading CUDA indices."""
        self.buffer.remove_second_newest(ix)
        mask = (self.inactive_sources == ix) | (self.inactive_targets == ix)
        self.inactive_sources -= (self.inactive_sources >= ix).long()
        self.inactive_targets -= (self.inactive_targets >= ix).long()
        keep = torch.where(~mask)[0].to(self.device)
        self.inactive_sources, self.inactive_targets = self.inactive_sources[~mask], self.inactive_targets[~mask]
        self.ii_inac, self.jj_inac = self.inactive_sources.to(self.device), self.inactive_targets.to(self.device)
        self.target_inac, self.weight_inac = self.target_inac[:, keep], self.weight_inac[:, keep]
        mask = (self.sources == ix) | (self.targets == ix)
        self.sources -= (self.sources >= ix).long()
        self.targets -= (self.targets >= ix).long()
        self.ii, self.jj = self.sources.to(self.device), self.targets.to(self.device)
        self.rm_factors(mask)
        self.prepared = None

    def add_neighborhood_factors(self, t0: int, t1: int, r: int = 3) -> None:
        ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1), indexing="ij")
        ii, jj = ii.flatten(), jj.flatten()
        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(self, t0: int = 0, t1: int = 0, rad: int = 2, nms: int = 2,
                              beta: float = 0.25, thresh: float = 16.0, remove: bool = False) -> None:
        """Original greedy NMS: one distance readback; all index data already lives on the host."""
        assert t0 >= t1
        t = self.buffer.n_frames
        ii, jj = torch.meshgrid(torch.arange(t0, t), torch.arange(t1, t), indexing="ij")
        ii, jj = ii.flatten(), jj.flatten()
        distances = self.buffer.frame_distance_dense_disp(ii.to(self.device), jj.to(self.device), beta=beta).mean(-1).cpu().numpy()
        ii_np, jj_np = ii.numpy(), jj.numpy()

        def suppress(i: int, j: int) -> None:
            if t0 <= i < t and t1 <= j < t:
                distances[(i - t0) * (t - t1) + (j - t1)] = np.inf

        def suppress_nms(i: int, j: int) -> None:
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        suppress(i + di, j + dj)

        for i, j in zip(self.sources.tolist() + self.inactive_sources.tolist(), self.targets.tolist() + self.inactive_targets.tolist(), strict=True):
            suppress_nms(i, j)
        distances[(ii_np - rad < jj_np) | (distances > thresh)] = np.inf
        edges: list[tuple[int, int]] = []
        for i in range(t0, t):
            for j in range(max(i - rad - 1, 0), i):
                edges.extend(((i, j), (j, i)))
                suppress(i, j)
        for k in np.argsort(distances):
            if distances[k] > thresh:
                continue
            if len(edges) > self.max_factors:
                break
            i, j = int(ii_np[k]), int(jj_np[k])
            edges.extend(((i, j), (j, i)))
            suppress_nms(i, j)
        if edges:
            sources, targets = torch.tensor(edges).unbind(-1)
            self.add_factors(sources, targets, remove)

    def _prepare(self, t0: int, t1: int, itrs: int, use_inactive: bool, motion_only: bool, limited_disp: bool) -> PreparedUpdate:
        groups, inverse = grouped_inverse(self.sources, self.device)
        inactive = torch.where((self.inactive_sources >= t0 - 3) & (self.inactive_targets >= t0 - 3))[0] if use_inactive else torch.empty(0, dtype=torch.int64)
        sources = torch.cat([self.inactive_sources[inactive], self.sources])
        targets = torch.cat([self.inactive_targets[inactive], self.targets])
        topology = prepare_topology(tuple(sources.tolist()), tuple(targets.tolist()), t0, t1, False, motion_only, self.device)
        topology.pose_system.prepare_device(self.device)
        assert self.corr is not None and self.f_net is not None and self.inp is not None
        correlations = copy.copy(self.corr)
        # CorrBlock.cat / __getitem__ replace list entries in place. Keep this bucket's input addresses alive.
        correlations.corr_pyramid = list(self.corr.corr_pyramid)
        inactive = inactive.to(self.device)
        state = StepState(self.ii, self.jj, self.f_net, self.inp, self.target, self.weight,
                          self.target_inac[:, inactive], self.weight_inac[:, inactive], correlations)
        plan = PreparedUpdate(groups.to(self.device), inverse, sources.to(self.device), targets.to(self.device),
                              topology, state, t0, t1, itrs, motion_only, limited_disp, use_inactive)
        key = plan.key()
        saved = self.buckets.get(key)
        if saved is not None:
            for destination, source in zip(saved.tensors(), plan.tensors(), strict=True):
                destination.copy_(source)
            saved.use_inactive = use_inactive
            self.buckets.move_to_end(key)
            self.bucket_reuses += 1
            return saved
        # The global topology cache is immutable: a shape bucket needs its own writable index tables.
        plan.topology = copy.deepcopy(topology)
        self.buckets[key] = plan
        while len(self.buckets) > MAX_CALIBRATION_BUCKETS:
            self.buckets.popitem(last=False)
        return plan

    @torch.amp.autocast("cuda", enabled=True)
    def _step(self, plan: PreparedUpdate) -> None:
        """All floating work of one update: projection, correlation, GRU, and every BA iteration."""
        state = plan.state
        with torch.autocast("cuda", enabled=False):
            coords = self._project(state.sources, state.targets)
            coords = coords[None]
            motion = torch.cat([coords - self.coords0, state.target - coords], -1).permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)
        corr = state.correlations(coords)
        net, delta, weight, damping, _ = self.net.update(state.hidden, state.context, corr, motion, ix=plan.inverse)
        weight = weight.masked_fill(self.buffer.masks[state.sources, 0][None, ..., None], 0.0)
        with torch.autocast("cuda", enabled=False):
            state.hidden.copy_(net)
            state.target.copy_(coords + delta.float())
            state.weight.copy_(weight.float())
            self.damping[plan.groups] = damping
            target = torch.cat([state.inactive_target, state.target], 1)
            weight = torch.cat([state.inactive_weight, state.weight], 1)
            self._ba(target, weight, plan.sources, plan.targets, plan.topology, plan.t0, plan.t1, plan.iterations,
                     1e-3, 0.1, False, plan.motion_only, plan.limited_disp, device_solve=True)

    def _ba(self, target: Tensor, weight: Tensor, ii: Tensor, jj: Tensor, topology: BaTopology,
            t0: int, t1: int, iterations: int, lm: float, ep: float, optimize_intrinsics: bool,
            motion_only: bool = False, limited_disp: bool = False, *, device_solve: bool = False) -> None:
        """The fused single-view BA contract, with topology validation done on the host."""
        buffer = self.buffer
        intrinsics = buffer.camera_type.build_camera_model(buffer.intrinsics).pinhole().scaled(1 / 8.0).intrinsics[0].contiguous()
        active = torch.ones(buffer.disps.shape[0], device=self.device)
        if limited_disp:
            active.zero_()
            active[t0:t1] = 1.0
        scalar_damping = 1e-6 * buffer.ba_config.get("intrinsics_damping_scale", 1.0)
        ba_extended(buffer.poses, buffer.disps[:, 0], intrinsics, buffer.disps_sens[:, 0],
                    target[0].permute(0, 3, 1, 2).contiguous(), weight[0].permute(0, 3, 1, 2).contiguous(),
                    (0.2 * self.damping[topology.kx] + 1e-7).contiguous(), ii, jj, active, t0, t1,
                    iterations, lm, ep, motion_only, float(buffer.ba_config.dense_disp_alpha), optimize_intrinsics,
                    scalar_damping, scalar_damping, 1 / 8.0, False, 0.001, topology=topology, device_solve=device_solve)
        if optimize_intrinsics:
            buffer.intrinsics[0, :4] = intrinsics / (1 / 8.0)
        buffer.disps.clamp_(min=0.001)

    def update(self, t0: int | None = None, t1: int | None = None, itrs: int = 3, use_inactive: bool = False,
               motion_only: bool = False, fixed_motion: bool = False, limited_disp: bool = False) -> None:
        if fixed_motion:
            raise CalibrationAccelerationError("SharedK calibration requires free camera poses")
        t0 = max(1, int(self.sources.min()) + 1) if t0 is None else t0
        t1 = max(int(self.sources.max()), int(self.targets.max())) + 1 if t1 is None else t1
        plan = self.prepared
        if plan is None or (plan.t0, plan.t1, plan.iterations, plan.motion_only, plan.limited_disp, plan.use_inactive) != (t0, t1, itrs, motion_only, limited_disp, use_inactive):
            plan = self.prepared = self._prepare(t0, t1, itrs, use_inactive, motion_only, limited_disp)
        if not plan.warmed:
            # First real update also warms every library/kernel; no algorithm steps are added or skipped.
            self._step(plan)
            plan.warmed = True
        else:
            if plan.graph is None:
                plan.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(plan.graph, capture_error_mode="thread_local"):
                    self._step(plan)
                self.capture_count += 1
            plan.graph.replay()
            self.replay_count += 1
        self.f_net, self.target, self.weight = plan.state.hidden, plan.state.target, plan.state.weight
        self.age += 1

    def _project(self, sources: Tensor, targets: Tensor) -> Tensor:
        """Use the existing geometry with a device grid prepared before capture."""
        buffer = self.buffer
        zero = torch.zeros_like(sources)
        coords, _, _, _, _ = geom.iproj_i_proj_j_disp(
            SE3(buffer.poses), buffer.flattened_disps[sources], self.coords0[None],
            buffer.camera_type.build_camera_model(buffer.intrinsics).scaled(1 / 8.0).intrinsics,
            buffer.camera_type, SE3(buffer.rig), sources, targets, zero, zero, None, False, False, False,
        )
        return coords

    @torch.amp.autocast("cuda", enabled=False)
    def update_batch(self, itrs: int, steps: int, optimize_intrinsics: bool, optimize_rig_rotation: bool, solver_verbose: bool = False) -> None:
        """Backend with the same batches and iterations, with topology and float64 LLT on the device."""
        if optimize_rig_rotation:
            raise CalibrationAccelerationError("SharedK calibration has a fixed identity rig")
        assert self.f_net is not None
        corr_op = AltCorrBlock(self.buffer.flattened_fmaps[None])
        batches: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        for start in range(0, int(self.targets.max()) + 1, 8):
            select = torch.where((self.sources >= start) & (self.sources < start + 8))[0]
            groups, inverse = grouped_inverse(self.sources[select], self.device)
            batches.append((select.to(self.device), self.sources[select].to(self.device), self.targets[select].to(self.device), groups.to(self.device), inverse))
        topology = prepare_topology(tuple(self.sources.tolist()), tuple(self.targets.tolist()), 1, self.buffer.n_frames,
                                     optimize_intrinsics, False, self.device)
        topology.pose_system.prepare_device(self.device)
        for _ in range(steps):
            coords, _ = self.buffer.reproject_dense_disp(self.ii, self.jj)
            coords = coords[None]
            motion = torch.cat([coords - self.coords0, self.target - coords], -1).permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)
            for select, sources, targets, groups, inverse in batches:
                corr = corr_op(coords[:, select], sources, targets)
                with torch.autocast("cuda", enabled=True):
                    net, delta, weight, damping, _ = self.net.update(self.f_net[:, select], self.buffer.inps[sources, 0][None], corr, motion[:, select], ix=inverse)
                    weight = weight.masked_fill(self.buffer.masks[sources, 0][None, ..., None], 0.0)
                self.f_net[:, select] = net
                self.target[:, select] = coords[:, select] + delta.float()
                self.weight[:, select] = weight.float()
                self.damping[groups] = damping
            self._ba(self.target, self.weight, self.ii, self.jj, topology, 1, self.buffer.n_frames, itrs,
                     1e-5, 1e-2, optimize_intrinsics, device_solve=True)


@contextmanager
def calibration_cuda_graphs(enabled: bool = True) -> Iterator[None]:
    """Install prepared single-view graphs for this run; live-view public tensors keep their meaning."""
    if not enabled:
        yield
        return
    old_frontend, old_backend = frontend.FactorGraph, backend.FactorGraph
    old_expand, old_agg = GraphBuffer.expand_edge_multiview, GraphAgg.forward

    @wraps(old_expand)
    def expand(self: GraphBuffer, ii: Tensor, jj: Tensor, cross: bool = True, view_offset: int = 0) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.n_views == 1:
            zero = torch.zeros_like(ii)
            return ii.reshape(-1), zero, ii.reshape(-1), jj.reshape(-1), zero, jj.reshape(-1)
        return old_expand(self, ii, jj, cross, view_offset)

    frontend.FactorGraph = CalibrationGraph
    backend.FactorGraph = CalibrationGraph
    GraphBuffer.expand_edge_multiview = expand
    GraphAgg.forward = grouped_aggregate
    try:
        yield
    finally:
        frontend.FactorGraph, backend.FactorGraph = old_frontend, old_backend
        GraphBuffer.expand_edge_multiview, GraphAgg.forward = old_expand, old_agg
        prepare_topology.cache_clear()  # its device index tables would outlive the run in an app process
