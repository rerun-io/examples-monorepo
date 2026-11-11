"""Main DPV-SLAM (Deep Patch Visual SLAM) inference class.

Implements the sliding-window visual odometry system described in
Teed et al. (2022), "Deep Patch Visual Odometry", extended with SLAM
features from Lipson et al. (2024), "Deep Patch Visual SLAM":

- **Proximity loop closure** (GPU-based): detects revisits via camera
  proximity and inserts long-range edges for global BA.
- **Classical loop closure** (DBoW2): image retrieval + keypoint matching
  + Sim(3) pose-graph optimization.
- **Enhanced global BA**: efficient Schur complement via block-E kernels.

The pipeline processes frames one at a time via :meth:`DPVO.__call__`:

1. **Patchify**: Extract features with the stride-4 CNN backbone and
   sample M = ``PATCHES_PER_FRAME`` sparse 3x3 patches using
   gradient-biased sampling.
2. **Motion model**: Predict the initial pose of the new frame using
   damped linear extrapolation on SE3.
3. **Graph construction**: Create measurement edges connecting the new
   frame's patches to recent frames (forward edges) and recent patches
   to the new frame (backward edges).
4. **Update**: Run one iteration of the recurrent GRU operator
   (:class:`~dpvo.net.Update`) to predict pixel displacements and
   confidence weights, then perform one Gauss-Newton bundle adjustment
   step (:func:`~dpvo.fastba.ba.BA`) to refine poses and depths.
5. **Keyframe management**: Remove redundant keyframes whose
   bidirectional pixel flow falls below ``KEYFRAME_THRESH``.
6. **Loop closure** (SLAM only): Periodically detect revisited locations
   and insert long-range edges for global bundle adjustment (proximity)
   or run Sim(3) pose-graph optimization (classical DBoW2).

The system maintains fixed-size buffers pre-allocated at construction
time via :class:`~dpvo.patchgraph.PatchGraph` on one device (CUDA by
default), avoiding dynamic memory allocation during tracking.

Extensions plug in through :mod:`dpvo.hooks`: a depth prior (patch-depth
initialisation and the BA unary), a graph policy (keyframe-cull veto, extra
loop edges) and observers (per-frame, cull, global-BA and terminate
callbacks). The defaults reproduce plain DPVO exactly.
"""

from collections import OrderedDict
from collections.abc import Generator, Sequence

import lietorch_mojo as lietorch
import numpy as np
import torch
import torch.nn.functional as F
from beartype.roar import BeartypeException
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Float64, Int, Int64, UInt8
from lietorch_mojo import SE3
from numpy import ndarray
from torch import Tensor

from . import altcorr, fastba
from . import projective_ops as pops
from .config import DPVOConfig
from .hooks import DefaultGraphPolicy, DepthPrior, DPVOObserver, FrameDepth, FrameEvent, GraphPolicy, RandomDepthInit, global_observers
from .net import VONet
from .patchgraph import PatchGraph
from .utils import DEFAULT_DEVICE, Timer, flatmeshgrid


class DPVO:
    """Sliding-window visual SLAM system using sparse deep patches.

    Manages all state for real-time visual odometry/SLAM: the
    :class:`PatchGraph` (poses, patches, edges), feature maps, and the
    neural network.  Designed to be called once per frame via
    :meth:`__call__`.

    Args:
        cfg: Tracker configuration.
        network: Checkpoint path or a ready :class:`VONet` (moved to ``device``).
        ht: Image height in pixels.
        wd: Image width in pixels.
        device: Device of the network and every buffer.
        depth_prior: Patch-depth strategy; ``None`` = :class:`~dpvo.hooks.RandomDepthInit`.
        graph_policy: Keyframe/loop-edge strategy; ``None`` = :class:`~dpvo.hooks.DefaultGraphPolicy`.
        observers: Run callbacks, called in order before the global observers
            (:func:`~dpvo.hooks.add_global_observer`).
    """

    def __init__(
        self,
        cfg: DPVOConfig,
        network: str | VONet,
        ht: int = 480,
        wd: int = 640,
        *,
        device: torch.device = DEFAULT_DEVICE,
        depth_prior: DepthPrior | None = None,
        graph_policy: GraphPolicy | None = None,
        observers: Sequence[DPVOObserver] = (),
    ) -> None:
        self.cfg: DPVOConfig = cfg
        self.device: torch.device = device
        self.depth_prior: DepthPrior = RandomDepthInit() if depth_prior is None else depth_prior
        self.graph_policy: GraphPolicy = DefaultGraphPolicy() if graph_policy is None else graph_policy
        self.observers: list[DPVOObserver] = list(observers)
        self.load_weights(network)
        self.is_initialized: bool = False
        self.enable_timing: bool = False

        self.M: int = self.cfg.patches_per_frame
        self.N: int = self.cfg.buffer_size

        self.ht: int = ht
        self.wd: int = wd

        DIM: int = self.DIM
        RES: int = self.RES

        ### state attributes ###
        self.tlist: list[int] = []
        self.counter: int = 0

        # Track global-BA calls to avoid redundant runs
        self.ran_global_ba: np.ndarray = np.zeros(100000, dtype=bool)
        # Last n at which the scheduled global BA (cfg.gba_schedule_k) ran
        self._gba_sched_last_n: int = -1

        # Identity SE3 element used as the relative pose for frames skipped during initialization
        self.identity: SE3 = SE3.Identity(1, device=self.device)

        # dummy image for visualization
        self.image_: UInt8[Tensor, "ht wd 3"] = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        self.feature_dtype: torch.dtype = torch.half if self.cfg.mixed_precision else torch.float
        self.kwargs: dict[str, object] = {"device": self.device, "dtype": self.feature_dtype}

        ### frame memory size ###
        self.pmem: int = 36
        if self.cfg.loop_closure:
            self.last_global_ba: int = -1000
            self.pmem = self.cfg.max_edge_age
        self.mem: int = min(self.pmem, self.cfg.fmap_memory)

        self.imap_: Float[Tensor, "pmem M DIM"] = torch.zeros(self.pmem, self.M, DIM, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]
        self.gmap_: Float[Tensor, "pmem M 128 P P"] = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]

        # PatchGraph holds all buffer state
        self.pg: PatchGraph = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, device=self.device, feature_dtype=self.feature_dtype)

        # Classical loop closure (optional)
        if self.cfg.classic_loop_closure:
            self.load_long_term_loop_closure()

        ht_feat: int = ht // RES
        wd_feat: int = wd // RES

        self.fmap1_: Float[Tensor, "1 mem 128 h4 w4"] = torch.zeros(1, self.mem, 128, ht_feat // 1, wd_feat // 1, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]
        self.fmap2_: Float[Tensor, "1 mem 128 h16 w16"] = torch.zeros(1, self.mem, 128, ht_feat // 4, wd_feat // 4, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]

        self.pyramid: tuple[Tensor, Tensor] = (self.fmap1_, self.fmap2_)

        # Process-wide observers last, so that their factories see a fully built tracker
        self.observers.extend(global_observers(self))

    def load_long_term_loop_closure(self) -> None:
        """Load the classical (DBoW2) loop closure module."""
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc: LongTermLoopClosure = LongTermLoopClosure(self.cfg, self.pg)
        except (ModuleNotFoundError, FileNotFoundError) as e:
            print(f"WARNING: Classical loop closure disabled: {e}")

    def load_weights(self, network: str | VONet) -> None:
        """Load network weights from a checkpoint file or use an existing VONet."""
        if isinstance(network, str):
            state_dict: dict[str, Tensor] = torch.load(network)
            new_state_dict: OrderedDict[str, Tensor] = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace("module.", "")] = v

            self.network: VONet = VONet()
            self.network.load_state_dict(new_state_dict)
        else:
            self.network: VONet = network

        self.DIM: int = self.network.DIM
        self.RES: int = self.network.RES
        self.P: int = self.network.P

        self.network.to(self.device)
        self.network.eval()

    @property
    def poses(self) -> Float32[Tensor, "1 N 7"]:
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self) -> Float32[Tensor, "1 NM 3 3 3"]:
        return self.pg.patches_.view(1, self.N * self.M, 3, 3, 3)

    @property
    def intrinsics(self) -> Float32[Tensor, "1 N 4"]:
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self) -> Int[Tensor, "NM"]:
        return self.pg.index_.view(-1)

    @property
    def imap(self) -> Float[Tensor, "1 pmem_times_M DIM"]:
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self) -> Float[Tensor, "1 pmem_times_M 128 3 3"]:
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self) -> int:
        return self.pg.n

    @n.setter
    def n(self, val: int) -> None:
        self.pg.n = val

    @property
    def m(self) -> int:
        return self.pg.m

    @m.setter
    def m(self, val: int) -> None:
        self.pg.m = val

    # Convenience aliases for direct buffer access (used by inference API)
    @property
    def poses_(self) -> Float32[Tensor, "N 7"]:
        return self.pg.poses_

    @property
    def points_(self) -> Float32[Tensor, "NM 3"]:
        return self.pg.points_

    @property
    def colors_(self) -> UInt8[Tensor, "N M 3"]:
        return self.pg.colors_

    @property
    def tstamps_(self) -> np.ndarray:
        return self.pg.tstamps_

    def get_pose(self, t: int) -> SE3:
        """Retrieve the pose for timestamp ``t``, interpolating removed frames."""
        if t in self.traj:
            return SE3(self.traj[t])

        t0: int
        dP: SE3
        t0, dP = self.pg.delta[t]
        result = dP * self.get_pose(t0)
        assert isinstance(result, SE3)
        return result

    def terminate(self) -> tuple[Float32[ndarray, "n_frames 7"], Float64[ndarray, "n_frames"]]:
        """Finalize tracking: run final BA, interpolate removed keyframes, return trajectory."""
        print("Terminating...")

        if self.cfg.classic_loop_closure and hasattr(self, "long_term_lc"):
            self.long_term_lc.terminate(self.n)

        if self.cfg.loop_closure:
            lii, ljj = self.pg.edges_loop()
            if lii.numel() > 0:
                self.append_factors(lii, ljj)
                self._append_extra_loop_edges(lii, ljj)

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        for observer in self.observers:
            observer.on_terminate(self)

        # Build lookup from internal timestamp -> pose for active keyframes
        self.traj: dict[int, Float32[Tensor, "7"]] = {}
        for i in range(self.n):
            self.traj[int(self.pg.tstamps_[i])] = self.pg.poses_[i]

        # Reconstruct poses for ALL timestamps
        poses_list: list[SE3] = [self.get_pose(t) for t in range(self.counter)]
        poses_stacked: SE3 = lietorch.stack(poses_list, dim=0)
        for observer in self.observers:
            poses_stacked = observer.refine_trajectory(self, poses_stacked)
        poses_inv = poses_stacked.inv()
        assert poses_inv is not None
        poses: Float32[ndarray, "n_frames 7"] = poses_inv.data.cpu().numpy()
        tstamps: Float64[ndarray, "n_frames"] = np.array(self.tlist, dtype=np.float64)
        print("Done!")

        return poses, tstamps

    def corr(
        self,
        coords: Float[Tensor, "1 n_edges 2 P P"],
        indicies: tuple[Int[Tensor, "n_edges"], Int[Tensor, "n_edges"]] | None = None,
    ) -> Float[Tensor, "1 n_edges corr_feat"]:
        """Compute local correlation volumes at two pyramid levels."""
        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1: Int[Tensor, "n_edges"] = ii % (self.M * self.pmem)
        jj1: Int[Tensor, "n_edges"] = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return rearrange(torch.stack([corr1, corr2], -1), "b e ... -> b e (...)")

    def reproject(
        self, indicies: tuple[Int[Tensor, "n_edges"], Int[Tensor, "n_edges"], Int[Tensor, "n_edges"]] | None = None
    ) -> Float[Tensor, "1 n_edges 2 P P"]:
        """Reproject patches from their source frames into target frames."""
        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        kk: Int[Tensor, "n_edges"]
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords: Float[Tensor, "1 n_edges P P 2"] = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk
        )
        return rearrange(coords, "b e p1 p2 c -> b e c p1 p2")

    def append_factors(self, ii: Int[Tensor, "n_new"], jj: Int[Tensor, "n_new"]) -> None:
        """Add new measurement edges to the factor graph."""
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net: Float[Tensor, "1 n_new DIM"] = torch.zeros(1, len(ii), self.DIM, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m: Bool[Tensor, "n_edges"], store: bool) -> None:
        """Remove edges from the factor graph by boolean mask.

        Args:
            m: Boolean mask; ``True`` for edges to remove.
            store: If ``True``, save removed edges as inactive (for global BA).
        """
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        kept: Int64[Tensor, "n_kept"] = (~m).nonzero().flatten()
        if store:
            removed: Int64[Tensor, "n_removed"] = m.nonzero().flatten()
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[removed]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[removed]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[removed]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:, removed]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:, removed]), dim=1)
        self.pg.weight = self.pg.weight[:, kept]
        self.pg.target = self.pg.target[:, kept]

        self.pg.ii = self.pg.ii[kept]
        self.pg.jj = self.pg.jj[kept]
        self.pg.kk = self.pg.kk[kept]
        self.pg.net = self.pg.net[:, kept]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def _append_extra_loop_edges(self, lii: Int64[Tensor, "n_loop"], ljj: Int64[Tensor, "n_loop"]) -> None:
        """Append the graph policy's extra edges for the proximity loop edges just added."""
        extra: tuple[Int64[Tensor, "n_extra"], Int64[Tensor, "n_extra"]] | None = self.graph_policy.extra_loop_edges(self, lii, ljj)
        if extra is not None and extra[0].numel() > 0:
            self.append_factors(extra[0], extra[1])

    def release_inactive_factors(self) -> None:
        """Drop every stored inactive factor (after the last global BA has used them)."""
        device: torch.device = self.pg.ii_inac.device
        self.pg.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.pg.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.pg.kk_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.pg.weight_inac = self.pg.weight_inac[:, :0]
        self.pg.target_inac = self.pg.target_inac[:, :0]

    @torch.no_grad()
    def refresh_factors(
        self,
        ii: Int64[Tensor, "n_edges"],
        jj: Int64[Tensor, "n_edges"],
        kk: Int64[Tensor, "n_edges"],
        target: Float32[Tensor, "1 n_edges 2"],
        weight: Float32[Tensor, "1 n_edges 2"],
        net: Float[Tensor, "1 n_edges DIM"],
        block_frames: int = 40,
    ) -> None:
        """Re-measure factors under the current poses and patches, in place.

        Re-runs the correlation lookup and the update operator for every edge whose
        features are still in the rings (frame ``jj`` within ``mem``, source
        ``ii`` within ``pmem`` of the newest keyframe), in blocks of
        ``block_frames`` source frames: the operator aggregates over edges that
        share ``kk`` and over edges that share ``(ii, jj)``, and both groups stay
        inside one block. ``target``, ``weight`` and the GRU state ``net`` are
        written in place; edges whose features left the rings keep their values.

        Args:
            ii: Source frame per edge.
            jj: Target frame per edge.
            kk: Patch index per edge.
            target: Measured target coordinates, updated in place.
            weight: Measurement weights, updated in place.
            net: GRU hidden state per edge, carried across calls; updated in place.
            block_frames: Source frames per operator block.
        """
        if ii.numel() == 0:
            return
        n: int = self.n
        resident: Bool[Tensor, "n_edges"] = ((n - 1 - jj) < self.mem) & ((n - 1 - ii) < self.pmem) & (jj >= 0) & (ii >= 0)
        P: int = self.P
        for b0 in range(0, n, block_frames):
            selected: Bool[Tensor, "n_edges"] = resident & (ii >= b0) & (ii < b0 + block_frames)
            idx: Int64[Tensor, "n_block"] = selected.nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            ii_b: Int64[Tensor, "n_block"] = ii[idx]
            jj_b: Int64[Tensor, "n_block"] = jj[idx]
            kk_b: Int64[Tensor, "n_block"] = kk[idx]
            coords: Float[Tensor, "1 n_block 2 P P"] = self.reproject(indicies=(ii_b, jj_b, kk_b))
            with torch.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
                corr: Float[Tensor, "1 n_block corr_feat"] = self.corr(coords, indicies=(kk_b, jj_b))
                ctx: Float[Tensor, "1 n_block DIM"] = self.imap[:, kk_b % (self.M * self.pmem)]
                net_b, (delta, weight_b, _) = self.network.update(net[:, idx], ctx, corr, None, ii_b, jj_b, kk_b)
            net[:, idx] = net_b.to(net.dtype)
            target[:, idx] = coords[..., P // 2, P // 2] + delta.float()
            weight[:, idx] = weight_b.float()

    def motion_probe(self) -> Float[Tensor, ""]:
        """Check if there is sufficient parallax for system initialization."""
        kk: Int[Tensor, "M"] = torch.arange(self.m - self.M, self.m, device=self.device)
        jj: Int[Tensor, "M"] = self.n * torch.ones_like(kk)
        ii: Int[Tensor, "M"] = self.ix[kk]

        net: Float[Tensor, "1 M DIM"] = torch.zeros(1, len(ii), self.DIM, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]
        coords: Float[Tensor, "1 M 2 P P"] = self.reproject(indicies=(ii, jj, kk))

        with torch.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
            corr: Float[Tensor, "1 M corr_feat"] = self.corr(coords, indicies=(kk, jj))
            ctx: Float[Tensor, "1 M DIM"] = self.imap[:, kk % (self.M * self.pmem)]
            net, (delta, weight, _) = self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i: int, j: int) -> float:
        """Compute mean pixel flow magnitude for all edges from frame i to frame j."""
        k: Int64[Tensor, "n_matched"] = ((self.pg.ii == i) & (self.pg.jj == j)).nonzero().flatten()
        ii: Int[Tensor, "n_matched"] = self.pg.ii[k]
        jj: Int[Tensor, "n_matched"] = self.pg.jj[k]
        kk: Int[Tensor, "n_matched"] = self.pg.kk[k]

        if self.device.type in ("cpu", "cuda"):
            from .mojo_backend.projective import flow_magnitude

            flow = flow_magnitude(self.poses, self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        else:
            flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self) -> None:
        """Remove redundant keyframes based on pixel flow magnitude."""
        i: int = self.n - self.cfg.keyframe_index - 1
        j: int = self.n - self.cfg.keyframe_index + 1
        m_flow: float = self.motionmag(i, j) + self.motionmag(j, i)
        k: int = self.n - self.cfg.keyframe_index
        low_flow: bool = m_flow / 2 < self.cfg.keyframe_thresh
        force_keep: bool = self.graph_policy.force_keep(self)

        if low_flow and not force_keep:
            t0: int = int(self.pg.tstamps_[k - 1])
            t1: int = int(self.pg.tstamps_[k])

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k - 1]).inv()
            assert isinstance(dP, SE3)
            self.pg.delta[t1] = (t0, dP)

            self.graph_policy.on_cull(self, k)
            for observer in self.observers:
                observer.on_cull(self, k)

            to_remove: Bool[Tensor, "n_edges"] = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for idx in range(k, self.n - 1):
                self.pg.tstamps_[idx] = self.pg.tstamps_[idx + 1]
                self.pg.colors_[idx] = self.pg.colors_[idx + 1]
                self.pg.poses_[idx] = self.pg.poses_[idx + 1]
                self.pg.patches_[idx] = self.pg.patches_[idx + 1]
                for buffer in self.pg.frame_attrs.values():
                    buffer[idx] = buffer[idx + 1]
                self.pg.intrinsics_[idx] = self.pg.intrinsics_[idx + 1]

                self.imap_[idx % self.pmem] = self.imap_[(idx + 1) % self.pmem]
                self.gmap_[idx % self.pmem] = self.gmap_[(idx + 1) % self.pmem]
                self.fmap1_[0, idx % self.mem] = self.fmap1_[0, (idx + 1) % self.mem]
                self.fmap2_[0, idx % self.mem] = self.fmap2_[0, (idx + 1) % self.mem]

            self.n -= 1
            self.m -= self.M
            for buffer in self.pg.frame_attrs.values():
                buffer[self.n].zero_()

            if self.cfg.classic_loop_closure and hasattr(self, "long_term_lc"):
                self.long_term_lc.keyframe(k)

        else:
            self.graph_policy.on_keep(self, k, forced=low_flow)

        # Prune stale edges
        to_remove: Bool[Tensor, "n_edges"] = self.ix[self.pg.kk] < self.n - self.cfg.removal_window
        if self.cfg.loop_closure:
            # Keep loop closure edges (long-range edges in the optimization window)
            lc_edges: Bool[Tensor, "n_edges"] = ((self.pg.jj - self.pg.ii) > 30) & (
                self.pg.jj > (self.n - self.cfg.optimization_window)
            )
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=self.cfg.store_inactive)

    def run_global_ba(self, t0: int | None = None) -> None:
        """Global bundle adjustment over the inactive and active edges.

        Observers' :meth:`~dpvo.hooks.DPVOObserver.on_global_ba` runs first. The
        depth scale is normalized first when ``cfg.normalize_global_ba``; the
        solve uses the depth prior's ``"global"`` prior.

        Args:
            t0: First optimized keyframe; ``None`` follows ``cfg.global_ba_t0``
                (``1`` for ``"first"``, the oldest active source frame for ``"active_min"``).
        """
        for observer in self.observers:
            observer.on_global_ba(self)
        full_target: Float[Tensor, "1 n_all 2"] = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight: Float[Tensor, "1 n_all 2"] = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii: Int[Tensor, "n_all"] = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj: Int[Tensor, "n_all"] = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk: Int[Tensor, "n_all"] = torch.cat((self.pg.kk_inac, self.pg.kk))

        if self.cfg.normalize_global_ba:
            self.pg.normalize()
        lmbda: Float[Tensor, "1"] = torch.as_tensor([1e-4], device=self.device)
        if t0 is None:
            t0 = 1 if self.cfg.global_ba_t0 == "first" else int(self.pg.ii.min().item())
        fastba.BA(
            self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk,
            t0, self.n, M=self.M, iterations=2, eff_impl=True,
            depth_prior=self.depth_prior.ba_prior(self, "global"),
        )
        self.ran_global_ba[self.n] = True

    def _gba_scheduled_now(self) -> bool:
        """Scheduled global BA every ``cfg.gba_schedule_k`` keyframes, once per value of ``n``, while inactive factors exist.

        Records ``n`` when it returns True, so call it only when the scheduled BA then runs.
        """
        k: int = self.cfg.gba_schedule_k
        if k <= 0:
            return False
        if self.n % k != 0 or self.n == self._gba_sched_last_n:
            return False
        if self.pg.ii_inac.numel() == 0:
            return False
        self._gba_sched_last_n = self.n
        return True

    def update(self) -> None:
        """Run one iteration of the DPVO tracking pipeline."""
        with Timer("other", enabled=self.enable_timing, device=self.device):
            coords: Float[Tensor, "1 n_edges 2 P P"] = self.reproject()

            with torch.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
                corr: Float[Tensor, "1 n_edges corr_feat"] = self.corr(coords)
                ctx: Float[Tensor, "1 n_edges DIM"] = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = self.network.update(
                    self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk
                )

            lmbda: Float[Tensor, "1"] = torch.as_tensor([1e-4], device=self.device)
            weight = weight.float()
            target: Float[Tensor, "1 n_edges 2"] = coords[..., self.P // 2, self.P // 2] + delta.float()

        # Store target/weight on PatchGraph for global BA
        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing, device=self.device):
            try:
                # Run global BA if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.removal_window - 1).any() and not self.ran_global_ba[self.n]:
                    self.run_global_ba()
                elif self._gba_scheduled_now():
                    # The scheduled global BA replaces this update's local BA (it includes the active edges)
                    self.run_global_ba()
                else:
                    t0: int = self.n - self.cfg.optimization_window if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(
                        self.poses, self.patches, self.intrinsics,
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk,
                        t0, self.n, M=self.M, iterations=2, eff_impl=False,
                        depth_prior=self.depth_prior.ba_prior(self, "local"),
                    )
            except BeartypeException:
                raise
            except Exception:
                if self.cfg.raise_on_ba_failure:
                    raise
                print("Warning BA failed...")

            points: Float[Tensor, "1 m P P 4"] = pops.point_cloud(
                SE3(self.poses), self.patches[:, : self.m], self.intrinsics, self.ix[: self.m]
            )
            points_3d: Float[Tensor, "m 3"] = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
            self.pg.points_[: len(points_3d)] = points_3d[:]

    def __edges_forw(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        """Generate forward edges: recent patches -> the newest frame."""
        r: int = self.cfg.patch_lifetime
        t0: int = self.M * max((self.n - r), 0)
        t1: int = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(self.n - 1, self.n, device=self.device),
            indexing="ij",
        )

    def __edges_back(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        """Generate backward edges: newest frame's patches -> recent frames."""
        r: int = self.cfg.patch_lifetime
        t0: int = self.M * max((self.n - 1), 0)
        t1: int = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(max(self.n - r, 0), self.n, device=self.device),
            indexing="ij",
        )

    def __call__(
        self,
        tstamp: int,
        image: UInt8[Tensor, "3 ht wd"],
        intrinsics: Float[Tensor, "4"],
        *,
        depth: Float32[Tensor, "ht wd"] | None = None,
        depth_mask: Bool[Tensor, "ht wd"] | None = None,
    ) -> None:
        """Process a new frame: extract patches, predict pose, update and optimize.

        Args:
            tstamp: Caller timestamp (ordinal frame index for every built-in source).
            image: BGR image on the tracker's device.
            intrinsics: ``fx fy cx cy`` in pixels.
            depth: Optional depth map for the depth prior; the default prior ignores it.
            depth_mask: Optional validity of ``depth``.

        Raises:
            ValueError: The depth prior rejected the frame; no state has changed.
        """

        # Feed image to classical loop closure retrieval (if enabled)
        if self.cfg.classic_loop_closure and hasattr(self, "long_term_lc"):
            self.long_term_lc(image, self.n)

        if (self.n + 1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N * 2}"')

        frame_index: int = self.counter
        keyframes_before: int = self.n
        frame_depth: FrameDepth = FrameDepth(depth=depth, mask=depth_mask)
        disps: Float32[Tensor, "1 1 h4 w4"] | None = self.depth_prior.prepare(self, (int(image.shape[1]), int(image.shape[2])), frame_depth)

        image_normalized: Float[Tensor, "1 1 3 ht wd"] = 2 * (image[None, None] / 255.0) - 0.5

        with torch.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
            fmap, gmap, imap, patches, _, clr = self.network.patchify(
                image_normalized,
                patches_per_image=self.cfg.patches_per_frame,
                disps=disps,
                gradient_bias=self.cfg.gradient_bias,
                return_color=True,
            )

        # Depth initialization (before any state write, so the prior can still reject the frame)
        self.depth_prior.init_patch_depth(self, patches)

        ### Update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        clr_rgb: Float[Tensor, "M 3"] = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr_rgb.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        # Motion model
        if self.n > 1:
            if self.cfg.motion_model == "DAMPED_LINEAR":
                P1: SE3 = SE3(self.pg.poses_[self.n - 1])
                P2: SE3 = SE3(self.pg.poses_[self.n - 2])

                relative = P1 * P2.inv()
                assert isinstance(relative, SE3)
                if self.cfg.motion_scale_by_tstamps:
                    # Scale by the ratio of the last two timestamp steps (varying camera rate)
                    *_, a, b, c = [1] * 3 + self.tlist
                    fac: float = (c - b) / (b - a)
                    xi: Float[Tensor, "6"] = self.cfg.motion_damping * fac * relative.log()
                else:
                    xi = self.cfg.motion_damping * relative.log()
                predicted = SE3.exp(xi) * P1
                assert isinstance(predicted, SE3)
                self.pg.poses_[self.n] = predicted.data
            else:
                self.pg.poses_[self.n] = self.poses[self.n - 1]

        self.pg.patches_[self.n] = patches

        ### Update network attributes (circular buffer) ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        # Motion check (pre-initialization only)
        if self.n > 0 and not self.is_initialized and self.motion_probe() < 2.0:
            self.pg.delta[self.counter - 1] = (self.counter - 2, self.identity[0])
            self._notify_frame(tstamp, frame_index, keyframes_before, False, image, intrinsics, frame_depth)
            return

        self.n += 1
        self.m += self.M

        # Proximity loop closure: add edges from old patches to recent frames
        if self.cfg.loop_closure and self.n - self.last_global_ba >= self.cfg.global_opt_freq:
            lii, ljj = self.pg.edges_loop()
            if lii.numel() > 0:
                self.last_global_ba = self.n
                self.append_factors(lii, ljj)
                self._append_extra_loop_edges(lii, ljj)

        # Add forward and backward edges
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True
            for _ in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

        # Classical loop closure attempt (if enabled)
        if self.cfg.classic_loop_closure and hasattr(self, "long_term_lc"):
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()

        self._notify_frame(tstamp, frame_index, keyframes_before, True, image, intrinsics, frame_depth)

    def _notify_frame(
        self,
        tstamp: int,
        frame_index: int,
        keyframes_before: int,
        keyframe_added: bool,
        image: UInt8[Tensor, "3 ht wd"],
        intrinsics: Float[Tensor, "4"],
        frame_depth: FrameDepth,
    ) -> None:
        """Fire every observer's ``on_frame`` for the frame that ``__call__`` just processed."""
        if not self.observers:
            return
        event: FrameEvent = FrameEvent(
            tstamp=tstamp,
            frame_index=frame_index,
            keyframes_before=keyframes_before,
            keyframe_added=keyframe_added,
            image=image,
            intrinsics=intrinsics,
            depth=frame_depth,
        )
        for observer in self.observers:
            observer.on_frame(self, event)
