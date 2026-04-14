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

The system maintains fixed-size GPU buffers pre-allocated at construction
time via :class:`~dpvo.patchgraph.PatchGraph`, avoiding dynamic memory
allocation during tracking.
"""

import functools
from collections import OrderedDict
from collections.abc import Generator

import lietorch
import numpy as np
import torch
import torch.nn.functional as F
from beartype.roar import BeartypeException
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Float64, Int, UInt8
from lietorch import SE3
from numpy import ndarray
from torch import Tensor

from . import altcorr, fastba
from . import projective_ops as pops
from .config import DPVOConfig
from .net import VONet
from .patchgraph import PatchGraph
from .utils import Timer, flatmeshgrid

autocast = functools.partial(torch.amp.autocast, "cuda")
Id: SE3 = SE3.Identity(1, device="cuda")
"""Pre-allocated identity SE3 element used as the relative pose for
skipped frames during initialization."""


class DPVO:
    """Sliding-window visual SLAM system using sparse deep patches.

    Manages all state for real-time visual odometry/SLAM: the
    :class:`PatchGraph` (poses, patches, edges), feature maps, and the
    neural network.  Designed to be called once per frame via
    :meth:`__call__`.
    """

    def __init__(self, cfg: DPVOConfig, network: str | VONet, ht: int = 480, wd: int = 640) -> None:
        self.cfg: DPVOConfig = cfg
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

        # dummy image for visualization
        self.image_: UInt8[Tensor, "ht wd 3"] = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        self.feature_dtype: torch.dtype = torch.half if self.cfg.mixed_precision else torch.float
        self.kwargs: dict[str, object] = {"device": "cuda", "dtype": self.feature_dtype}

        ### frame memory size ###
        self.pmem: int = 36
        self.mem: int = 36
        if self.cfg.loop_closure:
            self.last_global_ba: int = -1000
            self.pmem = self.cfg.max_edge_age

        self.imap_: Float[Tensor, "pmem M DIM"] = torch.zeros(self.pmem, self.M, DIM, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]
        self.gmap_: Float[Tensor, "pmem M 128 P P"] = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]

        # PatchGraph holds all buffer state
        self.pg: PatchGraph = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **self.kwargs)

        # Classical loop closure (optional)
        if self.cfg.classic_loop_closure:
            self.load_long_term_loop_closure()

        ht_feat: int = ht // RES
        wd_feat: int = wd // RES

        self.fmap1_: Float[Tensor, "1 mem 128 h4 w4"] = torch.zeros(1, self.mem, 128, ht_feat // 1, wd_feat // 1, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]
        self.fmap2_: Float[Tensor, "1 mem 128 h16 w16"] = torch.zeros(1, self.mem, 128, ht_feat // 4, wd_feat // 4, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]

        self.pyramid: tuple[Tensor, Tensor] = (self.fmap1_, self.fmap2_)

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

        self.network.cuda()
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

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        # Build lookup from internal timestamp -> pose for active keyframes
        self.traj: dict[int, Float32[Tensor, "7"]] = {}
        for i in range(self.n):
            self.traj[int(self.pg.tstamps_[i])] = self.pg.poses_[i]

        # Reconstruct poses for ALL timestamps
        poses_list: list[SE3] = [self.get_pose(t) for t in range(self.counter)]
        poses_stacked: SE3 = lietorch.stack(poses_list, dim=0)
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
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:, m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:, m]), dim=1)
        self.pg.weight = self.pg.weight[:, ~m]
        self.pg.target = self.pg.target[:, ~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:, ~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self) -> Float[Tensor, ""]:
        """Check if there is sufficient parallax for system initialization."""
        kk: Int[Tensor, "M"] = torch.arange(self.m - self.M, self.m, device="cuda")
        jj: Int[Tensor, "M"] = self.n * torch.ones_like(kk)
        ii: Int[Tensor, "M"] = self.ix[kk]

        net: Float[Tensor, "1 M DIM"] = torch.zeros(1, len(ii), self.DIM, **self.kwargs)  # pyrefly: ignore[bad-argument-type,no-matching-overload]
        coords: Float[Tensor, "1 M 2 P P"] = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.mixed_precision):
            corr: Float[Tensor, "1 M corr_feat"] = self.corr(coords, indicies=(kk, jj))
            ctx: Float[Tensor, "1 M DIM"] = self.imap[:, kk % (self.M * self.pmem)]
            net, (delta, weight, _) = self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i: int, j: int) -> float:
        """Compute mean pixel flow magnitude for all edges from frame i to frame j."""
        k: Bool[Tensor, "n_edges"] = (self.pg.ii == i) & (self.pg.jj == j)
        ii: Int[Tensor, "n_matched"] = self.pg.ii[k]
        jj: Int[Tensor, "n_matched"] = self.pg.jj[k]
        kk: Int[Tensor, "n_matched"] = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self) -> None:
        """Remove redundant keyframes based on pixel flow magnitude."""
        i: int = self.n - self.cfg.keyframe_index - 1
        j: int = self.n - self.cfg.keyframe_index + 1
        m_flow: float = self.motionmag(i, j) + self.motionmag(j, i)

        if m_flow / 2 < self.cfg.keyframe_thresh:
            k: int = self.n - self.cfg.keyframe_index
            t0: int = int(self.pg.tstamps_[k - 1])
            t1: int = int(self.pg.tstamps_[k])

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k - 1]).inv()
            assert isinstance(dP, SE3)
            self.pg.delta[t1] = (t0, dP)

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
                self.pg.intrinsics_[idx] = self.pg.intrinsics_[idx + 1]

                self.imap_[idx % self.pmem] = self.imap_[(idx + 1) % self.pmem]
                self.gmap_[idx % self.pmem] = self.gmap_[(idx + 1) % self.pmem]
                self.fmap1_[0, idx % self.mem] = self.fmap1_[0, (idx + 1) % self.mem]
                self.fmap2_[0, idx % self.mem] = self.fmap2_[0, (idx + 1) % self.mem]

            self.n -= 1
            self.m -= self.M

            if self.cfg.classic_loop_closure and hasattr(self, "long_term_lc"):
                self.long_term_lc.keyframe(k)

        # Prune stale edges
        to_remove: Bool[Tensor, "n_edges"] = self.ix[self.pg.kk] < self.n - self.cfg.removal_window
        if self.cfg.loop_closure:
            # Keep loop closure edges (long-range edges in the optimization window)
            lc_edges: Bool[Tensor, "n_edges"] = ((self.pg.jj - self.pg.ii) > 30) & (
                self.pg.jj > (self.n - self.cfg.optimization_window)
            )
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self) -> None:
        """Global bundle adjustment including both active and inactive edges."""
        full_target: Float[Tensor, "1 n_all 2"] = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight: Float[Tensor, "1 n_all 2"] = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii: Int[Tensor, "n_all"] = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj: Int[Tensor, "n_all"] = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk: Int[Tensor, "n_all"] = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda: Float[Tensor, "1"] = torch.as_tensor([1e-4], device="cuda")
        t0: int = int(self.pg.ii.min().item())
        fastba.BA(
            self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk,
            t0, self.n, M=self.M, iterations=2, eff_impl=True,
        )
        self.ran_global_ba[self.n] = True

    def update(self) -> None:
        """Run one iteration of the DPVO tracking pipeline."""
        with Timer("other", enabled=self.enable_timing):
            coords: Float[Tensor, "1 n_edges 2 P P"] = self.reproject()

            with autocast(enabled=True):
                corr: Float[Tensor, "1 n_edges corr_feat"] = self.corr(coords)
                ctx: Float[Tensor, "1 n_edges DIM"] = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = self.network.update(
                    self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk
                )

            lmbda: Float[Tensor, "1"] = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target: Float[Tensor, "1 n_edges 2"] = coords[..., self.P // 2, self.P // 2] + delta.float()

        # Store target/weight on PatchGraph for global BA
        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # Run global BA if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.removal_window - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0: int = self.n - self.cfg.optimization_window if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(
                        self.poses, self.patches, self.intrinsics,
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk,
                        t0, self.n, M=self.M, iterations=2, eff_impl=False,
                    )
            except BeartypeException:
                raise
            except Exception:
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
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        """Generate backward edges: newest frame's patches -> recent frames."""
        r: int = self.cfg.patch_lifetime
        t0: int = self.M * max((self.n - 1), 0)
        t1: int = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def __call__(self, tstamp: int, image: UInt8[Tensor, "3 ht wd"], intrinsics: Float[Tensor, "4"]) -> None:
        """Process a new frame: extract patches, predict pose, update and optimize."""

        # Feed image to classical loop closure retrieval (if enabled)
        if self.cfg.classic_loop_closure and hasattr(self, "long_term_lc"):
            self.long_term_lc(image, self.n)

        if (self.n + 1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N * 2}"')

        image_normalized: Float[Tensor, "1 1 3 ht wd"] = 2 * (image[None, None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.mixed_precision):
            fmap, gmap, imap, patches, _, clr = self.network.patchify(
                image_normalized,
                patches_per_image=self.cfg.patches_per_frame,
                gradient_bias=self.cfg.gradient_bias,
                return_color=True,
            )

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
                xi: Float[Tensor, "6"] = self.cfg.motion_damping * relative.log()
                predicted = SE3.exp(xi) * P1
                assert isinstance(predicted, SE3)
                self.pg.poses_[self.n] = predicted.data
            else:
                self.pg.poses_[self.n] = self.poses[self.n - 1]

        # Depth initialization
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized:
            s: Float[Tensor, ""] = torch.median(self.pg.patches_[self.n - 3 : self.n, :, 2])
            patches[:, :, 2] = s

        self.pg.patches_[self.n] = patches

        ### Update network attributes (circular buffer) ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        # Motion check (pre-initialization only)
        if self.n > 0 and not self.is_initialized and self.motion_probe() < 2.0:
            self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
            return

        self.n += 1
        self.m += self.M

        # Proximity loop closure: add edges from old patches to recent frames
        if self.cfg.loop_closure and self.n - self.last_global_ba >= self.cfg.global_opt_freq:
            lii, ljj = self.pg.edges_loop()
            if lii.numel() > 0:
                self.last_global_ba = self.n
                self.append_factors(lii, ljj)

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
