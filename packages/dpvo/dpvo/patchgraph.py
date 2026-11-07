"""PatchGraph -- data structure for DPV-SLAM state management.

Holds all buffer state (poses, patches, timestamps, edges, etc.) that
was previously inline in the :class:`~dpvo.dpvo.DPVO` class.  Extracted
into a standalone class to support:

- **Inactive edges**: edges from removed keyframes that are kept for
  global bundle adjustment.
- **Loop closure edges**: via :meth:`edges_loop`, which creates edges
  from old patches to recent frames based on camera proximity.
- **Depth normalization**: :meth:`normalize` for scale consistency.
- **Per-frame attributes**: :meth:`add_frame_attr` registers extra
  keyframe-slot buffers (e.g. per-patch depth priors) that
  :meth:`~dpvo.dpvo.DPVO.keyframe` compacts together with the patches.

See Lipson et al. (2024), "Deep Patch Visual SLAM" for details.
"""

import numpy as np
import torch
from einops import reduce, repeat
from jaxtyping import Float, Float32, Int, UInt8
from lietorch_mojo import SE3
from torch import Tensor

from . import projective_ops as pops
from .config import DPVOConfig
from .utils import flatmeshgrid


class PatchGraph:
    """Buffer state for the DPV-SLAM sliding-window system.

    Manages all per-keyframe buffers (poses, patches, intrinsics, etc.),
    the factor graph edge indices, and inactive edges for global BA.

    Args:
        cfg: DPVO configuration.
        P: Patch side length (typically 3).
        DIM: GRU hidden state dimensionality.
        pmem: Patch memory size (circular buffer size for imap/gmap).
        device: Device of every buffer.
        feature_dtype: Dtype of the GRU hidden state ``net`` (half under mixed precision).
    """

    def __init__(self, cfg: DPVOConfig, P: int, DIM: int, pmem: int, *, device: torch.device, feature_dtype: torch.dtype) -> None:
        self.cfg: DPVOConfig = cfg
        self.P: int = P
        self.pmem: int = pmem
        self.DIM: int = DIM
        self.device: torch.device = device

        self.n: int = 0  # number of active keyframes
        self.m: int = 0  # number of active patches

        self.M: int = self.cfg.patches_per_frame
        self.N: int = self.cfg.buffer_size

        # Fixed-size GPU buffers
        self.tstamps_: np.ndarray = np.zeros(self.N, dtype=np.int64)
        self.poses_: Float32[Tensor, "N 7"] = torch.zeros(self.N, 7, dtype=torch.float, device=device)
        self.patches_: Float32[Tensor, "N M 3 P P"] = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device=device)
        self.intrinsics_: Float32[Tensor, "N 4"] = torch.zeros(self.N, 4, dtype=torch.float, device=device)

        self.points_: Float32[Tensor, "NM 3"] = torch.zeros(self.N * self.M, 3, dtype=torch.float, device=device)
        self.colors_: UInt8[Tensor, "N M 3"] = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device=device)

        self.index_: Int[Tensor, "N M"] = torch.zeros(self.N, self.M, dtype=torch.long, device=device)
        self.index_map_: Int[Tensor, "N"] = torch.zeros(self.N, dtype=torch.long, device=device)

        # Initialize poses to identity (quaternion w=1)
        self.poses_[:, 6] = 1.0

        # Store relative poses for removed frames (for interpolation at terminate)
        self.delta: dict[int, tuple[int, SE3]] = {}

        # Extra per-keyframe-slot buffers registered by extensions (see add_frame_attr)
        self.frame_attrs: dict[str, Tensor] = {}

        ### Active edge information ###
        self.net: Float[Tensor, "1 n_edges DIM"] = torch.zeros(1, 0, DIM, device=device, dtype=feature_dtype)
        self.ii: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device=device)
        self.kk: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device=device)

        # Target and weight for active edges (set during update())
        self.target: Float[Tensor, "1 0 2"] = torch.zeros(1, 0, 2, device=device)
        self.weight: Float[Tensor, "1 0 2"] = torch.zeros(1, 0, 2, device=device)

        ### Inactive edge information (removed from GRU updates, kept for global BA) ###
        self.ii_inac: Int[Tensor, "n_inac"] = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac: Int[Tensor, "n_inac"] = torch.as_tensor([], dtype=torch.long, device=device)
        self.kk_inac: Int[Tensor, "n_inac"] = torch.as_tensor([], dtype=torch.long, device=device)
        self.weight_inac: Float[Tensor, "1 0 2"] = torch.zeros(1, 0, 2, dtype=torch.float, device=device)
        self.target_inac: Float[Tensor, "1 0 2"] = torch.zeros(1, 0, 2, dtype=torch.float, device=device)

    def add_frame_attr(self, name: str, tail: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        """Register a zero-initialised per-keyframe-slot buffer ``(N, *tail)`` on the graph's device.

        :meth:`~dpvo.dpvo.DPVO.keyframe` shifts every registered buffer with the
        patches when it removes a keyframe and zeroes the slot it vacates.

        Args:
            name: Registry key; must be new.
            tail: Shape after the keyframe-slot axis, e.g. ``(M,)`` for one value per patch.
            dtype: Buffer dtype.

        Returns:
            The registered buffer, also held in :attr:`frame_attrs`.

        Raises:
            ValueError: ``name`` is already registered.
        """
        if name in self.frame_attrs:
            raise ValueError(f"frame attribute {name!r} is already registered")
        buffer: Tensor = torch.zeros(self.N, *tail, dtype=dtype, device=self.device)
        self.frame_attrs[name] = buffer
        return buffer

    def edges_loop(self) -> tuple[Int[Tensor, "n_loop"], Int[Tensor, "n_loop"]]:
        """Create loop closure edges from old patches to recent frames.

        Finds pairs of (old patch, recent frame) where the camera has
        revisited an old location.  Filters by flow magnitude and applies
        NMS to avoid redundant edges.

        Returns:
            Tuple of ``(kk, jj)`` -- flat patch indices and target frame
            indices for the new loop closure edges.  Empty tensors if no
            valid edges found.
        """
        from einops import asnumpy

        from .loop_closure.optim_utils import reduce_edges

        lc_range: int = self.cfg.max_edge_age
        old_bound: int = self.n - self.cfg.removal_window  # upper bound for "old" patches

        if old_bound <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        # Create candidate edges: recent frames × old patches
        jj, kk = flatmeshgrid(
            torch.arange(self.n - self.cfg.global_opt_freq, self.n - self.cfg.keyframe_index, device=self.device),
            torch.arange(max(old_bound - lc_range, 0) * self.M, old_bound * self.M, device=self.device),
            indexing="ij",
        )
        ii = self.ix[kk]

        # Compute flow magnitude to filter out edges with too-large motion
        flow_mg, val = pops.flow_mag(
            SE3(self.poses),
            self.patches[..., 1, 1].view(1, -1, 3, 1, 1),
            self.intrinsics,
            ii, jj, kk,
            beta=0.5,
        )
        flow_mg_sum: Float[Tensor, "fl"] = reduce(flow_mg * val, "1 (fl M) 1 1 -> fl", "sum", M=self.M).float()
        num_val: Int[Tensor, "fl"] = reduce(val, "1 (fl M) 1 1 -> fl", "sum", M=self.M).clamp(min=1)
        flow_mag_vals: Float[Tensor, "fl"] = torch.where(num_val > (self.M * 0.75), flow_mg_sum / num_val, torch.inf)

        mask = flow_mag_vals < self.cfg.backend_thresh
        es = reduce_edges(
            asnumpy(flow_mag_vals[mask]),
            asnumpy(ii[:: self.M][mask]),
            asnumpy(jj[:: self.M][mask]),
            max_num_edges=1000,
            nms=1,
        )

        edges: Int[Tensor, "E 2"] = torch.as_tensor(es, device=ii.device)
        if edges.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        ii_out, jj_out = repeat(edges, "E ij -> ij E M", M=self.M, ij=2)
        kk_out = ii_out.mul(self.M) + torch.arange(self.M, device=ii.device)
        return kk_out.flatten(), jj_out.flatten()

    def normalize(self) -> None:
        """Normalize depth scale and re-center poses.

        Divides all inverse depths by their mean and scales translations
        accordingly, then re-centers the trajectory so pose[0] is identity.
        Updates the point cloud and delta entries.
        """
        s: Float[Tensor, ""] = self.patches_[: self.n, :, 2].mean()
        self.patches_[: self.n, :, 2] /= s
        self.poses_[: self.n, :3] *= s
        for t, (t0, dP) in self.delta.items():
            self.delta[t] = (t0, dP.scale(s))
        recentered = SE3(self.poses_[: self.n]) * SE3(self.poses_[[0]]).inv()
        assert recentered is not None
        self.poses_[: self.n] = recentered.data

        points: Float[Tensor, "1 m P P 4"] = pops.point_cloud(SE3(self.poses), self.patches[:, : self.m], self.intrinsics, self.ix[: self.m])
        points_3d: Float[Tensor, "m 3"] = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
        self.points_[: len(points_3d)] = points_3d[:]

    @property
    def poses(self) -> Float32[Tensor, "1 N 7"]:
        """All poses as a batched tensor ``(1, N, 7)``."""
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self) -> Float32[Tensor, "1 NM 3 3 3"]:
        """All patches flattened to ``(1, N*M, 3, 3, 3)``."""
        return self.patches_.view(1, self.N * self.M, 3, 3, 3)

    @property
    def intrinsics(self) -> Float32[Tensor, "1 N 4"]:
        """All intrinsics as a batched tensor ``(1, N, 4)``."""
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self) -> Int[Tensor, "NM"]:
        """Flat frame-ownership index for each patch slot."""
        return self.index_.view(-1)
