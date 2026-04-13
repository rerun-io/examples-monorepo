"""Main DPVO (Deep Patch Visual Odometry) inference class.

Implements the sliding-window visual odometry system described in
Teed et al. (2022), "Deep Patch Visual Odometry".

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
   step (:func:`~dpvo.ba.BA`) to refine poses and depths.
5. **Keyframe management**: Remove redundant keyframes whose
   bidirectional pixel flow falls below ``KEYFRAME_THRESH``.

The system maintains fixed-size GPU buffers pre-allocated at construction
time, avoiding dynamic memory allocation during tracking.
"""

from collections import OrderedDict
from collections.abc import Generator

import lietorch
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Float64, Int, UInt8
from lietorch import SE3
from torch import Tensor

from . import altcorr, fastba
from . import projective_ops as pops
from .config import DPVOConfig
from .net import VONet
from .utils import Timer, flatmeshgrid

autocast = torch.cuda.amp.autocast
Id: SE3 = SE3.Identity(1, device="cuda")
"""Pre-allocated identity SE3 element used as the relative pose for
skipped frames during initialization."""


class DPVO:
    """Sliding-window visual odometry system using sparse deep patches.

    Manages all state for real-time visual odometry: pose buffers,
    patch coordinates, feature maps, the measurement graph (factor graph),
    and the neural network.  Designed to be called once per frame via
    :meth:`__call__`.

    Attributes:
        cfg: DPVO configuration (see :class:`dpvo.config.DPVOConfig`).
        network: The :class:`~dpvo.net.VONet` neural network (in eval mode).
        is_initialized: ``True`` once 8 frames have been processed and the
            initial BA has converged.
        enable_timing: If ``True``, profile each update step with CUDA events.
        n: Number of active keyframes currently in the buffer.
        m: Total number of active patches (= ``n * M``).
        M: Patches per frame (from ``cfg.patches_per_frame``).
        N: Maximum buffer size (from ``cfg.buffer_size``).
        ht: Image height (pixels).
        wd: Image width (pixels).
        DIM: GRU hidden state dimensionality (from network).
        RES: Feature map stride (from network, typically 4).
        P: Patch side length (from network, typically 3).
        tlist: List of original timestamps for each processed frame.
        counter: Running count of all frames seen (including skipped ones).
        tstamps_: Per-keyframe internal timestamps, shape ``(N,)``.
        poses_: Per-keyframe SE3 poses as ``[tx, ty, tz, qx, qy, qz, qw]``,
            shape ``(N, 7)``.
        patches_: Per-keyframe patch coordinates ``(x, y, inv_depth)``,
            shape ``(N, M, 3, P, P)``.
        intrinsics_: Per-keyframe camera intrinsics ``(fx, fy, cx, cy)``
            at stride-4 resolution, shape ``(N, 4)``.
        points_: 3-D point cloud from backprojected patches, shape ``(N*M, 3)``.
        colors_: Per-patch RGB colors for visualization, shape ``(N, M, 3)``.
        imap_: Context features (from ``inet``), circular buffer of size
            ``(mem, M, DIM)``.
        gmap_: Patch feature descriptors (from ``fnet``), circular buffer
            of size ``(mem, M, 128, P, P)``.
        fmap1_: Dense feature maps at stride-4, circular buffer.
        fmap2_: Dense feature maps at stride-16, circular buffer.
        pyramid: 2-level feature pyramid ``(fmap1_, fmap2_)`` for correlation.
        net: GRU hidden states for all active edges, shape
            ``(1, n_edges, DIM)``.
        ii: Source frame index per edge (which frame the patch came from).
        jj: Target frame index per edge (which frame the patch is observed in).
        kk: Patch index per edge (flat index into the patch buffer).
        delta: Dictionary mapping removed-frame timestamps to
            ``(reference_timestamp, relative_SE3_pose)`` for interpolation
            at termination.
        mem: Circular buffer capacity for feature maps (default 32).
    """

    def __init__(self, cfg: DPVOConfig, network: str | VONet, ht: int = 480, wd: int = 640) -> None:
        self.cfg: DPVOConfig = cfg
        self.load_weights(network)
        self.is_initialized: bool = False
        self.enable_timing: bool = False

        self.n: int = 0  # number of active keyframes
        self.m: int = 0  # total number of active patches
        self.M: int = self.cfg.patches_per_frame
        self.N: int = self.cfg.buffer_size

        self.ht: int = ht  # image height
        self.wd: int = wd  # image width

        DIM: int = self.DIM
        RES: int = self.RES

        ### state attributes ###
        self.tlist: list[int] = []
        self.counter: int = 0

        # dummy image for visualization
        self.image_: UInt8[Tensor, "ht wd 3"] = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        # Fixed-size GPU buffers pre-allocated for the sliding window
        # tstamps_ is float64 because evaluation datasets (EuRoC, TUM-RGBD)
        # pass real-valued timestamps (nanoseconds / seconds).  The dpvo
        # video/image streams happen to use integer frame indices, but the
        # original DPVO evaluation code relies on float timestamps for
        # trajectory alignment via evo's associate_trajectories().
        self.tstamps_: Float64[Tensor, "N"] = torch.zeros(self.N, dtype=torch.float64, device="cuda")
        self.poses_: Float32[Tensor, "N 7"] = torch.zeros(self.N, 7, dtype=torch.float32, device="cuda")
        self.patches_: Float32[Tensor, "N M 3 P P"] = torch.zeros(
            self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda"
        )
        self.intrinsics_: Float32[Tensor, "N 4"] = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_: Float32[Tensor, "NM 3"] = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_: UInt8[Tensor, "N M 3"] = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_: Int[Tensor, "N M"] = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_: Int[Tensor, "N"] = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        # Circular buffer size for feature maps (to avoid storing all frames)
        self.mem: int = 32

        # Feature dtype: half for mixed precision, float otherwise
        self.feature_dtype: torch.dtype = torch.half if self.cfg.mixed_precision else torch.float

        # Circular buffers for per-frame network features
        self.imap_: Float[Tensor, "mem M DIM"] = torch.zeros(self.mem, self.M, DIM, device="cuda", dtype=self.feature_dtype)
        self.gmap_: Float[Tensor, "mem M 128 P P"] = torch.zeros(self.mem, self.M, 128, self.P, self.P, device="cuda", dtype=self.feature_dtype)

        ht: int = ht // RES
        wd: int = wd // RES

        # Two-level feature pyramid for correlation computation
        self.fmap1_: Float[Tensor, "1 mem 128 h4 w4"] = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, device="cuda", dtype=self.feature_dtype)
        self.fmap2_: Float[Tensor, "1 mem 128 h16 w16"] = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, device="cuda", dtype=self.feature_dtype)

        self.pyramid: tuple[Float[Tensor, "1 mem 128 h4 w4"], Float[Tensor, "1 mem 128 h16 w16"]] = (self.fmap1_, self.fmap2_)

        # GRU hidden state and factor graph edge indices (dynamically sized)
        self.net: Float[Tensor, "1 n_edges DIM"] = torch.zeros(1, 0, DIM, device="cuda", dtype=self.feature_dtype)
        self.ii: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device="cuda")

        # Initialize all poses to identity (quaternion w=1)
        self.poses_[:, 6] = 1.0

        # Store relative poses for removed keyframes, used for interpolation
        # at termination. Maps timestamp -> (reference_timestamp, relative_SE3).
        self.delta: dict[int, tuple[int, SE3]] = {}

    def load_weights(self, network: str | VONet) -> None:
        """Load network weights from a checkpoint file or use an existing VONet.

        When loading from a checkpoint string path, the state dict is
        cleaned: ``"module."`` prefixes (from ``DataParallel``) are stripped,
        and the deprecated ``update.lmbda`` key is removed.

        After loading, network attributes ``DIM``, ``RES``, and ``P`` are
        copied to the DPVO instance, and the network is moved to CUDA in
        eval mode.

        Args:
            network: Either a file path to a ``.pth`` checkpoint or an
                already-instantiated :class:`~dpvo.net.VONet`.
        """
        # load network from checkpoint file
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

        # Copy network architecture constants to the DPVO instance
        self.DIM: int = self.network.DIM
        self.RES: int = self.network.RES
        self.P: int = self.network.P

        self.network.cuda()
        self.network.eval()

    @property
    def poses(self) -> Float32[Tensor, "1 N 7"]:
        """All poses as a batched tensor ``(1, N, 7)`` for projective_ops."""
        return rearrange(self.poses_, "n c -> 1 n c")

    @property
    def patches(self) -> Float32[Tensor, "1 NM 3 3 3"]:
        """All patches flattened to ``(1, N*M, 3, 3, 3)`` for projective_ops."""
        return rearrange(self.patches_, "n m c p1 p2 -> 1 (n m) c p1 p2")

    @property
    def intrinsics(self) -> Float32[Tensor, "1 N 4"]:
        """All intrinsics as a batched tensor ``(1, N, 4)``."""
        return rearrange(self.intrinsics_, "n c -> 1 n c")

    @property
    def ix(self) -> Int[Tensor, "NM"]:
        """Flat frame-ownership index for each patch slot.

        ``ix[k]`` gives the frame index that patch ``k`` belongs to.
        Used to map from flat patch index to source frame.
        """
        return self.index_.view(-1)

    @property
    def imap(self) -> Float[Tensor, "1 mem_times_M DIM"]:
        """Context features from the circular buffer, flattened for indexing."""
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self) -> Float[Tensor, "1 mem_times_M 128 3 3"]:
        """Patch descriptors from the circular buffer, flattened for correlation."""
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t: int) -> SE3:
        """Retrieve the pose for timestamp ``t``, interpolating removed frames.

        If ``t`` is in :attr:`traj` (i.e. it is an active keyframe), the
        pose is returned directly.  Otherwise, ``t`` was a removed
        keyframe and its relative pose delta stored in :attr:`delta` is
        composed with the reference frame's pose (which may itself be
        removed, so this recurses).

        This is called during :meth:`terminate` to reconstruct poses for
        all original timestamps.

        Args:
            t: Timestamp to look up.

        Returns:
            SE3 pose for timestamp ``t``.
        """
        if t in self.traj:
            return SE3(self.traj[t])

        t0: int
        dP: SE3
        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self) -> tuple[Float32[np.ndarray, "n_frames 7"], Float64[np.ndarray, "n_frames"]]:
        """Finalize tracking: interpolate removed keyframes and return full trajectory.

        After the input stream ends, this method reconstructs the complete
        trajectory for all original timestamps (including frames that were
        removed by keyframe management).  Removed frames have their poses
        recovered via the relative pose deltas stored in :attr:`delta`.

        The returned poses are *camera-to-world* (inverted from the
        internal world-to-camera representation).

        Returns:
            A 2-tuple of:
            - ``poses``: Camera-to-world poses as ``[tx, ty, tz, qx, qy, qz, qw]``,
              shape ``(n_frames, 7)``.
            - ``tstamps``: Original timestamps, shape ``(n_frames,)``.
        """
        print("Terminating...")
        # Build lookup from internal timestamp -> pose for active keyframes
        self.traj: dict[int, Float32[Tensor, "7"]] = {}
        for i in range(self.n):
            current_t: int = int(self.tstamps_[i].item())
            self.traj[current_t] = self.poses_[i]

        # Reconstruct poses for ALL timestamps (including removed keyframes)
        poses: list[SE3] = [self.get_pose(t) for t in range(self.counter)]
        poses: SE3 = lietorch.stack(poses, dim=0)
        # Invert: internal representation is world-to-camera, output is camera-to-world
        poses: Float[np.ndarray, "n_frames 7"] = poses.inv().data.cpu().numpy()
        tstamps: Float64[np.ndarray, "n_frames"] = np.array(self.tlist, dtype=np.float64)
        print("Done!")

        return poses, tstamps

    def corr(self, coords: Float[Tensor, "1 n_edges 2 P P"], indicies: tuple[Int[Tensor, "n_edges"], Int[Tensor, "n_edges"]] | None = None) -> Float[Tensor, "1 n_edges corr_feat"]:
        """Compute local correlation volumes at two pyramid levels.

        For each edge, computes normalized cross-correlation between the
        patch descriptor ``gmap[ii]`` and the feature map of frame ``jj``
        in a 7x7 neighbourhood (radius R=3) around the reprojected
        coordinates.  This is done at two scales:

        - Level 1 (stride-4 ``fmap1_``): Fine-grained matching.
        - Level 2 (stride-16 ``fmap2_``): Coarse contextual matching.

        The two correlation volumes are stacked to form a feature vector
        of dimension ``2 * (2*3+1)^2 = 98`` per patch pixel.
        See Sec. 3.2 of Teed et al. (2022).

        Note: Indices are taken modulo the circular buffer size ``mem``
        to correctly index into the feature ring buffers.

        Args:
            coords: Reprojected 2-D coordinates at stride-4 resolution,
                shape ``(1, n_edges, 2, P, P)``.
            indicies: Optional ``(patch_indices, frame_indices)`` tuple.
                If ``None``, defaults to ``(self.kk, self.jj)``.

        Returns:
            Stacked correlation features, shape
            ``(1, n_edges, 2 * 49 * P^2)``.
        """
        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        # Map to circular buffer indices
        ii1: Int[Tensor, "n_edges"] = ii % (self.M * self.mem)
        jj1: Int[Tensor, "n_edges"] = jj % (self.mem)
        # Level 1: stride-4 features (coords as-is)
        # Output shape: (1, n_edges, 2R+1, 2R+1, P, P) where R=3, P=3
        corr1: Float[Tensor, "1 n_edges neighborhood neighborhood P P"] = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        # Level 2: stride-16 features (coords scaled by 1/4)
        corr2: Float[Tensor, "1 n_edges neighborhood neighborhood P P"] = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return rearrange(torch.stack([corr1, corr2], -1), "b e ... -> b e (...)")

    def reproject(self, indicies: tuple[Int[Tensor, "n_edges"], Int[Tensor, "n_edges"], Int[Tensor, "n_edges"]] | None = None) -> Float[Tensor, "1 n_edges 2 P P"]:
        """Reproject patches from their source frames into target frames.

        For each edge ``(ii, jj, kk)``, transforms patch ``kk`` (which
        lives in frame ``ii``'s coordinate system) into frame ``jj``
        using the current pose estimates, then projects to 2-D pixel
        coordinates.

        The output is permuted from ``(..., P, P, 2)`` to ``(..., 2, P, P)``
        to match the channel-first layout expected by the correlation code.

        Args:
            indicies: Optional ``(ii, jj, kk)`` edge tuple.  If ``None``,
                uses the current graph edges ``(self.ii, self.jj, self.kk)``.

        Returns:
            Reprojected 2-D coordinates, shape ``(1, n_edges, 2, P, P)``.
        """
        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        kk: Int[Tensor, "n_edges"]
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords: Float[Tensor, "1 n_edges P P 2"] = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk
        )
        # Move xy-coordinate dim before patch spatial dims for the correlation code
        return rearrange(coords, "b e p1 p2 c -> b e c p1 p2")

    def append_factors(self, ii: Int[Tensor, "n_new"], jj: Int[Tensor, "n_new"]) -> None:
        """Add new measurement edges to the factor graph.

        Each edge connects a patch (indexed by ``ii`` as a flat patch index)
        to a target frame (indexed by ``jj``).  The source frame index is
        looked up from ``self.ix[ii]`` (which frame owns that patch).

        New edges get zero-initialized GRU hidden states.

        Args:
            ii: Flat patch indices for the new edges.
            jj: Target frame indices for the new edges.
        """
        self.jj: Int[Tensor, "n_edges"] = torch.cat([self.jj, jj])
        self.kk: Int[Tensor, "n_edges"] = torch.cat([self.kk, ii])
        self.ii: Int[Tensor, "n_edges"] = torch.cat([self.ii, self.ix[ii]])

        net: Float[Tensor, "1 n_new DIM"] = torch.zeros(1, len(ii), self.DIM, device="cuda", dtype=self.feature_dtype)
        self.net: Float[Tensor, "1 n_edges DIM"] = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m: Bool[Tensor, "n_edges"]) -> None:
        """Remove edges from the factor graph by boolean mask.

        Edges where ``m`` is ``True`` are removed.  The corresponding
        GRU hidden states are also dropped.

        Args:
            m: Boolean mask of shape ``(n_edges,)``; ``True`` for edges
                to remove.
        """
        self.ii: Int[Tensor, "n_remaining"] = self.ii[~m]
        self.jj: Int[Tensor, "n_remaining"] = self.jj[~m]
        self.kk: Int[Tensor, "n_remaining"] = self.kk[~m]
        self.net: Float[Tensor, "1 n_remaining DIM"] = self.net[:, ~m]

    def motion_probe(self) -> Float[Tensor, ""]:
        """Check if there is sufficient parallax for system initialization.

        Before initialization (first 8 frames), this function is called to
        verify that the new frame has enough motion relative to the previous
        keyframe.  It runs a single GRU update iteration on the most recent
        patches and returns the median predicted displacement.

        If the median displacement is less than 2 pixels, the frame is
        skipped (stored as a relative pose delta for later interpolation).
        This prevents the system from initializing with degenerate
        (near-zero baseline) geometry.

        Returns:
            Median L2 norm of the predicted pixel displacements (scalar).
        """
        # Use patches from the most recent frame only
        kk: Int[Tensor, "M"] = torch.arange(self.m - self.M, self.m, device="cuda")
        jj: Int[Tensor, "M"] = self.n * torch.ones_like(kk)
        ii: Int[Tensor, "M"] = self.ix[kk]

        net: Float[Tensor, "1 M DIM"] = torch.zeros(1, len(ii), self.DIM, device="cuda", dtype=self.feature_dtype)
        coords: Float[Tensor, "1 M 2 P P"] = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.mixed_precision):
            corr: Float[Tensor, "1 M corr_feat"] = self.corr(coords, indicies=(kk, jj))
            ctx: Float[Tensor, "1 M DIM"] = self.imap[:, kk % (self.M * self.mem)]
            net: Float[Tensor, "1 M DIM"]
            delta: Float[Tensor, "1 M 2"]
            weight: Float[Tensor, "1 M 2"]
            net, (delta, weight, _) = self.network.update(
                net, ctx, corr, None, ii, jj, kk
            )

        # Median displacement in pixels -- threshold is 2.0 in __call__
        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i: int, j: int) -> float:
        """Compute mean pixel flow magnitude for all edges from frame i to frame j.

        Filters the factor graph for edges where ``ii == i`` and ``jj == j``,
        then computes the blended (translation + rotation) flow magnitude
        via :func:`~dpvo.projective_ops.flow_mag`.

        Used by :meth:`keyframe` to assess whether a candidate keyframe
        has sufficient parallax to be worth keeping.

        Args:
            i: Source frame index.
            j: Target frame index.

        Returns:
            Mean flow magnitude in pixels (scalar float).
        """
        k: Bool[Tensor, "n_edges"] = (self.ii == i) & (self.jj == j)
        ii: Int[Tensor, "n_matched"] = self.ii[k]
        jj: Int[Tensor, "n_matched"] = self.jj[k]
        kk: Int[Tensor, "n_matched"] = self.kk[k]

        flow: Float[Tensor, "..."] = pops.flow_mag(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5
        )
        return flow.mean().item()

    def keyframe(self) -> None:
        """Remove redundant keyframes based on pixel flow magnitude.

        Checks whether the candidate keyframe at position
        ``n - KEYFRAME_INDEX`` has sufficient motion (parallax) relative
        to its neighbours.  If the average bidirectional flow is below
        ``KEYFRAME_THRESH`` (default 12.5 px), the frame is removed:

        1. Store the relative pose ``dP = P_k * P_{k-1}^{-1}`` in
           :attr:`delta` for later interpolation.
        2. Remove all factor graph edges touching frame ``k``.
        3. Shift all subsequent frame data (poses, patches, features,
           timestamps) down by one slot.
        4. Update edge indices accordingly.

        After the optional removal, edges whose source patch belongs to
        a frame older than ``n - REMOVAL_WINDOW`` are also pruned.

        See Sec. 3.4 of Teed et al. (2022) for keyframe management.
        """
        # Check flow between frames flanking the candidate
        i: int = self.n - self.cfg.keyframe_index - 1
        j: int = self.n - self.cfg.keyframe_index + 1
        # Average bidirectional flow (i->j and j->i)
        m: float = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.keyframe_thresh:
            # Insufficient parallax: remove the candidate keyframe
            k: int = self.n - self.cfg.keyframe_index
            t0: int = int(self.tstamps_[k - 1].item())
            t1: int = int(self.tstamps_[k].item())

            # Store relative pose for interpolation at termination
            dP: SE3 = SE3(self.poses_[k]) * SE3(self.poses_[k - 1]).inv()
            self.delta[t1] = (t0, dP)

            # Remove all edges incident to the removed frame
            to_remove: Bool[Tensor, "n_edges"] = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            # Shift edge indices: patches after frame k move down by M,
            # frame indices after k move down by 1
            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            # Compact the buffer: shift all data after position k down by one
            for i in range(k, self.n - 1):
                self.tstamps_[i] = self.tstamps_[i + 1]
                self.colors_[i] = self.colors_[i + 1]
                self.poses_[i] = self.poses_[i + 1]
                self.patches_[i] = self.patches_[i + 1]
                self.intrinsics_[i] = self.intrinsics_[i + 1]

                # Also shift circular buffer entries for network features
                self.imap_[i % self.mem] = self.imap_[(i + 1) % self.mem]
                self.gmap_[i % self.mem] = self.gmap_[(i + 1) % self.mem]
                self.fmap1_[0, i % self.mem] = self.fmap1_[0, (i + 1) % self.mem]
                self.fmap2_[0, i % self.mem] = self.fmap2_[0, (i + 1) % self.mem]

            self.n -= 1
            self.m -= self.M

        # Prune stale edges: remove edges whose source patch is too old
        to_remove: Bool[Tensor, "n_edges"] = self.ix[self.kk] < self.n - self.cfg.removal_window
        self.remove_factors(to_remove)

    def update(self) -> None:
        """Run one iteration of the DPVO tracking pipeline.

        This is the core per-step operation that combines neural network
        inference with geometric optimization:

        1. **Reproject** all active patches into their target frames using
           current pose estimates.
        2. **Correlate**: Compute multi-scale correlation features between
           patch descriptors and target frame feature maps.
        3. **GRU update**: Run the recurrent update operator to predict
           pixel displacements (``delta``) and confidence weights.
        4. **Bundle adjustment**: Solve for pose and depth updates using
           the C++ fast BA backend (``fastba.BA``).  Only the most recent
           ``OPTIMIZATION_WINDOW`` poses are optimised (after init); pose 0
           is always fixed (gauge freedom).
        5. **Point cloud**: Update the 3-D point cloud by backprojecting
           the refined patches.
        """
        with Timer("other", enabled=self.enable_timing):
            # Step 1: Reproject all patches through current poses
            coords: Float[Tensor, "1 n_edges 2 P P"] = self.reproject()

            with autocast(enabled=True):
                # Step 2: Compute correlation features
                corr: Float[Tensor, "1 n_edges corr_feat"] = self.corr(coords)
                # Look up context features from the circular buffer
                ctx: Float[Tensor, "1 n_edges DIM"] = self.imap[:, self.kk % (self.M * self.mem)]
                # Step 3: GRU update -> predicted displacement and confidence
                delta: Float[Tensor, "1 n_edges 2"]
                weight: Float[Tensor, "1 n_edges 2"]
                self.net, (delta, weight, _) = self.network.update(
                    self.net, ctx, corr, None, self.ii, self.jj, self.kk
                )

            # Damping parameter for the inverse-depth diagonal in BA
            lmbda: Float[Tensor, "1"] = torch.as_tensor([1e-4], device="cuda")
            weight: Float[Tensor, "1 n_edges 2"] = weight.float()
            # Target = current reprojection at patch center + learned delta
            target: Float[Tensor, "1 n_edges 2"] = coords[..., self.P // 2, self.P // 2] + delta.float()

        with Timer("BA", enabled=self.enable_timing):
            # Step 4: Bundle adjustment via the fast C++ backend
            # Only optimize poses from t0 to self.n (sliding window)
            t0: int = self.n - self.cfg.optimization_window if self.is_initialized else 1
            t0: int = max(t0, 1)

            try:
                fastba.BA(
                    self.poses,
                    self.patches,
                    self.intrinsics,
                    target,
                    weight,
                    lmbda,
                    self.ii,
                    self.jj,
                    self.kk,
                    t0,
                    self.n,
                    2,  # number of BA iterations
                )
            except Exception:
                print("Warning BA failed...")

            # Step 5: Update 3-D point cloud from refined patches
            # Backproject patches to world coordinates using center pixel
            points: Float[Tensor, "1 m P P 4"] = pops.point_cloud(
                SE3(self.poses),
                self.patches[:, : self.m],
                self.intrinsics,
                self.ix[: self.m],
            )
            # Extract center pixel (1,1) and convert from homogeneous to 3D
            points: Float[Tensor, "m 3"] = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
            self.points_[: len(points)] = points[:]

    def __edges_all(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        """Generate edges connecting ALL patches to ALL frames (fully connected).

        Returns:
            Generator yielding ``(patch_indices, frame_indices)`` as flat
            tensors from the Cartesian product.
        """
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_forw(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        """Generate forward edges: recent patches -> the newest frame.

        Creates edges from patches belonging to frames within the last
        ``PATCH_LIFETIME`` frames (excluding the newest frame itself) to
        the newest frame.  These allow existing patches to be observed in
        the new frame.

        Returns:
            Generator yielding ``(patch_indices, frame_indices)`` for the
            forward edges.
        """
        r: int = self.cfg.patch_lifetime
        t0: int = self.M * max((self.n - r), 0)
        t1: int = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        """Generate backward edges: newest frame's patches -> recent frames.

        Creates edges from the newest frame's patches to all frames within
        the last ``PATCH_LIFETIME`` frames.  These allow the new patches
        to refine their depth estimates by observing them in older frames.

        Returns:
            Generator yielding ``(patch_indices, frame_indices)`` for the
            backward edges.
        """
        r: int = self.cfg.patch_lifetime
        t0: int = self.M * max((self.n - 1), 0)
        t1: int = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def __call__(self, tstamp: int, image: UInt8[Tensor, "3 ht wd"], intrinsics: Float[Tensor, "4"]) -> None:
        """Process a new frame: extract patches, predict pose, update and optimize.

        This is the main entry point called once per frame.  The full
        pipeline is:

        1. **Feature extraction**: Run the Patchifier CNN to extract
           features and sample M sparse patches.
        2. **Motion model**: Predict the new frame's initial pose using
           damped linear extrapolation on SE3 (see Sec. 3.3 of
           Teed et al. 2022)::

               xi = MOTION_DAMPING * log(P_{n-1} * P_{n-2}^{-1})
               P_n = exp(xi) * P_{n-1}

        3. **Depth initialization**: Set patch inverse-depths to random
           values (pre-init) or the median of recent patches (post-init).
        4. **Motion check** (pre-initialization only): If the median
           predicted displacement is < 2 px, skip this frame (insufficient
           parallax) and store an identity relative pose for later
           interpolation.
        5. **Graph construction**: Create forward edges (old patches ->
           new frame) and backward edges (new patches -> recent frames).
        6. **Optimization**: Either run 12 BA iterations for
           initialization (when n == 8) or 1 iteration + keyframe
           management (after initialization).

        Args:
            tstamp: Timestamp for this frame (typically a frame counter).
            image: RGB image tensor, shape ``(3, ht, wd)``, uint8-scaled
                float values in [0, 255].
            intrinsics: Camera intrinsics ``[fx, fy, cx, cy]`` at full
                image resolution.

        Raises:
            Exception: If the buffer is full (``n + 1 >= N``).
        """
        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"'
            )

        # Normalize image to [-0.5, 0.5] and add batch + sequence dims
        image: Float[Tensor, "1 1 3 ht wd"] = 2 * (image[None, None] / 255.0) - 0.5

        # Step 1: Feature extraction and patch sampling
        with autocast(enabled=self.cfg.mixed_precision):
            fmap: Float[Tensor, "1 1 128 h4 w4"]
            gmap: Float[Tensor, "1 M 128 P P"]
            imap: Float[Tensor, "1 M DIM 1 1"]
            patches: Float[Tensor, "1 M 3 P P"]
            clr: Float[Tensor, "1 M 3"]
            fmap, gmap, imap, patches, _, clr = self.network.patchify(
                image,
                patches_per_image=self.cfg.patches_per_frame,
                gradient_bias=self.cfg.gradient_bias,
                return_color=True,
            )

        ### Update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        # Scale intrinsics to match stride-4 feature resolution
        self.intrinsics_[self.n] = intrinsics / self.RES

        # Convert colors from normalised BGR to uint8 RGB for visualization
        clr: Float[Tensor, "M 3"] = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        # Step 2: Motion model -- predict initial pose for new frame
        if self.n > 1:
            if self.cfg.motion_model == "DAMPED_LINEAR":
                # Damped constant-velocity model on SE3:
                # P_new = exp(damping * log(P_{n-1} * P_{n-2}^{-1})) * P_{n-1}
                P1: SE3 = SE3(self.poses_[self.n - 1])
                P2: SE3 = SE3(self.poses_[self.n - 2])

                xi: Float[Tensor, "6"] = self.cfg.motion_damping * (P1 * P2.inv()).log()
                tvec_qvec: Float[Tensor, "7"] = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                # Constant position model (copy previous pose)
                tvec_qvec: Float[Tensor, "7"] = self.poses[self.n - 1]
                self.poses_[self.n] = tvec_qvec

        # Step 3: Depth initialization
        # Before initialization: random inverse depths
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized:
            # After initialization: use median of recent patches for stability
            s: Float[Tensor, ""] = torch.median(self.patches_[self.n - 3 : self.n, :, 2])
            patches[:, :, 2] = s

        self.patches_[self.n] = patches

        ### Update network attributes (circular buffer) ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        # Build two-level feature pyramid: stride-4 (identity pool) and stride-16
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        # Step 4: Motion check (pre-initialization only)
        # Skip frames with insufficient parallax to avoid degenerate geometry
        if self.n > 0 and not self.is_initialized and self.motion_probe() < 2.0:
            # Store identity relative pose for later interpolation
            self.delta[self.counter - 1] = (self.counter - 2, Id[0])
            return

        self.n += 1
        self.m += self.M

        # Step 5: Create measurement edges in the factor graph
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        # Step 6: Optimization
        if self.n == 8 and not self.is_initialized:
            # Initialization: run 12 BA iterations to bootstrap geometry
            self.is_initialized = True

            for _itr in range(12):
                self.update()

        elif self.is_initialized:
            # Normal tracking: one update iteration + keyframe management
            self.update()
            self.keyframe()
