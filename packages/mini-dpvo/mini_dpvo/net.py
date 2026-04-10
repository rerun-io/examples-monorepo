"""Neural network architecture for DPVO (Deep Patch Visual Odometry).

Implements the three main components of the DPVO neural network:

- :class:`Patchifier` -- Feature extraction + gradient-biased patch sampling.
  Runs two stride-4 CNNs (feature net ``fnet`` and context net ``inet``)
  on each frame, then selects M sparse 3x3 patches.
- :class:`Update` -- Recurrent GRU update operator that takes correlation
  features + context and predicts pixel displacement ``delta`` and
  confidence ``weight``.  Uses :class:`~mini_dpvo.blocks.SoftAgg` for
  neighbor communication.
- :class:`CorrBlock` -- Multi-scale correlation computation wrapper.
- :class:`VONet` -- Top-level module combining Patchifier + Update, with
  a training-time forward pass that runs the full optimization loop.

See Teed et al. (2022), "Deep Patch Visual Odometry", for the full
architecture description (Sec. 3).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from lietorch import SE3
from torch import Tensor

from . import altcorr, fastba
from . import projective_ops as pops
from .ba import BA
from .blocks import GatedResidual, GradientClip, SoftAgg
from .extractor import BasicEncoder4
from .utils import coords_grid_with_index, flatmeshgrid, pyramidify, set_depth

autocast = torch.cuda.amp.autocast

DIM: int = 384
"""Hidden state dimensionality for the GRU update operator.  This is also
the output dimension of the context network (``inet``)."""


class Update(nn.Module):
    """Recurrent GRU update operator for predicting pixel displacements and confidence.

    This module implements the core recurrent update described in Sec. 3.2
    of Teed et al. (2022).  Each iteration:

    1. **Inject** correlation features and context features into the hidden
       state via addition: ``net = net + inp + corr_embed(corr)``.
    2. **Neighbor communication** via two mechanisms:
       - Direct message passing from graph neighbours (``c1``, ``c2``),
         found via ``fastba.neighbors(kk, jj)``.
       - Attention-weighted aggregation via :class:`SoftAgg`:
         * ``agg_kk``: groups edges by patch index ``kk`` (spatial: same
           patch observed in different frames).
         * ``agg_ij``: groups edges by the hash ``ii*12345 + jj``
           (temporal: different patches connecting the same frame pair).
    3. **GRU update** via two stacked :class:`GatedResidual` blocks with
       layer normalization.
    4. **Prediction heads**: ``d`` predicts pixel displacement (delta),
       ``w`` predicts sigmoid confidence (weight).

    Attributes:
        c1: MLP for message passing from spatial neighbours.
        c2: MLP for message passing from temporal neighbours.
        norm: Layer normalization applied after injection.
        agg_kk: Spatial neighbor aggregation (group by patch index).
        agg_ij: Temporal neighbor aggregation (group by frame pair hash).
        gru: Two stacked GatedResidual blocks with layer norms.
        corr: MLP embedding for correlation features.  Input size is
            ``2 * (2R+1)^2 * P^2 = 2 * 49 * 9 = 882`` for R=3, P=3
            (two pyramid levels, 7x7 search window, 3x3 patch).
        d: Delta (pixel displacement) prediction head with gradient
            clipping.
        w: Weight (confidence) prediction head with gradient clipping
            and sigmoid activation.
    """

    def __init__(self, p: int) -> None:
        super().__init__()

        # Neighbor communication MLPs
        self.c1: nn.Sequential = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2: nn.Sequential = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.norm: nn.LayerNorm = nn.LayerNorm(DIM, eps=1e-3)

        # Attention-weighted scatter aggregation modules
        self.agg_kk: SoftAgg = SoftAgg(DIM)
        self.agg_ij: SoftAgg = SoftAgg(DIM)

        # Gated residual GRU (replaces a standard ConvGRU)
        self.gru: nn.Sequential = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        # Correlation feature embedding.
        # Input dim = 2 levels * (2*3+1)^2 * p^2 = 2 * 49 * p^2
        self.corr: nn.Sequential = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        # Pixel displacement prediction head (2D: du, dv)
        self.d: nn.Sequential = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        # Confidence weight prediction head (2D: wu, wv), sigmoid-activated
        self.w: nn.Sequential = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(
        self,
        net: Float[Tensor, "1 n_edges dim"],
        inp: Float[Tensor, "1 n_edges dim"],
        corr: Float[Tensor, "1 n_edges corr_feat"],
        flow: None,
        ii: Int[Tensor, "n_edges"],
        jj: Int[Tensor, "n_edges"],
        kk: Int[Tensor, "n_edges"],
    ) -> tuple[Float[Tensor, "1 n_edges dim"], tuple[Float[Tensor, "1 n_edges 2"], Float[Tensor, "1 n_edges 2"], None]]:
        """Run one iteration of the recurrent update operator.

        Args:
            net: Hidden state from previous iteration, shape
                ``(1, n_edges, DIM)``.
            inp: Context features for each edge (from ``inet``), shape
                ``(1, n_edges, DIM)``.
            corr: Multi-scale correlation features, shape
                ``(1, n_edges, 2 * 49 * P^2)``.
            flow: Unused (kept for API compatibility).
            ii: Source frame index per edge.
            jj: Target frame index per edge.
            kk: Patch index per edge.

        Returns:
            A 2-tuple of:
            - Updated hidden state ``net``, shape ``(1, n_edges, DIM)``.
            - A 3-tuple ``(delta, weight, None)`` where ``delta`` is the
              predicted pixel displacement ``(1, n_edges, 2)`` and ``weight``
              is the sigmoid confidence ``(1, n_edges, 2)``.
        """
        # Step 1: Inject correlation and context into hidden state
        net: Float[Tensor, "1 n_edges dim"] = net + inp + self.corr(corr)
        net = self.norm(net)

        # Step 2a: Direct neighbor message passing
        # ix, jx are indices of the "other" edge sharing the same (kk, jj)
        ix: Int[Tensor, "n_edges"]
        jx: Int[Tensor, "n_edges"]
        ix, jx = fastba.neighbors(kk, jj)
        # Mask out invalid neighbours (index -1 means no neighbour found)
        mask_ix: Float[Tensor, "1 n_edges 1"] = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx: Float[Tensor, "1 n_edges 1"] = (jx >= 0).float().reshape(1, -1, 1)

        # Message from spatial neighbour (same target frame jj, adjacent patch)
        net = net + self.c1(mask_ix * net[:,ix])
        # Message from temporal neighbour (same patch kk, adjacent frame)
        net = net + self.c2(mask_jx * net[:,jx])

        # Step 2b: Attention-weighted scatter aggregation
        # Spatial: pool across all edges sharing the same patch index kk
        net = net + self.agg_kk(net, kk)
        # Temporal: pool across edges connecting the same frame pair (ii, jj)
        # The hash ii*12345+jj maps each (ii,jj) pair to a unique group ID
        net = net + self.agg_ij(net, ii*12345 + jj)

        # Step 3: Gated residual GRU update
        net = self.gru(net)

        # Step 4: Prediction heads
        delta: Float[Tensor, "1 n_edges 2"] = self.d(net)
        weight: Float[Tensor, "1 n_edges 2"] = self.w(net)

        return net, (delta, weight, None)


class Patchifier(nn.Module):
    """Feature extraction and gradient-biased patch sampling.

    Runs two parallel stride-4 CNN encoders on the input images:

    - ``fnet`` (feature network): Produces 128-dim features used for
      computing correlation volumes.  Uses instance normalization.
    - ``inet`` (context / injection network): Produces DIM-dim (384)
      features injected into the GRU hidden state.  No normalization.

    Then selects M sparse patch locations per frame and extracts:
    - ``gmap``: Local feature descriptors (128-dim, 3x3 patches) from ``fnet``.
    - ``imap``: Context descriptors (DIM-dim, point samples) from ``inet``.
    - ``patches``: Coordinate patches ``(x, y, inv_depth)`` of size 3x3.

    See Sec. 3.1 of Teed et al. (2022) for patch selection details.

    Attributes:
        patch_size: Side length of square patches (default 3).
        fnet: Feature network (stride-4, 128-dim output, instance norm).
        inet: Context network (stride-4, DIM-dim output, no norm).
    """

    def __init__(self, patch_size: int = 3) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.fnet: BasicEncoder4 = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet: BasicEncoder4 = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images: Float[Tensor, "1 n 3 h w"]) -> Float[Tensor, "1 n h4 w4"]:
        """Compute image gradient magnitude at stride-4 resolution.

        Converts to grayscale, computes finite-difference gradients in
        x and y, takes the L2 magnitude, then average-pools 4x to match
        the feature map resolution.  Used by gradient-biased patch
        selection.

        Args:
            images: Normalised input images (in [-0.5, 0.5] range).

        Returns:
            Gradient magnitude map at stride-4 resolution.
        """
        # Convert normalised images back to [0, 255] grayscale
        gray: Float[Tensor, "1 n h w"] = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        # Finite differences
        dx: Float[Tensor, "1 n h_m1 w_m1"] = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy: Float[Tensor, "1 n h_m1 w_m1"] = gray[...,1:,:-1] - gray[...,:-1,:-1]
        # L2 gradient magnitude
        g: Float[Tensor, "1 n h_m1 w_m1"] = torch.sqrt(dx**2 + dy**2)
        # Downsample to match stride-4 feature map resolution
        g: Float[Tensor, "1 n h4 w4"] = F.avg_pool2d(g, 4, 4)
        return g

    def forward(
        self,
        images: Float[Tensor, "1 n 3 h w"],
        patches_per_image: int = 80,
        disps: Float[Tensor, "1 n h w"] | None = None,
        gradient_bias: bool = False,
        return_color: bool = False,
    ) -> tuple[Float[Tensor, "1 n 128 h4 w4"], Float[Tensor, "1 num_patches 128 p p"], Float[Tensor, "1 num_patches dim 1 1"], Float[Tensor, "1 num_patches 3 p p"], Int[Tensor, "num_patches"]] | tuple[Float[Tensor, "1 n 128 h4 w4"], Float[Tensor, "1 num_patches 128 p p"], Float[Tensor, "1 num_patches dim 1 1"], Float[Tensor, "1 num_patches 3 p p"], Int[Tensor, "num_patches"], Float[Tensor, "1 num_patches 3"]]:
        """Extract features and sample sparse patches from input images.

        Patch selection strategy (see Sec. 3.1 of Teed et al. 2022):

        - **Random** (``gradient_bias=False``): Uniformly sample M locations.
        - **Gradient-biased** (``gradient_bias=True``): Sample 3M candidate
          locations uniformly, evaluate the image gradient magnitude at each,
          and keep the top-M by gradient strength.  This concentrates patches
          on textured regions where correlation matching is more reliable.

        Args:
            images: Normalised images, shape ``(1, n, 3, h, w)``.
            patches_per_image: Number of patches M to select per frame.
            disps: Optional inverse-depth maps for initializing the depth
                channel of patches.  If ``None``, defaults to ones.
            gradient_bias: Whether to use gradient-biased patch selection.
            return_color: Whether to also return per-patch RGB colors
                (for visualization).

        Returns:
            A tuple of:
            - ``fmap``: Feature maps from ``fnet``, ``(1, n, 128, h/4, w/4)``.
            - ``gmap``: Per-patch feature descriptors, ``(1, M*n, 128, P, P)``.
            - ``imap``: Per-patch context features, ``(1, M*n, DIM, 1, 1)``.
            - ``patches``: Patch coordinates ``(x, y, d)``, ``(1, M*n, 3, P, P)``.
            - ``index``: Source frame index per patch, ``(M*n,)``.
            - ``clr`` (optional): Per-patch colors, ``(1, M*n, 3)``.
        """
        # Run both encoders; divide by 4 for numerical scaling
        fmap: Float[Tensor, "1 n 128 h4 w4"] = self.fnet(images) / 4.0
        imap: Float[Tensor, "1 n dim h4 w4"] = self.inet(images) / 4.0

        b: int
        n: int
        c: int
        h: int
        w: int
        b, n, c, h, w = fmap.shape
        P: int = self.patch_size

        # Patch selection: choose (x, y) locations in stride-4 feature space
        if gradient_bias:
            # Gradient-biased sampling: sample 3x candidates, keep top-M
            g: Float[Tensor, "1 n h4 w4"] = self.__image_gradient(images)
            x: Int[Tensor, "n candidates"] = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y: Int[Tensor, "n candidates"] = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords: Float[Tensor, "n candidates 2"] = torch.stack([x, y], dim=-1).float()
            # Evaluate gradient magnitude at each candidate location
            g: Float[Tensor, "n candidates"] = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)

            # Sort by gradient magnitude and keep the top-M
            ix: Int[Tensor, "n candidates"] = torch.argsort(g, dim=1)
            x: Int[Tensor, "n patches_per_image"] = torch.gather(x, 1, ix[:, -patches_per_image:])
            y: Int[Tensor, "n patches_per_image"] = torch.gather(y, 1, ix[:, -patches_per_image:])

        else:
            # Uniform random sampling
            x: Int[Tensor, "n patches_per_image"] = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y: Int[Tensor, "n patches_per_image"] = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        coords: Float[Tensor, "n patches_per_image 2"] = torch.stack([x, y], dim=-1).float()
        # Extract context features at patch centers (radius 0 = point sample)
        imap: Float[Tensor, "1 num_patches dim 1 1"] = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        # Extract feature descriptors as PxP patches (radius P//2 = 1 for P=3)
        gmap: Float[Tensor, "1 num_patches 128 p p"] = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)

        clr: Float[Tensor, "1 num_patches 3"]
        if return_color:
            # Sample RGB colors from the original image at patch locations
            # (scale coords by 4 to go from stride-4 to pixel resolution)
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            # Default inverse-depth: ones (unit depth)
            disps: Float[Tensor, "1 n h w"] = torch.ones(b, n, h, w, device="cuda")

        # Build coordinate grid (x, y, d) and extract PxP patches
        grid: Float[Tensor, "1 n 3 h w"]
        _index: Float[Tensor, "1 n 1 h w"]
        grid, _index = coords_grid_with_index(disps, device=fmap.device)
        patches: Float[Tensor, "1 num_patches 3 p p"] = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)

        # Frame index for each patch (which frame it came from)
        index: Int[Tensor, "n 1"] = torch.arange(n, device="cuda").view(n, 1)
        index: Int[Tensor, "num_patches"] = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    """Multi-scale local correlation volume computation.

    Precomputes a feature pyramid from the dense feature map ``fmap`` and
    computes local normalized cross-correlation between per-patch
    descriptors ``gmap`` and the pyramid levels at given 2-D coordinates.

    DPVO uses two pyramid levels (``levels=[1, 4]``):
    - Level 1 (stride-4): 1:1 feature map, captures fine detail.
    - Level 4 (stride-16): 4x pooled feature map, captures coarse context.

    At each level, correlation is computed in a ``(2R+1) x (2R+1)``
    neighborhood (R=3 by default, giving 7x7 = 49 values per level).
    The total correlation feature dimension per patch pixel is
    ``2 * 49 * P^2 = 882`` for P=3.  See Sec. 3.2 of Teed et al. (2022).

    Attributes:
        dropout: Dropout probability applied during correlation computation
            (training only).
        radius: Search radius R for the correlation neighborhood.
        levels: List of pooling factors for the feature pyramid.
        gmap: Per-patch feature descriptors, shape ``(1, M, 128, P, P)``.
        pyramid: List of feature maps at each pyramid level.
    """

    def __init__(
        self,
        fmap: Float[Tensor, "1 n 128 h4 w4"],
        gmap: Float[Tensor, "1 num_patches 128 p p"],
        radius: int = 3,
        dropout: float = 0.2,
        levels: list[int] | None = None,
    ) -> None:
        if levels is None:
            levels = [1, 4]
        self.dropout: float = dropout
        self.radius: int = radius
        self.levels: list[int] = levels

        self.gmap: Float[Tensor, "1 num_patches 128 p p"] = gmap
        self.pyramid: list[Float[Tensor, "1 n 128 h_l w_l"]] = pyramidify(fmap, lvls=levels)

    def __call__(
        self,
        ii: Int[Tensor, "n_edges"],
        jj: Int[Tensor, "n_edges"],
        coords: Float[Tensor, "1 n_edges 2 p p"],
    ) -> Float[Tensor, "1 n_edges corr_feat"]:
        """Compute multi-scale correlation features for a set of edges.

        For each edge ``(ii[e], jj[e])``, the patch descriptor ``gmap[ii[e]]``
        is correlated against the feature map of frame ``jj[e]`` in a local
        window around the reprojected coordinates ``coords[e]``.

        Args:
            ii: Patch indices for each edge.
            jj: Target frame indices for each edge.
            coords: Reprojected 2-D coordinates, shape
                ``(1, n_edges, 2, P, P)`` at stride-4 resolution.

        Returns:
            Stacked correlation features from all pyramid levels,
            shape ``(1, n_edges, 2 * (2R+1)^2 * P^2)``.
        """
        corrs: list[Float[Tensor, "..."]] = []
        for i in range(len(self.levels)):
            # Scale coordinates to match pyramid level resolution
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        # Stack levels along last dim, then flatten
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    """Top-level DPVO network combining feature extraction and recurrent update.

    This module orchestrates:
    1. :class:`Patchifier` for feature extraction and patch sampling.
    2. :class:`Update` for the recurrent GRU-based displacement prediction.

    During inference, only the :class:`Patchifier` (via ``self.patchify``)
    and :class:`Update` (via ``self.update``) are called individually by
    the :class:`~mini_dpvo.dpvo.DPVO` class.

    The ``forward`` method implements the *training-time* loop: given
    a sequence of images with known poses and depths, it runs the full
    optimization pipeline (patchify -> correlate -> GRU update -> BA)
    for ``STEPS`` iterations and returns intermediate trajectory
    predictions for computing the training loss.

    Attributes:
        P: Patch size (default 3).
        patchify: :class:`Patchifier` module for feature extraction.
        update: :class:`Update` module for recurrent displacement prediction.
        DIM: Hidden state dimensionality (384).
        RES: Feature map stride relative to the input image (4).
    """

    def __init__(self, _use_viewer: bool = False) -> None:
        super().__init__()
        self.P: int = 3
        self.patchify: Patchifier = Patchifier(self.P)
        self.update: Update = Update(self.P)

        self.DIM: int = DIM
        self.RES: int = 4


    @autocast(enabled=False)
    def forward(
        self,
        images: Float[Tensor, "1 n_frames 3 h w"],
        poses: SE3,
        disps: Float[Tensor, "1 n_frames h w"],
        intrinsics: Float[Tensor, "1 n_frames 4"],
        M: int = 1024,
        STEPS: int = 12,
        P: int = 1,
        structure_only: bool = False,
        _rescale: bool = False,
    ) -> list[tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."], SE3, SE3, Tensor]]:
        """Training forward pass: run the full DPVO optimization loop.

        This implements the end-to-end differentiable training pipeline:

        1. Extract features and sample patches via :class:`Patchifier`.
        2. Initialize random inverse depths (ground truth is available but
           not used for depth initialization to force learning).
        3. Iteratively expand the measurement graph (adding one frame at
           a time after 8 warmup steps) and run:
           a. Reproject all patches.
           b. Compute correlation features via :class:`CorrBlock`.
           c. Run the :class:`Update` GRU to predict displacements.
           d. Run 2 iterations of :func:`~mini_dpvo.ba.BA` bundle adjustment.
        4. Collect trajectory snapshots for loss computation.

        Args:
            images: Input images (uint8 [0, 255]), shape ``(1, N, 3, H, W)``.
            poses: Ground-truth SE3 poses (used for supervision targets).
            disps: Ground-truth inverse-depth maps, shape ``(1, N, H, W)``.
            intrinsics: Camera intrinsics ``(fx, fy, cx, cy)`` per frame.
            M: Maximum number of patches (not used directly here).
            STEPS: Number of optimization iterations to run.
            P: Not used (patch size is fixed at ``self.P``).
            structure_only: If True, only optimize structure (not poses).
            rescale: Reserved for future use.

        Returns:
            List of ``STEPS`` trajectory snapshots, each a tuple of
            ``(valid, coords, coords_gt, Gs, Ps, kl)`` for loss computation.
        """
        # Normalize images to [-0.5, 0.5] range
        images: Float[Tensor, "1 n_frames 3 h w"] = 2 * (images / 255.0) - 0.5
        # Scale intrinsics to match stride-4 feature map resolution
        intrinsics: Float[Tensor, "1 n_frames 4"] = intrinsics / 4.0
        # Subsample depth maps to stride-4 resolution
        disps: Float[Tensor, "1 n_frames h4 w4"] = disps[:, :, 1::4, 1::4].float()

        fmap: Float[Tensor, "1 n_frames 128 h4 w4"]
        gmap: Float[Tensor, "1 num_patches 128 p p"]
        imap: Float[Tensor, "1 num_patches dim 1 1"]
        patches: Float[Tensor, "1 num_patches 3 p p"]
        ix: Int[Tensor, "num_patches"]
        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn: CorrBlock = CorrBlock(fmap, gmap)

        b: int
        N: int
        c: int
        h: int
        w: int
        b, N, c, h, w = fmap.shape
        p: int = self.P

        # Save ground-truth patches for loss computation
        patches_gt: Float[Tensor, "1 num_patches 3 p p"] = patches.clone()
        Ps: SE3 = poses

        # Randomize inverse depths (training: learn to recover from random init)
        d: Float[Tensor, "1 num_patches"] = patches[..., 2, p//2, p//2]
        patches: Float[Tensor, "1 num_patches 3 p p"] = set_depth(patches, torch.rand_like(d))

        # Initialize measurement graph: connect all patches from first 8 frames
        kk: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        ii: Int[Tensor, "n_edges"] = ix[kk]

        imap: Float[Tensor, "1 num_patches dim"] = imap.view(b, -1, DIM)
        net: Float[Tensor, "1 n_edges dim"] = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)

        # Initialize estimated poses to identity
        Gs: SE3 = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj: list[tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."], SE3, SE3, Tensor]] = []
        # Relaxed image bounds (allow slight out-of-frame projections)
        bounds: list[int] = [-64, -64, w + 64, h + 64]

        while len(traj) < STEPS:
            # Detach to limit backprop through time (TBPTT-style)
            Gs = Gs.detach()
            patches = patches.detach()

            n: int = ii.max() + 1
            # After 8 warmup steps, incrementally add one frame per iteration
            if len(traj) >= 8 and n < images.shape[1]:
                # Initialize new frame's pose from the previous frame
                if not structure_only:
                    Gs.data[:,n] = Gs.data[:,n-1]
                # Forward edges: all existing patches -> new frame
                kk1: Int[Tensor, "new_edges1"]
                jj1: Int[Tensor, "new_edges1"]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                # Backward edges: new frame's patches -> all existing frames
                kk2: Int[Tensor, "new_edges2"]
                jj2: Int[Tensor, "new_edges2"]
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1: Float[Tensor, "1 new_edges dim"] = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                # Randomly prune edges to frame (n-4) for training augmentation
                if np.random.rand() < 0.1:
                    k: Tensor = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                # Initialize new frame's depth from median of recent frames
                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            # Reproject all patches and compute correlation
            coords: Float[Tensor, "1 n_edges p p 2"] = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1: Float[Tensor, "1 n_edges 2 p p"] = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr: Float[Tensor, "1 n_edges corr_feat"] = corr_fn(kk, jj, coords1)
            # GRU update: predict displacement and confidence
            delta: Float[Tensor, "1 n_edges 2"]
            weight: Float[Tensor, "1 n_edges 2"]
            _damping: None
            net, (delta, weight, _damping) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda: float = 1e-4
            # Target = current reprojection + learned delta
            target: Float[Tensor, "1 n_edges 2"] = coords[...,p//2,p//2,:] + delta

            # Run 2 iterations of bundle adjustment
            ep: int = 10
            for _itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk,
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            # Compute loss targets: reproject with estimated and GT poses
            kl: Tensor = torch.as_tensor(0)
            dij: Int[Tensor, "n_edges"] = (ii - jj).abs()
            # Only compute loss on edges with temporal distance 1 or 2
            k: Tensor = (dij > 0) & (dij <= 2)

            coords: Float[Tensor, "..."] = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt: Float[Tensor, "..."]
            valid: Float[Tensor, "..."]
            _jacobians: tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]
            coords_gt, valid, _jacobians = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj
