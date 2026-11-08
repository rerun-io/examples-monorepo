"""Hexels' CUDA extensions in Mojo, as drop-in replacements in the hexels-src recipe (patches 0002-0004).

- `hexels_mojo.fused_ssim.fused_ssim`: fused-ssim (Image-GS's loss, the 2D scorer).
- `hexels_mojo.simple_knn.distCUDA2`: simple-knn (2DGS's initial scales).
- `hexels_mojo.prune.removal_scores_cuda`: the 2D prune scorer that upstream JIT-compiles from
  `scripts/image_compression_prune_scores.cu`.
- `hexels_mojo.raster2d`: hexel-kernel's 2D hexel rasterizers (flat and color-gradient) and tile binning (patch 0005
  overlays them on hexel-kernel's `gsplat` module).
- `hexels_mojo.gaussians2d`: Image-GS's Gaussian projection and top-K rasterizer (`imagegs-2d` baseline).
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
