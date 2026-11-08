"""Hexels' CUDA extensions in Mojo, as drop-in replacements for the hexels-src recipe.

- `hexels_mojo.prune.removal_scores_cuda`: the 2D prune scorer that upstream JIT-compiles from
  `scripts/image_compression_prune_scores.cu` (recipes/hexels-src patches it to import this).
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
