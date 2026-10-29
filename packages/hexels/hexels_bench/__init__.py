"""hexels-bench: Hexels (honeycomb-surfels/hexels) beside Brush in the gsplat path.

Upstream code runs from the `hexels-src` conda outputs (packages/hexels-src): the `hexels` and
`scripts` modules, 2DGS camera loading and the hexel-kernel CUDA rasterizer (Python module `gsplat`).
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
