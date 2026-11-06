"""MetricTrail (NVIDIA ViPE) in the monorepo.

Upstream ViPE + MetricTrail is the `vipe` module from `recipes/vipe-src`; this package holds our code around it.
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
