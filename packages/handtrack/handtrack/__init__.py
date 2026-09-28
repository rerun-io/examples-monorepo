"""handtrack: MEgATrack's hand-tracking pipeline (DetNet-F, KeyNet-F, the LM pose fit, the tracker) on the UmeTrack hand model."""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
