"""The RoboCap hands catalog layer: robocap-live's Rust hand pipeline (``robocap_live._core``) fed from the Rerun catalog."""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
