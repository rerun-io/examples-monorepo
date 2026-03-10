"""robocap-slam: Python package for cuVSLAM multicamera visual odometry and SLAM."""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
