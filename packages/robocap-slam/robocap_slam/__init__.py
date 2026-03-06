"""robocap-slam: Python package for cuVSLAM multicamera visual odometry and SLAM."""

import os

if os.environ.get("PIXI_ENVIRONMENT_NAME") == "dev":
    try:
        from beartype.claw import beartype_this_package

        beartype_this_package()
    except ImportError:
        pass
