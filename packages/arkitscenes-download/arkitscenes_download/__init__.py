"""ARKitScenes dataset downloader.

Mirrors the official Apple downloader (https://github.com/apple/ARKitScenes)
and the Rerun example port, restructured to follow the project's Python
conventions with Tyro and Rich CLIs, import-cheap SimpleCV print utilities,
and the scientific and Rerun dependencies used by ingestion.
"""

import os

# Full runtime type checking only in the pixi `dev` environment; zero overhead
# otherwise. Never add @beartype decorators manually — the claw covers all.
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
