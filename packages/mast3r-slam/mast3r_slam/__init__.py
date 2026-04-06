import os
import sys
from pathlib import Path

# The Mojo backend (mast3r_slam_mojo_backends.so) is built as a bare .so in
# the package root — its PyInit symbol is baked in at compile time and can't
# be renamed.  Adding the package root to sys.path lets a plain `import
# mast3r_slam_mojo_backends` find it.  (The CUDA backend is properly
# namespaced as mast3r_slam._backends and needs no path hack.)
_pkg_root: str = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
