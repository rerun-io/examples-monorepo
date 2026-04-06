import os
import sys
from pathlib import Path

# The compiled backend extensions (mast3r_slam_backends.so,
# mast3r_slam_mojo_backends.so) are built in-place in the package root
# via `python setup.py build_ext --inplace`.  The editable install only
# registers the mast3r_slam *package*, so those top-level .so files are
# invisible unless the package root is on sys.path.  Adding it here
# means any module in this package can `import mast3r_slam_backends`
# without per-file sys.path hacks.
_pkg_root: str = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
