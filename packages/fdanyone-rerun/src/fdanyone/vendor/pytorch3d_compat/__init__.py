"""Minimal PyTorch3D 0.7.6 compatibility surface for classic GVHMR.

The official GVHMR inference path needs rotation conversions but imports the
compiled PyTorch3D package through broader training/demo modules. 4DAnyone
registers this BSD-licensed, inference-only subset when full PyTorch3D is not
installed, allowing GVHMR and generation to share the same modern PyTorch
environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys


def install_if_needed() -> bool:
    """Expose the compatibility modules as ``pytorch3d`` when it is absent."""

    if importlib.util.find_spec("pytorch3d") is not None:
        return False

    root = sys.modules[__name__]
    transforms = importlib.import_module(f"{__name__}.transforms")
    ops = importlib.import_module(f"{__name__}.ops")
    knn = importlib.import_module(f"{__name__}.ops.knn")
    sys.modules.update(
        {
            "pytorch3d": root,
            "pytorch3d.transforms": transforms,
            "pytorch3d.ops": ops,
            "pytorch3d.ops.knn": knn,
        }
    )
    return True


__all__ = ["install_if_needed"]
