"""Single source of truth for layer names and default raw assets.

Standard-library only: the slim Modal container interpreters import this module
directly, so it must never pull in the rerun/rich-heavy ingest package.
"""

from typing import Final

REQUIRED_LAYER_NAMES: Final[tuple[str, ...]] = (
    "base",
    "calibration",
    "video_wide",
    "video_ultrawide",
    "arkit_depth",
    "imu",
    "arkit_mesh",
    "gt_boxes",
)
"""The eight layers every ingested capture publishes; ``base`` carries the recording properties."""

GT_POSES_LAYER: Final[str] = "gt_poses"
GT_DEPTH_LAYER: Final[str] = "gt_depth"
OPTIONAL_LAYER_NAMES: Final[tuple[str, str]] = (GT_POSES_LAYER, GT_DEPTH_LAYER)
"""Laser-GT layers produced by the CA-1M tool, present only on covered captures; absence = no GT."""

ALL_LAYER_NAMES: Final[tuple[str, ...]] = REQUIRED_LAYER_NAMES + OPTIONAL_LAYER_NAMES

DEPTH_RANGE_MM: Final[tuple[float, float]] = (0.0, 4000.0)
"""Explicit viewer colormap range for encoded indoor depth, in native millimetres."""

DEFAULT_ASSETS: Final[tuple[str, ...]] = (
    "mov",
    "annotation",
    "mesh",
    "lowres_wide.traj",
    "confidence",
    "lowres_depth",
    "lowres_wide_intrinsics",
    "ultrawide_intrinsics",
)
"""Raw assets required by the layered ARKitScenes ingest."""
