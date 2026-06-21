"""Shared visualization constants for SAM3 segmentation overlays."""

import numpy as np
from jaxtyping import UInt8

BOX_PALETTE: UInt8[np.ndarray, "n_colors 4"] = np.array(
    [
        [255, 99, 71, 255],  # tomato
        [65, 105, 225, 255],  # royal blue
        [60, 179, 113, 255],  # medium sea green
        [255, 215, 0, 255],  # gold
        [138, 43, 226, 255],  # blue violet
        [255, 140, 0, 255],  # dark orange
        [220, 20, 60, 255],  # crimson
        [70, 130, 180, 255],  # steel blue
    ],
    dtype=np.uint8,
)

# Use a separate id range for segmentation classes to avoid clobbering the person class (id=0).
# Segmentation IDs use 100=background, 101-N for person instances (uint8 compatible, avoids keypoint IDs 0-70).
SEG_CLASS_OFFSET: int = 100

SEG_OVERLAY_ALPHA: int = 242  # ~0.95 opacity (242/255)
"""Alpha value for segmentation mask overlay (0-255). Higher = more opaque."""
