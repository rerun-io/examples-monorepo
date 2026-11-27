"""The tracked keypoints' entity, colours and size on the base camera images."""


import numpy as np
from jaxtyping import Int64, UInt8, UInt64
from numpy import ndarray

from slam_rs.rig import RIG_ENTITY

KEYPOINT_RADIUS_PX: float = 2.0
"""Radius of a tracked keypoint, in image pixels."""


def camera_entity(index: int) -> str:
    """The dataset's own path for one camera's image plane."""
    return f"{RIG_ENTITY}/cam_{index:02d}/pinhole"


_rising: UInt8[ndarray, " 255"] = np.arange(255, dtype=np.uint8)
_falling: UInt8[ndarray, " 255"] = 255 - _rising
_full: UInt8[ndarray, " 255"] = np.full(255, 255, dtype=np.uint8)
_zero: UInt8[ndarray, " 255"] = np.zeros(255, dtype=np.uint8)
HUE_RAMP: UInt8[ndarray, "1275 3"] = np.concatenate(
    [
        np.stack([_full, _rising, _zero], axis=1),
        np.stack([_falling, _full, _zero], axis=1),
        np.stack([_zero, _full, _rising], axis=1),
        np.stack([_zero, _falling, _full], axis=1),
        np.stack([_rising, _zero, _full], axis=1),
    ]
)
"""Five fixed hue ramps, from red through blue."""


def track_colors(ids: Int64[ndarray, " n_tracks"]) -> UInt8[ndarray, "n_tracks 3"]:
    """A stable, saturated colour per track id, from a 32-bit hash of the id.

    Hue only, so every colour is equally readable over grey imagery, and two
    neighbouring ids get unrelated hues — which is what makes an id swap visible
    rather than a smooth gradient.

    Args:
        ids: Track ids.

    Returns:
        One ``uint8`` RGB triple per id.
    """
    hashed: UInt64[ndarray, " n_tracks"] = (ids.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return HUE_RAMP[hashed % np.uint64(len(HUE_RAMP))]
