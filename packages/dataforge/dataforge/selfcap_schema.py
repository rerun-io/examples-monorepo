"""SelfCap preview-layer names and rig facts shared by ingest, layouts and the curate tools.

Outside ``dataforge.datasets`` on purpose: importing ``dataforge.datasets.*`` needs httpx, which the selfcap-soma
environment lacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import schema

EGO_STREAM_LAYOUTS: tuple[tuple[str, str], ...] = (("rgb", "center"), ("left", "left"), ("right", "right"))
"""OAK stream name → ``positional_layout`` in ``ego/calibration.json``, in cam_00..cam_02 order.

Keying on ``positional_layout`` rather than ``camera_id`` because the OAK's
``CAM_A``/``CAM_B``/``CAM_C`` lettering is device-firmware detail.
"""
QUEST_STREAM_LAYOUTS: tuple[tuple[str, str], ...] = (("left", "left"), ("right", "right"))
"""Quest stream name → ``positional_layout``, in cam_00..cam_01 order."""
IMU_DEVICE: int = 0
"""The OAK ships one IMU, so the ego rig has a single ``imu_00``."""
DEFAULT_EXO_CAMERAS: int = 4
"""Exo device count in 2937 of the 3085 readable ``cut`` episodes; shapes the dataset-wide blueprint."""
PREVIEW_LAYER: str = "preview"
"""Layer that holds a segment's card content (``selfcap-preview``); the table is the card table once any segment holds it."""
PREVIEW_TIMELINE: str = "preview_time"
"""Timeline of the preview layer: same values as ``video_time``, but only the window exists there."""
PREVIEW_VIDEO_SIZE: tuple[int, int] = (480, 270)
"""Width and height of the preview videos, in pixels."""
PREVIEW_VIDEO_GAP_PX: int = 4
"""Gap between the exo and ego videos stacked in one 2D view."""
PREVIEW_CENTER: str = "/preview/center"
"""Static transform to the window's mean skeleton position: the card's 3D view origin."""
PREVIEW_BODY: str = "/preview/body"
"""Parent of the window copies of the body: the card's 3D view shows all of it."""
PREVIEW_SKELETON: str = f"{PREVIEW_BODY}/skeleton"
"""Window copy of ``/world/skeleton``."""
PREVIEW_SOMA_XLO: str = f"{PREVIEW_BODY}/soma_xlo"
"""Window copy of the low-vertex SOMA mesh."""
PREVIEW_EXO_CAM: str = "/preview/exo_cam"
"""The exo phone's static camera pose."""
PREVIEW_EXO_PINHOLE: str = f"{PREVIEW_EXO_CAM}/pinhole"
"""The exo phone's pinhole, scaled to the preview video; the videos sit below it."""
PREVIEW_IMAGE_PLANE_DISTANCE: float = 0.06
"""Frustum depth of every camera in the card's 3D view, in metres."""


def rig_label_path(rig: int) -> str:
    """``.../rig_NN/cam_00/rig_label`` — the small labelled point that names a phone rig, at its camera."""
    return f"{schema.cam_path(rig, 0)}/rig_label"


HeadConvention: TypeAlias = Literal["RDF", "RUB"]


@dataclass(frozen=True, slots=True)
class HeadAxes:
    """Read-side orientations in RDF, with the source convention for provenance."""

    quaternions: Float64[ndarray, "n 4"]
    """Head-to-world xyzw orientations; invalid rows keep their source values."""
    convention: HeadConvention
    """Axes stored in the base recording."""

    @property
    def recorded_T_rdf(self) -> Float64[ndarray, "4 4"]:
        """RDF head frame to the unchanged base rig frame."""
        return np.diag([1.0, -1.0, -1.0, 1.0]) if self.convention == "RUB" else np.eye(4)


def rdf_head(quaternions: Float64[ndarray, "n 4"], source_convention: HeadConvention = "RDF") -> HeadAxes:
    """Normalize SelfCap head axes without changing the stored recording.

    Camera-50 captures can store RUB axes. A worn headset's down axis points
    down in the Y-up world; a positive median world-Y component marks RUB.
    Right-multiplication by a 180-degree x rotation converts it to RDF.

    Args:
        quaternions: Float64[n, 4] head-to-world xyzw orientations.
        source_convention: Keep the recorded axes when an earlier catalog read
            has already normalized the quaternions.

    Returns:
        RDF quaternions and the basis needed to attach recorded extrinsics.
    """
    valid: Bool[ndarray, "n"] = np.isfinite(quaternions).all(-1) & (np.linalg.norm(quaternions, axis=-1) > 0.5)
    if not valid.any() or np.median(Rotation.from_quat(quaternions[valid]).as_matrix()[:, 1, 1]) <= 0.0:
        return HeadAxes(quaternions, source_convention)
    x, y, z, w = quaternions.T
    return HeadAxes(np.stack([w, z, -y, -x], axis=1), "RUB")
