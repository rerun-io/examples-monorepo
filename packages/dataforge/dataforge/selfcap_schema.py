"""SelfCap preview-layer names and rig facts shared by ingest, layouts and the curate tools.

Outside ``dataforge.datasets`` on purpose: importing ``dataforge.datasets.*`` needs httpx, which the selfcap-soma
environment lacks.
"""

from __future__ import annotations

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
