"""Entity paths of the labeler's catalog layers.

Labels sit at the dataforge schema's reserved box path (exoego schema section 13:
``/world/rig_NN/cam_MM/pinhole/boxes/<label>``), so readers of the ``hand_boxes`` layer need no
labeler code. Proposals sit beside them under ``.../pinhole/proposals/<model>``.
"""

from __future__ import annotations

from typing import Literal

from posekit.labeler.schema import HandSide

LEFT_HAND_CLASS_ID: int = 100
"""Class id of a left-hand box (dataforge schema section 13)."""
RIGHT_HAND_CLASS_ID: int = 101
"""Class id of a right-hand box."""


def pinhole_path(cam: int, rig: int = 0) -> str:
    """``/world/rig_NN/cam_MM/pinhole`` — the image plane of one camera."""
    return f"/world/rig_{rig:02d}/cam_{cam:02d}/pinhole"


def video_path(cam: int, rig: int = 0) -> str:
    """``.../pinhole/video`` — the camera's VideoStream on ``video_time``."""
    return f"{pinhole_path(cam, rig)}/video"


def label_path(cam: int, side: HandSide, rig: int = 0) -> str:
    """``.../pinhole/boxes/{left_hand,right_hand}`` — one reviewed hand box."""
    return f"{pinhole_path(cam, rig)}/boxes/{side}_hand"


def proposal_path(cam: int, model: Literal["egohand", "hands_layer"], rig: int = 0) -> str:
    """``.../pinhole/proposals/<model>`` — every box one proposer made in this camera."""
    return f"{pinhole_path(cam, rig)}/proposals/{model}"


def class_id(side: HandSide) -> int:
    """Class id for a hand side."""
    return LEFT_HAND_CLASS_ID if side == "left" else RIGHT_HAND_CLASS_ID
