"""Clock and anatomical correspondence rules for the prototype."""

import numpy as np
import rerun as rr
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray


def quest_sample_times(video_times: Float64[ndarray, "n"], offset_ms: float) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n"]]:
    """Sample head at capture time, and delayed hand reports 48 ms later."""
    head: Float64[ndarray, "n"] = video_times - offset_ms / 1000.0
    return head, head + 0.048


def coco_joint_names() -> dict[int, str]:
    """Map joint centers only; omit facial and foot surface landmarks.

    SOMA's index/middle/ring/pinky 1 joints are metacarpals, proximal to
    COCO's MCP. Thumb 1 already corresponds to COCO's thumb base.
    """
    names: dict[int, str] = {
        5: "LeftArm",
        6: "RightArm",
        7: "LeftForeArm",
        8: "RightForeArm",
        9: "LeftHand",
        10: "RightHand",
        11: "LeftLeg",
        12: "RightLeg",
        13: "LeftShin",
        14: "RightShin",
        15: "LeftFoot",
        16: "RightFoot",
    }
    for side, base in (("Left", 91), ("Right", 112)):
        names[base] = f"{side}Hand"
        for finger, digit in enumerate(("Thumb", "Index", "Middle", "Ring", "Pinky")):
            suffixes: tuple[str, ...] = ("1", "2", "3", "End") if digit == "Thumb" else ("2", "3", "4", "End")
            for joint, suffix in enumerate(suffixes):
                names[base + 1 + finger * 4 + joint] = f"{side}Hand{digit}{suffix}"
    return names


def overlay_points(points: Float32[ndarray, "n j 3"], names: tuple[str, ...], upper_body_only: bool = True) -> Float32[ndarray, "n 133 3"]:
    """Projectable COCO body/hand joints only; internal and face joints stay absent."""
    result: Float32[ndarray, "n 133 3"] = np.full((len(points), 133, 3), np.nan, dtype=np.float32)
    for key, name in coco_joint_names().items():
        if name in names and (not upper_body_only or not 13 <= key <= 22):
            result[:, key] = points[:, names.index(name)]
    return result


def require_full_rate_mesh(mesh_times: Int64[ndarray, "m"], joint_times: Int64[ndarray, "n"]) -> None:
    """Reject decimated or shifted meshes before creating a publishable layer."""
    if not np.array_equal(np.sort(mesh_times), np.sort(joint_times)):
        raise ValueError("mesh timestamps must exactly equal all fitted joint timestamps; regenerate with mesh_stride=1")


def video_time_column(times: Float64[ndarray, "n"]) -> rr.TimeColumn:
    """Round back to source nanoseconds; avoid Rerun's float-seconds truncation."""
    return rr.TimeColumn("video_time", duration=np.rint(times * 1e9).astype("timedelta64[ns]"))
