"""Clock and anatomical correspondence rules for the prototype."""

from jaxtyping import Float64
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
