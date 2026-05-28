from typing import Final

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

HAND_LINKS = (
    (5, 6),
    (6, 7),
    (7, 0),  # Thumb
    (5, 8),
    (8, 9),
    (9, 10),
    (10, 1),
    (5, 11),  # index
    (11, 12),
    (12, 13),
    (13, 2),
    (5, 14),  # ring
    (14, 15),
    (15, 16),
    (16, 3),
    (5, 17),  # middle
    (17, 18),
    (18, 19),
    (19, 4),  # pinky
)

HAND_ID2NAME: dict[int, str] = {
    0: "THUMB_TIP",
    1: "INDEX_FINGER_TIP",
    2: "MIDDLE_FINGER_TIP",
    3: "RING_FINGER_TIP",
    4: "PINKY_TIP",
    5: "WRIST",
    6: "THUMB_CMC",
    7: "THUMB_MCP",
    8: "INDEX_FINGER_MCP",
    9: "INDEX_FINGER_PIP",
    10: "INDEX_FINGER_DIP",
    11: "MIDDLE_FINGER_MCP",
    12: "MIDDLE_FINGER_PIP",
    13: "MIDDLE_FINGER_DIP",
    14: "RING_FINGER_MCP",
    15: "RING_FINGER_PIP",
    16: "RING_FINGER_DIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PALM",
}

HAND_IDS: list[int] = [int(key) for key in HAND_ID2NAME]

# ------------------------------------------------------------------
# Assembly-Hands ↔ COCO-133 index tables
# ------------------------------------------------------------------
# single-hand map:  index in Assembly-Hands  →  index/indices in COCO-133
# (left hand; right hand is +21)
_ASM2COCO: Final[dict[int, tuple[int, ...]]] = {
    5: (91, 9),  # wrist  → hand root & body wrist
    # thumb1 is synthesized later by interpolating wrist↔CMC
    6: (93,),  # thumb CMC  → thumb2
    7: (94,),  # thumb MCP  → thumb3
    0: (95,),  # thumb tip  → thumb4
    8: (96,),
    9: (97,),
    10: (98,),
    1: (99,),  # index 1-4
    11: (100,),
    12: (101,),
    13: (102,),
    2: (103,),  # middle 1-4
    14: (104,),
    15: (105,),
    16: (106,),
    3: (107,),  # ring   1-4
    17: (108,),
    18: (109,),
    19: (110,),
    4: (111,),  # pinky  1-4
    # 20 = palm → not used in COCO 133
}

# produce right-hand mapping by offsetting +21
_ASM2COCO_R: Final[dict[int, tuple[int, ...]]] = {
    k: tuple((cid + 21) if cid != 9 else 10 for cid in v) for k, v in _ASM2COCO.items()
}
_ASM2COCO_R[5] = (112, 10)  # right wrist duplicates → 10


# ------------------------------------------------------------------
# main helper
# ------------------------------------------------------------------
def assembly21_to_coco133(
    kpts_lr: Float32[ndarray, "2 21 3"],
) -> Float32[ndarray, "133 4"]:
    """
    Convert one frame of Assembly-Hands (L,R) → COCO-WholeBody 133.

    Any joint not present in the source is filled with NaN/0-conf.
    """
    coco_133: Float32[ndarray, "133 4"] = np.zeros((133, 4), dtype=np.float32)
    coco_133[:] = np.nan  # xyz defaults to NaN
    # confidence defaults to 0.0 so we leave out[:,3] as zeros

    # left hand ------------------------------------------------------
    for asm_id, coco_ids in _ASM2COCO.items():
        for cid in coco_ids:
            coco_133[cid, :3] = kpts_lr[0, asm_id]
            coco_133[cid, 3] = 1.0

    # Assembly-Hands omits the thumb base joint; synthesize it by averaging wrist↔CMC
    left_wrist: Float32[ndarray, "3"] = kpts_lr[0, 5]
    left_thumb_cmc: Float32[ndarray, "3"] = kpts_lr[0, 6]
    left_thumb_base_valid: bool = not (
        np.isnan(left_wrist).any() or np.isnan(left_thumb_cmc).any()
    )
    if left_thumb_base_valid:
        left_thumb_base: Float32[ndarray, "3"] = (left_wrist + left_thumb_cmc) * np.float32(0.5)
        coco_133[92, :3] = left_thumb_base
        coco_133[92, 3] = np.float32(1.0)

    # right hand -----------------------------------------------------
    for asm_id, coco_ids in _ASM2COCO_R.items():
        for cid in coco_ids:
            coco_133[cid, :3] = kpts_lr[1, asm_id]
            coco_133[cid, 3] = 1.0

    # Same interpolation for the right thumb base
    right_wrist: Float32[ndarray, "3"] = kpts_lr[1, 5]
    right_thumb_cmc: Float32[ndarray, "3"] = kpts_lr[1, 6]
    right_thumb_base_valid: bool = not (
        np.isnan(right_wrist).any() or np.isnan(right_thumb_cmc).any()
    )
    if right_thumb_base_valid:
        right_thumb_base: Float32[ndarray, "3"] = (right_wrist + right_thumb_cmc) * np.float32(0.5)
        coco_133[113, :3] = right_thumb_base
        coco_133[113, 3] = np.float32(1.0)

    return coco_133
