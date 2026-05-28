from typing import Final

import numpy as np
from jaxtyping import Float

AVP_LINKS = (
    # Spine
    (0, 61),  # hip -> spine1
    (61, 62),  # spine1 -> spine2
    (62, 63),  # spine2 -> spine3
    (63, 64),  # spine3 -> spine4
    (64, 65),  # spine4 -> spine5
    (65, 66),  # spine5 -> spine6
    (66, 67),  # spine6 -> spine7
    # Neck
    (67, 29),  # spine7 -> neck1
    (29, 30),  # neck1 -> neck2
    (30, 31),  # neck2 -> neck3
    (31, 32),  # neck3 -> neck4
    # Left Arm
    (67, 24),  # spine7 -> leftShoulder
    (24, 1),  # leftShoulder -> leftArm
    (1, 2),  # leftArm -> leftForearm
    (2, 3),  # leftForearm -> leftHand (wrist)
    # Right Arm
    (67, 56),  # spine7 -> rightShoulder
    (56, 33),  # rightShoulder -> rightArm
    (33, 34),  # rightArm -> rightForearm
    (34, 35),  # rightForearm -> rightHand (wrist)
    # Left Hand
    # Thumb
    (3, 27),  # leftHand -> leftThumbKnuckle
    (27, 25),  # leftThumbKnuckle -> leftThumbIntermediateBase
    (25, 26),  # leftThumbIntermediateBase -> leftThumbIntermediateTip
    (26, 28),  # leftThumbIntermediateTip -> leftThumbTip
    # Index Finger
    (3, 7),  # leftHand -> leftIndexFingerMetacarpal
    (7, 6),  # leftIndexFingerMetacarpal -> leftIndexFingerKnuckle
    (6, 4),  # leftIndexFingerKnuckle -> leftIndexFingerIntermediateBase
    (4, 5),  # leftIndexFingerIntermediateBase -> leftIndexFingerIntermediateTip
    (5, 8),  # leftIndexFingerIntermediateTip -> leftIndexFingerTip
    # Middle Finger
    (3, 17),  # leftHand -> leftMiddleFingerMetacarpal
    (17, 16),  # leftMiddleFingerMetacarpal -> leftMiddleFingerKnuckle
    (16, 14),  # leftMiddleFingerKnuckle -> leftMiddleFingerIntermediateBase
    (14, 15),  # leftMiddleFingerIntermediateBase -> leftMiddleFingerIntermediateTip
    (15, 18),  # leftMiddleFingerIntermediateTip -> leftMiddleFingerTip
    # Ring Finger
    (3, 22),  # leftHand -> leftRingFingerMetacarpal
    (22, 21),  # leftRingFingerMetacarpal -> leftRingFingerKnuckle
    (21, 19),  # leftRingFingerKnuckle -> leftRingFingerIntermediateBase
    (19, 20),  # leftRingFingerIntermediateBase -> leftRingFingerIntermediateTip
    (20, 23),  # leftRingFingerIntermediateTip -> leftRingFingerTip
    # Little Finger
    (3, 12),  # leftHand -> leftLittleFingerMetacarpal
    (12, 11),  # leftLittleFingerMetacarpal -> leftLittleFingerKnuckle
    (11, 9),  # leftLittleFingerKnuckle -> leftLittleFingerIntermediateBase
    (9, 10),  # leftLittleFingerIntermediateBase -> leftLittleFingerIntermediateTip
    (10, 13),  # leftLittleFingerIntermediateTip -> leftLittleFingerTip
    # Right Hand
    # Thumb
    (35, 59),  # rightHand -> rightThumbKnuckle
    (59, 57),  # rightThumbKnuckle -> rightThumbIntermediateBase
    (57, 58),  # rightThumbIntermediateBase -> rightThumbIntermediateTip
    (58, 60),  # rightThumbIntermediateTip -> rightThumbTip
    # Index Finger
    (35, 39),  # rightHand -> rightIndexFingerMetacarpal
    (39, 38),  # rightIndexFingerMetacarpal -> rightIndexFingerKnuckle
    (38, 36),  # rightIndexFingerKnuckle -> rightIndexFingerIntermediateBase
    (36, 37),  # rightIndexFingerIntermediateBase -> rightIndexFingerIntermediateTip
    (37, 40),  # rightIndexFingerIntermediateTip -> rightIndexFingerTip
    # Middle Finger
    (35, 49),  # rightHand -> rightMiddleFingerMetacarpal
    (49, 48),  # rightMiddleFingerMetacarpal -> rightMiddleFingerKnuckle
    (48, 46),  # rightMiddleFingerKnuckle -> rightMiddleFingerIntermediateBase
    (46, 47),  # rightMiddleFingerIntermediateBase -> rightMiddleFingerIntermediateTip
    (47, 50),  # rightMiddleFingerIntermediateTip -> rightMiddleFingerTip
    # Ring Finger
    (35, 54),  # rightHand -> rightRingFingerMetacarpal
    (54, 53),  # rightRingFingerMetacarpal -> rightRingFingerKnuckle
    (53, 51),  # rightRingFingerKnuckle -> rightRingFingerIntermediateBase
    (51, 52),  # rightRingFingerIntermediateBase -> rightRingFingerIntermediateTip
    (52, 55),  # rightRingFingerIntermediateTip -> rightRingFingerTip
    # Little Finger
    (35, 44),  # rightHand -> rightLittleFingerMetacarpal
    (44, 43),  # rightLittleFingerMetacarpal -> rightLittleFingerKnuckle
    (43, 41),  # rightLittleFingerKnuckle -> rightLittleFingerIntermediateBase
    (41, 42),  # rightLittleFingerIntermediateBase -> rightLittleFingerIntermediateTip
    (42, 45),  # rightLittleFingerIntermediateTip -> rightLittleFingerTip
)

AVP_ID2NAME: dict[int, str] = {
    0: "hip",
    1: "leftArm",
    2: "leftForearm",
    3: "leftHand",  # Typically wrist
    4: "leftIndexFingerIntermediateBase",
    5: "leftIndexFingerIntermediateTip",
    6: "leftIndexFingerKnuckle",
    7: "leftIndexFingerMetacarpal",
    8: "leftIndexFingerTip",
    9: "leftLittleFingerIntermediateBase",
    10: "leftLittleFingerIntermediateTip",
    11: "leftLittleFingerKnuckle",
    12: "leftLittleFingerMetacarpal",
    13: "leftLittleFingerTip",
    14: "leftMiddleFingerIntermediateBase",
    15: "leftMiddleFingerIntermediateTip",
    16: "leftMiddleFingerKnuckle",
    17: "leftMiddleFingerMetacarpal",
    18: "leftMiddleFingerTip",
    19: "leftRingFingerIntermediateBase",
    20: "leftRingFingerIntermediateTip",
    21: "leftRingFingerKnuckle",
    22: "leftRingFingerMetacarpal",
    23: "leftRingFingerTip",
    24: "leftShoulder",
    25: "leftThumbIntermediateBase",
    26: "leftThumbIntermediateTip",
    27: "leftThumbKnuckle",
    28: "leftThumbTip",
    29: "neck1",
    30: "neck2",
    31: "neck3",
    32: "neck4",
    33: "rightArm",
    34: "rightForearm",
    35: "rightHand",  # Typically wrist
    36: "rightIndexFingerIntermediateBase",
    37: "rightIndexFingerIntermediateTip",
    38: "rightIndexFingerKnuckle",
    39: "rightIndexFingerMetacarpal",
    40: "rightIndexFingerTip",
    41: "rightLittleFingerIntermediateBase",
    42: "rightLittleFingerIntermediateTip",
    43: "rightLittleFingerKnuckle",
    44: "rightLittleFingerMetacarpal",
    45: "rightLittleFingerTip",
    46: "rightMiddleFingerIntermediateBase",
    47: "rightMiddleFingerIntermediateTip",
    48: "rightMiddleFingerKnuckle",
    49: "rightMiddleFingerMetacarpal",
    50: "rightMiddleFingerTip",
    51: "rightRingFingerIntermediateBase",
    52: "rightRingFingerIntermediateTip",
    53: "rightRingFingerKnuckle",
    54: "rightRingFingerMetacarpal",
    55: "rightRingFingerTip",
    56: "rightShoulder",
    57: "rightThumbIntermediateBase",
    58: "rightThumbIntermediateTip",
    59: "rightThumbKnuckle",
    60: "rightThumbTip",
    61: "spine1",
    62: "spine2",
    63: "spine3",
    64: "spine4",
    65: "spine5",
    66: "spine6",
    67: "spine7",
}
AVP_IDS: list[int] = [int(key) for key in AVP_ID2NAME]


AVP2COCO133: Final[dict[int, tuple[int, ...]]] = {
    # ---- shoulders & elbows (COCO-17) ----
    1: (5,),  # leftArm       → left_shoulder
    33: (6,),  # rightArm      → right_shoulder
    2: (7,),  # leftForearm   → left_elbow
    34: (8,),  # rightForearm  → right_elbow
    # ---- wrists (duplicate!) -------------
    3: (9, 91),  # leftHand / wrist  → body-wrist + left_hand_root
    35: (10, 112),  # rightHand / wrist → body-wrist + right_hand_root
    # ---- LEFT HAND -----------------------
    27: (92,),
    25: (93,),
    26: (94,),
    28: (95,),  # thumb 1-4
    6: (96,),
    4: (97,),
    5: (98,),
    8: (99,),  # index 1-4
    16: (100,),
    14: (101,),
    15: (102,),
    18: (103,),  # middle 1-4
    21: (104,),
    19: (105,),
    20: (106,),
    23: (107,),  # ring 1-4
    11: (108,),
    9: (109,),
    10: (110,),
    13: (111,),  # pinky 1-4
    # ---- RIGHT HAND ----------------------
    59: (113,),
    57: (114,),
    58: (115,),
    60: (116,),  # thumb 1-4
    38: (117,),
    36: (118,),
    37: (119,),
    40: (120,),  # index 1-4
    48: (121,),
    46: (122,),
    47: (123,),
    50: (124,),  # middle 1-4
    53: (125,),
    51: (126,),
    52: (127,),
    55: (128,),  # ring 1-4
    43: (129,),
    41: (130,),
    42: (131,),
    45: (132,),  # pinky 1-4
}


def avp_to_coco_hands(
    xyz_avp: Float[np.ndarray, "N 68 3"],  # shape: (N, 68, 3)
    conf_avp: Float[np.ndarray, "N 68 1"] | None,  # shape: (N, 68, 1)  (can be None → filled with 1.0)
) -> tuple[Float[np.ndarray, "N 133 3"], Float[np.ndarray, "N 133 1"]]:
    """
    Convert the 42 hand joints from AVP (68‑joint model) to COCO‑133 layout.

    Parameters
    ----------
    xyz_avp : float32[N, 68, 3]
        Joint xyz positions in AVP order.
    conf_avp : float32[N, 68, 1]
        Per‑joint confidence (broadcastable).  If you pass None, a tensor of 1.0 is used.

    Returns
    -------
    xyz_coco : float32[N, 133, 3]
        COCO‑133 xyz array with *all* 133 joints.  Non‑hand joints are zero.
    conf_coco : float32[N, 133, 1]
        Matching confidence array (zeros where we did not fill anything).
    """
    N = xyz_avp.shape[0]

    xyz_coco = np.full((N, 133, 3), np.nan, dtype=np.float32)
    conf_coco = np.full((N, 133, 1), np.nan, dtype=np.float32)

    if conf_avp is None:
        conf_avp = np.ones((N, 68, 1), dtype=np.float32)

    for avp_id, coco_ids in AVP2COCO133.items():
        if isinstance(coco_ids, int):
            coco_ids = (coco_ids,)
        for cid in coco_ids:
            xyz_coco[:, cid, :] = xyz_avp[:, avp_id, :]
            conf_coco[:, cid, :] = conf_avp[:, avp_id, :]

    return xyz_coco, conf_coco
