"""Map SMPL-X regressed joints to the COCO-133 (COCO-WholeBody) keypoint layout.

The ``smplx`` package's SMPL-X forward regresses 127 joints (55 kinematic tree
joints, then vertex-picked landmarks: nose/eyes/ears, toes/heels, fingertips,
and the 51 iBUG face landmarks), or 144 with ``use_face_contour=True`` (adds
the 17 jawline contour points). Every COCO-133 keypoint has a named SMPL-X
counterpart, so the mapping is built by name against
``smplx.joint_names.JOINT_NAMES`` — an index shift in either vocabulary fails
loudly instead of silently mismapping.
"""

from functools import lru_cache

import numpy as np
from jaxtyping import Bool, Float, Float32, Int
from numpy import ndarray

from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME

# COCO-133 names that differ from their SMPL-X joint name (identical names —
# body joints, toes/heels — map implicitly).
_COCO_TO_SMPLX_NAME: dict[str, str] = {
    "left_hand_root": "left_wrist",
    "right_hand_root": "right_wrist",
    # COCO finger chains are <finger>1..4 (root->tip); SMPL-X names the three
    # kinematic segments <finger>1..3 and the vertex-picked tip "<side>_<finger>".
    "left_thumb4": "left_thumb",
    "right_thumb4": "right_thumb",
    "left_forefinger1": "left_index1",
    "left_forefinger2": "left_index2",
    "left_forefinger3": "left_index3",
    "left_forefinger4": "left_index",
    "right_forefinger1": "right_index1",
    "right_forefinger2": "right_index2",
    "right_forefinger3": "right_index3",
    "right_forefinger4": "right_index",
    "left_middle_finger1": "left_middle1",
    "left_middle_finger2": "left_middle2",
    "left_middle_finger3": "left_middle3",
    "left_middle_finger4": "left_middle",
    "right_middle_finger1": "right_middle1",
    "right_middle_finger2": "right_middle2",
    "right_middle_finger3": "right_middle3",
    "right_middle_finger4": "right_middle",
    "left_ring_finger1": "left_ring1",
    "left_ring_finger2": "left_ring2",
    "left_ring_finger3": "left_ring3",
    "left_ring_finger4": "left_ring",
    "right_ring_finger1": "right_ring1",
    "right_ring_finger2": "right_ring2",
    "right_ring_finger3": "right_ring3",
    "right_ring_finger4": "right_ring",
    "left_pinky_finger1": "left_pinky1",
    "left_pinky_finger2": "left_pinky2",
    "left_pinky_finger3": "left_pinky3",
    "left_pinky_finger4": "left_pinky",
    "right_pinky_finger1": "right_pinky1",
    "right_pinky_finger2": "right_pinky2",
    "right_pinky_finger3": "right_pinky3",
    "right_pinky_finger4": "right_pinky",
}

# The 68 iBUG face points ("face-0".."face-67" in COCO-133): 0-16 are the
# jawline contour, 17-67 the inner landmarks. SMPL-X's JOINT_NAMES lists the 51
# inner landmarks (indices 76..126) and the contour (127..143) each in iBUG
# order, so face-n maps positionally within those two blocks.
_SMPLX_FACE_LANDMARKS_START: int = 76
_SMPLX_FACE_CONTOUR_START: int = 127
_NUM_FACE_CONTOUR: int = 17


@lru_cache(maxsize=1)
def _coco133_to_smplx_indices() -> tuple[Int[ndarray, "n_mapped"], Int[ndarray, "n_mapped"]]:
    """Build (coco_ids, smplx_ids) index arrays for all 133 mappable keypoints."""
    from smplx.joint_names import JOINT_NAMES

    smplx_name_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    coco_ids: list[int] = []
    smplx_ids: list[int] = []
    for coco_id in range(133):
        coco_name: str = COCO_133_ID2NAME[coco_id]
        if coco_name.startswith("face-"):
            face_idx: int = int(coco_name.removeprefix("face-"))
            smplx_idx: int = (
                _SMPLX_FACE_CONTOUR_START + face_idx if face_idx < _NUM_FACE_CONTOUR else _SMPLX_FACE_LANDMARKS_START + face_idx - _NUM_FACE_CONTOUR
            )
        else:
            smplx_idx = smplx_name_to_idx[_COCO_TO_SMPLX_NAME.get(coco_name, coco_name)]
        coco_ids.append(coco_id)
        smplx_ids.append(smplx_idx)
    return np.asarray(coco_ids, dtype=np.int64), np.asarray(smplx_ids, dtype=np.int64)


def smplx_joints_to_coco133_xyzc(joints: Float[ndarray, "n_frames n_joints 3"]) -> Float32[ndarray, "n_frames 133 4"]:
    """Scatter SMPL-X regressed joints into a COCO-133 ``[x, y, z, conf]`` stack.

    Args:
        joints: SMPL-X forward joints in meters — 127 per frame (or 144 with
            ``use_face_contour=True``; without the contour the 17 jawline
            keypoints stay NaN with confidence 0).

    Returns:
        COCO-133 keypoint stack; mapped joints carry confidence 1.0, unmapped
        entries are NaN with confidence 0.0.
    """
    indices: tuple[Int[ndarray, "n_mapped"], Int[ndarray, "n_mapped"]] = _coco133_to_smplx_indices()
    coco_ids: Int[ndarray, "n_mapped"] = indices[0]
    smplx_ids: Int[ndarray, "n_mapped"] = indices[1]
    num_joints: int = joints.shape[1]
    available: Bool[ndarray, "n_mapped"] = smplx_ids < num_joints
    xyzc_stack: Float32[ndarray, "n_frames 133 4"] = np.full((joints.shape[0], 133, 4), np.nan, dtype=np.float32)
    xyzc_stack[..., 3] = np.float32(0.0)
    xyzc_stack[:, coco_ids[available], :3] = joints[:, smplx_ids[available]].astype(np.float32)
    xyzc_stack[:, coco_ids[available], 3] = np.float32(1.0)
    return xyzc_stack
