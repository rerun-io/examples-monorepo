"""HOT3D UmeTrack FK and skinning using the same model math as SHOW3D."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy, skin_landmarks, skin_mesh, wrist_for_hand

from dataforge import hands
from dataforge.datasets.hot3d_source import UmeFrame
from dataforge.datasets.show3d_hands import HAND_SIDES


@dataclass(frozen=True, slots=True)
class HandBatch:
    """Dense parameters and landmarks; mesh vertices only for present hands."""

    positions: Float32[ndarray, "n 133 3"]
    """COCO world positions in metres; NaN for absent slots."""
    confidence: Float32[ndarray, "n 133"]
    """Source confidence or zero for missing slots."""
    angles: Float32[ndarray, "2 n 22"]
    """Source angles; NaN for missing hands."""
    wrists: Float32[ndarray, "2 n 4 4"]
    """World-from-wrist in metres, before handedness mirroring."""
    scores: Float32[ndarray, "2 n"]
    """Source hand confidence or zero."""


def evaluate_hands(model: HandModelNumpy, rows: dict[int, UmeFrame], times_ns: Int64[ndarray, "n"]) -> HandBatch:
    """Exact timestamp lookup and FK; the profile is millimetres, wrists are metres."""
    count: int = len(times_ns)
    angles: Float32[ndarray, "2 n 22"] = np.full((2, count, 22), np.nan, dtype=np.float32)
    wrists: Float32[ndarray, "2 n 4 4"] = np.full((2, count, 4, 4), np.nan, dtype=np.float32)
    scores: Float32[ndarray, "2 n"] = np.zeros((2, count), dtype=np.float32)
    landmarks: Float32[ndarray, "n 2 21 3"] = np.full((count, 2, 21, 3), np.nan, dtype=np.float32)
    for index, stamp in enumerate(times_ns):
        row: UmeFrame | None = rows.get(int(stamp))
        for side_index, side in enumerate(HAND_SIDES):
            if row is not None and side.key in row.hand_poses:
                pose = row.hand_poses[side.key]
                angles[side_index, index] = pose.joint_angles
                wrists[side_index, index] = pose.wrist_xform.matrix()
                scores[side_index, index] = pose.hand_confidence
    for side_index in range(len(HAND_SIDES)):
        present = np.isfinite(angles[side_index]).all(axis=1)
        if np.any(present):
            landmarks[present, side_index] = skin_landmarks(
                model, angles[side_index, present], umetrack_wrists(wrists[side_index, present], side_index)
            ) * np.float32(0.001)
    positions: Float32[ndarray, "n 133 3"] = np.full((count, 133, 3), np.nan, dtype=np.float32)
    confidence: Float32[ndarray, "n 133"] = np.zeros((count, 133), dtype=np.float32)
    for index in range(count):
        positions[index], confidence[index] = hands.coco133_from_hands(landmarks[index], scores[:, index])
    return HandBatch(positions, confidence, angles, wrists, scores)


def umetrack_wrists(wrists: Float32[ndarray, "n 4 4"], side_index: int) -> Float32[ndarray, "n 4 4"]:
    """Metre world-from-wrist to the millimetre, handedness-mirrored transform the UmeTrack model expects."""
    transforms: Float32[ndarray, "n 4 4"] = wrists.copy()
    transforms[:, :3, 3] *= np.float32(1000.0)
    return wrist_for_hand(transforms, HAND_SIDES[side_index].model_index)


def mesh_vertices(
    model: HandModelNumpy, angles: Float32[ndarray, "n 22"], wrists: Float32[ndarray, "n 4 4"], side_index: int
) -> Float32[ndarray, "n v 3"]:
    """Skin a present-hand batch into world metres, using SHOW3D's handedness rule."""
    return skin_mesh(model, angles, umetrack_wrists(wrists, side_index)) * np.float32(0.001)
