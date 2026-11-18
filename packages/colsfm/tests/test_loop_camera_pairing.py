"""`colsfm.loop_pairs.loop_camera_pairing`: which target camera each source anchor of a loop measurement is matched against.

Synthetic rigs only: a rig is a set of optical axes, a rig frame is a prior pose, and the retrieval index is a hand-written
score matrix. Nothing here matches an image.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32, Float64
from loop_helpers import same_camera_image_pairs
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.database import Keypoints
from colsfm.loop_pairs import CameraPairing, loop_camera_pairing, required_image_pairs
from colsfm.retrieval import RetrievalConfig, RetrievalIndex
from colsfm.rig_geometry import RigFrameIndex, RigGeometry

CUBE_AXES: Final[dict[int, tuple[float, float, float]]] = {
    0: (1.0, 0.0, 0.0),
    1: (-1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, -1.0, 0.0),
    4: (0.0, 0.0, 1.0),
    5: (0.0, 0.0, -1.0),
}
"""Camera id -> optical axis in the rig frame: the six faces of a cube."""

TURNED_HALF_WAY: Final[dict[int, int]] = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4, 5: 5}
"""Under a 180 deg turn about z, the target face that looks where each source face looks."""


def cam_from_rig(axis: tuple[float, float, float]) -> pycolmap.Rigid3d:
    """A camera at the rig origin whose optical axis (row 2 of `cam_R_rig`) is `axis`."""
    z: Float64[ndarray, "3"] = np.asarray(axis, dtype=np.float64)
    helper: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x: Float64[ndarray, "3"] = np.cross(helper, z)
    x /= np.linalg.norm(x)
    rotation: Float64[ndarray, "3 3"] = np.stack([x, np.cross(z, x), z])
    return pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_matrix(rotation).as_quat()), np.zeros(3))


def make_rig(axes: dict[int, tuple[float, float, float]], stereo_partner: dict[int, int] | None = None) -> RigGeometry:
    """A rig of pinhole cameras with the given optical axes and stereo declarations."""
    camera: pycolmap.Camera = pycolmap.Camera(model="PINHOLE", width=800, height=800, params=[400.0, 400.0, 400.0, 400.0])
    camera_ids: tuple[int, ...] = tuple(sorted(axes))
    return RigGeometry(
        camera_ids=camera_ids,
        cams_from_rig=tuple(cam_from_rig(axes[camera_id]) for camera_id in camera_ids),
        cameras=tuple(camera for _ in camera_ids),
        stereo_partner={} if stereo_partner is None else stereo_partner,
    )


def make_frames(geometry: RigGeometry, yaws_deg: list[float]) -> RigFrameIndex:
    """Rig frames 0..n-1 at the origin, turned about z by `yaws_deg`; image id = 10 * rig id + camera id."""
    return RigFrameIndex(
        sequence=tuple(range(len(yaws_deg))),
        world_T_rig={
            rig_id: pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_euler("z", yaw, degrees=True).as_quat()), np.zeros(3))
            for rig_id, yaw in enumerate(yaws_deg)
        },
        keyframe_by_rig_camera={(rig_id, camera_id): 10 * rig_id + camera_id for rig_id in range(len(yaws_deg)) for camera_id in geometry.camera_ids},
    )


def make_retrieval(image_ids: list[int], similarity: Float32[ndarray, "n n"]) -> RetrievalIndex:
    """A retrieval index that answers with the given score matrix."""
    return RetrievalIndex(
        image_ids=tuple(image_ids), similarity=similarity, descriptors_per_image=1, config=RetrievalConfig(), build_seconds=0.0
    )


def random_retrieval(index: RigFrameIndex, seed: int) -> RetrievalIndex:
    """A retrieval index with random symmetric scores over every image of `index`."""
    image_ids: list[int] = sorted(index.keyframe_by_rig_camera.values())
    scores: Float32[ndarray, "n n"] = np.random.default_rng(seed).random((len(image_ids), len(image_ids))).astype(np.float32)
    scores = (scores + scores.T) / np.float32(2.0)
    np.fill_diagonal(scores, np.float32(1.0))
    return make_retrieval(image_ids, scores)


def all_keypoints(index: RigFrameIndex) -> dict[int, Keypoints]:
    """One keypoint per image, so every image can be paired."""
    return {image_id: np.zeros((1, 2), dtype=np.float64) for image_id in index.keyframe_by_rig_camera.values()}


@pytest.mark.parametrize("by_prior", [False, True])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "geometry",
    [
        make_rig({0: (0.0, 0.0, 1.0)}),
        make_rig({0: (0.0, 0.0, 1.0), 1: (0.0, 0.0, 1.0)}, {0: 1, 1: 0}),
        # A toed-in stereo pair: calibration noise must not move the primary to the partner.
        make_rig({0: (0.0, 0.05, 1.0), 1: (0.0, -0.05, 1.0)}, {0: 1, 1: 0}),
    ],
    ids=["mono", "stereo", "stereo-toed-in"],
)
def test_mono_and_stereo_plans_give_exactly_mains_pair_list(geometry: RigGeometry, seed: int, by_prior: bool) -> None:
    """Random scores and random headings change nothing for a rig whose cameras look one way."""
    index: RigFrameIndex = make_frames(geometry, list(np.random.default_rng(seed).uniform(-180.0, 180.0, size=8)))
    retrieval: RetrievalIndex = random_retrieval(index, seed)
    keypoints: dict[int, Keypoints] = all_keypoints(index)
    rig_pairs: list[tuple[int, int]] = [(0, 5), (1, 6), (2, 7), (1, 4)]
    pairing: CameraPairing = loop_camera_pairing(index, geometry, retrieval, keypoints, rig_pairs, by_prior)
    assert required_image_pairs(index, geometry, keypoints, 1, rig_pairs, pairing) == same_camera_image_pairs(index, geometry, keypoints, 1, rig_pairs)


def test_the_prior_pairs_a_cube_face_with_the_face_that_looks_the_same_way_after_a_half_turn() -> None:
    """A revisit walked the other way: each face pairs with the opposite face, up and down stay."""
    geometry: RigGeometry = make_rig(CUBE_AXES)
    index: RigFrameIndex = make_frames(geometry, [0.0, 180.0])
    retrieval: RetrievalIndex = random_retrieval(index, 0)
    pairing: CameraPairing = loop_camera_pairing(index, geometry, retrieval, all_keypoints(index), [(0, 1)], by_prior=True)
    assert {camera_id: pairing[(0, 1, camera_id)] for camera_id in CUBE_AXES} == {c: (t,) for c, t in TURNED_HALF_WAY.items()}


def test_the_best_retrieval_score_pairs_a_cube_face_with_the_face_that_sees_the_same_wall() -> None:
    """Without a prior: retrieval scores the facing face highest, so that face is chosen, whatever the prior says."""
    geometry: RigGeometry = make_rig(CUBE_AXES)
    index: RigFrameIndex = make_frames(geometry, [0.0, 0.0])  # the prior says "same heading"; retrieval knows better
    image_ids: list[int] = sorted(index.keyframe_by_rig_camera.values())
    position: dict[int, int] = {image_id: row for row, image_id in enumerate(image_ids)}
    scores: Float32[ndarray, "n n"] = np.full((len(image_ids), len(image_ids)), 0.05, dtype=np.float32)
    np.fill_diagonal(scores, np.float32(1.0))
    for source_camera, target_camera in TURNED_HALF_WAY.items():
        scores[position[source_camera], position[10 + target_camera]] = scores[position[10 + target_camera], position[source_camera]] = 0.4
    pairing: CameraPairing = loop_camera_pairing(index, geometry, make_retrieval(image_ids, scores), all_keypoints(index), [(0, 1)], by_prior=False)
    assert {camera_id: pairing[(0, 1, camera_id)] for camera_id in CUBE_AXES} == {c: (t,) for c, t in TURNED_HALF_WAY.items()}


def _forward_rig_scores(score_by_target_camera: dict[int, float]) -> tuple[RigGeometry, RigFrameIndex, RetrievalIndex]:
    """Two forward stereo pairs (0-1, 2-3) and two rig frames; camera 0's anchor scores as given against the target."""
    geometry: RigGeometry = make_rig({0: (0.0, 0.0, 1.0), 1: (0.0, 0.0, 1.0), 2: (1.0, 0.0, 0.0), 3: (1.0, 0.0, 0.0)}, {0: 1, 1: 0, 2: 3, 3: 2})
    index: RigFrameIndex = make_frames(geometry, [0.0, 0.0])
    image_ids: list[int] = sorted(index.keyframe_by_rig_camera.values())
    position: dict[int, int] = {image_id: row for row, image_id in enumerate(image_ids)}
    scores: Float32[ndarray, "n n"] = np.zeros((len(image_ids), len(image_ids)), dtype=np.float32)
    np.fill_diagonal(scores, np.float32(1.0))
    for target_camera, score in score_by_target_camera.items():
        scores[position[0], position[10 + target_camera]] = scores[position[10 + target_camera], position[0]] = score
    return geometry, index, make_retrieval(image_ids, scores)


def test_equal_scores_keep_the_same_camera() -> None:
    geometry, index, retrieval = _forward_rig_scores({0: 0.3, 2: 0.3, 3: 0.3})
    pairing: CameraPairing = loop_camera_pairing(index, geometry, retrieval, all_keypoints(index), [(0, 1)], by_prior=False)
    assert pairing[(0, 1, 0)] == (0, 1)


def test_the_stereo_partner_never_becomes_the_primary() -> None:
    geometry, index, retrieval = _forward_rig_scores({0: 0.1, 1: 0.9})
    pairing: CameraPairing = loop_camera_pairing(index, geometry, retrieval, all_keypoints(index), [(0, 1)], by_prior=False)
    assert pairing[(0, 1, 0)] == (0, 1)


def test_a_better_scoring_camera_becomes_the_primary_and_brings_its_partner() -> None:
    geometry, index, retrieval = _forward_rig_scores({0: 0.1, 3: 0.5})
    pairing: CameraPairing = loop_camera_pairing(index, geometry, retrieval, all_keypoints(index), [(0, 1)], by_prior=False)
    assert pairing[(0, 1, 0)] == (3, 2)


def test_the_prior_keeps_the_same_camera_on_a_tie() -> None:
    """Two forward cameras with the same axis that are not a stereo pair: the prior cannot tell them apart."""
    geometry: RigGeometry = make_rig({0: (0.0, 0.0, 1.0), 1: (0.0, 0.0, 1.0)})
    index: RigFrameIndex = make_frames(geometry, [0.0, 0.0])
    pairing: CameraPairing = loop_camera_pairing(index, geometry, random_retrieval(index, 0), all_keypoints(index), [(0, 1)], by_prior=True)
    assert pairing[(0, 1, 0)] == (0,)
    assert pairing[(0, 1, 1)] == (1,)


def test_a_camera_without_an_image_is_never_chosen_and_no_pair_is_dropped() -> None:
    """The rule chooses a camera; it never removes the source camera's entry, even when the facing face is missing."""
    geometry: RigGeometry = make_rig(CUBE_AXES)
    index: RigFrameIndex = make_frames(geometry, [0.0, 180.0])
    keypoints: dict[int, Keypoints] = all_keypoints(index)
    del keypoints[10 + TURNED_HALF_WAY[0]]
    pairing: CameraPairing = loop_camera_pairing(index, geometry, random_retrieval(index, 0), keypoints, [(0, 1)], by_prior=True)
    assert set(pairing) == {(0, 1, camera_id) for camera_id in CUBE_AXES}
    assert pairing[(0, 1, 0)] != (TURNED_HALF_WAY[0],)
