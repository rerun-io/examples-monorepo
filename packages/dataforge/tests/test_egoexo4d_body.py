"""Ego-Exo4D-HM fits: npz decode, OpenPose-67 -> COCO-133, the body_pose layer, and SMPL-H parity with the release."""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES, column_rows, raw_asset, read_back, read_chunks
from jaxtyping import Float32, Int64
from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX

from dataforge import paths, schema, writing
from dataforge.datasets.egoexo4d_body import (
    BODY_MESH_STRIDE,
    SMPLH_FILE,
    SMPLX_FILE,
    HmFit,
    SmplhModel,
    coco133_from_openpose67,
    read_fit,
    write_body_mesh,
    write_body_pose,
)

FIT: Path = FIXTURES / "egoexo4d" / "cmu_bike02_4-first12.npz"
MODEL_ROOT: Path = Path(os.environ.get("DATAFORGE_EGOEXO4D_MODEL_ROOT", str(paths.raw_root() / "egoexo4d")))
# SLAHMR's smpl_to_openpose("smplh", use_hands=True, use_face=False, openpose_format="coco25") over SMPL-H's 52 joints
# plus the smplx vertex selector (slahmr/body_model/specs.py): the joint order of the release's joints3d.
SMPLH_TO_OPENPOSE67: list[int] = [
    52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    20, 34, 35, 36, 63, 22, 23, 24, 64, 25, 26, 27, 65, 31, 32, 33, 66, 28, 29, 30, 67,
    21, 49, 50, 51, 68, 37, 38, 39, 69, 40, 41, 42, 70, 46, 47, 48, 71, 43, 44, 45, 72,
]  # fmt: skip


def frame_clock(count: int) -> tuple[Int64[np.ndarray, "t"], Int64[np.ndarray, "t"]]:
    """A 30 fps clock starting at 1 s, and the frame indices it stamps."""
    frames: Int64[np.ndarray, "t"] = np.arange(count, dtype=np.int64)
    return 1_000_000_000 + frames * 1_000_000_000 // 30, frames


def test_read_fit_drops_the_person_axis() -> None:
    fit: HmFit = read_fit(FIT)
    assert fit.trans.shape == (12, 3)
    assert fit.hand_pose.shape == (12, 90)
    assert fit.betas.shape == (12, 16)
    assert fit.joints3d.shape == (12, 67, 3)
    assert fit.valid.dtype == np.bool_ and fit.valid.all()
    np.testing.assert_array_equal(fit.chunk_ranges, [[0, 12]])


def test_read_fit_names_the_file_on_untiled_chunks(tmp_path: Path) -> None:
    with np.load(FIT) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["chunk_ranges"] = np.array([[0, 5], [6, 12]], dtype=np.int64)
    broken: Path = tmp_path / "broken.npz"
    np.savez(broken, **arrays)
    with pytest.raises(ValueError, match="broken.npz"):
        read_fit(broken)


def test_openpose67_maps_body_feet_and_hands() -> None:
    joints: Float32[np.ndarray, "1 67 3"] = np.arange(67 * 3, dtype=np.float32).reshape(1, 67, 3)
    coco: Float32[np.ndarray, "1 133 3"] = coco133_from_openpose67(joints)
    # (COCO slot, OpenPose joint): nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, feet.
    pairs = [(0, 0), (1, 16), (2, 15), (3, 18), (4, 17), (5, 5), (6, 2), (7, 6), (8, 3), (9, 7), (10, 4), (11, 12), (12, 9)]
    pairs += [(13, 13), (14, 10), (15, 14), (16, 11), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (22, 24)]
    for slot, joint in pairs:
        np.testing.assert_array_equal(coco[0, slot], joints[0, joint])
    np.testing.assert_array_equal(coco[0, LEFT_HAND_IDX], joints[0, 25:46])
    np.testing.assert_array_equal(coco[0, RIGHT_HAND_IDX], joints[0, 46:67])
    assert np.isnan(coco[0, 23:91]).all()  # no face in the release
    used = {joint for _, joint in pairs} | set(range(25, 67))
    assert sorted(set(range(67)) - used) == [1, 8]  # neck and mid-hip have no COCO slot


def test_body_pose_layer(tmp_path: Path) -> None:
    fit: HmFit = read_fit(FIT)
    invalid: np.ndarray = fit.valid.copy()
    invalid[3] = False
    fit = dataclasses.replace(fit, valid=invalid)
    times, frames = frame_clock(12)
    target: Path = tmp_path / "body_pose.rrd"
    with writing.atomic_recording(target, recording_id="egoexo4d__cmu_bike02_4", default_blueprint=None, send_properties=False) as recording:
        write_body_pose(recording, fit, times, frames)
    chunks = read_chunks(target)
    keypoints = [chunk for chunk in chunks if str(chunk.entity_path) == schema.coco133_xyz_path() and not chunk.is_static]
    assert sum(chunk.num_rows for chunk in keypoints) == 12
    parameters = [chunk for chunk in chunks if str(chunk.entity_path) == schema.body_path("smplh") and not chunk.is_static]
    assert sum(chunk.num_rows for chunk in parameters) == 12
    store = read_back(target)
    confidence = column_rows(store, f"{schema.coco133_xyz_path()}:simplecv.KeypointConfidence3D:confidences")
    rows = confidence.column(1).to_pylist()
    assert not any(rows[3])  # the invalid frame keeps no keypoint
    assert all(value == 1.0 for value in rows[0][:23])  # no shipped confidence -> 1.0
    assert not any(rows[0][23:91])  # face slots are missing


@pytest.mark.golden
def test_smplh_reproduces_shipped_joints() -> None:
    """SMPL-H male + SMPL-X hand PCA (SLAHMR's model) regresses the release's joints3d to < 1e-5 m."""
    raw_asset("SMPL-H male model (dataforge-download egoexo4d)", MODEL_ROOT / SMPLH_FILE)
    raw_asset("SMPL-X neutral model (dataforge-download egoexo4d)", MODEL_ROOT / SMPLX_FILE)
    fit: HmFit = read_fit(FIT)
    model = SmplhModel(MODEL_ROOT)
    joints: Float32[np.ndarray, "t 73 3"] = model.forward(fit, np.arange(12, dtype=np.int64)).joints
    openpose: Float32[np.ndarray, "t 67 3"] = joints[:, SMPLH_TO_OPENPOSE67].copy()
    # SLAHMR moves both hips toward ViTPose's hip definition before reporting joints3d_op (optim/base_scene.py).
    hips, other = openpose[:, [9, 12]], openpose[:, [12, 9]]
    openpose[:, [9, 12]] = hips + 0.25 * (hips - other) + 0.5 * (openpose[:, [8]] - 0.5 * (hips + other))
    np.testing.assert_allclose(openpose, fit.joints3d, atol=1e-5)


@pytest.mark.integration
def test_body_mesh_layer_is_10hz_with_empty_invalid_rows(tmp_path: Path) -> None:
    raw_asset("SMPL-H male model (dataforge-download egoexo4d)", MODEL_ROOT / SMPLH_FILE)
    raw_asset("SMPL-X neutral model (dataforge-download egoexo4d)", MODEL_ROOT / SMPLX_FILE)
    fit: HmFit = read_fit(FIT)
    invalid: np.ndarray = fit.valid.copy()
    invalid[[1, 2, 6]] = False  # 1-2 lie between stride frames: their start and end still get rows
    fit = dataclasses.replace(fit, valid=invalid)
    times, frames = frame_clock(12)
    target: Path = tmp_path / "body_mesh.rrd"
    with writing.atomic_recording(target, recording_id="egoexo4d__cmu_bike02_4", default_blueprint=None, send_properties=False) as recording:
        write_body_mesh(recording, SmplhModel(MODEL_ROOT), fit, times, frames)
    mesh = [chunk for chunk in read_chunks(target) if str(chunk.entity_path) == schema.body_path("mesh") and not chunk.is_static]
    assert BODY_MESH_STRIDE == 3
    table = column_rows(read_back(target), f"{schema.body_path('mesh')}:Mesh3D:vertex_positions")
    assert sum(chunk.num_rows for chunk in mesh) == table.num_rows == 6
    # frames 0, 1 (turns invalid), 3 (valid again, on the stride), 6 (invalid, on the stride), 7 (valid again), 9
    assert [len(row) for row in table.column(1).to_pylist()] == [6890, 0, 6890, 0, 6890, 6890]
