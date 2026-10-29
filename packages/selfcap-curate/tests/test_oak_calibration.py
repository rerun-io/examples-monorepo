"""CPU OAK geometry, factory conventions and rotation pooling."""

from dataclasses import replace

import numpy as np
import pytest
from dataforge.datasets.selfcap_raw import Distortion, FactoryCalibration, FactoryExtrinsic
from frozen_v2 import session_episode
from scipy.spatial.transform import Rotation
from serde import from_dict

import selfcap_curate.oak_mount as module
from selfcap_curate.device_tables import pool_rotations
from selfcap_curate.factory import cy_shift_px, intrinsic_matrix, oak_stereo, quest_block, transform
from selfcap_curate.oak_mount import gyro_mount, solve_oak
from selfcap_curate.quest_fit import load_quest_hands
from selfcap_curate.records import CatalogHands, GyroFit, OakDetections, OakFit


def factory(quest: bool = True) -> FactoryCalibration:
    intrinsic = {
        "camera_id": "50",
        "positional_layout": "right",
        "lens_intrinsics": {
            "focal_length_x": 850.0,
            "focal_length_y": 850.0,
            "principal_point_x": 640.0,
            "principal_point_y": 642.0,
            "available": True,
        },
        "sensor_info": {"pixel_array_size": {"width": 1280, "height": 1280}},
        "capture_resolution": {"width": 1280, "height": 960},
        "distortion": {},
    }
    extrinsics = (
        []
        if quest
        else [
            {"type": name, "R": np.eye(3).tolist(), "T": translation}
            for name, translation in (("left_to_rgb", [2.0, 0.0, 0.0]), ("left_to_right", [7.5, 0.0, 0.0]))
        ]
    )
    return from_dict(FactoryCalibration, {"device_id": "device", "intrinsics": [intrinsic], "extrinsics": extrinsics})


def test_factory_converts_oak_centimetres_and_december_quest_crop() -> None:
    stereo = oak_stereo(factory(False))
    np.testing.assert_allclose(stereo["rgb_T_left"][:3, 3], [-0.02, 0.0, 0.0])
    np.testing.assert_allclose(stereo["rgb_T_right"][:3, 3], [0.055, 0.0, 0.0])
    quest = quest_block(factory())
    assert quest.mode == "faulty-unresolved"
    assert intrinsic_matrix(quest.pinholes["1"])[1, 2] == 482.0
    assert cy_shift_px(quest.pinholes["1"]) == 160.0


def test_broken_left_requires_explicit_borrow_and_measured_lever_arm() -> None:
    borrowed = FactoryExtrinsic("device_to_left", np.eye(3), np.array([-0.03, 0.01, -0.07]), "50")
    lever = np.eye(4)
    lever[0, 3] = 0.03
    result = quest_block(factory(), camera50=borrowed, rig_T_device=lever)
    assert result.mode == "faulty-left-borrowed-cam50"
    np.testing.assert_allclose(result.rig_T_cam["1"][:3, 3], [0.0, 0.01, -0.07], atol=1e-12)


@pytest.mark.parametrize(
    "flags,accepted",
    [
        ([], True),
        (["quest_unsynced_kept_as_cut"], False),
        (["quest_pose_unsynced"], False),
        (["oak_imu_unsynced"], False),
        (["quest_unsynced_kept_as_cut", "oak_imu_via_pose"], True),
        (["quest_pose_unsynced", "oak_imu_via_pose"], True),
        (["quest_unsynced_kept_as_cut", "oak_imu_unsynced"], False),
        (["quest_unsynced_kept_as_cut", "oak_imu_via_pose", "oak_imu_unsynced"], False),
    ],
)
def test_gyro_mount_recovers_known_axis_rotation(flags, accepted, fixture_sync) -> None:

    times = np.arange(0.0, 5.0, 0.005)
    rot = Rotation.from_euler("xyz", np.c_[0.3 * np.sin(times), 0.4 * np.sin(1.4 * times), 0.25 * np.cos(0.8 * times)])
    omega = (rot[:-1].inv() * rot[1:]).as_rotvec() / np.diff(times)[:, None]
    midpoint = (times[:-1] + times[1:]) / 2.0
    expected = Rotation.from_euler("xyz", [0.2, -0.1, 0.3])
    imu = np.column_stack([np.interp(times, midpoint, expected.inv().apply(omega)[:, k]) for k in range(3)])
    hands = CatalogHands(times, np.full((len(times), 133, 3), np.nan), times, np.zeros((len(times), 3)), rot.as_quat())
    sync = fixture_sync()
    sync = replace(sync, offsets_ms={"quest_pose": 0.0, "oak_imu": -125.0})
    sync = replace(sync, flags=flags)
    result = gyro_mount("synthetic", [session_episode(sync, hands, times + 0.125, imu)])
    if not accepted:
        assert not result.ok
        return
    assert result.ok
    assert np.degrees((expected.inv() * Rotation.from_matrix(result.R_head_from_imu)).magnitude()) < 0.01


def test_rotation_pool_trims_bad_mounts_and_preserves_provenance() -> None:

    free, gyro = [], {}
    for i, angle in enumerate([1.0, 2.0, 3.0, 40.0]):
        session = f"session-{i}"
        free.append(OakFit(session, True, "oak", 10.0, np.eye(4), 0.8))
        gyro[session] = GyroFit(session, True, Rotation.from_euler("z", angle, degrees=True).as_matrix(), 0.2)
    result = pool_rotations(free, gyro, "2026-10-26").devices["oak"]
    assert Rotation.from_matrix(result.rgb_R_imu).as_euler("xyz", degrees=True)[2] == pytest.approx(2.0)
    assert result.source_sessions == tuple(f"session-{i}" for i in range(4))
    assert result.used_sessions == tuple(f"session-{i}" for i in range(3))


@pytest.mark.parametrize("flag", [None, "quest_unsynced_kept_as_cut", "quest_pose_unsynced"])
@pytest.mark.parametrize("latency_ms", [48.0, 60.0])
def test_hand_pnp_recovers_mount_with_fixed_gyro_rotation(flag, latency_ms, monkeypatch, fixture_sync) -> None:

    monkeypatch.setattr(module, "HAND_LATENCY_MS", latency_ms, raising=False)
    rng = np.random.default_rng(25)
    times = np.arange(0.0, 6.0, 0.01)
    geometry = rng.normal(scale=0.03, size=(21, 3)) + np.array([0.0, 0.0, 0.65])

    def joints(t, shift):
        return geometry[None] + np.stack([0.1 * np.sin(t), 0.08 * np.cos(1.1 * t), 0.05 * np.sin(0.7 * t)], axis=1)[:, None] + shift

    positions = np.full((len(times), 133, 3), np.nan)
    for base, shift in ((91, 0.0), (112, 0.1)):
        positions[:, base : base + 21] = joints(times, shift)
    raw = CatalogHands(times, positions, times, np.zeros((len(times), 3)), np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1)))
    sync = fixture_sync()
    sync = replace(sync, offsets_ms={"quest_pose": 0.0, "oak_imu": 0.0})
    sync = replace(sync, flags=[] if flag is None else [flag])
    motion = session_episode(sync, raw, times, np.zeros((len(times), 3)), quest=load_quest_hands(raw))
    intrinsic = replace(factory(False).intrinsics[0], distortion=Distortion((0.0,) * 14))
    expected = transform(Rotation.from_euler("x", 0.2).as_matrix(), np.array([0.03, 0.04, -0.02]))
    vt = np.linspace(0.5, 5.5, 32)
    xy = np.full((len(vt), 133, 2), np.nan)
    scores = np.zeros((len(vt), 133))
    for base, shift in ((91, 0.0), (112, 0.1)):
        rgb = (joints(vt + latency_ms / 1000.0, shift) - expected[:3, 3]) @ expected[:3, :3]
        pixels = rgb @ intrinsic_matrix(intrinsic).T
        xy[:, base : base + 21] = pixels[..., :2] / pixels[..., 2:3]
        scores[:, base : base + 21] = 1.0
    result = solve_oak("synthetic", [replace(motion, oak=OakDetections(vt, xy, scores))], intrinsic, "oak", expected[:3, :3])
    if flag is not None:
        assert not result.ok
        return
    assert result.ok and result.median_px < 0.1
    np.testing.assert_allclose(result.questrig_T_rgb, expected, atol=1e-4)


def test_wrong_camera_borrow_is_unresolved() -> None:
    wrong = FactoryExtrinsic("device_to_left", np.eye(3), np.zeros(3), "51")
    assert quest_block(factory(), camera50=wrong, rig_T_device=np.eye(4)).mode == "faulty-unresolved"
