"""Correctness tests for the COLMAP-style ego rig factoring.

The load-bearing invariant of the rig schema is that recomposing the logged
``world_T_rig(t)`` (on the rig node) with each camera's static ``rig_T_cam``
reproduces the camera's original ``world_T_cam(t)``. These tests assert that
directly on :func:`simplecv.data.exoego.base_exoego._build_ego_rigs`, covering the
rigid-collapse path, NaN tracking dropouts, and the non-rigid fallback.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exoego.base_exoego import _build_ego_rigs
from simplecv.rig import Rig


def _random_se3(rng: np.random.Generator) -> Float[ndarray, "4 4"]:
    """A random rigid transform (proper rotation + translation)."""
    axis: Float[ndarray, "3"] = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle: float = float(rng.uniform(0.0, np.pi))
    skew: Float[ndarray, "3 3"] = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation: Float[ndarray, "3 3"] = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    transform: Float[ndarray, "4 4"] = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = rng.normal(size=3)
    return transform


def _intrinsics() -> Intrinsics:
    return Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=500.0, fl_y=500.0, cx=320.0, cy=240.0, height=480, width=640)


def _pinhole_from_world_T_cam(name: str, world_T_cam: Float[ndarray, "4 4"]) -> PinholeParameters:
    """Build a PinholeParameters from a world_T_cam pose (NaN-safe)."""
    return PinholeParameters(name=name, extrinsics=Extrinsics(world_R_cam=world_T_cam[:3, :3].copy(), world_t_cam=world_T_cam[:3, 3].copy()), intrinsics=_intrinsics())


def _fake_ego_sequence(ego_cam_dict: dict[str, list[PinholeParameters]]) -> SimpleNamespace:
    """Minimal stand-in exposing only what ``_build_ego_rigs`` consumes."""
    return SimpleNamespace(image_plane_distance=0.1, ego_cam_dict=ego_cam_dict, ego_video_names=list(ego_cam_dict.keys()))


def _recompose_world_T_cam(rig: Rig, cam_index: int, frame: int) -> Float[ndarray, "4 4"]:
    """Replicate Rerun's tree composition: world_T_rig(t) @ rig_T_cam.

    The rig node stores world_T_rig (parent-from-child). The camera node stores
    rig_T_cam via ``from_parent=True`` (the logged values are ``cam_T_world`` of the
    sensor's rig-frame extrinsics), so going camera->rig->world is
    ``world_T_rig @ inv(cam_T_world) == world_T_rig @ world_T_cam``.
    """
    pose = rig.pose_stream
    assert pose is not None
    world_T_rig: Float[ndarray, "4 4"] = np.eye(4)
    world_T_rig[:3, :3] = pose.world_R_rig[frame]
    world_T_rig[:3, 3] = pose.world_t_rig[frame]
    sensor_extrinsics: Extrinsics = rig.calibration.cameras[cam_index].pinhole.extrinsics
    return world_T_rig @ sensor_extrinsics.world_T_cam


def test_rigid_multicam_ego_collapses_to_one_rig_and_recomposes() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    n_frames: int = 12
    # A moving device pose per frame, with three rigidly-attached cameras.
    world_T_device: list[Float[ndarray, "4 4"]] = [_random_se3(rng) for _ in range(n_frames)]
    device_T_cam: dict[str, Float[ndarray, "4 4"]] = {"camera-rgb": np.eye(4), "slam-left": _random_se3(rng), "slam-right": _random_se3(rng)}

    ego_cam_dict: dict[str, list[PinholeParameters]] = {}
    truth: dict[str, list[Float[ndarray, "4 4"]]] = {}
    for name, dev_T_cam in device_T_cam.items():
        poses: list[Float[ndarray, "4 4"]] = [world_T_device[t] @ dev_T_cam for t in range(n_frames)]
        truth[name] = poses
        ego_cam_dict[name] = [_pinhole_from_world_T_cam(name, pose) for pose in poses]

    rigs, cam_paths = _build_ego_rigs(_fake_ego_sequence(ego_cam_dict), start_index=2, world_path=Path("world"))

    assert len(rigs) == 1, "rigidly-coupled multi-cam ego should collapse to ONE rig"
    rig: Rig = rigs[0]
    assert rig.index == 2
    assert len(rig.calibration.cameras) == 3
    # reference is the rgb-named camera (index 0 here)
    assert rig.calibration.reference_index == 0
    assert rig.calibration.cameras[0].kind == "rgb" and rig.calibration.cameras[1].kind == "grayscale"
    assert set(cam_paths) == set(ego_cam_dict)
    assert str(cam_paths["camera-rgb"]) == "world/rig_02/cam_00"

    for cam_index, name in enumerate(ego_cam_dict):
        for frame in range(n_frames):
            recomposed: Float[ndarray, "4 4"] = _recompose_world_T_cam(rig, cam_index, frame)
            np.testing.assert_allclose(recomposed, truth[name][frame], atol=1e-9, err_msg=f"recompose mismatch for {name} @ {frame}")


def test_single_cam_ego_is_one_rig_with_identity_offset() -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    n_frames: int = 8
    poses: list[Float[ndarray, "4 4"]] = [_random_se3(rng) for _ in range(n_frames)]
    ego_cam_dict: dict[str, list[PinholeParameters]] = {"hololens": [_pinhole_from_world_T_cam("hololens", p) for p in poses]}

    rigs, _ = _build_ego_rigs(_fake_ego_sequence(ego_cam_dict), start_index=9, world_path=Path("world"))

    assert len(rigs) == 1 and rigs[0].index == 9
    assert len(rigs[0].calibration.cameras) == 1
    # identity rig_T_cam for the single camera
    np.testing.assert_allclose(rigs[0].calibration.cameras[0].pinhole.extrinsics.cam_T_world, np.eye(4), atol=1e-12)
    for frame in range(n_frames):
        np.testing.assert_allclose(_recompose_world_T_cam(rigs[0], 0, frame), poses[frame], atol=1e-9)


def test_nan_dropout_propagates_to_rig_pose() -> None:
    rng: np.random.Generator = np.random.default_rng(2)
    n_frames: int = 10
    poses: list[Float[ndarray, "4 4"]] = [_random_se3(rng) for _ in range(n_frames)]
    nan_pose: Float[ndarray, "4 4"] = np.full((4, 4), np.nan)
    poses[4] = nan_pose  # a tracking dropout frame
    poses[5] = nan_pose
    ego_cam_dict: dict[str, list[PinholeParameters]] = {"hololens": [_pinhole_from_world_T_cam("hololens", p) for p in poses]}

    rigs, _ = _build_ego_rigs(_fake_ego_sequence(ego_cam_dict), start_index=0, world_path=Path("world"))
    pose = rigs[0].pose_stream
    assert pose is not None and pose.valid is not None
    assert not pose.valid[4] and not pose.valid[5]
    assert bool(pose.valid[0]) and bool(pose.valid[9])
    # dropout frames carry NaN translation so the rig (and all its frusta) disappears
    assert np.isnan(pose.world_t_rig[4]).all()
    assert np.isfinite(pose.world_t_rig[0]).all()


def test_nonrigid_ego_falls_back_to_per_camera_rigs() -> None:
    rng: np.random.Generator = np.random.default_rng(3)
    n_frames: int = 12
    # Two cameras whose RELATIVE pose changes over time -> not rigidly factorable.
    world_T_device: list[Float[ndarray, "4 4"]] = [_random_se3(rng) for _ in range(n_frames)]
    ego_cam_dict: dict[str, list[PinholeParameters]] = {
        "camera-rgb": [_pinhole_from_world_T_cam("camera-rgb", world_T_device[t]) for t in range(n_frames)],
        # second camera wanders independently each frame (large drift > RIGIDITY_TOL)
        "slam-left": [_pinhole_from_world_T_cam("slam-left", world_T_device[t] @ _random_se3(rng)) for t in range(n_frames)],
    }

    rigs, cam_paths = _build_ego_rigs(_fake_ego_sequence(ego_cam_dict), start_index=5, world_path=Path("world"))

    assert len(rigs) == 2, "non-rigid multi-cam ego should fall back to one rig per camera"
    assert [r.index for r in rigs] == [5, 6]
    assert all(len(r.calibration.cameras) == 1 for r in rigs)
    assert str(cam_paths["camera-rgb"]) == "world/rig_05/cam_00"
    assert str(cam_paths["slam-left"]) == "world/rig_06/cam_00"
