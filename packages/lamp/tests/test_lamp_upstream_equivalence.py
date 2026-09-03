"""Numerical equivalence between the owned fork and pristine upstream source."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from numpy import ndarray
from torch import Tensor, nn

from lamptrack.third_party.lamp.core.types import Detection2D, Skeleton
from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings, SnippetData
from lamptrack.third_party.lamp.tracking.tracker import LampTracker

REFERENCE_DIR = Path(__file__).parent / "reference_data" / "lamp"


def _load_module(name: str, filename: str) -> ModuleType:
    """Load one pristine source file under its original package name."""
    path = REFERENCE_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream fixture {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class _CalibrationStub:
    """Import-only replacement for the pristine Aria calibration container."""


def _load_upstream_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load pristine lifter, tracker, and core types without Aria dependencies."""
    for package_name in ("lamp", "lamp.core", "lamp.io", "lamp.models", "lamp.tracking"):
        package = ModuleType(package_name)
        package.__path__ = [str(REFERENCE_DIR)]
        sys.modules[package_name] = package
    _load_module("lamp.core.se3", "upstream_core_se3.py")
    types_module = _load_module("lamp.core.types", "upstream_core_types.py")
    sensor_module = ModuleType("lamp.io.sensor_io")
    sensor_module.PerCameraCalibration = _CalibrationStub
    sys.modules["lamp.io.sensor_io"] = sensor_module
    _load_module("lamp.models.model_utils", "upstream_models_model_utils.py")
    _load_module("lamp.models.blocks", "upstream_models_blocks.py")
    _load_module("lamp.models.model", "upstream_models_model.py")
    _load_module("lamp.models.model_loader", "upstream_models_model_loader.py")
    lifter_module = _load_module("lamp.models.lifter", "upstream_models_lifter.py")
    _load_module("lamp.tracking.smoothing", "upstream_tracking_smoothing.py")
    _load_module("lamp.tracking.tracking_utils", "upstream_tracking_tracking_utils.py")
    _load_module("lamp.tracking.snippets", "upstream_tracking_snippets.py")
    tracker_module = _load_module("lamp.tracking.tracker", "upstream_tracking_tracker.py")
    return lifter_module, tracker_module, types_module


class _DeterministicLampNet(nn.Module):
    """Small model double at the external learned-model boundary."""

    def forward(
        self,
        keypoints: list[Tensor],
        camera_params: list[Tensor],
        camera_poses: list[Tensor],
        ground_planes: Tensor,
    ) -> dict[str, Tensor]:
        """Return deterministic SMPL-shaped tensors from every input family."""
        batch = int(keypoints[0].shape[0])
        steps = int(keypoints[0].shape[1])
        keypoint_signal = torch.stack([view[..., :2].mean(dim=(2, 3)) for view in keypoints]).mean(dim=0)
        camera_signal = torch.stack([view[..., :4].mean(dim=2) for view in camera_params]).mean(dim=0)
        pose_signal = torch.stack([view[..., :3, 3].mean(dim=2) for view in camera_poses]).mean(dim=0)
        floor_signal = torch.nan_to_num(ground_planes, nan=0.0).mean(dim=(1, 2))[:, None]
        signal = keypoint_signal + camera_signal * 1e-5 + pose_signal * 1e-4 + floor_signal * 1e-4
        translation = torch.stack((signal, signal * 0.5, signal * 0.25), dim=-1)
        rotations = torch.eye(3, dtype=torch.float32).expand(batch, steps, 24, 3, 3).clone()
        joints = translation[:, :, None, :].expand(batch, steps, 24, 3).clone()
        return {
            "skel_w": joints,
            "transl": translation,
            "global_orient_rotmat": rotations[:, :, :1],
            "body_pose_rotmat": rotations[:, :, 1:],
            "betas": signal.mean(dim=1, keepdim=True).expand(batch, 10).clone(),
        }


def _snippet(snippet_type: type, *, person_id: int, steps: int, camera_width: int) -> object:
    """Build one seeded four-view snippet through either implementation."""
    generator = np.random.default_rng(42 + person_id)
    keypoints: list[ndarray] = []
    camera_params: list[ndarray] = []
    camera_poses: list[ndarray] = []
    for view_idx in range(4):
        keypoints_view = generator.uniform(0.0, 512.0, size=(steps, 17, 3)).astype(np.float32)
        keypoints_view[..., 2] = 1.0
        keypoints.append(keypoints_view)
        params = generator.normal(size=(steps, camera_width)).astype(np.float32)
        params[:, :4] += np.array([500.0, 500.0, 256.0, 256.0], dtype=np.float32)
        camera_params.append(params)
        poses = np.tile(np.eye(4, dtype=np.float32), (steps, 1, 1))
        poses[:, 0, 3] = float(view_idx) * 0.1
        camera_poses.append(poses)
    return snippet_type(
        person_id=person_id,
        snippet_timestamps_ns=[1_000_000_000 + i * 100_000_000 for i in range(steps)],
        view_cam_indices=[0, 1, 2, 3],
        kp2ds_per_view=keypoints,
        Ts_gw_cam_per_view=camera_poses,
        cam_params_per_view=camera_params,
        T_gravityWorld_world=np.eye(4, dtype=np.float32),
    )


@pytest.mark.parametrize("steps", [5, 20])
@pytest.mark.parametrize("camera_width", [4, 16])
@pytest.mark.parametrize("floor_z", [None, 0.0])
@pytest.mark.parametrize("batch", [1, 3])
def test_lifter_forward_matches_pristine_upstream(steps: int, camera_width: int, floor_z: float | None, batch: int) -> None:
    """All supported host-side input combinations remain bit-identical."""
    upstream_lifter_module, _, _ = _load_upstream_modules()
    upstream = upstream_lifter_module.Lifter(
        _DeterministicLampNet(), torch.device("cpu"), upstream_lifter_module.LifterSettings(snippet_length=steps)
    )
    owned = Lifter(_DeterministicLampNet(), torch.device("cpu"), LifterSettings(snippet_length=steps))
    upstream.set_floor_plane(floor_z)
    owned.set_floor_plane(floor_z)
    upstream_snippets = {
        person_id: _snippet(upstream_lifter_module.SnippetData, person_id=person_id, steps=steps, camera_width=camera_width)
        for person_id in range(1, batch + 1)
    }
    owned_snippets = {
        person_id: _snippet(SnippetData, person_id=person_id, steps=steps, camera_width=camera_width)
        for person_id in range(1, batch + 1)
    }

    upstream_outputs = upstream.lift_all_steps_batched(upstream_snippets)
    owned_outputs = owned.lift_all_steps_batched(owned_snippets)

    assert upstream_outputs.keys() == owned_outputs.keys()
    for person_id in upstream_outputs:
        for (upstream_ts, upstream_skeleton), (owned_ts, owned_skeleton) in zip(
            upstream_outputs[person_id], owned_outputs[person_id], strict=True
        ):
            assert upstream_ts == owned_ts
            assert np.array_equal(upstream_skeleton.kp_world, owned_skeleton.kp_world)
            assert np.array_equal(upstream_skeleton.T_world_pelvis, owned_skeleton.T_world_pelvis)
            assert np.array_equal(upstream_skeleton.shape, owned_skeleton.shape)
            assert np.array_equal(upstream_skeleton.joints_rot_mat, owned_skeleton.joints_rot_mat)


def _detection(detection_type: type, timestamp_ns: int, offset: float) -> object:
    """Build an association record for either implementation."""
    return detection_type(
        box_xyxy=np.array([10.0 + offset, 20.0, 110.0 + offset, 220.0], dtype=np.float32),
        box_score=0.9,
        keypoints=np.zeros((17, 3), dtype=np.float32),
        cam_idx=0,
        timestamp_ns=timestamp_ns,
        has_keypoints=False,
    )


def _skeleton(skeleton_type: type, shift: float) -> object:
    """Build a valid upright SMPL skeleton for smoothing parity."""
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[:, 0] = shift
    joints[:, 2] = 2.0
    joints[[4, 5], 2] = 1.4
    joints[[7, 8], 2] = 0.8
    root = np.eye(4, dtype=np.float32)
    root[0, 3] = shift
    return skeleton_type(
        kp_world=joints,
        kp_score=np.ones(24, dtype=np.float32),
        T_world_pelvis=root,
        shape=np.full(10, shift, dtype=np.float32),
        joints_rot_mat=np.tile(np.eye(3, dtype=np.float32), (24, 1, 1)),
    )


def test_tracker_and_smoothing_match_pristine_upstream() -> None:
    """Synthetic association and repeated-window fusion remain bit-identical."""
    _, upstream_tracker_module, upstream_types = _load_upstream_modules()
    upstream = upstream_tracker_module.LampTracker(num_cameras=1)
    owned = LampTracker(num_cameras=1)
    identity = np.eye(4, dtype=np.float32)
    timestamps = [1_000_000_000, 1_100_000_000]
    for timestamp, offset in zip(timestamps, [0.0, 2.0], strict=True):
        upstream_detection = _detection(upstream_types.Detection2D, timestamp, offset)
        owned_detection = _detection(Detection2D, timestamp, offset)
        upstream.track_frameset({0: [upstream_detection]}, {0: identity}, {0: None}, timestamp)
        owned.track_frameset({0: [owned_detection]}, {0: identity}, {0: None}, timestamp)
        assert upstream_detection.track_id == owned_detection.track_id == 1

    for shift in (0.0, 0.2):
        upstream.attach_skeletons(
            1,
            [(timestamp, _skeleton(upstream_types.Skeleton, shift)) for timestamp in timestamps],
            {},
            identity,
            min_pose_depth=0.0,
            max_pose_depth=float("inf"),
        )
        owned.attach_skeletons(
            1,
            [(timestamp, _skeleton(Skeleton, shift)) for timestamp in timestamps],
            {},
            identity,
            min_pose_depth=0.0,
            max_pose_depth=float("inf"),
        )

    upstream_person = upstream.people[1]
    owned_person = owned.people[1]
    assert np.array_equal(upstream_person.shape_estimate, owned_person.shape_estimate)
    for timestamp in timestamps:
        upstream_skeleton = upstream_person.ts_to_states[timestamp].skeleton
        owned_skeleton = owned_person.ts_to_states[timestamp].skeleton
        assert upstream_skeleton is not None and owned_skeleton is not None
        assert np.array_equal(upstream_skeleton.kp_world, owned_skeleton.kp_world)
        assert np.array_equal(upstream_skeleton.T_world_pelvis, owned_skeleton.T_world_pelvis)
        assert np.array_equal(upstream_skeleton.shape, owned_skeleton.shape)
        assert np.array_equal(upstream_skeleton.joints_rot_mat, owned_skeleton.joints_rot_mat)
