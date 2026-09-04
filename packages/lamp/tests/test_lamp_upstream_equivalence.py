"""Keep the vendored LAMP lifter numerically equivalent to pristine upstream."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor, nn

from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings, SnippetData

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "lamp"
FixtureMap: TypeAlias = dict[str, ModuleType]


def _load_module(name: str, filename: str) -> ModuleType:
    """Load one pristine source file under its original package name."""
    path: Path = REFERENCE_DIR / filename
    spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream fixture {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_lifter() -> ModuleType:
    """Load the pristine lifter and its direct dependencies as ``lamp.*``."""
    for package_name in ("lamp", "lamp.core", "lamp.models"):
        package: ModuleType = ModuleType(package_name)
        package.__path__ = [str(REFERENCE_DIR)]
        sys.modules[package_name] = package
    _load_module("lamp.core.se3", "upstream_core_se3.py")
    _load_module("lamp.core.types", "upstream_core_types.py")
    return _load_module("lamp.models.lifter", "upstream_models_lifter.py")


class _DeterministicLampNet(nn.Module):
    """Small model double at the external learned-model boundary."""

    def forward(
        self,
        keypoints: list[Tensor],
        camera_params: list[Tensor],
        camera_poses: list[Tensor],
        ground_planes: Tensor,
    ) -> dict[str, Tensor]:
        """Return deterministic SMPL-shaped tensors from the snippet inputs."""
        del camera_params, camera_poses, ground_planes
        batch: int = int(keypoints[0].shape[0])
        steps: int = int(keypoints[0].shape[1])
        signal: Tensor = torch.stack([view[..., :2].mean(dim=(2, 3)) for view in keypoints], dim=0).mean(dim=0)
        translation: Tensor = torch.stack((signal, signal * 0.5, signal * 0.25), dim=-1)
        rotations: Tensor = torch.eye(3, dtype=torch.float32).expand(batch, steps, 24, 3, 3).clone()
        joints: Tensor = translation[:, :, None, :].expand(batch, steps, 24, 3).clone()
        return {
            "skel_w": joints,
            "transl": translation,
            "global_orient_rotmat": rotations[:, :, :1],
            "body_pose_rotmat": rotations[:, :, 1:],
            "betas": signal.mean(dim=1, keepdim=True).expand(batch, 10).clone(),
        }


def _snippet(snippet_type: type, *, seed: int = 42, steps: int = 5) -> object:
    """Build one seeded four-view pinhole snippet through either implementation."""
    generator: np.random.Generator = np.random.default_rng(seed)
    keypoints: list[ndarray] = []
    camera_params: list[ndarray] = []
    camera_poses: list[ndarray] = []
    for view_idx in range(4):
        keypoints_view: ndarray = generator.uniform(0.0, 512.0, size=(steps, 17, 3)).astype(np.float32)
        keypoints_view[..., 2] = 1.0
        keypoints.append(keypoints_view)
        params: ndarray = np.tile(np.array([500.0, 500.0, 256.0, 256.0], dtype=np.float32), (steps, 1))
        camera_params.append(params)
        poses: ndarray = np.tile(np.eye(4, dtype=np.float32), (steps, 1, 1))
        poses[:, 0, 3] = float(view_idx) * 0.1
        camera_poses.append(poses)
    return snippet_type(
        person_id=7,
        snippet_timestamps_ns=[1_000_000_000 + i * 100_000_000 for i in range(steps)],
        view_cam_indices=[0, 1, 2, 3],
        kp2ds_per_view=keypoints,
        Ts_gw_cam_per_view=camera_poses,
        cam_params_per_view=camera_params,
        T_gravityWorld_world=np.eye(4, dtype=np.float32),
    )


def test_seeded_lifter_forward_matches_pristine_upstream() -> None:
    """The vendored host-side lifter produces bit-identical window states."""
    upstream: ModuleType = _load_upstream_lifter()
    upstream_lifter: object = upstream.Lifter(_DeterministicLampNet(), torch.device("cpu"), upstream.LifterSettings(snippet_length=5))
    owned_lifter: Lifter = Lifter(_DeterministicLampNet(), torch.device("cpu"), LifterSettings(snippet_length=5))

    upstream_steps: list[tuple[int, object]] = upstream_lifter.lift_all_steps(_snippet(upstream.SnippetData))
    owned_steps: list[tuple[int, object]] = owned_lifter.lift_all_steps(_snippet(SnippetData))

    assert [timestamp for timestamp, _ in upstream_steps] == [timestamp for timestamp, _ in owned_steps]
    for (_, upstream_skeleton), (_, owned_skeleton) in zip(upstream_steps, owned_steps, strict=True):
        assert np.array_equal(upstream_skeleton.kp_world, owned_skeleton.kp_world)
        assert np.array_equal(upstream_skeleton.T_world_pelvis, owned_skeleton.T_world_pelvis)
        assert np.array_equal(upstream_skeleton.shape, owned_skeleton.shape)
        assert np.array_equal(upstream_skeleton.joints_rot_mat, owned_skeleton.joints_rot_mat)
