"""The upstream init wrapper: cameras go out as PINHOLE world->camera, the arrays come back as an InitCloud the trainer reads."""

from types import ModuleType

import numpy as np
import pytest
import torch

TRAINER_KEYS: frozenset[str] = frozenset({"positions", "velocities", "colors", "times", "frame_start", "frame_end", "keyframe_step"})
"""Keys ``build_canonical_from_npz`` requires (w1-facts A1 (b))."""


def _upstream_arrays() -> dict[str, np.ndarray]:
    """Upstream's keyframes NPZ layout for source frames 100..106, keyframe step 3, sub-keyframe 101 appended at the end."""
    rng = np.random.default_rng(0)
    source_frames = np.concatenate([np.full(4, 100), np.full(3, 103), np.full(2, 106), np.full(2, 101)]).astype(np.int32).reshape(-1, 1)
    n = source_frames.shape[0]
    return {
        "positions": rng.normal(size=(n, 3)).astype(np.float32),
        "colors": rng.random((n, 3), dtype=np.float32),
        "velocities": rng.normal(scale=0.01, size=(n, 3)).astype(np.float32),
        "accelerations": np.zeros((n, 3), np.float32),
        "has_acceleration": np.ones(n, bool),
        "track_flags": np.zeros(n, np.uint8),
        "times": ((source_frames - 100) / 6.0).astype(np.float32),
        "durations": np.full((n, 1), 0.25, np.float32),
        "has_velocity": np.ones(n, bool),
        "is_static": np.zeros(n, bool),
        "dynamic_protect": np.zeros(n, bool),
        "dynamic_allowed": np.ones(n, bool),
        "stage_keep": np.ones(n, bool),
        "frame_start": np.array(0, np.int32),
        "frame_end": np.array(6, np.int32),
        "n_keyframes": np.array(3, np.int32),
        "keyframe_step": np.array(3, np.int32),
        "source_frame_start": np.array(100, np.int32),
        "source_frame_end": np.array(106, np.int32),
        "source_frames": source_frames,
        "keyframe_frames": np.array([100, 103, 106], np.int32),
        "fps": np.array(30.0, np.float32),
        "velocity_units": np.array("world_per_video_frame"),
        "time_units": np.array("normalized_0_1_over_source_frame_range"),
        "local_stride": np.full(n, 3, np.int32),
    }


def test_upstream_arrays_round_trip_to_trigs_mapping() -> None:
    from open4dgs.roma_init import METHOD, init_cloud_from_upstream

    arrays = _upstream_arrays()
    cloud = init_cloud_from_upstream(arrays, dataset="d", segment_id="s")
    assert cloud.method == METHOD and cloud.source_frame_start == 100
    assert cloud.frames.tolist() == [0, 0, 0, 0, 1, 1, 3, 3, 3, 6, 6]
    order = np.argsort(arrays["source_frames"].reshape(-1), kind="stable")
    mapping = cloud.to_trigs_mapping()
    assert mapping.keys() >= TRAINER_KEYS
    np.testing.assert_array_equal(mapping["positions"], arrays["positions"][order])
    np.testing.assert_array_equal(mapping["velocities"], arrays["velocities"][order])
    np.testing.assert_array_equal(mapping["colors"], arrays["colors"][order])
    # Units: velocities stay world units per VIDEO frame, times stay normalised over the logical window.
    assert str(mapping["velocity_units"]) == "world_per_video_frame"
    np.testing.assert_allclose(mapping["times"].reshape(-1), cloud.frames / 6.0, rtol=0, atol=1e-6)
    assert (int(mapping["frame_start"]), int(mapping["frame_end"]), int(mapping["keyframe_step"]), float(mapping["fps"])) == (0, 6, 3, 30.0)
    assert mapping["durations"].shape == (11, 1) and mapping["has_velocity"].dtype == np.bool_
    assert "source_frame_start" not in mapping


def test_upstream_arrays_reject_keyframe_interval_velocities() -> None:
    from open4dgs.roma_init import init_cloud_from_upstream

    arrays = _upstream_arrays() | {"velocity_units": np.array("world_per_keyframe_interval")}
    with pytest.raises(ValueError, match="velocity_units"):
        init_cloud_from_upstream(arrays, dataset="d", segment_id="s")


def _clip():
    from open4dgs.catalog_clip import MultiViewClip

    angle = 0.3
    rotation = torch.tensor([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]], dtype=torch.float32)
    world_T_cam = torch.eye(4).repeat(2, 1, 1)
    world_T_cam[1, :3, :3] = rotation
    world_T_cam[1, :3, 3] = torch.tensor([1.0, -0.5, 2.0])
    K = torch.tensor([[100.0, 0.0, 31.5], [0.0, 110.0, 23.5], [0.0, 0.0, 1.0]]).repeat(2, 1, 1)
    frames = (torch.zeros(3, 48, 64, 3, dtype=torch.uint8), torch.full((3, 48, 64, 3), 200, dtype=torch.uint8))
    return MultiViewClip(
        camera_names=("rig_01/cam_00", "rig_02/cam_00"),
        frames=frames,
        valid=tuple(torch.ones(48, 64, dtype=torch.bool) for _ in frames),
        K=K,
        world_T_cam=world_T_cam,
        fps=30.0,
        frame_indices=torch.arange(10, 13),
        people=None,
    )


def test_reference_cameras_are_pinhole_world_to_camera() -> None:
    from open4dgs.roma_init import reference_cameras

    clip = _clip()
    cameras = reference_cameras(clip)
    assert [camera["folder"] for camera in cameras] == ["rig_01__cam_00", "rig_02__cam_00"]
    camera = cameras[1]
    assert camera["camera_model_name"] == "PINHOLE" and (camera["width"], camera["height"]) == (64, 48)
    assert camera["params"] == pytest.approx([100.0, 110.0, 31.5, 23.5])
    point_cam = np.array([0.2, -0.1, 3.0])
    world_T_cam = clip.world_T_cam[1].double().numpy()
    point_world = world_T_cam[:3, :3] @ point_cam + world_T_cam[:3, 3]
    np.testing.assert_allclose(np.asarray(camera["R"]) @ point_world + np.asarray(camera["t"]), point_cam, atol=1e-6)


def test_materialise_writes_frames_and_binary_masks(tmp_path) -> None:
    import cv2

    from open4dgs.roma_init import materialise

    clip = _clip()
    masks = (torch.zeros(3, 48, 64, dtype=torch.bool), torch.ones(3, 48, 64, dtype=torch.bool))
    materialise(clip, masks, tmp_path)
    assert sorted(path.name for path in (tmp_path / "images" / "rig_02__cam_00").iterdir()) == ["000010.jpg", "000011.jpg", "000012.jpg"]
    mask = cv2.imread(str(tmp_path / "masks" / "rig_02__cam_00" / "000011.png"), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (48, 64) and int(mask.min()) == 255
    assert int(cv2.imread(str(tmp_path / "masks" / "rig_01__cam_00" / "000011.png"), cv2.IMREAD_GRAYSCALE).max()) == 0


def test_run_upstream_captures_the_final_arrays_only(tmp_path) -> None:
    from open4dgs.roma_init import run_upstream

    output_path = tmp_path / "keyframes.npz"
    written: list[str] = []
    module = ModuleType("fake_upstream")

    def original(path, *, compressed: bool = False, **arrays) -> None:
        written.append(str(path))

    def main() -> None:
        import sys

        assert sys.argv[-1] == "--flag"
        module._atomic_savez(tmp_path / "cache" / "kf_000100.npz", positions=np.zeros((1, 3)))
        module._atomic_savez(output_path, compressed=True, positions=np.ones((2, 3)))

    module._atomic_savez = original
    module.main = main
    arrays = run_upstream(module, ["prog", "--flag"], output_path)
    assert arrays["positions"].shape == (2, 3)
    assert written == [str(tmp_path / "cache" / "kf_000100.npz")]
    assert module._atomic_savez is original and not output_path.exists()
