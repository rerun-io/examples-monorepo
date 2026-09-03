"""CPU contracts for the direct gsplat trainer."""

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from gauss_surf.apis.train_gsplat import Config
from gauss_surf.train_gsplat.cache import load_training_cameras, scene_scale_from_camera_poses
from gauss_surf.train_gsplat.core import (
    decode_normal_uint8,
    holdout_hash,
    median_scale_regularization,
    opacity_export_mask,
    per_frame_average_psnr,
    positive_neighbor_distances,
    save_checkpoint,
)
from gauss_surf.train_gsplat.publish import metric_splats_from_live, quantize_depth_m_to_mm, quantize_normal_to_uint8
from gauss_surf.train_gsplat.renderer import fill_depth_holes, render_splats, render_splats_two_call
from gauss_surf.train_gsplat.trainer import camera_sample_indices, metric_training_constants, read_metric_seed, select_training_renderer


def _require_local_artifact(path: Path) -> Path:
    """Skip golden-data checks when the untracked local run is unavailable."""
    if not path.is_file():
        pytest.skip(f"local Gauss Surf artifact is unavailable: {path}")
    return path


def test_training_renderer_defaults_to_accepted_two_call_path() -> None:
    """Fused training needs an explicit quality-round opt-in."""
    config = Config(video_id="segment")

    assert config.dataset_name == "arkitscenes-v2"
    assert config.bundle_dir is None
    assert config.timestamp == "run"
    assert (config.splat_output_dir, config.depth_output_dir, config.triage_output_dir) == (
        Path("data/splat"),
        Path("data/splat_depth"),
        Path("data/splat_triage"),
    )
    assert config.fused_training is False
    assert select_training_renderer(fused_training=config.fused_training) is render_splats_two_call
    assert select_training_renderer(fused_training=True) is render_splats


@given(scene_scale=st.sampled_from((0.25, 0.5, 1.0, 2.0, 4.0, 8.0)))
def test_metric_training_constants_preserve_weighted_depth_scale(scene_scale: float) -> None:
    """Metric depth scales by S, so dividing its weight by S preserves loss."""
    constants = metric_training_constants(scene_scale)
    normalized_depth_error: torch.Tensor = torch.tensor(0.375, dtype=torch.float64)
    metric_depth_error: torch.Tensor = normalized_depth_error * scene_scale

    assert constants.means_lr == 1.6e-4 * scene_scale
    assert constants.means_lr_final == 1.6e-6 * scene_scale
    assert constants.depth_regularization_weight == 3.2 / scene_scale
    assert constants.flat_regularization_weight == 1.0 / scene_scale
    assert torch.equal(
        metric_depth_error * constants.depth_regularization_weight,
        normalized_depth_error * 3.2,
    )


def test_training_cameras_keep_metric_centers_and_reproduce_applied_scale(tmp_path: Path) -> None:
    """Metric cameras stay unchanged while scale follows centered pose axes."""
    centers_n3: np.ndarray = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 4.0, 0.0)), dtype=np.float32)
    frames: list[dict[str, object]] = []
    for index, center_3 in enumerate(centers_n3):
        world_from_camera_44: np.ndarray = np.eye(4, dtype=np.float32)
        world_from_camera_44[:3, 3] = center_3
        frames.append(
            {
                "file_path": f"images/wide_{index:06d}.png",
                "camera": "wide",
                "holdout": index == 2,
                "timestamp_ns": index,
                "fl_x": 100.0,
                "fl_y": 100.0,
                "cx": 10.0,
                "cy": 10.0,
                "w": 20,
                "h": 20,
                "transform_matrix": world_from_camera_44.tolist(),
            }
        )
    (tmp_path / "transforms.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")

    cameras, scene_scale = load_training_cameras(tmp_path)
    recovered_centers_n3: torch.Tensor = torch.stack(
        tuple(torch.linalg.inv(camera.viewmat_44)[:3, 3] for camera in cameras)
    )

    torch.testing.assert_close(recovered_centers_n3, torch.from_numpy(centers_n3))
    np.testing.assert_allclose(scene_scale, np.float32(8.0 / 3.0), rtol=0.0, atol=2.0 * np.finfo(np.float32).eps)


def test_part10_scene_scale_is_inverse_of_saved_applied_scale() -> None:
    """The local formula reproduces the historical float32 parser artifact."""
    package_root: Path = Path(__file__).parents[1]
    bundle_dir: Path = package_root / "data/training_bundle_part10/47115416"
    transform_path: Path = _require_local_artifact(
        package_root / "data/splat_runs/47115416-gaussurf/gaussurf-arkit/part10/dataparser_transforms.json"
    )
    _require_local_artifact(bundle_dir / "transforms.json")
    artifact: dict[str, object] = json.loads(transform_path.read_text(encoding="utf-8"))

    _cameras, scene_scale = load_training_cameras(bundle_dir)
    expected_scene_scale: float = 1.0 / float(artifact["scale"])

    assert np.float32(scene_scale) == np.float32(expected_scene_scale)


def test_scene_scale_is_translation_invariant() -> None:
    """A common world translation cannot change the automatic scene scale."""
    poses_n44: torch.Tensor = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
    poses_n44[:, :3, 3] = torch.tensor(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 4.0, 0.0)))
    translated_poses_n44: torch.Tensor = poses_n44.clone()
    translated_poses_n44[:, :3, 3] += torch.tensor((17.0, -11.0, 5.0))

    np.testing.assert_allclose(
        scene_scale_from_camera_poses(poses_n44),
        scene_scale_from_camera_poses(translated_poses_n44),
        rtol=0.0,
        atol=4.0 * np.finfo(np.float32).eps,
    )


def test_metric_seed_loader_preserves_world_coordinates(tmp_path: Path) -> None:
    """Seed vertices are already metric and need no dataparser transform."""
    seed_path: Path = tmp_path / "seed.ply"
    seed_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "1.0 2.0 3.0 255 0 128\n"
        "-4.0 5.0 6.0 0 255 64\n",
        encoding="ascii",
    )

    points_n3, colors_n3 = read_metric_seed(seed_path)

    assert np.array_equal(points_n3, np.asarray(((1.0, 2.0, 3.0), (-4.0, 5.0, 6.0)), dtype=np.float32))
    assert np.array_equal(colors_n3, np.asarray(((1.0, 0.0, 128.0 / 255.0), (0.0, 1.0, 64.0 / 255.0)), dtype=np.float32))


def test_camera_sample_indices_match_accepted_seeded_shuffled_epochs() -> None:
    """Every camera appears once per epoch in the accepted seed-42 order."""
    sampled_indices: tuple[int, ...] = camera_sample_indices(num_train_data=5, iteration_count=12, seed=42)

    assert sampled_indices == (3, 1, 2, 4, 0, 3, 2, 0, 4, 1, 3, 1)
    assert sorted(sampled_indices[:5]) == list(range(5))
    assert sorted(sampled_indices[5:10]) == list(range(5))


def test_normal_dequantization_restores_central_code_to_exact_zero() -> None:
    """Code 128 denotes signed zero, not the affine inverse 1/255."""
    encoded: torch.Tensor = torch.tensor([0, 127, 128, 129, 255], dtype=torch.uint8)

    decoded: torch.Tensor = decode_normal_uint8(encoded)

    assert torch.equal(decoded[2], torch.tensor(0.0))
    assert torch.allclose(decoded, torch.tensor([-1.0, -1.0 / 255.0, 0.0, 3.0 / 255.0, 1.0]))


def test_psnr_averages_frame_scores_instead_of_pooling_mse() -> None:
    """Match the accepted evaluation on unequal per-frame errors."""
    prediction: torch.Tensor = torch.zeros((2, 2, 2, 1), dtype=torch.float64)
    target: torch.Tensor = torch.stack(
        (
            torch.full((2, 2, 1), 0.1, dtype=torch.float64),
            torch.full((2, 2, 1), 0.5, dtype=torch.float64),
        )
    )

    psnr: torch.Tensor = per_frame_average_psnr(prediction, target)

    assert torch.isclose(psnr, torch.tensor(13.010299956639813, dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_holdout_hash_is_order_sensitive_and_stable() -> None:
    """The split fingerprint changes if frame identity or order changes."""
    assert holdout_hash(("wide_000007", "wide_000015")) == "10447155eb2b0d40b728e256bcabfb9c918c7fea9741587e91248c30de8cdcde"
    assert holdout_hash(("wide_000015", "wide_000007")) != holdout_hash(("wide_000007", "wide_000015"))


def test_export_filter_uses_the_required_one_over_255_threshold() -> None:
    """The exporter keeps exact-threshold opacity and drops only lower values."""
    threshold: float = torch.logit(torch.tensor(1.0 / 255.0)).item()
    logits: torch.Tensor = torch.tensor([threshold - 1e-4, threshold, threshold + 1e-4])

    assert torch.equal(opacity_export_mask(logits), torch.tensor([False, True, True]))


def test_checkpoint_loads_with_weights_only(tmp_path: Path) -> None:
    """Direct checkpoints contain tensors and primitive metadata only."""
    checkpoint_path: Path = tmp_path / "step-000000001.pt"
    splats = torch.nn.ParameterDict({"means": torch.nn.Parameter(torch.ones((2, 3)))})

    save_checkpoint(checkpoint_path, step=1, splats=splats, metadata={"seed": 42})
    payload: dict[str, object] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert payload["step"] == 1
    assert payload["metadata"] == {"seed": 42}
    assert torch.equal(payload["splats"]["means"], torch.ones((2, 3)))  # type: ignore[index]


def test_scale_regularization_ignores_only_all_zero_duplicate_seeds() -> None:
    """Four duplicate mesh seeds must not turn the first geometry step into NaN."""
    log_scales: torch.Tensor = torch.tensor(
        [[float("-inf"), float("-inf"), float("-inf")], [0.0, 0.0, torch.log(torch.tensor(3.0)).item()]]
    )

    loss: torch.Tensor = median_scale_regularization(log_scales, maximum_ratio=2.0)

    assert torch.isfinite(loss)
    assert torch.equal(loss, torch.tensor(0.5))


def test_duplicate_seed_neighbor_distance_uses_smallest_observed_positive_value() -> None:
    """Duplicate mesh vertices keep finite log-scales without changing seed count."""
    distances: np.ndarray = np.asarray([[0.0], [2e-7], [5e-7]], dtype=np.float32)

    repaired: np.ndarray = positive_neighbor_distances(distances)

    assert np.array_equal(repaired, np.asarray([[2e-7], [2e-7], [5e-7]], dtype=np.float32))


def test_auxiliary_center_depth_fills_zero_alpha_holes_with_detached_maximum() -> None:
    """Keep Splatfacto's ED postprocessing without using ED as surface depth."""
    depth: torch.Tensor = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)
    alpha: torch.Tensor = torch.tensor([[[1.0], [0.0]], [[0.5], [0.0]]])

    filled: torch.Tensor = fill_depth_holes(depth, alpha)
    filled.sum().backward()

    assert torch.equal(filled, torch.tensor([[[1.0], [4.0]], [[3.0], [4.0]]]))
    assert torch.equal(depth.grad, torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]]))


def test_publish_quantization_matches_depth_and_normal_storage_contracts() -> None:
    """GPU-side product quantization preserves sentinels, rounding, and saturation."""
    depth_m_hw: torch.Tensor = torch.tensor(
        [[float("nan"), -1.0, 0.0, 1.2344, 1.2346, 70.0]], dtype=torch.float32
    )
    normals_hw3: torch.Tensor = torch.tensor(
        [[[-1.0, -1.0 / 255.0, 0.0], [3.0 / 255.0, 1.0, 0.0]]], dtype=torch.float32
    )

    depth_mm_hw: torch.Tensor = quantize_depth_m_to_mm(depth_m_hw)
    normal_rgb_hw3: torch.Tensor = quantize_normal_to_uint8(normals_hw3)

    assert depth_mm_hw.dtype == torch.uint16
    assert torch.equal(depth_mm_hw, torch.tensor([[0, 0, 0, 1234, 1235, 65535]], dtype=torch.uint16))
    assert normal_rgb_hw3.dtype == torch.uint8
    assert torch.equal(normal_rgb_hw3, torch.tensor([[[0, 127, 128], [129, 255, 128]]], dtype=torch.uint8))


def test_live_metric_splats_need_no_parent_coordinate_transform() -> None:
    """In-process splat publication preserves metric centers and natural values."""
    splats: dict[str, torch.Tensor] = {
        "means": torch.tensor([[1.0, 2.0, 3.0]]),
        "scales": torch.log(torch.tensor([[1.0, 2.0, 3.0]])),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "opacities": torch.tensor([[0.0]]),
        "sh0": torch.zeros((1, 1, 3)),
        "shN": torch.zeros((1, 15, 3)),
    }

    published = metric_splats_from_live(splats)

    assert np.array_equal(published.centers_n3, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert np.array_equal(published.scales_n3, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert np.array_equal(published.quaternions_xyzw_n4, np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(published.colors_rgba_n4, np.asarray([[128, 128, 128, 128]], dtype=np.uint8))


@pytest.mark.skipif(
    os.environ.get("GAUSS_SURF_RUN_FUSED_PARITY") != "1" or not torch.cuda.is_available(),
    reason="requires CUDA and the accepted Stage-3b checkpoint",
)
def test_fused_raster_matches_two_call_reference_on_trained_scene() -> None:
    """One 8-channel pass preserves RGB and raw plane channels within atomics noise."""
    package_root: Path = Path(__file__).parents[1]
    bundle_dir: Path = package_root / "data/training_bundle_part10/47115416"
    checkpoint_path: Path = (
        package_root
        / "data/splat_runs/47115416-gaussurf/gsplat-direct/part8-stage3b/checkpoints/step-000006999.pt"
    )
    checkpoint: dict[str, object] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_splats: dict[str, torch.Tensor] = cast(dict[str, torch.Tensor], checkpoint["splats"])
    device: torch.device = torch.device("cuda")
    splats: dict[str, torch.Tensor] = {name: value.to(device) for name, value in saved_splats.items()}
    cameras, _scene_scale = load_training_cameras(bundle_dir)
    camera = cameras[len(cameras) // 2]
    background_3: torch.Tensor = torch.tensor((0.1490, 0.1647, 0.2157), dtype=torch.float32, device=device)

    with torch.inference_mode():
        reference = render_splats_two_call(
            splats,
            camera,
            downscale=4,
            sh_degree=2,
            background_3=background_3,
            absgrad=False,
        )
        fused = render_splats(
            splats,
            camera,
            downscale=4,
            sh_degree=2,
            background_3=background_3,
            absgrad=False,
        )

    rgb_max_abs: float = float((fused.rgb_hw3 - reference.rgb_hw3).abs().max().item())
    plane_max_abs: float = float((fused.plane_features_hw4 - reference.plane_features_hw4).abs().max().item())
    print({"rgb_max_abs": rgb_max_abs, "plane_max_abs": plane_max_abs})
    torch.testing.assert_close(fused.rgb_hw3, reference.rgb_hw3, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(fused.plane_features_hw4, reference.plane_features_hw4, rtol=2e-5, atol=2e-5)
