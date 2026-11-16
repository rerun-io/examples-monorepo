"""Probe (d): bundle adjustment — what can be held constant, and what BA recovers."""

from __future__ import annotations

import numpy as np
import pyceres
import pycolmap
import pycolmap.pyceres as colmap_ceres
import pytest
from jaxtyping import Float
from numpy import ndarray

from colsfm.reconstruction import rig_from_world, rig_sensor_from_rig
from probes.synthetic import (
    REF_CAMERA_ID,
    SECOND_CAMERA_ID,
    SyntheticScene,
    add_ground_truth_points,
)


def _all_images_config(reconstruction: pycolmap.Reconstruction) -> pycolmap.BundleAdjustmentConfig:
    """Build a config that includes every registered image.

    Args:
        reconstruction: The model to adjust.

    Returns:
        A `BundleAdjustmentConfig` with all images added.
    """
    config: pycolmap.BundleAdjustmentConfig = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(image_id)
    return config


def _quiet_options() -> pycolmap.BundleAdjustmentOptions:
    """Bundle adjustment options with the solver summary suppressed."""
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    options.print_summary = False
    return options


def test_constant_poses_move_only_points(scene: SyntheticScene) -> None:
    """With every `rig_from_world` constant, BA moves the 3D points and nothing else.

    Poses are frozen per *frame*, not per image: `set_constant_rig_from_world_pose`
    takes a `frame_id`.  Rig extrinsics are frozen separately, per sensor, with
    `set_constant_sensor_from_rig_pose(sensor_t)`.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    add_ground_truth_points(scene)

    rng: np.random.Generator = np.random.default_rng(5)
    for point3D_id in reconstruction.point3D_ids():
        point: pycolmap.Point3D = reconstruction.point3D(point3D_id)
        point.xyz = point.xyz + rng.normal(0.0, 0.05, 3)
    # `compute_mean_reprojection_error` only averages points whose `error` field
    # has been computed; freshly added points carry error == -1 and are skipped,
    # so the mean reads 0.0 until `update_point_3d_errors` is called.
    assert reconstruction.compute_mean_reprojection_error() == pytest.approx(0.0)
    reconstruction.update_point_3d_errors()
    before_error: float = reconstruction.compute_mean_reprojection_error()
    assert before_error > 1.0

    poses_before: dict[int, Float[ndarray, "3 4"]] = {
        frame_id: rig_from_world(reconstruction, frame_id).matrix().copy() for frame_id in sorted(reconstruction.frames)
    }
    intrinsics_before: dict[int, Float[ndarray, " n_params"]] = {
        camera_id: reconstruction.camera(camera_id).params.copy() for camera_id in sorted(reconstruction.cameras)
    }

    config: pycolmap.BundleAdjustmentConfig = _all_images_config(reconstruction)
    for frame_id in sorted(reconstruction.frames):
        config.set_constant_rig_from_world_pose(frame_id)
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)
    for camera_id in (REF_CAMERA_ID, SECOND_CAMERA_ID):
        config.set_constant_sensor_from_rig_pose(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id))
    assert config.num_constant_rig_from_world_poses() == reconstruction.num_frames()
    assert config.num_constant_cam_intrinsics() == reconstruction.num_cameras()
    assert config.num_constant_sensor_from_rig_poses() == 2

    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(_quiet_options(), config, reconstruction)
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    assert summary.termination_type != pycolmap.BundleAdjustmentTerminationType.FAILURE

    reconstruction.update_point_3d_errors()
    after_error: float = reconstruction.compute_mean_reprojection_error()
    print(f"[probe] constant-pose BA: mean reproj {before_error:.4f} -> {after_error:.6f} px")
    assert after_error < 1e-3
    for frame_id, matrix in poses_before.items():
        assert np.array_equal(rig_from_world(reconstruction, frame_id).matrix(), matrix)
    for camera_id, params in intrinsics_before.items():
        assert np.array_equal(reconstruction.camera(camera_id).params, params)
    for point3D_id in reconstruction.point3D_ids():
        recovered: Float[ndarray, "3"] = reconstruction.point3D(point3D_id).xyz
        assert float(np.min(np.linalg.norm(scene.points_xyz - recovered, axis=1))) < 1e-4


def test_free_poses_recover_a_perturbed_frame(scene: SyntheticScene) -> None:
    """With poses free and points constant, BA pulls a perturbed frame back.

    The gauge is fixed by holding the points constant, which is the natural
    formulation when a known map is being localised against.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    point3D_ids: list[int] = add_ground_truth_points(scene)

    perturbed_frame_id: int = 3
    truth: Float[ndarray, "3 4"] = rig_from_world(reconstruction, perturbed_frame_id).matrix().copy()
    perturbed: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.01, -0.02, 0.015])) * rig_from_world(reconstruction, perturbed_frame_id).rotation,
        rig_from_world(reconstruction, perturbed_frame_id).translation + np.array([0.08, -0.05, 0.06]),
    )
    reconstruction.frame(perturbed_frame_id).rig_from_world = perturbed
    start_error: float = float(np.linalg.norm(perturbed.matrix() - truth))
    assert start_error > 0.05

    config: pycolmap.BundleAdjustmentConfig = _all_images_config(reconstruction)
    for point3D_id in point3D_ids:
        config.add_constant_point(point3D_id)
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)
    for camera_id in (REF_CAMERA_ID, SECOND_CAMERA_ID):
        config.set_constant_sensor_from_rig_pose(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id))

    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(_quiet_options(), config, reconstruction)
    adjuster.solve()

    final_error: float = float(np.linalg.norm(rig_from_world(reconstruction, perturbed_frame_id).matrix() - truth))
    print(f"[probe] free-pose BA: frame error {start_error:.4f} -> {final_error:.3e}")
    assert final_error < 1e-6


def test_rig_extrinsics_can_be_refined_or_frozen(scene: SyntheticScene) -> None:
    """`refine_sensor_from_rig` / `set_constant_sensor_from_rig_pose` control the rig.

    This is the pycolmap equivalent of cuSFM's `--optimize_extrinsics`.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    point3D_ids: list[int] = add_ground_truth_points(scene)
    second_sensor: pycolmap.sensor_t = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, SECOND_CAMERA_ID)
    rig: pycolmap.Rig = reconstruction.rig(1)
    truth: Float[ndarray, "3"] = rig_sensor_from_rig(rig, second_sensor).translation.copy()

    rig.set_sensor_from_rig(
        second_sensor,
        pycolmap.Rigid3d(pycolmap.Rotation3d(), truth + np.array([0.03, 0.0, 0.0])),
    )

    config: pycolmap.BundleAdjustmentConfig = _all_images_config(reconstruction)
    for point3D_id in point3D_ids:
        config.add_constant_point(point3D_id)
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)
    for frame_id in sorted(reconstruction.frames):
        config.set_constant_rig_from_world_pose(frame_id)

    options: pycolmap.BundleAdjustmentOptions = _quiet_options()
    assert options.refine_sensor_from_rig is True
    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(options, config, reconstruction)
    adjuster.solve()
    refined: Float[ndarray, "3"] = rig_sensor_from_rig(rig, second_sensor).translation
    print(f"[probe] rig extrinsic recovered to {np.linalg.norm(refined - truth):.3e} m")
    assert np.allclose(refined, truth, atol=1e-6)


def test_bundle_adjustment_options_surface() -> None:
    """Pin the BA option fields the pipeline configures, and their defaults."""
    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    assert options.backend == pycolmap.BundleAdjustmentBackend.CERES
    assert options.refine_focal_length is True
    assert options.refine_principal_point is False
    assert options.refine_extra_params is True
    assert options.refine_rig_from_world is True
    assert options.refine_sensor_from_rig is True
    assert options.refine_points3D is True
    assert options.constant_rig_from_world_rotation is False
    assert options.min_track_length == 0
    assert options.print_summary is True

    ceres: pycolmap.CeresBundleAdjustmentOptions = options.ceres
    assert ceres.loss_function_type == pycolmap.LossFunctionType.TRIVIAL
    assert ceres.loss_function_scale == pytest.approx(1.0)
    assert ceres.use_gpu is False
    assert ceres.gpu_index == "-1"
    assert ceres.auto_select_solver_type is True
    assert ceres.min_num_images_gpu_solver == 50
    assert ceres.min_num_residuals_for_cpu_multi_threading == 50_000
    assert ceres.max_num_images_direct_dense_cpu_solver == 50
    assert ceres.max_num_images_direct_sparse_cpu_solver == 1_000

    solver = ceres.solver_options
    assert solver.max_num_iterations == 100
    assert solver.num_threads == -1
    assert solver.function_tolerance == pytest.approx(0.0)
    assert solver.gradient_tolerance == pytest.approx(1e-4)
    assert solver.parameter_tolerance == pytest.approx(0.0)

    # There are exactly four robust losses; no Tukey, no GemanMcClure.
    loss_names: set[str] = {name for name in dir(pycolmap.LossFunctionType) if not name.startswith("_")} - {"name", "value"}
    assert loss_names == {"TRIVIAL", "SOFT_L1", "CAUCHY", "HUBER"}

    # Only two backends, and CASPAR is the GPU one.
    backend_names: set[str] = {name for name in dir(pycolmap.BundleAdjustmentBackend) if not name.startswith("_")} - {"name", "value"}
    assert backend_names == {"CERES", "CASPAR"}


def test_robust_loss_and_solver_settings_are_honoured(scene: SyntheticScene) -> None:
    """A Cauchy loss with a small scale keeps a gross outlier from dragging the fit."""
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    point3D_ids: list[int] = add_ground_truth_points(scene)

    # Move one point far away; without a robust loss it drags the whole solve.
    outlier_id: int = point3D_ids[0]
    reconstruction.point3D(outlier_id).xyz = reconstruction.point3D(outlier_id).xyz + np.array([3.0, 3.0, 3.0])

    options: pycolmap.BundleAdjustmentOptions = _quiet_options()
    options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    options.ceres.loss_function_scale = 1.0
    options.ceres.auto_select_solver_type = False
    options.ceres.solver_options.linear_solver_type = colmap_ceres.LinearSolverType.SPARSE_SCHUR
    options.ceres.solver_options.num_threads = 2
    options.ceres.solver_options.max_num_iterations = 40

    config: pycolmap.BundleAdjustmentConfig = _all_images_config(reconstruction)
    for frame_id in sorted(reconstruction.frames):
        config.set_constant_rig_from_world_pose(frame_id)
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)

    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(options, config, reconstruction)
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    assert summary.termination_type != pycolmap.BundleAdjustmentTerminationType.FAILURE
    assert adjuster.options.ceres.loss_function_type == pycolmap.LossFunctionType.CAUCHY
    inlier_errors: list[float] = [
        float(np.min(np.linalg.norm(scene.points_xyz - reconstruction.point3D(pid).xyz, axis=1))) for pid in point3D_ids[1:]
    ]
    print(f"[probe] Cauchy BA: worst inlier point error {max(inlier_errors):.3e} m")
    assert max(inlier_errors) < 1e-3


def test_ceres_gpu_solver_switch_exists(scene: SyntheticScene) -> None:
    """`options.ceres.use_gpu` is bindable and the solve still succeeds.

    conda-forge ships `ceres-solver` in a `gpu*` build here, so the CUDA linear
    algebra library is compiled in.  Ceres only picks a GPU solver once the
    problem passes `min_num_images_gpu_solver` (50 images), so a five-frame
    probe still runs on the CPU; the flag is verified as *accepted*, not as
    exercised.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    add_ground_truth_points(scene)
    options: pycolmap.BundleAdjustmentOptions = _quiet_options()
    options.ceres.use_gpu = True
    options.ceres.gpu_index = "0"
    assert options.ceres.use_gpu is True

    config: pycolmap.BundleAdjustmentConfig = _all_images_config(reconstruction)
    for frame_id in sorted(reconstruction.frames):
        config.set_constant_rig_from_world_pose(frame_id)
    adjuster: pycolmap.BundleAdjuster = pycolmap.create_default_bundle_adjuster(options, config, reconstruction)
    summary: pycolmap.BundleAdjustmentSummary = adjuster.solve()
    assert summary.termination_type != pycolmap.BundleAdjustmentTerminationType.FAILURE


def test_pose_prior_bundle_adjustment(scene: SyntheticScene) -> None:
    """`create_pose_prior_bundle_adjuster` adds 3-DoF position priors to BA.

    Signature::

        create_pose_prior_bundle_adjuster(options, prior_options, config,
                                          pose_priors, reconstruction) -> BundleAdjuster

    Priors are *position only* (plus an optional gravity direction); there is no
    6-DoF absolute pose prior in the BA path, and no relative-pose prior at all.
    `PosePriorBundleAdjustmentOptions.prior_position_fallback_stddev` covers
    priors with no covariance.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    point3D_ids: list[int] = add_ground_truth_points(scene)
    reference_sensor: pycolmap.sensor_t = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, REF_CAMERA_ID)

    priors: list[pycolmap.PosePrior] = []
    for frame_id in sorted(reconstruction.frames):
        frame: pycolmap.Frame = reconstruction.frame(frame_id)
        reference_image_id: int = next(data.id for data in frame.data_ids if data.sensor_id == reference_sensor)
        prior: pycolmap.PosePrior = pycolmap.PosePrior()
        prior.corr_data_id = pycolmap.data_t(reference_sensor, reference_image_id)
        prior.position = reconstruction.image(reference_image_id).projection_center()
        prior.position_covariance = np.diag([1e-4, 1e-4, 1e-4])
        prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        priors.append(prior)

    # Shift one frame away; the prior should pull it back.
    perturbed_frame_id: int = 4
    truth: Float[ndarray, "3"] = rig_from_world(reconstruction, perturbed_frame_id).translation.copy()
    reconstruction.frame(perturbed_frame_id).rig_from_world = pycolmap.Rigid3d(
        rig_from_world(reconstruction, perturbed_frame_id).rotation,
        truth + np.array([0.2, 0.0, 0.0]),
    )

    config: pycolmap.BundleAdjustmentConfig = _all_images_config(reconstruction)
    for point3D_id in point3D_ids:
        config.add_constant_point(point3D_id)
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)

    prior_options: pycolmap.PosePriorBundleAdjustmentOptions = pycolmap.PosePriorBundleAdjustmentOptions()
    assert prior_options.prior_position_fallback_stddev == pytest.approx(1.0)
    assert prior_options.ceres.prior_position_loss_function_type == pycolmap.LossFunctionType.TRIVIAL
    adjuster: pycolmap.BundleAdjuster = pycolmap.create_pose_prior_bundle_adjuster(_quiet_options(), prior_options, config, priors, reconstruction)
    adjuster.solve()

    recovered: Float[ndarray, "3"] = rig_from_world(reconstruction, perturbed_frame_id).translation
    print(f"[probe] pose-prior BA: residual {np.linalg.norm(recovered - truth):.3e} m")
    assert np.allclose(recovered, truth, atol=1e-4)


def test_pycolmap_embeds_a_cut_down_pyceres() -> None:
    """`pycolmap.pyceres` and the conda `pyceres` module are *different* bindings.

    pycolmap embeds a cut-down `pyceres` (`pycolmap.pyceres`, importable but
    deliberately excluded from `pycolmap.__all__`) that exposes only
    `SolverOptions`, `SolverSummary` and the enums.  It has no `Problem`, no
    `CostFunction` and no `Manifold`, which is what makes
    `pycolmap.cost_functions` unusable from the standalone `pyceres` — see
    `test_pose_graph.py`.

    Scalar enums do cross the boundary (pybind11 converts them through their
    integer value), so `solver_options.linear_solver_type` accepts either
    module's enum.  Only the opaque C++ object types are incompatible.
    """
    assert not hasattr(colmap_ceres, "Problem")
    assert not hasattr(colmap_ceres, "CostFunction")
    assert not hasattr(colmap_ceres, "Manifold")
    assert hasattr(pyceres, "Problem") and hasattr(pyceres, "CostFunction")
    assert "pyceres" not in pycolmap.__all__

    options: pycolmap.BundleAdjustmentOptions = pycolmap.BundleAdjustmentOptions()
    assert options.ceres.solver_options.linear_solver_type == colmap_ceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    # `options.ceres.solver_options` hands back a live reference, so in-place
    # mutation sticks.
    options.ceres.solver_options.linear_solver_type = colmap_ceres.LinearSolverType.SPARSE_SCHUR
    assert options.ceres.solver_options.linear_solver_type == colmap_ceres.LinearSolverType.SPARSE_SCHUR
    options.ceres.solver_options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    assert options.ceres.solver_options.linear_solver_type == colmap_ceres.LinearSolverType.DENSE_QR
