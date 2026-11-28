"""The pycolmap <-> array-contract bridge on tiny synthetic models: no GPU, no solver."""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64
from numpy import ndarray


def _tiny(model: str = "PINHOLE", num_cameras: int = 1) -> pycolmap.Reconstruction:
    from colsfm.apis.ba_synth import synthetic_reconstruction

    return synthetic_reconstruction(6, 400, model, num_cameras)  # type: ignore[arg-type]


def _matrix(pose: pycolmap.Rigid3d | None) -> Float64[ndarray, "3 4"]:
    assert pose is not None
    return np.asarray(pose.matrix())


def _config(reconstruction: pycolmap.Reconstruction) -> pycolmap.BundleAdjustmentConfig:
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.reg_image_ids()):
        config.add_image(image_id)
    config.set_constant_rig_from_world_pose(min(reconstruction.reg_frame_ids()))
    return config


def _options(refine_intrinsics: bool = False, refine_rig: bool = False) -> pycolmap.BundleAdjustmentOptions:
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = refine_intrinsics
    options.refine_extra_params = refine_intrinsics
    options.refine_principal_point = False
    options.refine_sensor_from_rig = refine_rig
    return options


def _rotation(quat: Float64[ndarray, "n 4"]) -> Float64[ndarray, "n 3 3"]:
    x, y, z, w = (quat / np.linalg.norm(quat, axis=1, keepdims=True)).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        1,
    )


def _pinhole_residuals(reconstruction: pycolmap.Reconstruction) -> tuple[Float64[ndarray, "n 2"], object]:
    from colsfm.ba_mojo import problem_from_reconstruction

    bridged = problem_from_reconstruction(reconstruction, _config(reconstruction), _options())
    p = bridged.problem
    frame, camera, point = p.observation_frame, p.observation_camera, p.observation_point
    in_rig = np.einsum("nij,nj->ni", _rotation(p.rig_from_world[frame, :4]), p.points[point]) + p.rig_from_world[frame, 4:]
    in_cam = np.einsum("nij,nj->ni", _rotation(p.sensor_from_rig[camera, :4]), in_rig) + p.sensor_from_rig[camera, 4:]
    fx, fy, cx, cy = p.intrinsics[camera, :4].T
    projected = np.stack([fx * in_cam[:, 0] / in_cam[:, 2] + cx, fy * in_cam[:, 1] / in_cam[:, 2] + cy], -1)
    return projected - p.pixels, bridged


@pytest.mark.parametrize("num_cameras", [1, 2])
def test_pinhole_residuals_match_colmap_point_errors(num_cameras: int) -> None:
    reconstruction = _tiny(num_cameras=num_cameras)
    residuals, bridged = _pinhole_residuals(reconstruction)
    reconstruction.update_point_3d_errors()
    norms = np.linalg.norm(residuals, axis=1)
    point = bridged.problem.observation_point  # type: ignore[attr-defined]
    mean_per_point = np.bincount(point, norms) / np.bincount(point)
    colmap_errors = np.array([reconstruction.point3D(int(i)).error for i in bridged.ids.point3D_ids])  # type: ignore[attr-defined]
    np.testing.assert_allclose(mean_per_point, colmap_errors, rtol=1e-9)
    # COLMAP's mean reprojection error is the mean of the per-point means.
    assert np.isclose(mean_per_point.mean(), reconstruction.compute_mean_reprojection_error(), rtol=1e-9)


def test_problem_follows_caspar_rules() -> None:
    from caspar_mojo.problem import CameraModel

    from colsfm.ba_mojo import problem_from_reconstruction

    reconstruction = _tiny("SIMPLE_RADIAL", num_cameras=2)
    bridged = problem_from_reconstruction(reconstruction, _config(reconstruction), _options(refine_intrinsics=True, refine_rig=True))
    p, ids = bridged.problem, bridged.ids
    assert ids.frame_ids.tolist() == sorted(reconstruction.reg_frame_ids())
    assert p.pose_free.tolist() == [False] + [True] * 5
    assert p.sensor_free.tolist() == [False, True]
    np.testing.assert_array_equal(p.sensor_from_rig[0], [0, 0, 0, 1, 0, 0, 0])
    assert p.intrinsics_free.all() and (p.camera_model == CameraModel.SIMPLE_RADIAL).all()
    assert p.point_free.all() and len(p.pixels) == reconstruction.compute_num_observations()
    # Camera-first, then image, then point2D index.
    camera_ids = ids.camera_ids[p.observation_camera]
    keys = np.stack([camera_ids, ids.observation_image_ids, ids.observation_point2D_idxs], 1)
    assert (np.diff(keys[:, 0]) >= 0).all()
    assert np.array_equal(np.lexsort(keys.T[::-1]), np.arange(len(keys)))
    image = reconstruction.image(int(ids.observation_image_ids[7]))
    np.testing.assert_array_equal(p.pixels[7], image.points2D[int(ids.observation_point2D_idxs[7])].xy)


def test_point_outside_config_is_constant_and_constant_observations_drop() -> None:
    from colsfm.ba_mojo import problem_from_reconstruction

    reconstruction = _tiny()
    config = _config(reconstruction)
    config.remove_image(max(reconstruction.reg_image_ids()))
    bridged = problem_from_reconstruction(reconstruction, config, _options())
    seen_outside = {element.point3D_id for element in reconstruction.image(max(reconstruction.reg_image_ids())).get_observation_points2D()}
    free = dict(zip(bridged.ids.point3D_ids.tolist(), bridged.problem.point_free.tolist(), strict=True))
    assert all(not free[i] for i in seen_outside if i in free)
    assert any(free.values())
    # An observation of a constant point in the gauge frame has nothing free: CASPAR leaves it out.
    gauge = bridged.problem.observation_frame == 0
    assert bridged.problem.point_free[bridged.problem.observation_point[gauge]].all()


def test_round_trip_leaves_the_model_unchanged() -> None:
    from caspar_mojo.problem import BaResult

    from colsfm.ba_mojo import apply_result, problem_from_reconstruction

    reconstruction = _tiny("OPENCV_FISHEYE", num_cameras=2)
    before = pycolmap.Reconstruction(reconstruction)
    bridged = problem_from_reconstruction(reconstruction, _config(reconstruction), _options(refine_intrinsics=True, refine_rig=True))
    p = bridged.problem
    result = BaResult(
        intrinsics=p.intrinsics.copy(),
        sensor_from_rig=p.sensor_from_rig.copy(),
        rig_from_world=p.rig_from_world.copy(),
        points=p.points.copy(),
        initial_cost=0.0,
        final_cost=0.0,
        iterations=0,
        pcg_iterations=0,
        exit_reason="MAX_ITERATIONS",
        solve_seconds=0.0,
        total_seconds=0.0,
    )
    apply_result(reconstruction, bridged, result)
    for point3D_id in before.point3D_ids():
        np.testing.assert_allclose(reconstruction.point3D(point3D_id).xyz, before.point3D(point3D_id).xyz, rtol=0, atol=0)
    for frame_id in before.frames:
        np.testing.assert_allclose(_matrix(reconstruction.frame(frame_id).rig_from_world), _matrix(before.frame(frame_id).rig_from_world), atol=1e-12)
    for camera_id in before.cameras:
        np.testing.assert_allclose(reconstruction.camera(camera_id).params, before.camera(camera_id).params, rtol=0, atol=0)
    sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 2)
    np.testing.assert_allclose(_matrix(reconstruction.rig(1).sensor_from_rig(sensor)), _matrix(before.rig(1).sensor_from_rig(sensor)), atol=1e-12)


def test_apply_result_writes_free_nodes_only_and_keeps_principal_point() -> None:
    from caspar_mojo.problem import BaResult

    from colsfm.ba_mojo import apply_result, problem_from_reconstruction

    reconstruction = _tiny("SIMPLE_RADIAL")
    bridged = problem_from_reconstruction(reconstruction, _config(reconstruction), _options(refine_intrinsics=True))
    p = bridged.problem
    gauge_pose = _matrix(reconstruction.frame(int(bridged.ids.frame_ids[0])).rig_from_world)
    rig_from_world = p.rig_from_world + 0.01
    result = BaResult(
        intrinsics=p.intrinsics + 1.0,
        sensor_from_rig=p.sensor_from_rig.copy(),
        rig_from_world=rig_from_world,
        points=p.points + 0.5,
        initial_cost=0.0,
        final_cost=0.0,
        iterations=1,
        pcg_iterations=1,
        exit_reason="MAX_ITERATIONS",
        solve_seconds=0.0,
        total_seconds=0.0,
    )
    apply_result(reconstruction, bridged, result)
    np.testing.assert_array_equal(_matrix(reconstruction.frame(int(bridged.ids.frame_ids[0])).rig_from_world), gauge_pose)
    written = reconstruction.frame(int(bridged.ids.frame_ids[1])).rig_from_world
    assert written is not None
    np.testing.assert_allclose(written.rotation.quat, rig_from_world[1, :4] / np.linalg.norm(rig_from_world[1, :4]))
    f, cx, cy, k = p.intrinsics[0, :4]
    np.testing.assert_allclose(reconstruction.camera(1).params, [f + 1.0, cx, cy, k + 1.0])
    np.testing.assert_allclose(reconstruction.point3D(int(bridged.ids.point3D_ids[0])).xyz, p.points[0] + 0.5)


def test_unsupported_camera_model_names_it() -> None:
    from colsfm.ba_mojo import problem_from_reconstruction

    reconstruction = _tiny()
    camera = reconstruction.camera(1)
    reconstruction.camera(1).model = pycolmap.CameraModelId.OPENCV
    reconstruction.camera(1).params = [*camera.params[:4], 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="OPENCV"):
        problem_from_reconstruction(reconstruction, _config(reconstruction), _options())


def test_refined_principal_point_is_rejected() -> None:
    from colsfm.ba_mojo import problem_from_reconstruction

    reconstruction = _tiny()
    options = _options()
    options.refine_principal_point = True
    with pytest.raises(ValueError, match="principal_point"):
        problem_from_reconstruction(reconstruction, _config(reconstruction), options)


def test_compare_solutions_of_identical_and_moved_models() -> None:
    from colsfm.ba_compare import compare_solutions
    from colsfm.ba_mojo import problem_from_reconstruction

    a = _tiny(num_cameras=2)
    residuals, _ = _pinhole_residuals(a)
    bridged = problem_from_reconstruction(a, _config(a), _options())
    same = compare_solutions(a, pycolmap.Reconstruction(a), bridged)
    assert np.isclose(same.cost_a, 0.5 * np.sum(residuals**2), rtol=1e-9) and same.cost_relative_difference == 0.0
    assert same.rotation_difference_max_deg < 1e-6 and same.center_difference_max == 0.0 and same.point_difference_max == 0.0
    assert same.registered_images_a == same.registered_images_b == 12

    b = pycolmap.Reconstruction(a)
    frame = b.frame(3)
    pose = frame.rig_from_world
    assert pose is not None
    frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.deg2rad([0.0, 1.0, 0.0])) * pose.rotation, np.asarray(pose.translation))
    moved = compare_solutions(a, b, bridged)
    assert np.isclose(moved.rotation_difference_max_deg, 1.0, atol=1e-6) and moved.rotation_difference_median_deg < 1e-6
    assert moved.cost_b > moved.cost_a and moved.reprojection_relative_difference > 0


def test_mojo_options_carry_caspars_parameters() -> None:
    from colsfm.apis.caspar_baseline import caspar_options
    from colsfm.ba_mojo import mojo_options

    options = caspar_options()
    options.caspar.reduced_pcg_eta = 0.25
    options.caspar.solver_iter_max = 17
    mojo = mojo_options(options)
    assert mojo.reduced_pcg_eta == 0.25 and mojo.solver_iter_max == 17 and mojo.reduced_camera_system


@pytest.mark.integration
def test_solve_with_mojo_matches_caspar_on_a_tiny_pinhole_model() -> None:
    solver = pytest.importorskip("caspar_mojo.solver", reason="caspar_mojo.solver is not on this branch yet")
    if not hasattr(solver, "solve"):
        pytest.skip("caspar_mojo.solver.solve is not written yet")
    from colsfm.apis.caspar_baseline import caspar_options
    from colsfm.ba_compare import compare_solutions
    from colsfm.ba_mojo import problem_from_reconstruction, solve_with_mojo

    start = _tiny()
    caspar, mojo = pycolmap.Reconstruction(start), pycolmap.Reconstruction(start)
    options = caspar_options()
    try:
        pycolmap.create_default_bundle_adjuster(options, _config(caspar), caspar).solve()
        summary = solve_with_mojo(mojo, _config(mojo), options)
    except (RuntimeError, ValueError) as error:
        pytest.skip(f"no GPU answered: {error}")
    comparison = compare_solutions(caspar, mojo, problem_from_reconstruction(start, _config(start), options))
    assert summary.num_residuals > 0
    assert abs(comparison.reprojection_relative_difference) < 0.02 and abs(comparison.cost_relative_difference) < 0.02
