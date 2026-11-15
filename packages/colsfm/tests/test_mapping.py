"""What the mapper does to a scene: correspondences in, triangulated poses out.

The seams are `colsfm.mapping.load_correspondences` and
`colsfm.mapping.run_mapping`, observed through the `pycolmap.Reconstruction` they
leave behind and the `MappingResult` they return. Every input here is
**synthetic** — a rig whose geometry, points and observation noise are known — so
a recovered point count or a pose error is measured against ground truth rather
than against the code's own opinion.

This file owns the mapper's core behaviour only. The four concerns that used to
share it, and now do not, are:

* `test_mapping_ba_backend.py` — which solver a run gets, and the Ceres polish.
* `test_mapping_point_filters.py` — the two pre-solve point guards.
* `test_mapping_extrinsics.py` — `--optimize_extrinsics` on real Galileo matches.
* `test_mapping_galileo_parity.py` — the integration run against the cuSFM blob.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from conftest import pose_delta
from jaxtyping import Float64
from mapping_helpers import (
    PERTURBATION_DEG,
    PERTURBATION_M,
    SYNTHETIC_NOISE_PX,
    SyntheticRig,
    build_synthetic_rig,
    every_round,
    match_tracks_to_truth,
    quiet_options,
    random_offset,
    synthetic_matches,
    write_database,
)
from numpy import ndarray

from colsfm.config import CusfmConfig, VisionMappingConfig
from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.mapping import MappingResult, RoundStats, load_correspondences, pixel_error_schedule, run_mapping
from colsfm.reconstruction import RIG_ID, PosedModel, build_reconstruction, camera_sensor_id, gauge_rig_frame


def test_pixel_error_schedule_is_the_isaac_ramp(isaac_config: CusfmConfig) -> None:
    """The gate decays linearly across the outer loop: 25, 20, 15, 10, 5 (§5.6).

    Read off a `--v=3` re-run of the blob, not derived from the code.
    """
    assert pixel_error_schedule(isaac_config.vision_mapping) == (25.0, 20.0, 15.0, 10.0, 5.0)


def test_load_correspondences_fills_points2d_and_the_graph(
    synthetic_rig: SyntheticRig, synthetic_database: Path
) -> None:
    """Keypoints land on the images and the verified pairs become the graph.

    Nothing is re-estimated: the poses the reconstruction went in with come back
    out bit-identical.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    poses_before: dict[int, Float64[ndarray, "3 4"]] = {
        frame_id: reconstruction.frame(frame_id).rig_from_world.matrix().copy() for frame_id in sorted(reconstruction.frames)
    }

    correspondences = load_correspondences(reconstruction, synthetic_database, min_num_matches=15)

    assert correspondences.num_images == reconstruction.num_images() == 40
    assert correspondences.num_image_pairs > 100
    for keyframe in synthetic_rig.frames_meta.keyframes:
        image: pycolmap.Image = reconstruction.image(keyframe.keyframe_id)
        expected: Float64[ndarray, "n 2"] = synthetic_rig.keypoints_px[keyframe.keyframe_id]
        assert image.num_points2D() == len(expected)
        assert np.allclose(np.array([point.xy for point in image.points2D]), expected, atol=1e-4)
        assert correspondences.graph.num_correspondences_for_image(keyframe.keyframe_id) > 0
    for frame_id, matrix in poses_before.items():
        assert np.array_equal(reconstruction.frame(frame_id).rig_from_world.matrix(), matrix)


def test_load_correspondences_rejects_a_database_with_other_image_ids(
    synthetic_rig: SyntheticRig, tmp_path: Path
) -> None:
    """A database that numbers the same image differently is an error, not a silent miss.

    The correspondence graph is keyed by database image id, so a mismatch would
    triangulate one image's keypoints against another's.
    """
    database_path: Path = tmp_path / "shifted.db"
    write_database(
        database_path,
        synthetic_rig.frames_meta,
        synthetic_rig.keypoints_px,
        synthetic_matches(synthetic_rig),
        image_id_offset=1000,
    )

    reconstruction: pycolmap.Reconstruction = build_reconstruction(synthetic_rig.frames_meta).reconstruction
    with pytest.raises(ValueError, match="disagree on image ids"):
        load_correspondences(reconstruction, database_path)


def test_run_mapping_recovers_the_synthetic_points(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """With exact poses, at least 95 % of the points come back at the noise floor.

    Ground truth is the point cloud the observations were generated from, so a
    recovered point is only counted when it lands within a centimetre of one.
    """
    model: PosedModel = build_reconstruction(synthetic_rig.frames_meta)
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    result: MappingResult = run_mapping(model, synthetic_database, mapping_config, quiet_options())

    recovered, mixed_tracks, relative_errors = match_tracks_to_truth(result.reconstruction, synthetic_rig)
    fraction: float = len(recovered) / len(synthetic_rig.points_xyz)
    print(
        f"[colsfm] synthetic: {result.num_points3D} points, {fraction:.1%} of the truth recovered, "
        f"{mixed_tracks} tracks mixing two points, reprojection {result.mean_reprojection_error_px:.4f} px, "
        f"median depth error {np.median(relative_errors) * 100:.3f} % of range"
    )
    assert fraction >= 0.95
    assert mixed_tracks == 0, "a track must not merge observations of two different points"
    # A 0.3 px noise on a 0.5 m stereo baseline puts a 25 m point a few
    # decimetres out along the ray, so the position check is relative to range.
    assert float(np.median(relative_errors)) < 0.01
    assert result.mean_reprojection_error_px <= 1.5 * SYNTHETIC_NOISE_PX
    assert result.num_registered_images == 40
    assert result.num_images_with_observations == 40
    schedule: tuple[float, ...] = pixel_error_schedule(mapping_config)
    assert [round_stats.max_pixel_error for round_stats in result.rounds] == list(schedule[: len(result.rounds)])
    assert all(round_stats.ba_termination != "FAILURE" for round_stats in result.rounds)


def test_the_outer_loop_can_run_to_the_end_or_exit_early(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """On data where nothing moves, the observation-change test ends the loop at once.

    `observation_change = (merged + completed + filtered) / observations` (§5.4).
    On exact synthetic geometry no track merges, completes or gets filtered, so
    the statistic is 0 and the loop stops after one round. cuSFM's own log says
    `use num_ba_iterations to stop BA`, i.e. its two stopping rules are
    exclusive, and a `max_observation_change` of 0.0 is how the configuration asks
    for every round regardless.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    early: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta), synthetic_database, mapping_config, quiet_options()
    )
    assert len(early.rounds) == 1
    assert early.rounds[0].observation_change < mapping_config.max_observation_change
    # A Ceres round reports its own progress, so none of the three is None here.
    # The CASPAR backend writes no Ceres line at all; those rounds carry None
    # rather than the 0 iterations at 0.0 cost the old parser fabricated
    # (`colsfm.solver_report`).
    assert early.rounds[0].report.is_available
    assert early.rounds[0].report.initial_cost is not None
    assert early.rounds[0].report.final_cost is not None

    full: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta),
        synthetic_database,
        every_round(mapping_config),
        quiet_options(),
    )
    assert len(full.rounds) == mapping_config.num_ba_iterations == 5
    assert [round_stats.max_pixel_error for round_stats in full.rounds] == [25.0, 20.0, 15.0, 10.0, 5.0]


def _perturbed_metadata(rig: SyntheticRig, seed: int = 11) -> FramesMeta:
    """Knock every rig frame but the gauge one 1 cm and 0.5 deg off its true pose.

    The offset is applied to the **vehicle** pose and then pushed through the
    extrinsics, so the perturbed metadata stays rig-exact and the only thing
    bundle adjustment has to undo is the rig pose itself.

    Args:
        rig: The synthetic rig holding the true poses.
        seed: PRNG seed for the offset directions.

    Returns:
        A copy of the metadata with perturbed `camera_to_world` poses.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    keyframe_by_id: dict[int, KeyframeMeta] = rig.frames_meta.keyframe_by_id()
    gauge_sample_id: int = gauge_rig_frame(rig.frames_meta).synced_sample_id
    perturbed_poses: dict[int, pycolmap.Rigid3d] = {}
    for rig_frame in rig.frames_meta.rig_frames():
        offset: pycolmap.Rigid3d = (
            pycolmap.Rigid3d()
            if rig_frame.synced_sample_id == gauge_sample_id
            else random_offset(rng, PERTURBATION_M, PERTURBATION_DEG)
        )
        world_T_vehicle: pycolmap.Rigid3d = rig_frame.world_T_vehicle * offset
        for keyframe_id in rig_frame.keyframe_ids:
            camera_params_id: int = keyframe_by_id[keyframe_id].camera_params_id
            perturbed_poses[keyframe_id] = world_T_vehicle * rig.frames_meta.cameras[camera_params_id].vehicle_T_cam
    return rig.frames_meta.with_camera_to_world(perturbed_poses)


@pytest.mark.parametrize(
    ("noise_px", "max_translation_m", "max_rotation_deg"),
    [(0.0, 1e-5, 1e-4), (0.1, 1e-3, 0.05)],
    ids=["exact", "0.1px-noise"],
)
def test_bundle_adjustment_pulls_perturbed_rig_poses_back(
    tmp_path: Path, isaac_config: CusfmConfig, noise_px: float, max_translation_m: float, max_rotation_deg: float
) -> None:
    """Poses knocked 1 cm and 0.5 deg off come back to the observation noise floor.

    The gauge frame is held constant, so the recovered trajectory is directly
    comparable with the truth — no alignment step hides the error. What is left
    is estimator variance, and it scales with the pixel noise: measured
    0.0 px -> 0 mm, 0.1 px -> 0.85 mm, 0.3 px -> 2.5 mm on this geometry. The
    exact case is the real proof that the perturbation itself is removed.

    Extrinsics must not move: `optimize_extrinsics` is off.
    """
    rig: SyntheticRig = build_synthetic_rig(noise_px=noise_px)
    database_path: Path = tmp_path / "perturbed.db"
    write_database(database_path, rig.frames_meta, rig.keypoints_px, synthetic_matches(rig))
    perturbed_meta: FramesMeta = _perturbed_metadata(rig)

    model: PosedModel = build_reconstruction(perturbed_meta)
    reconstruction: pycolmap.Reconstruction = model.reconstruction
    extrinsics_before: dict[int, Float64[ndarray, "3 4"]] = {
        camera_params_id: reconstruction.rig(RIG_ID).sensor_from_rig(camera_sensor_id(camera_params_id)).matrix().copy()
        for camera_params_id in sorted(perturbed_meta.cameras)
    }
    start_translation_m: float = max(
        pose_delta(reconstruction.frame(rig_frame.synced_sample_id).rig_from_world.inverse(), rig_frame.world_T_vehicle)[0]
        for rig_frame in rig.frames_meta.rig_frames()
    )
    assert start_translation_m == pytest.approx(PERTURBATION_M, abs=1e-6)

    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    result: MappingResult = run_mapping(
        model, database_path, every_round(mapping_config), quiet_options()
    )

    worst_translation_m: float = 0.0
    worst_rotation_deg: float = 0.0
    for rig_frame in rig.frames_meta.rig_frames():
        frame: pycolmap.Frame = result.reconstruction.frame(rig_frame.synced_sample_id)
        translation_m, rotation_deg = pose_delta(frame.rig_from_world.inverse(), rig_frame.world_T_vehicle)
        worst_translation_m = max(worst_translation_m, translation_m)
        worst_rotation_deg = max(worst_rotation_deg, rotation_deg)
    print(
        f"[colsfm] perturbed rig at {noise_px} px noise: worst pose error "
        f"{worst_translation_m * 1e3:.3f} mm, {worst_rotation_deg:.4f} deg after {len(result.rounds)} rounds"
    )
    assert worst_translation_m < max_translation_m
    assert worst_rotation_deg < max_rotation_deg
    for camera_params_id, matrix in extrinsics_before.items():
        after: Float64[ndarray, "3 4"] = (
            result.reconstruction.rig(RIG_ID).sensor_from_rig(camera_sensor_id(camera_params_id)).matrix()
        )
        assert np.array_equal(after, matrix), "extrinsics must not move when optimize_extrinsics is off"


def test_run_mapping_is_deterministic_on_one_thread(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """Two single-threaded runs agree on every point count and every round.

    cuSFM's own config says the same thing about its triangulator: a re-run at
    eight threads gave 1281 points against a shipped 1275 (§5.3).
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    results: list[MappingResult] = [
        run_mapping(
            build_reconstruction(synthetic_rig.frames_meta),
            synthetic_database,
            mapping_config,
            quiet_options(num_threads=1),
        )
        for _ in range(2)
    ]
    assert results[0].num_points3D == results[1].num_points3D
    assert results[0].num_observations == results[1].num_observations
    assert [stats.num_points3D for stats in results[0].rounds] == [stats.num_points3D for stats in results[1].rounds]
    assert results[0].mean_reprojection_error_px == pytest.approx(results[1].mean_reprojection_error_px, abs=1e-9)


def test_fisheye_rig_triangulates(tmp_path: Path, isaac_config: CusfmConfig) -> None:
    """An OPENCV_FISHEYE rig produces points, with the prior focal length declared.

    This is the guard on pycolmap-capabilities.md §5: a fisheye camera whose
    `has_prior_focal_length` is left at its False default degrades to
    `DEGENERATE` with zero inliers, and nothing warns.
    """
    rig: SyntheticRig = build_synthetic_rig(num_frames=8, num_points=200, fisheye=True, seed=3)
    database_path: Path = tmp_path / "fisheye.db"
    write_database(database_path, rig.frames_meta, rig.keypoints_px, synthetic_matches(rig))

    model: PosedModel = build_reconstruction(rig.frames_meta)
    reconstruction: pycolmap.Reconstruction = model.reconstruction
    assert {camera.model.name for camera in reconstruction.cameras.values()} == {"OPENCV_FISHEYE"}
    assert all(camera.has_prior_focal_length for camera in reconstruction.cameras.values())

    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    result: MappingResult = run_mapping(model, database_path, mapping_config, quiet_options())
    print(f"[colsfm] fisheye: {result.num_points3D} points, reprojection {result.mean_reprojection_error_px:.4f} px")
    assert result.num_points3D > 100
    assert result.mean_reprojection_error_px < 1.0


def test_the_round_callback_sees_every_round_as_it_finishes(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """`MappingOptions.round_callback` fires once per round, with that round's model.

    The hook a live viewer needs. It runs after the round's filter and solve and
    before the early-exit check, so the number of calls equals the number of
    `RoundStats` the mapper returns, and the reconstruction it is handed already
    holds `RoundStats.num_points3D` points.
    """
    seen: list[tuple[int, int, int]] = []

    def record(stats: RoundStats, reconstruction: pycolmap.Reconstruction) -> None:
        """Record the round index, its reported point count and the live one."""
        seen.append((stats.round_index, stats.num_points3D, reconstruction.num_points3D()))

    model: PosedModel = build_reconstruction(synthetic_rig.frames_meta)
    result: MappingResult = run_mapping(
        model, synthetic_database, isaac_config.vision_mapping, quiet_options(round_callback=record)
    )
    assert len(seen) == len(result.rounds)
    assert [entry[0] for entry in seen] == [stats.round_index for stats in result.rounds]
    assert all(reported == live for _index, reported, live in seen)
    assert seen[-1][1] == result.reconstruction.num_points3D()
