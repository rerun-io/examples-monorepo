"""Probe (c): triangulation with known poses, both end-to-end and through the triangulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from colsfm.reconstruction import rig_from_world
from probes.synthetic import SyntheticScene, matches_between


def test_triangulate_points_end_to_end(scene_on_disk: tuple[SyntheticScene, Path, Path], tmp_path: Path) -> None:
    """`triangulate_points` triangulates a model whose poses are already known.

    Signature (COLMAP 4.2)::

        triangulate_points(reconstruction, database_path, image_path, output_path,
                           clear_points=True,
                           options=IncrementalPipelineOptions(),
                           refine_intrinsics=False,
                           cancellation_token=None) -> Reconstruction

    The input reconstruction must carry registered frames with poses; the
    database supplies keypoints and verified two-view geometries.  Poses are
    held constant throughout (see
    `test_triangulate_points_preserves_the_input_poses`).
    """
    scene, database_path, image_dir = scene_on_disk
    output_dir: Path = tmp_path / "sparse"
    output_dir.mkdir()

    options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions()
    triangulated: pycolmap.Reconstruction = pycolmap.triangulate_points(
        scene.reconstruction, database_path, image_dir, output_dir, True, options, False
    )

    assert triangulated.num_reg_frames() == scene.reconstruction.num_frames()
    assert triangulated.num_points3D() == len(scene.points_xyz)
    assert triangulated.compute_mean_reprojection_error() < 0.01
    # COLMAP 4.2 writes five files, not three: rigs and frames are first-class.
    written: set[str] = {path.name for path in output_dir.iterdir()}
    assert written == {"cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"}

    recovered: dict[int, np.ndarray] = {point_id: triangulated.point3D(point_id).xyz for point_id in triangulated.point3D_ids()}
    nearest: float = max(float(np.min(np.linalg.norm(scene.points_xyz - xyz, axis=1))) for xyz in recovered.values())
    assert nearest < 1e-3, f"worst triangulated point is {nearest:.4f} m from ground truth"


def test_triangulate_points_preserves_the_input_poses(scene_on_disk: tuple[SyntheticScene, Path, Path], tmp_path: Path) -> None:
    """`triangulate_points` is pose-preserving: the input trajectory survives bit-exact.

    That is the property the cuSFM contract needs — the supplied trajectory is a
    prior, not an initialisation.  COLMAP does run a global bundle adjustment at
    the end of this call (it says so in its log), but with every `rig_from_world`
    held constant, so only the points move.  Measured, not assumed: a frame is
    pushed 6 cm off its true pose and comes back byte-identical to the
    perturbed value.
    """
    scene, database_path, image_dir = scene_on_disk
    output_dir: Path = tmp_path / "sparse"
    output_dir.mkdir()

    perturbed_frame_id: int = 3
    scene.reconstruction.frame(perturbed_frame_id).rig_from_world = pycolmap.Rigid3d(
        rig_from_world(scene.reconstruction, perturbed_frame_id).rotation,
        rig_from_world(scene.reconstruction, perturbed_frame_id).translation + np.array([0.05, -0.03, 0.02]),
    )
    poses_before: dict[int, np.ndarray] = {
        frame_id: rig_from_world(scene.reconstruction, frame_id).matrix().copy() for frame_id in sorted(scene.reconstruction.frames)
    }

    triangulated: pycolmap.Reconstruction = pycolmap.triangulate_points(
        scene.reconstruction,
        database_path,
        image_dir,
        output_dir,
        True,
        pycolmap.IncrementalPipelineOptions(),
        False,
    )
    shifts: list[float] = [float(np.abs(rig_from_world(triangulated, frame_id).matrix() - matrix).max()) for frame_id, matrix in poses_before.items()]
    print(
        f"[probe] triangulate_points max pose change {max(shifts):.3e} "
        f"with one frame 6 cm off; reproj {triangulated.compute_mean_reprojection_error():.4f} px"
    )
    assert max(shifts) == 0.0
    # The perturbation does show up as reprojection error, so the model really
    # was triangulated against the wrong pose rather than the call being a no-op.
    assert triangulated.compute_mean_reprojection_error() > 0.1


def test_incremental_triangulator_only_triangulates(scene: SyntheticScene) -> None:
    """`IncrementalTriangulator` adds points without touching poses.

    It is built from a `CorrespondenceGraph` plus the `Reconstruction`, and it is
    the API to use when the input trajectory must be honoured exactly.  Both
    objects must outlive the triangulator, so keep references to them.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    graph: pycolmap.CorrespondenceGraph = pycolmap.CorrespondenceGraph()
    for image_id in sorted(reconstruction.images):
        graph.add_image(image_id, len(scene.keypoints_px[image_id]))
    image_ids: list[int] = sorted(reconstruction.images)
    for index, image_id1 in enumerate(image_ids):
        for image_id2 in image_ids[index + 1 :]:
            matches = matches_between(scene, image_id1, image_id2)
            if len(matches) < 20:
                continue
            geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
            geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
            geometry.inlier_matches = matches
            graph.add_two_view_geometry(image_id1, image_id2, geometry)
    graph.finalize()

    observations: pycolmap.ObservationManager = pycolmap.ObservationManager(reconstruction, graph)
    triangulator: pycolmap.IncrementalTriangulator = pycolmap.IncrementalTriangulator(graph, reconstruction, observations)
    options: pycolmap.IncrementalTriangulatorOptions = pycolmap.IncrementalTriangulatorOptions()
    options.min_angle = 0.5
    options.ignore_two_view_tracks = False

    poses_before: dict[int, np.ndarray] = {
        frame_id: rig_from_world(reconstruction, frame_id).matrix().copy() for frame_id in sorted(reconstruction.frames)
    }
    num_new: int = sum(triangulator.triangulate_image(options, image_id) for image_id in image_ids)
    merged: int = triangulator.merge_all_tracks(options)
    completed: int = triangulator.complete_all_tracks(options)
    print(f"[probe] triangulated {num_new}, merged {merged}, completed {completed}")

    assert reconstruction.num_points3D() > 0
    assert reconstruction.compute_mean_reprojection_error() < 0.01
    for frame_id, matrix in poses_before.items():
        assert np.array_equal(rig_from_world(reconstruction, frame_id).matrix(), matrix)


def test_incremental_triangulator_options_defaults() -> None:
    """Pin the `IncrementalTriangulatorOptions` fields and defaults.

    These are the knobs that map onto cuSFM's `vision_mapping_config.pb.txt`.
    """
    options: pycolmap.IncrementalTriangulatorOptions = pycolmap.IncrementalTriangulatorOptions()
    assert options.min_angle == pytest.approx(1.5)
    assert options.max_transitivity == 1
    assert options.create_max_angle_error == pytest.approx(2.0)
    assert options.continue_max_angle_error == pytest.approx(2.0)
    assert options.merge_max_reproj_error == pytest.approx(4.0)
    assert options.complete_max_reproj_error == pytest.approx(4.0)
    assert options.complete_max_transitivity == 5
    assert options.re_max_angle_error == pytest.approx(5.0)
    assert options.re_min_ratio == pytest.approx(0.2)
    assert options.re_max_trials == 1
    assert options.min_focal_length_ratio == pytest.approx(0.1)
    assert options.max_focal_length_ratio == pytest.approx(10.0)
    assert options.max_extra_param == pytest.approx(1.0)
    assert options.ignore_two_view_tracks is True
    assert options.random_seed == -1
    assert options.check()
    # The field is `re_max_angle_error`, not the `re_max_angle` some older
    # COLMAP documentation uses.
    assert not hasattr(options, "re_max_angle")


def test_estimate_triangulation_is_the_standalone_multi_view_solver(
    scene: SyntheticScene,
) -> None:
    """`estimate_triangulation(points, cams_from_world, cameras, options)` is LO-RANSAC.

    It returns a dict with `xyz` and an inlier mask, or None when it fails, and
    needs no database or reconstruction at all.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    image_ids: list[int] = sorted(reconstruction.images)
    point_index: int = 0
    pixels: list[np.ndarray] = []
    poses: list[pycolmap.Rigid3d] = []
    cameras: list[pycolmap.Camera] = []
    for image_id in image_ids:
        where = np.flatnonzero(scene.point_indices[image_id] == point_index)
        if len(where) == 0:
            continue
        pixels.append(scene.keypoints_px[image_id][where[0]])
        poses.append(reconstruction.image(image_id).cam_from_world())
        cameras.append(reconstruction.camera(reconstruction.image(image_id).camera_id))
    assert len(pixels) >= 3

    result: dict | None = pycolmap.estimate_triangulation(np.asarray(pixels), poses, cameras, pycolmap.EstimateTriangulationOptions())
    assert result is not None
    assert np.allclose(result["xyz"], scene.points_xyz[point_index], atol=1e-6)
