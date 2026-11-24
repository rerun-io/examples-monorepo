"""`--optimize_extrinsics`: what an unregularised rig refinement recovers, and what it costs.

Its own file because it is the one mapping subject that cannot be measured on a
synthetic rig. The claim — that a 20 mm perturbation of one camera's
`vehicle_T_cam` is *removed* rather than merely reduced — is settled by three
mappings of the same real Galileo matches that differ only in their start point
and the flag, and that comparison is what the module-scoped `extrinsic_runs`
fixture exists for. The bounds below are measurements, and each one carries the
number it was measured at.

The synthetic rig appears only once here, to pin the refusal: a vehicle-referenced
rig cannot refine extrinsics at all, and says so rather than silently no-opping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from conftest import pose_delta
from mapping_helpers import (
    PERTURBATION_DEG,
    SyntheticRig,
    quiet_options,
    random_offset,
    synthetic_matches,
    write_database,
)

from colsfm.config import CusfmConfig, VisionMappingConfig
from colsfm.extrinsic_refinement import ExtrinsicRefinementOptions, ExtrinsicRefinementResult, refine_extrinsics
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.mapping import MappingResult, run_mapping
from colsfm.reconstruction import build_reconstruction, gauge_camera_params_id


@pytest.fixture(scope="module")
def synthetic_database(synthetic_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic rig's observations as a COLMAP database."""
    database_path: Path = tmp_path_factory.mktemp("synthetic") / "database.db"
    write_database(
        database_path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, synthetic_matches(synthetic_rig)
    )
    return database_path

WEAKLY_CONSTRAINED_MOVE_M: float = 0.03
"""Above this an unregularised refinement is drifting, not correcting; see
`test_the_unregularised_refinement_drifts_on_weakly_covisible_cameras`."""

EXTRINSIC_PERTURBATION_M: float = 0.02
"""20 mm of translation error the refinement has to remove."""

EXTRINSIC_RECOVERY_M: float = 2e-3
"""How close to the unperturbed refinement the perturbed one must land: 2 mm.

Not 1 mm, which is what this was and which made the test flaky. Neither side of
this comparison is a fixed point: both runs are the same unregularised
block-coordinate descent over the same matches, but they start 20 mm apart and
stop on their own tolerances, so the gap between them is a property of the
descent, not of the optimum. Measured 0.660 mm on the committed Galileo database
and 1.453 mm on a regenerated one — a factor of 2.2 between two artifacts of the
same run, so the bound has to sit above both."""

EXTRINSIC_RECOVERY_DEG: float = 0.05
"""How close to the unperturbed refinement the perturbed one must land: 0.05 deg.

Measured 0.0019 deg, i.e. 26x inside the bound; rotation is far better
conditioned here than translation."""

EXTRINSIC_RESIDUAL_FRACTION: float = 0.5
"""Fraction of the perturbation that may survive against the *calibration*.

This bound is a fraction rather than a millimetre figure on purpose. The
refinement is unregularised — colsfm has no equivalent of cuSFM's absolute and
relative extrinsic priors (NOTES.md deviation 9) — so its optimum does not sit on
the factory calibration and there is no reason for it to: on the committed
Galileo database the *unperturbed* refinement already lands 7.07 mm and 0.30 deg
away from it, while cutting the mean reprojection error from 1.027 px to 0.860 px.
So "came back to the calibration" is not the claim this test can make; "removed
most of a 20 mm perturbation" is, and 7.07 mm of 20 mm is inside this bound with
room to spare."""


def _stereo_partner(frames_meta: FramesMeta, camera_params_id: int) -> int:
    """The other camera of the declared stereo pair holding this one.

    The refinement tests perturb the reference camera's stereo partner, because
    that is the one relative extrinsic Galileo's imagery genuinely determines:
    the pair shares a whole field of view, whereas two cameras facing different
    ways are linked only through the trajectory.

    Args:
        frames_meta: The collection, for its `stereo_pair` list.
        camera_params_id: One camera of a declared pair.

    Returns:
        The partner's `camera_params_id`.

    Raises:
        ValueError: When the camera is in no declared stereo pair.
    """
    for pair in frames_meta.stereo_pairs:
        if pair.left_camera_params_id == camera_params_id:
            return pair.right_camera_params_id
        if pair.right_camera_params_id == camera_params_id:
            return pair.left_camera_params_id
    raise ValueError(f"camera {camera_params_id} is in no declared stereo pair")


@pytest.fixture(scope="module")
def galileo_pose_graph_meta(repo_root: Path) -> FramesMeta:
    """The rig-exact 226-keyframe Galileo poses a colsfm run's pose-graph stage wrote."""
    path: Path = repo_root / "data" / "runs" / "galileo" / "pose_graph" / "frames_meta.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return read_frames_meta(path)


@pytest.fixture(scope="module")
def galileo_colsfm_database(repo_root: Path) -> Path:
    """The COLMAP database that same run's matching stage wrote."""
    path: Path = repo_root / "data" / "runs" / "galileo" / "database.db"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return path


def _perturbed_extrinsic(pose: pycolmap.Rigid3d, seed: int = 5) -> pycolmap.Rigid3d:
    """Knock one extrinsic 20 mm and 0.5 deg off.

    Args:
        pose: The extrinsic to disturb.
        seed: PRNG seed for the offset directions.

    Returns:
        The perturbed transform.
    """
    return pose * random_offset(np.random.default_rng(seed), EXTRINSIC_PERTURBATION_M, PERTURBATION_DEG)


@dataclass(frozen=True, slots=True)
class ExtrinsicRuns:
    """Three Galileo mappings that differ only in the start point and the flag."""

    fixed: MappingResult
    """Perturbed metadata, `optimize_extrinsics` off: the control that keeps the perturbation."""
    refined: MappingResult
    """Perturbed metadata, `optimize_extrinsics` on."""
    refined_from_clean: MappingResult
    """Unperturbed metadata, `optimize_extrinsics` on: where the refinement's own optimum is."""
    perturbed_meta: FramesMeta
    """The metadata the first two runs started from."""
    reference_camera_params_id: int
    """The rig origin, which is also the fixed camera."""
    perturbed_camera_params_id: int
    """The camera the first two runs started 20 mm and 0.5 deg off."""


@pytest.fixture(scope="module")
def extrinsic_runs(
    galileo_pose_graph_meta: FramesMeta, galileo_colsfm_database: Path, isaac_config: CusfmConfig
) -> ExtrinsicRuns:
    """Map Galileo three times over the same matches; see `ExtrinsicRuns`.

    Args:
        galileo_pose_graph_meta: The rig-exact input poses.
        galileo_colsfm_database: The matches every run reads.
        isaac_config: The mapping configuration.

    Returns:
        The three results and what they started from.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    reference_camera_params_id: int = gauge_camera_params_id(galileo_pose_graph_meta)
    perturbed_camera_params_id: int = _stereo_partner(galileo_pose_graph_meta, reference_camera_params_id)
    perturbed_meta: FramesMeta = galileo_pose_graph_meta.with_extrinsics(
        {perturbed_camera_params_id: _perturbed_extrinsic(
            galileo_pose_graph_meta.cameras[perturbed_camera_params_id].vehicle_T_cam
        )}
    )

    def _map(frames_meta: FramesMeta, optimize_extrinsics: bool) -> MappingResult:
        return run_mapping(
            build_reconstruction(frames_meta, reference_camera_params_id=reference_camera_params_id),
            galileo_colsfm_database,
            mapping_config,
            quiet_options(optimize_extrinsics=optimize_extrinsics, num_threads=1),
        )

    return ExtrinsicRuns(
        fixed=_map(perturbed_meta, False),
        refined=_map(perturbed_meta, True),
        refined_from_clean=_map(galileo_pose_graph_meta, True),
        perturbed_meta=perturbed_meta,
        reference_camera_params_id=reference_camera_params_id,
        perturbed_camera_params_id=perturbed_camera_params_id,
    )


def test_refinement_removes_a_20_mm_extrinsic_perturbation(
    extrinsic_runs: ExtrinsicRuns, galileo_pose_graph_meta: FramesMeta
) -> None:
    """Camera 1, knocked 20 mm and 0.5 deg off, lands where the clean refinement lands.

    The substantive claim is that the perturbation is *removed*, not merely
    reduced, and the evidence for that is the unperturbed refinement over the
    same matches: two descents starting 20 mm apart stop within
    `EXTRINSIC_RECOVERY_M` of each other. The calibration cannot serve as that
    reference, because this refinement is **unregularised** — colsfm has no
    equivalent of cuSFM's absolute and relative extrinsic priors (§7) — so its
    optimum is several millimetres off the factory numbers with nothing perturbed
    at all (7.07 mm on the committed Galileo database, at a *lower* reprojection
    error). Against the calibration the test therefore asserts only the weak,
    artifact-stable claim: at most `EXTRINSIC_RESIDUAL_FRACTION` of the
    perturbation survives.

    This test was flaky at a 1 mm refinement-against-refinement bound: it
    measured 0.660 mm on the committed database and 1.453 mm on a regenerated
    one. Neither number is an error — the descent stops on a tolerance, not on a
    fixed point — so the bound is 2 mm rather than a seed or a tightened
    tolerance, which would only hide the same variation.

    The fixed-extrinsics run over the same matches is the control: it keeps all
    20 mm, which is the regression this whole change exists for.
    """
    truth: pycolmap.Rigid3d = galileo_pose_graph_meta.cameras[extrinsic_runs.perturbed_camera_params_id].vehicle_T_cam
    started: pycolmap.Rigid3d = extrinsic_runs.perturbed_meta.cameras[extrinsic_runs.perturbed_camera_params_id].vehicle_T_cam
    assert pose_delta(started, truth)[0] == pytest.approx(EXTRINSIC_PERTURBATION_M, abs=1e-9)

    assert extrinsic_runs.fixed.refined_extrinsics is None, "nothing is refined when the flag is off"
    kept: dict[int, pycolmap.Rigid3d] = extrinsic_runs.fixed.reference.vehicle_T_cam_by_camera_params_id(
        extrinsic_runs.fixed.reconstruction
    )
    assert pose_delta(kept[extrinsic_runs.perturbed_camera_params_id], truth)[0] == pytest.approx(EXTRINSIC_PERTURBATION_M, abs=1e-9)

    assert extrinsic_runs.refined.refined_extrinsics is not None
    assert extrinsic_runs.refined_from_clean.refined_extrinsics is not None
    recovered: pycolmap.Rigid3d = extrinsic_runs.refined.refined_extrinsics[extrinsic_runs.perturbed_camera_params_id]
    clean: pycolmap.Rigid3d = extrinsic_runs.refined_from_clean.refined_extrinsics[extrinsic_runs.perturbed_camera_params_id]
    translation_m, rotation_deg = pose_delta(recovered, clean)
    residual_m, residual_deg = pose_delta(recovered, truth)
    print(
        f"[colsfm] extrinsic refinement: camera {extrinsic_runs.perturbed_camera_params_id} started "
        f"{EXTRINSIC_PERTURBATION_M * 1e3:.1f} mm / {PERTURBATION_DEG} deg off, landed "
        f"{translation_m * 1e3:.3f} mm / {rotation_deg:.4f} deg from the clean refinement and "
        f"{residual_m * 1e3:.3f} mm / {residual_deg:.4f} deg from the calibration; reprojection "
        f"{extrinsic_runs.fixed.mean_reprojection_error_px:.4f} -> "
        f"{extrinsic_runs.refined.mean_reprojection_error_px:.4f} px"
    )
    assert translation_m < EXTRINSIC_RECOVERY_M
    assert rotation_deg < EXTRINSIC_RECOVERY_DEG
    assert residual_m < EXTRINSIC_RESIDUAL_FRACTION * EXTRINSIC_PERTURBATION_M
    assert extrinsic_runs.refined.mean_reprojection_error_px <= extrinsic_runs.fixed.mean_reprojection_error_px


def test_the_fixed_camera_extrinsic_is_untouched(
    extrinsic_runs: ExtrinsicRuns, galileo_pose_graph_meta: FramesMeta
) -> None:
    """The fixed camera is the rig origin, so its `vehicle_T_cam` is the input one exactly.

    Bundle adjustment holds no parameter block for the reference sensor at all —
    COLMAP fixes it to identity — which is what makes it the exact bridge back to
    the vehicle frame.
    """
    assert extrinsic_runs.refined.refined_extrinsics is not None
    camera_params_id: int = extrinsic_runs.reference_camera_params_id
    translation_m, rotation_deg = pose_delta(
        extrinsic_runs.refined.refined_extrinsics[camera_params_id],
        galileo_pose_graph_meta.cameras[camera_params_id].vehicle_T_cam,
    )
    assert translation_m == pytest.approx(0.0, abs=1e-12)
    assert rotation_deg == pytest.approx(0.0, abs=1e-9)


def test_the_unregularised_refinement_drifts_on_weakly_covisible_cameras(
    extrinsic_runs: ExtrinsicRuns, galileo_pose_graph_meta: FramesMeta
) -> None:
    """`refined_extrinsics` covers the whole rig, and records how far each camera walked.

    This pins the measured cost of having no extrinsic priors. cuSFM regularises
    its refinement with absolute and relative extrinsic residuals (§7), which
    pycolmap's bundle adjuster cannot express, so colsfm's optimum is the
    reprojection optimum alone. On Galileo's 0.66 m sweep that is well determined
    only inside a stereo pair — the reference camera's partner moves millimetres,
    the four cameras facing other directions tens of millimetres, because nothing
    but the trajectory links them to the reference.
    """
    refined: dict[int, pycolmap.Rigid3d] | None = extrinsic_runs.refined_from_clean.refined_extrinsics
    assert refined is not None
    assert sorted(refined) == sorted(extrinsic_runs.refined_from_clean.reconstruction.cameras)
    moved_m: dict[int, float] = {
        camera_params_id: pose_delta(pose, galileo_pose_graph_meta.cameras[camera_params_id].vehicle_T_cam)[0]
        for camera_params_id, pose in refined.items()
    }
    print(
        "[colsfm] unregularised extrinsic moves (mm): "
        + ", ".join(f"{key}={value * 1e3:.2f}" for key, value in sorted(moved_m.items()))
    )
    reference_camera_params_id: int = extrinsic_runs.reference_camera_params_id
    assert moved_m[reference_camera_params_id] == pytest.approx(0.0, abs=1e-12)
    partner: int = _stereo_partner(galileo_pose_graph_meta, reference_camera_params_id)
    assert 0.0 < moved_m[partner] < WEAKLY_CONSTRAINED_MOVE_M, "the reference's own stereo pair stays put"
    elsewhere: list[float] = [
        value for key, value in moved_m.items() if key not in {reference_camera_params_id, partner}
    ]
    assert max(elsewhere) > WEAKLY_CONSTRAINED_MOVE_M, "the other cameras drift; there is no prior holding them"


def test_refining_extrinsics_on_a_vehicle_referenced_rig_is_refused(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A vehicle-referenced rig cannot refine extrinsics, and says so instead of no-opping.

    COLMAP freezes every `sensor_from_rig` of a rig whose reference sensor is not
    itself parameterised, and the vehicle body owns no images, so the request
    would otherwise be silently ignored.
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    with pytest.raises(ValueError, match="reference sensor"):
        run_mapping(
            build_reconstruction(synthetic_rig.frames_meta),
            synthetic_database,
            mapping_config,
            quiet_options(optimize_extrinsics=True),
        )


def test_the_unregularised_refinement_runs_on_an_already_mapped_model(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """`--no-regularised-extrinsics` refines the model stage 7 left behind instead of crashing.

    Stage 7b hands `refine_extrinsics` a reconstruction whose images already hold the database keypoints; the
    unregularised path used to load them a second time, which pycolmap refuses (`image.cc:60 points2D_.empty()`).
    """
    mapping_config: VisionMappingConfig = isaac_config.vision_mapping
    reference_camera_params_id: int = gauge_camera_params_id(synthetic_rig.frames_meta)
    mapped: MappingResult = run_mapping(
        build_reconstruction(synthetic_rig.frames_meta, reference_camera_params_id=reference_camera_params_id),
        synthetic_database,
        mapping_config,
        quiet_options(num_threads=1),
    )
    num_points: int = mapped.num_points3D
    refinement: ExtrinsicRefinementResult = refine_extrinsics(
        mapped,
        synthetic_database,
        mapping_config,
        quiet_options(num_threads=1),
        {camera_params_id: camera.vehicle_T_cam for camera_params_id, camera in synthetic_rig.frames_meta.cameras.items()},
        ExtrinsicRefinementOptions(regularised=False, verbose=False),
    )
    assert refinement.mapping.refined_extrinsics is not None
    assert refinement.mapping.num_points3D >= 0.9 * num_points
    assert refinement.mapping.mean_reprojection_error_px < 1.0
