"""The accuracy matrix's GPU-free parts: the perturbed start and the table bookkeeping."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pycolmap
import pytest

if TYPE_CHECKING:
    from colsfm.apis.ba_accuracy import AccuracyRow


def _tiny() -> pycolmap.Reconstruction:
    from colsfm.apis.ba_synth import synthetic_reconstruction

    return synthetic_reconstruction(6, 200, "PINHOLE", 1)  # type: ignore[arg-type]


def _exactly_zero(reconstruction: pycolmap.Reconstruction) -> pycolmap.Reconstruction:
    """A copy whose every linked observation is the exact projection of its point: a reference at zero cost."""
    exact = pycolmap.Reconstruction(reconstruction)
    for image_id in exact.reg_image_ids():
        image: pycolmap.Image = exact.image(image_id)
        camera: pycolmap.Camera = exact.camera(image.camera_id)
        pose: pycolmap.Rigid3d = image.cam_from_world()
        for point2D in image.points2D:
            if point2D.has_point3D():
                in_cam = np.asarray(pose.rotation.matrix()) @ np.asarray(exact.point3D(point2D.point3D_id).xyz) + np.asarray(pose.translation)
                point2D.xy = np.asarray(camera.img_from_cam(in_cam[None]))[0]
    return exact


def _row() -> AccuracyRow:
    from colsfm.apis.ba_accuracy import AccuracyRow

    relative = {"reprojection_relative_difference", "cost_relative_difference"}
    numbers = {f.name: 0.5 for f in dataclasses.fields(AccuracyRow) if f.type in ("float", float) and f.name not in relative}
    lists = {"caspar_lm_trials": [3], "caspar_pcg_iterations": [9], "mojo_lm_trials": [3], "mojo_pcg_iterations": [9]}
    seconds = {"caspar_solver_seconds": [0.1], "caspar_solve_call_seconds": [0.2], "mojo_solve_seconds": [0.1], "mojo_call_seconds": [0.2]}
    return AccuracyRow(
        row="p1",
        gpu="RTX 3060",
        commit="abc",
        date="today",
        observations=10,
        reprojection_relative_difference=0.5,
        cost_relative_difference=0.5,
        registered_images_caspar=2,
        registered_images_mojo=2,
        caspar_exit="CONVERGED_DIAG_EXIT",
        mojo_exit="CONVERGED_DIAG_EXIT",
        **numbers,
        **lists,
        **seconds,
    )


def test_perturb_moves_every_free_pose_and_point_by_the_stated_amounts() -> None:
    from colsfm.apis.ba_accuracy import perturb
    from colsfm.apis.ba_synth import synthetic_reconstruction
    from colsfm.apis.caspar_baseline import gauge_config

    start: pycolmap.Reconstruction = synthetic_reconstruction(6, 200, "PINHOLE", 1)  # type: ignore[arg-type]
    moved = pycolmap.Reconstruction(start)
    config = gauge_config(moved)
    perturb(moved, config, seed=0, translation=0.02, rotation_deg=0.1, point=0.03)
    for frame_id in start.reg_frame_ids():
        before, after = start.frame(frame_id).rig_from_world, moved.frame(frame_id).rig_from_world
        assert before is not None and after is not None
        shift: float = float(np.linalg.norm(np.asarray(after.translation) - np.asarray(before.translation)))
        turn: float = np.rad2deg(before.rotation.angle_to(after.rotation))
        expected = (0.0, 0.0) if config.has_constant_rig_from_world_pose(frame_id) else (0.02, 0.1)
        assert np.allclose((shift, turn), expected, atol=1e-9)
    shifts = [np.linalg.norm(np.asarray(moved.point3D(i).xyz) - np.asarray(start.point3D(i).xyz)) for i in start.point3D_ids()]
    assert np.allclose(shifts, 0.03)


def test_a_merged_row_replaces_the_row_of_the_same_name_and_gpu() -> None:
    from colsfm.apis.ba_accuracy import AccuracyTable, markdown, merged

    row = _row()
    newer = dataclasses.replace(row, commit="def")
    other_gpu = dataclasses.replace(row, gpu="GB10")
    table = merged(merged(AccuracyTable(rows=[row]), [other_gpu]), [newer])
    assert [(r.gpu, r.commit) for r in table.rows] == [("GB10", "abc"), ("RTX 3060", "def")]
    assert "**>2 %**" in markdown(table)


def test_a_failed_adjuster_build_leaves_pycolmaps_logging_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    from colsfm.apis import caspar_baseline

    def refuse(*_: object) -> pycolmap.BundleAdjuster:
        raise ValueError("refine_sensor_from_rig=true is not supported by CASPAR")

    monkeypatch.setattr(caspar_baseline.pycolmap, "create_default_bundle_adjuster", refuse)
    before: tuple[int, bool] = (pycolmap.logging.verbose_level, pycolmap.logging.logtostderr)
    start: pycolmap.Reconstruction = _tiny()
    with pytest.raises(ValueError, match="not supported"):
        caspar_baseline.traced_solve(start, caspar_baseline.gauge_config(start), caspar_baseline.caspar_options())
    assert (pycolmap.logging.verbose_level, pycolmap.logging.logtostderr) == before


@pytest.mark.parametrize("runs", [0, -1])
def test_a_run_count_below_one_is_refused_before_any_model_is_read(runs: int, tmp_path: Path) -> None:
    from colsfm.apis import ba_accuracy, caspar_baseline

    with pytest.raises(ValueError, match="runs"):
        ba_accuracy.Config(model_dir=tmp_path / "missing", gpu="x", output=tmp_path / "t.json", runs=runs)
    with pytest.raises(ValueError, match="runs"):
        caspar_baseline.Config(model_dir=tmp_path / "missing", runs=runs)


def test_an_exact_reference_compares_without_dividing_by_zero() -> None:
    from colsfm.apis.caspar_baseline import caspar_options, gauge_config
    from colsfm.ba_compare import compare_solutions
    from colsfm.ba_mojo import problem_from_reconstruction

    exact: pycolmap.Reconstruction = _exactly_zero(_tiny())
    options = caspar_options()
    bridged = problem_from_reconstruction(exact, gauge_config(exact), options)
    same = compare_solutions(exact, pycolmap.Reconstruction(exact), bridged)
    assert same.cost_a == 0.0 and same.cost_relative_difference == 0.0

    moved = pycolmap.Reconstruction(exact)
    point3D_id: int = int(bridged.ids.point3D_ids[0])
    moved.point3D(point3D_id).xyz = np.asarray(moved.point3D(point3D_id).xyz) + 0.1
    away = compare_solutions(exact, moved, bridged)
    assert away.cost_b > 0.0 and away.cost_relative_difference is None


def test_rows_of_the_same_model_with_different_perturbations_are_kept_apart() -> None:
    from colsfm.apis.ba_accuracy import AccuracyTable, merged

    row = dataclasses.replace(_row(), model_dir="/models/p1")
    stronger = dataclasses.replace(row, perturb_seed=0, perturb_translation=0.05, perturb_rotation_deg=0.1, perturb_point=0.02)
    weaker = dataclasses.replace(stronger, perturb_translation=0.02)
    table: AccuracyTable = merged(AccuracyTable(rows=[row]), [stronger, weaker, dataclasses.replace(weaker, commit="def")])
    assert len(table.rows) == 3 and [r.commit for r in table.rows if r.perturb_translation == 0.02] == ["def"]


def test_rows_written_before_the_inputs_were_recorded_still_load() -> None:
    import json

    from serde.json import from_json, to_json

    from colsfm.apis.ba_accuracy import AccuracyTable

    stored = json.loads(to_json(AccuracyTable(rows=[_row()])))
    for name in ("model_dir", "refine_intrinsics", "refine_sensor_from_rig", "perturb_seed", "perturb_translation", "perturb_rotation_deg", "perturb_point"):
        del stored["rows"][0][name]
    legacy: AccuracyRow = from_json(AccuracyTable, json.dumps(stored)).rows[0]
    assert legacy.case_key() == ("RTX 3060", "p1")


def test_a_cost_or_registration_miss_is_flagged_like_a_reprojection_miss() -> None:
    from colsfm.apis.ba_accuracy import AccuracyTable, markdown

    within = dataclasses.replace(_row(), reprojection_relative_difference=0.0, cost_relative_difference=0.0)
    assert "**>2 %**" not in markdown(AccuracyTable(rows=[within]))
    assert "**>2 %**" in markdown(AccuracyTable(rows=[dataclasses.replace(within, cost_relative_difference=0.03)]))
    assert "**>2 %**" in markdown(AccuracyTable(rows=[dataclasses.replace(within, registered_images_mojo=1)]))
