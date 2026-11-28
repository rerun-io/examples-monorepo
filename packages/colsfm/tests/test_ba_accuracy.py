"""The accuracy matrix's GPU-free parts: the perturbed start and the table bookkeeping."""

from __future__ import annotations

import dataclasses

import numpy as np
import pycolmap


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
    from colsfm.apis.ba_accuracy import AccuracyRow, AccuracyTable, markdown, merged

    numbers = {
        f.name: 0.5
        for f in dataclasses.fields(AccuracyRow)
        if "float" in str(f.type) and "list" not in str(f.type) and f.default is dataclasses.MISSING
    }
    lists = {"caspar_lm_trials": [3], "caspar_pcg_iterations": [9], "mojo_lm_trials": [3], "mojo_pcg_iterations": [9]}
    seconds = {"caspar_solver_seconds": [0.1], "caspar_solve_call_seconds": [0.2], "mojo_solve_seconds": [0.1], "mojo_call_seconds": [0.2]}
    row = AccuracyRow(
        row="p1",
        gpu="RTX 3060",
        commit="abc",
        date="today",
        observations=10,
        registered_images_caspar=2,
        registered_images_mojo=2,
        caspar_exit="CONVERGED_DIAG_EXIT",
        mojo_exit="CONVERGED_DIAG_EXIT",
        **numbers,
        **lists,
        **seconds,
    )
    newer = dataclasses.replace(row, commit="def")
    other_gpu = dataclasses.replace(row, gpu="GB10")
    table = merged(merged(AccuracyTable(rows=[row]), [other_gpu]), [newer])
    assert [(r.gpu, r.commit) for r in table.rows] == [("GB10", "abc"), ("RTX 3060", "def")]
    assert "**>2 %**" in markdown(table)
