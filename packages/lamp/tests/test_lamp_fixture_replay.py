"""Slow equivalence against the fork-recorded Aria fixture."""

from pathlib import Path

import numpy as np
import pytest
from test_lamp_upstream_equivalence import _load_upstream_modules

from lamptrack.apis.lamp_replay import fixture_path, load_lift_record, skeleton_arrays
from lamptrack.third_party.lamp.core.types import Person, PersonState, Skeleton
from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings
from lamptrack.third_party.lamp.tracking.tracker import LampTracker

PACKAGE_ROOT = Path(__file__).parents[1]
FIXTURE_DIR = PACKAGE_ROOT / "data" / "fixtures" / "test-library"
CHECKPOINT = PACKAGE_ROOT / "data" / "checkpoints" / "lamp_smpl_aria_gen2.pt"
SMPL_MODEL = PACKAGE_ROOT / "data" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"


def test_fixture_path_matches_published_archive() -> None:
    """The replay resolves the exact archive published by the reviewer."""
    assert fixture_path(FIXTURE_DIR) == FIXTURE_DIR / "test-library_fixture.npz"


@pytest.mark.slow
def test_fixture_lifter_and_smoothing_equivalence() -> None:
    """Pristine/owned CPU outputs are exact; upstream GPU smoothing is within 0.1 mm."""
    try:
        path = fixture_path(FIXTURE_DIR)
    except FileNotFoundError as exc:
        pytest.skip(f"fork-recorded LAMP fixture absent: {exc}")
    missing_artifacts = [artifact for artifact in (CHECKPOINT, SMPL_MODEL) if not artifact.is_file()]
    if missing_artifacts:
        pytest.skip(f"LAMP model artifacts absent: {missing_artifacts}")

    # Call four is the first accepted upstream lift, so it exercises the
    # fixture's post-smoothing values without depending on earlier fusion.
    record = load_lift_record(path, 4)
    upstream_lifter_module, _, _ = _load_upstream_modules()
    upstream = upstream_lifter_module.Lifter.from_checkpoint(
        CHECKPOINT,
        SMPL_MODEL,
        device="cpu",
        settings=upstream_lifter_module.LifterSettings(snippet_length=20),
        capture_cuda_graph=False,
    )
    owned = Lifter.from_checkpoint(
        CHECKPOINT,
        SMPL_MODEL,
        device="cpu",
        settings=LifterSettings(snippet_length=20),
        capture_cuda_graph=False,
    )

    upstream_outputs = upstream.lift_all_steps_batched(record.snippets)
    owned_outputs = owned.lift_all_steps_batched(record.snippets)
    upstream_rows = [[skeleton for _, skeleton in upstream_outputs[int(person_id)]] for person_id in record.person_ids]
    upstream_arrays = (
        np.stack([np.stack([skeleton.kp_world for skeleton in row]) for row in upstream_rows]),
        np.stack([row[0].shape for row in upstream_rows]),
        np.stack([np.stack([skeleton.T_world_pelvis for skeleton in row]) for row in upstream_rows]),
        np.stack([np.stack([skeleton.joints_rot_mat for skeleton in row]) for row in upstream_rows]),
    )
    owned_arrays = skeleton_arrays(record.person_ids, owned_outputs)
    for upstream_array, owned_array in zip(upstream_arrays, owned_arrays, strict=True):
        assert np.array_equal(upstream_array, owned_array)

    tracker = LampTracker(num_cameras=4)
    for row, person_id_raw in enumerate(record.person_ids):
        person_id = int(person_id_raw)
        person = tracker.people.setdefault(person_id, Person(person_id))
        for timestamp_raw in record.timestamps_ns[row]:
            person.ts_to_states.setdefault(int(timestamp_raw), PersonState(detection2ds=[]))
        snippet = record.snippets[person_id]
        world_T_cameras = {
            int(camera_index): (
                np.linalg.inv(snippet.T_gravityWorld_world) @ snippet.Ts_gw_cam_per_view[view_index][-1]
            ).astype(np.float32)
            for view_index, camera_index in enumerate(snippet.view_cam_indices)
            if camera_index is not None
        }
        fixture_skeletons = [
            (
                int(timestamp),
                Skeleton(
                    kp_world=record.expected_skel_w[row, time_index].copy(),
                    kp_score=np.ones(24, dtype=np.float32),
                    T_world_pelvis=record.expected_root_transform[row, time_index].copy(),
                    shape=record.expected_betas[row].copy(),
                    joints_rot_mat=record.expected_rotations[row, time_index].copy(),
                ),
            )
            for time_index, timestamp in enumerate(record.timestamps_ns[row])
        ]
        tracker.attach_skeletons(
            person_id,
            fixture_skeletons,
            world_T_cameras,
            snippet.T_gravityWorld_world,
            min_pose_depth=1.0,
            max_pose_depth=5.0,
        )
        accepted = tracker.people[person_id].last_lifted_ts == int(record.timestamps_ns[row, -1])
        assert accepted == bool(record.expected_accepted[row])
        if accepted:
            smoothed = np.stack(
                [tracker.people[person_id].ts_to_states[int(timestamp)].skeleton.kp_world for timestamp in record.timestamps_ns[row]]  # type: ignore[union-attr]
            )
            assert np.allclose(smoothed, record.expected_smoothed_skel_w[row], atol=1e-4, rtol=0.0)
