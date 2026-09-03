"""Replay the fork-recorded LAMP lifter and smoothing seam without Aria."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from lamptrack.rerun_logging import LivePeopleLogger
from lamptrack.third_party.lamp.core.types import SMPL_SKELETON_EDGES, Person, PersonState, Skeleton, color_from_id
from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings, SnippetData
from lamptrack.third_party.lamp.tracking.tracker import LampTracker

_PACKAGE_ROOT = Path(__file__).parents[2]
_FIXTURE_NAME = "test-library_fixture.npz"


@dataclass(frozen=True, slots=True)
class Config:
    """Paths, execution bounds, and Rerun output for fixture replay."""

    fixture_dir: Path = _PACKAGE_ROOT / "data" / "fixtures" / "test-library"
    """Directory containing the fork-recorded ``test-library_fixture.npz``."""
    checkpoint: Path = _PACKAGE_ROOT / "data" / "checkpoints" / "lamp_smpl_aria_gen2.pt"
    """Pinned released LAMP checkpoint."""
    smpl_model_path: Path = _PACKAGE_ROOT / "data" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"
    """Chumpy-free neutral SMPL body model."""
    device: str = "cuda"
    """Torch device for lifter inference."""
    max_lift_calls: int | None = None
    """Optional prefix length for a quick replay; ``None`` runs all 623 calls."""
    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(application_id="lamp_fixture_replay", headless=True))
    """Viewer, connection, or save configuration."""


@dataclass(frozen=True, slots=True)
class LiftFixtureRecord:
    """One exact batched upstream lifter call and its recorded outputs."""

    call_timestamp_ns: int
    person_ids: Int64[ndarray, "batch"]
    timestamps_ns: Int64[ndarray, "batch time"]
    snippets: dict[int, SnippetData]
    expected_skel_w: Float32[ndarray, "batch time 24 3"]
    expected_betas: Float32[ndarray, "batch 10"]
    expected_root_transform: Float32[ndarray, "batch time 4 4"]
    expected_rotations: Float32[ndarray, "batch time 24 3 3"]
    expected_accepted: Bool[ndarray, "batch"]
    expected_smoothed_skel_w: Float32[ndarray, "batch time 24 3"]


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    """Measured replay coverage, timing, and fixture differences."""

    lift_calls: int
    person_inputs: int
    accepted_outputs: int
    acceptance_mismatches: int
    lifter_mean_ms: float
    max_raw_joint_error_m: float
    max_smoothed_joint_error_m: float


def fixture_path(fixture_dir: Path) -> Path:
    """Resolve the required replay archive or fail with download guidance.

    Args:
        fixture_dir: Directory expected to contain the compressed NPZ.

    Returns:
        Existing fixture path.

    Raises:
        FileNotFoundError: If the reviewer fixture is not available.
    """
    path = fixture_dir / _FIXTURE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"LAMP replay fixture is missing: {path}\n"
            "Run `pixi run -e lamp _lamp-download-fixture` to download the pinned reviewer fixture."
        )
    return path


def fixture_lift_call_count(path: Path) -> int:
    """Return the number of recorded active lifter calls."""
    with np.load(path, allow_pickle=True) as fixture:
        _validate_fixture(fixture.files, path)
        return int(len(fixture["lift_call_timestamp_ns"]))


def load_lift_record(path: Path, index: int) -> LiftFixtureRecord:
    """Load one trusted fork-produced object-array record.

    Args:
        path: Compressed fixture archive copied from the pixified fork.
        index: Zero-based active-lifter call index.

    Returns:
        Exact model inputs and pre/post-smoothing reference arrays.
    """
    with np.load(path, allow_pickle=True) as fixture:
        _validate_fixture(fixture.files, path)
        return _lift_record_from_fixture(fixture, index)


def load_lift_records(path: Path, count: int) -> list[LiftFixtureRecord]:
    """Load a prefix without reopening and decompressing the archive per call."""
    with np.load(path, allow_pickle=True) as fixture:
        _validate_fixture(fixture.files, path)
        available = len(fixture["lift_call_timestamp_ns"])
        if count < 0 or count > available:
            raise ValueError(f"Requested {count} lift calls from a fixture with {available}")
        return [_lift_record_from_fixture(fixture, index) for index in range(count)]


def _lift_record_from_fixture(fixture: np.lib.npyio.NpzFile, index: int) -> LiftFixtureRecord:
    """Decode one record from an already-open trusted fixture archive."""
    person_ids = np.asarray(fixture["lift_person_ids"][index], dtype=np.int64)
    timestamps = np.asarray(fixture["lift_timestamps_ns"][index], dtype=np.int64)
    view_indices = np.asarray(fixture["lift_view_cam_indices"][index], dtype=np.int64)
    keypoints = np.asarray(fixture["lift_view_keypoints"][index], dtype=np.float32)
    camera_poses = np.asarray(fixture["lift_Ts_gravityWorld_cam"][index], dtype=np.float32)
    camera_params = np.asarray(fixture["lift_view_cameras"][index], dtype=np.float32)
    gravity_transforms = np.asarray(fixture["lift_T_gravityWorld_world"][index], dtype=np.float32)
    snippets: dict[int, SnippetData] = {}
    for row, person_id_raw in enumerate(person_ids):
        person_id = int(person_id_raw)
        snippets[person_id] = SnippetData(
            person_id=person_id,
            snippet_timestamps_ns=[int(value) for value in timestamps[row]],
            view_cam_indices=[None if value < 0 else int(value) for value in view_indices[row]],
            kp2ds_per_view=[keypoints[row, view].copy() for view in range(4)],
            Ts_gw_cam_per_view=[camera_poses[row, view].copy() for view in range(4)],
            cam_params_per_view=[camera_params[row, view].copy() for view in range(4)],
            T_gravityWorld_world=gravity_transforms[row].copy(),
        )
    return LiftFixtureRecord(
        call_timestamp_ns=int(fixture["lift_call_timestamp_ns"][index]),
        person_ids=person_ids,
        timestamps_ns=timestamps,
        snippets=snippets,
        expected_skel_w=np.asarray(fixture["lift_skel_w"][index], dtype=np.float32),
        expected_betas=np.asarray(fixture["lift_betas"][index], dtype=np.float32),
        expected_root_transform=np.asarray(fixture["lift_root_transform"][index], dtype=np.float32),
        expected_rotations=np.asarray(fixture["lift_rotations"][index], dtype=np.float32),
        expected_accepted=np.asarray(fixture["lift_accepted"][index], dtype=np.bool_),
        expected_smoothed_skel_w=np.asarray(fixture["lift_smoothed_skel_w"][index], dtype=np.float32),
    )


def _validate_fixture(files: list[str], path: Path) -> None:
    """Reject archives that do not implement the frozen fixture version."""
    required = {
        "fixture_version",
        "lift_call_timestamp_ns",
        "lift_person_ids",
        "lift_timestamps_ns",
        "lift_view_cam_indices",
        "lift_view_keypoints",
        "lift_Ts_gravityWorld_cam",
        "lift_view_cameras",
        "lift_T_gravityWorld_world",
        "lift_skel_w",
        "lift_betas",
        "lift_root_transform",
        "lift_rotations",
        "lift_accepted",
        "lift_smoothed_skel_w",
    }
    missing = sorted(required - set(files))
    if missing:
        raise ValueError(f"Fixture {path} is missing required arrays: {missing}")


def skeleton_arrays(
    person_ids: Int64[ndarray, "batch"], lifted: dict[int, list[tuple[int, Skeleton]]]
) -> tuple[
    Float32[ndarray, "batch time 24 3"],
    Float32[ndarray, "batch 10"],
    Float32[ndarray, "batch time 4 4"],
    Float32[ndarray, "batch time 24 3 3"],
]:
    """Stack mutable upstream skeleton objects in fixture batch order."""
    rows = [[skeleton for _, skeleton in lifted[int(person_id)]] for person_id in person_ids]
    return (
        np.stack([np.stack([skeleton.kp_world for skeleton in row]) for row in rows]).astype(np.float32),
        np.stack([row[0].shape for row in rows]).astype(np.float32),
        np.stack([np.stack([skeleton.T_world_pelvis for skeleton in row]) for row in rows]).astype(np.float32),
        np.stack([np.stack([skeleton.joints_rot_mat for skeleton in row]) for row in rows]).astype(np.float32),
    )


def replay(config: Config) -> ReplayMetrics:
    """Replay recorded inputs through the owned lifter and tracker smoother."""
    path = fixture_path(config.fixture_dir)
    total_calls = fixture_lift_call_count(path)
    call_count = total_calls if config.max_lift_calls is None else min(total_calls, config.max_lift_calls)
    if call_count <= 0:
        raise ValueError(f"max_lift_calls must select at least one of {total_calls} calls")
    lifter = Lifter.from_checkpoint(
        config.checkpoint,
        config.smpl_model_path,
        device=config.device,
        settings=LifterSettings(snippet_length=20),
        capture_cuda_graph=False,
    )
    tracker = LampTracker(num_cameras=4)
    edges = np.asarray(SMPL_SKELETON_EDGES, dtype=np.int64)
    trails: dict[int, list[Float32[ndarray, "3"]]] = {}
    people_logger = LivePeopleLogger()
    lifter_times: list[float] = []
    raw_errors: list[float] = []
    smoothed_errors: list[float] = []
    person_inputs = 0
    accepted_outputs = 0
    acceptance_mismatches = 0
    records = load_lift_records(path, call_count)
    for record in records:
        live_people: list[tuple[int, Person]] = []
        for row, person_id_raw in enumerate(record.person_ids):
            person_id = int(person_id_raw)
            person = tracker.people.setdefault(person_id, Person(person_id))
            for timestamp_raw in record.timestamps_ns[row]:
                person.ts_to_states.setdefault(int(timestamp_raw), PersonState(detection2ds=[]))

        started = time.perf_counter()
        lifted = lifter.lift_all_steps_batched(record.snippets)
        lifter_times.append((time.perf_counter() - started) * 1000.0)
        raw, _, _, _ = skeleton_arrays(record.person_ids, lifted)
        raw_errors.append(float(np.max(np.abs(raw - record.expected_skel_w))))
        person_inputs += len(record.person_ids)

        for row, person_id_raw in enumerate(record.person_ids):
            person_id = int(person_id_raw)
            snippet = record.snippets[person_id]
            world_T_cameras = {
                int(camera_index): (
                    np.linalg.inv(snippet.T_gravityWorld_world) @ snippet.Ts_gw_cam_per_view[view_index][-1]
                ).astype(np.float32)
                for view_index, camera_index in enumerate(snippet.view_cam_indices)
                if camera_index is not None
            }
            tracker.attach_skeletons(
                person_id,
                _recorded_skeletons(record, row),
                world_T_cameras,
                snippet.T_gravityWorld_world,
                min_pose_depth=1.0,
                max_pose_depth=5.0,
            )
            person = tracker.people[person_id]
            accepted = person.last_lifted_ts == int(record.timestamps_ns[row, -1])
            if accepted != bool(record.expected_accepted[row]):
                acceptance_mismatches += 1
            if accepted:
                accepted_outputs += 1
                actual = np.stack(
                    [person.ts_to_states[int(timestamp)].skeleton.kp_world for timestamp in record.timestamps_ns[row]]  # type: ignore[union-attr]
                )
                expected = record.expected_smoothed_skel_w[row]
                finite = np.isfinite(expected)
                if finite.any():
                    smoothed_errors.append(float(np.max(np.abs(actual[finite] - expected[finite]))))
                state = person.ts_to_states.get(record.call_timestamp_ns)
                if state is not None and state.skeleton is not None:
                    live_people.append((person_id, person))

        rr.set_time("fixture_time", timestamp=record.call_timestamp_ns * 1e-9)
        people_logger.update([person_id for person_id, _ in live_people])
        for person_id, person in live_people:
            _log_person(person_id, person, record.call_timestamp_ns, lifter, edges, trails)

    return ReplayMetrics(
        lift_calls=call_count,
        person_inputs=person_inputs,
        accepted_outputs=accepted_outputs,
        acceptance_mismatches=acceptance_mismatches,
        lifter_mean_ms=float(np.mean(lifter_times)),
        max_raw_joint_error_m=max(raw_errors, default=0.0),
        max_smoothed_joint_error_m=max(smoothed_errors, default=0.0),
    )


def _recorded_skeletons(record: LiftFixtureRecord, row: int) -> list[tuple[int, Skeleton]]:
    """Rebuild the upstream GPU skeletons so smoothing is device-independent."""
    return [
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


def _log_person(
    person_id: int,
    person: Person,
    timestamp_ns: int,
    lifter: Lifter,
    edges: Int64[ndarray, "edges 2"],
    trails: dict[int, list[Float32[ndarray, "3"]]],
) -> None:
    """Log one accepted person's latest skeleton, mesh, and pelvis trail."""
    state = person.ts_to_states.get(timestamp_ns)
    if state is None or state.skeleton is None:
        return
    skeleton = state.skeleton
    color_float = color_from_id(person_id)
    color = tuple(round(channel * 255.0) for channel in color_float)
    rr.set_time("fixture_time", timestamp=timestamp_ns * 1e-9)
    rr.log(f"world/people/{person_id}/skeleton", rr.Points3D(skeleton.kp_world, colors=color, radii=0.025))
    rr.log(
        f"world/people/{person_id}/skeleton/edges",
        rr.LineStrips3D(skeleton.kp_world[edges], colors=color, radii=0.012),
    )
    trail = trails.setdefault(person_id, [])
    trail.append(skeleton.kp_world[0].copy())
    rr.log(f"world/people/{person_id}/pelvis_trail", rr.LineStrips3D([np.stack(trail)], colors=color, radii=0.01))
    if skeleton.shape.shape == (10,) and skeleton.joints_rot_mat.shape == (24, 3, 3):
        _, vertices = lifter.forward_smpl_geometry(
            betas=skeleton.shape[None],
            global_orient_rotmat=skeleton.joints_rot_mat[None, 0],
            body_pose_rotmat=skeleton.joints_rot_mat[None, 1:],
            transl=skeleton.T_world_pelvis[None, :3, 3],
        )
        if lifter.smpl_faces is not None:
            rr.log(
                f"world/people/{person_id}/mesh",
                rr.Mesh3D(vertex_positions=vertices[0], triangle_indices=lifter.smpl_faces, albedo_factor=color),
            )


def main(config: Config) -> None:
    """Replay the fixture and print measured coverage, timing, and errors."""
    metrics = replay(config)
    print(f"fixture={fixture_path(config.fixture_dir)}")
    print(f"lift_calls={metrics.lift_calls}")
    print(f"person_inputs={metrics.person_inputs}")
    print(f"accepted_outputs={metrics.accepted_outputs}")
    print(f"acceptance_mismatches={metrics.acceptance_mismatches}")
    print(f"lifter_mean_ms={metrics.lifter_mean_ms:.3f}")
    print(f"max_raw_joint_error_m={metrics.max_raw_joint_error_m:.9f}")
    print(f"max_smoothed_joint_error_m={metrics.max_smoothed_joint_error_m:.9f}")


__all__ = (
    "Config",
    "LiftFixtureRecord",
    "ReplayMetrics",
    "fixture_lift_call_count",
    "fixture_path",
    "load_lift_record",
    "load_lift_records",
    "main",
    "replay",
    "skeleton_arrays",
)
