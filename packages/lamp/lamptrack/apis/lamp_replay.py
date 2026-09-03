"""Replay recorded LAMP lifter inputs without images or Aria dependencies."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from lamptrack.third_party.lamp.core.types import SMPL_SKELETON_EDGES, Detection2D
from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings, SnippetData
from lamptrack.third_party.lamp.tracking.tracker import LampTracker

_PACKAGE_ROOT = Path(__file__).parents[2]
_FIXTURE_NAME = "lamp_fixture.npz"


@dataclass(frozen=True, slots=True)
class Config:
    """Paths and Rerun output for fixture replay."""

    fixture_dir: Path = _PACKAGE_ROOT / "data" / "fixtures" / "test-library"
    """Directory containing the fork-recorded ``lamp_fixture.npz``."""
    checkpoint: Path = _PACKAGE_ROOT / "data" / "checkpoints" / "lamp_smpl_aria_gen2.pt"
    """Pinned released LAMP checkpoint."""
    smpl_model_path: Path = _PACKAGE_ROOT / "data" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"
    """Chumpy-free neutral SMPL body model."""
    device: str = "cuda"
    """Torch device for lifter inference."""
    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(application_id="lamp_fixture_replay", headless=True))
    """Viewer, connection, or save configuration."""


def fixture_path(fixture_dir: Path) -> Path:
    """Resolve the required replay archive or fail with download guidance."""
    path = fixture_dir / _FIXTURE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"LAMP replay fixture is missing: {path}\n"
            "Run `pixi run -e lamp _lamp-download-fixture` after the reviewer publishes "
            "pablovela5620/lamp-fixtures/test-library/."
        )
    return path


def load_snippets(path: Path) -> tuple[dict[int, SnippetData], ndarray | None, ndarray | None]:
    """Load the stable NPZ fixture contract emitted by the pixified fork.

    The archive stores ``person_ids`` and ``timestamps_ns`` plus three arrays
    per view: ``keypoints_vN``, ``camera_params_vN``, and ``camera_poses_vN``.
    Optional ``expected_skel_w`` and ``expected_smoothed_joints`` arrays support
    the slow equivalence test.
    """
    with np.load(path, allow_pickle=False) as fixture:
        required = {"person_ids", "timestamps_ns"} | {
            f"{prefix}_v{view}" for prefix in ("keypoints", "camera_params", "camera_poses") for view in range(4)
        }
        missing = sorted(required - set(fixture.files))
        if missing:
            raise ValueError(f"Fixture {path} is missing required arrays: {missing}")
        person_ids = np.asarray(fixture["person_ids"], dtype=np.int64)
        timestamps = np.asarray(fixture["timestamps_ns"], dtype=np.int64)
        if timestamps.ndim == 1:
            timestamps = np.broadcast_to(timestamps[None], (len(person_ids), len(timestamps)))
        snippets: dict[int, SnippetData] = {}
        for row, person_id_raw in enumerate(person_ids):
            person_id = int(person_id_raw)
            snippets[person_id] = SnippetData(
                person_id=person_id,
                snippet_timestamps_ns=[int(value) for value in timestamps[row]],
                view_cam_indices=[0, 1, 2, 3],
                kp2ds_per_view=[np.asarray(fixture[f"keypoints_v{view}"][row], dtype=np.float32) for view in range(4)],
                Ts_gw_cam_per_view=[np.asarray(fixture[f"camera_poses_v{view}"][row], dtype=np.float32) for view in range(4)],
                cam_params_per_view=[np.asarray(fixture[f"camera_params_v{view}"][row], dtype=np.float32) for view in range(4)],
                T_gravityWorld_world=np.eye(4, dtype=np.float32),
            )
        expected_lifter = np.asarray(fixture["expected_skel_w"], dtype=np.float32) if "expected_skel_w" in fixture else None
        expected_smoothed = (
            np.asarray(fixture["expected_smoothed_joints"], dtype=np.float32) if "expected_smoothed_joints" in fixture else None
        )
    return snippets, expected_lifter, expected_smoothed


def replay(config: Config) -> tuple[ndarray, ndarray]:
    """Run the recorded snippets through the released lifter and LAMP smoother."""
    path = fixture_path(config.fixture_dir)
    snippets, _, _ = load_snippets(path)
    if not snippets:
        raise ValueError(f"Fixture {path} contains no people.")
    steps = len(next(iter(snippets.values())).snippet_timestamps_ns)
    lifter = Lifter.from_checkpoint(
        config.checkpoint,
        config.smpl_model_path,
        device=config.device,
        settings=LifterSettings(snippet_length=steps),
        capture_cuda_graph=False,
    )
    tracker = LampTracker(num_cameras=4)
    identity = np.eye(4, dtype=np.float32)
    fixture_ids = list(snippets)
    for time_index in range(steps):
        timestamp_ns = snippets[fixture_ids[0]].snippet_timestamps_ns[time_index]
        detections = []
        for person_index, fixture_id in enumerate(fixture_ids):
            x0 = float(100 * person_index)
            keypoints = snippets[fixture_id].kp2ds_per_view[0][time_index]
            detections.append(
                Detection2D(
                    box_xyxy=np.array([x0, 0.0, x0 + 50.0, 100.0], dtype=np.float32),
                    box_score=1.0,
                    keypoints=keypoints.copy(),
                    cam_idx=0,
                    timestamp_ns=timestamp_ns,
                )
            )
        tracker.track_frameset({0: detections}, {0: identity}, {0: None}, timestamp_ns)

    internal_ids = sorted(tracker.people)
    if len(internal_ids) != len(fixture_ids):
        raise RuntimeError(f"Fixture replay created {len(internal_ids)} tracks for {len(fixture_ids)} people.")
    remapped = {internal_id: snippets[fixture_id] for internal_id, fixture_id in zip(internal_ids, fixture_ids, strict=True)}
    lifted = lifter.lift_all_steps_batched(remapped)
    for person_id, skeletons in lifted.items():
        tracker.attach_skeletons(person_id, skeletons, {}, np.eye(4, dtype=np.float32), min_pose_depth=0.0, max_pose_depth=float("inf"))

    lifted_rows = []
    smoothed_rows = []
    edges = np.asarray(SMPL_SKELETON_EDGES, dtype=np.int32)
    for person_id in internal_ids:
        person = tracker.people[person_id]
        rows = [(timestamp, state.skeleton) for timestamp, state in sorted(person.ts_to_states.items()) if state.skeleton is not None]
        person_lifted = np.stack([skeleton.kp_world for _, skeleton in lifted[person_id]])
        person_smoothed = np.stack([skeleton.kp_world for _, skeleton in rows])
        lifted_rows.append(person_lifted)
        smoothed_rows.append(person_smoothed)
        trail = []
        for timestamp, skeleton in rows:
            rr.set_time("fixture_time", timestamp=timestamp * 1e-9)
            points = skeleton.kp_world
            trail.append(points[0].copy())
            rr.log(f"world/people/{person_id}/skeleton", rr.Points3D(points), rr.LineStrips3D(points[edges]))
            rr.log(f"world/people/{person_id}/pelvis_trail", rr.LineStrips3D([np.asarray(trail)]))
            vertices = skeleton.verts_w
            if vertices.shape == (6890, 3) and lifter.smpl_faces is not None:
                rr.log(f"world/people/{person_id}/mesh", rr.Mesh3D(vertex_positions=vertices, triangle_indices=lifter.smpl_faces))
    return np.stack(lifted_rows), np.stack(smoothed_rows)


def main(config: Config) -> None:
    """Replay the fixture and report its exact output dimensions."""
    lifted, smoothed = replay(config)
    print(f"fixture={fixture_path(config.fixture_dir)}")
    print(f"lifter_output_shape={lifted.shape}")
    print(f"smoothed_joints_shape={smoothed.shape}")


__all__ = ("Config", "fixture_path", "load_snippets", "main", "replay")
