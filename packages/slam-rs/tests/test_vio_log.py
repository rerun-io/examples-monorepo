"""What the estimator's Rerun layer promises: the frustum geometry, and what ``log`` writes.

The logging contract is checked against a recording, not against a mock: the test
drives the whole pipeline over the synthetic rig, saves what
:class:`slam_rs.vio_log.VioLogger` logged into the global recording (D03: the
tools own no :class:`rerun.RecordingStream`) and reads the chunks back with
:class:`rerun.experimental.RrdReader` — the same rows a viewer would receive. An
entity path or a ``video_time`` value that never reached the file fails here.

The reference trajectories are synthetic straight lines rather than a segment's:
what is under test is that a reference is drawn up to the cursor and no further,
which a straight line says as well as a real one and in milliseconds.

The rig and the pipeline are :mod:`conftest` fixtures, which pytest injects; the
factory aliases and the recording reader below are declared here rather than
imported from another test module, because ``tests`` is not on the typechecker's
search path and every module in this directory therefore stands alone.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, TypeAlias

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
import rerun.experimental as rx
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import TIMELINE, CameraCalib
from slam_rs.trajectory import Trajectory, empty_trajectory
from slam_rs.vio_log import (
    CPP_ENTITY,
    GT_ENTITY,
    IDENTITY,
    RUN_ENTITY,
    STATS_ENTITY,
    VioLogger,
    alignment_onto,
    frustum_strip,
    vio_blueprint,
)

FRAME_INTERVAL_NS: int = 33_000_000
"""One 30 Hz frameset to the next."""
IMU_PERIOD_NS: int = 1_000_000
"""Synthetic IMU period: 1 kHz."""
FRAMESETS: int = 12
"""Enough framesets for the estimator to initialise, optimise and fill a window."""

CameraFactory: TypeAlias = Callable[[int, float], CameraCalib]
"""One camera of the synthetic rig, by rig index and baseline in metres."""
PipelineFactory: TypeAlias = Callable[[int], _core.Vio]
"""The whole pipeline on a rig of the given camera count."""
TextureFactory: TypeAlias = Callable[[int, int], UInt8[ndarray, "h w"]]
"""The synthetic scene, shifted by whole pixels in x and y."""
Rows: TypeAlias = dict[str, list[tuple[int, dict[str, list]]]]
"""Per entity path, its non-static rows as ``(video_time ns, components by short name)``."""


def read_rows(path: Path) -> Rows:
    """Read every non-static row of a recording, grouped by entity path.

    A component a row did not set comes back as a null and is dropped, so a
    missing key here means the row really did not carry that component.

    Args:
        path: An ``.rrd`` written by :func:`rerun.save`.

    Returns:
        Entity path to its rows, in ``video_time`` order.
    """
    rows: Rows = {}
    for chunk in rx.RrdReader(path).stream().collect().stream():
        if chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        if TIMELINE not in batch.schema.names:
            # A row written before the caller set a cursor sits on no timeline.
            continue
        times: list = batch.column(TIMELINE).to_pylist()
        components: dict[str, list] = {
            name: batch.column(name).to_pylist() for name in batch.schema.names if ":" in name and not name.startswith("rerun.")
        }
        for index, time in enumerate(times):
            values: dict[str, list] = {name: column[index] for name, column in components.items() if column[index] is not None}
            rows.setdefault(chunk.entity_path, []).append((int(np.timedelta64(time, "ns").astype(np.int64)), values))
    for entity in rows:
        rows[entity].sort(key=lambda row: row[0])
    return rows


class Logged(NamedTuple):
    """What one logged run gives back."""

    rows: Rows
    """Every non-static row of the recording, by entity path."""
    tracked: list[int]
    """``video_time`` of every frameset that tracked, in order."""
    logger: VioLogger
    """The logger that produced them, for its accumulated estimate."""


def straight_line(t_ns: Int64[ndarray, " n"]) -> Trajectory:
    """A trajectory running along +x at 1 m/s, one pose per timestamp."""
    seconds: Float64[ndarray, " n"] = t_ns.astype(np.float64) / 1e9
    return Trajectory(
        t_ns=t_ns,
        position_m=np.column_stack([seconds, np.zeros_like(seconds), np.zeros_like(seconds)]),
        quaternion_wxyz=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(t_ns), 1)),
    )


@pytest.fixture
def logged(pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path) -> Logged:
    """Drive the whole pipeline over the synthetic rig and read back what was logged."""
    vio: _core.Vio = pipeline(2)
    cameras: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    reference_t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_INTERVAL_NS, FRAME_INTERVAL_NS, dtype=np.int64)
    output: Path = tmp_path / "vio.rrd"
    rr.init("slam-rs-vio-log-test", recording_id="vio-log")
    rr.save(output)
    logger: VioLogger = VioLogger(cameras=cameras, ground_truth=straight_line(reference_t_ns), cpp=straight_line(reference_t_ns))
    tracked: list[int] = []
    for step in range(FRAMESETS):
        t_ns: int = step * FRAME_INTERVAL_NS
        vio.push_imu_batch(
            np.arange(t_ns, t_ns + FRAME_INTERVAL_NS, IMU_PERIOD_NS, dtype=np.int64),
            np.zeros((FRAME_INTERVAL_NS // IMU_PERIOD_NS, 3), dtype=np.float64),
            np.tile(np.array([0.0, 0.0, 9.81]), (FRAME_INTERVAL_NS // IMU_PERIOD_NS, 1)),
        )
        rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
        # Camera 1 is the scene shifted one pixel along the baseline, so the
        # stereo pass matches: a shift across it matches nothing and camera 1
        # would carry no keypoint at all.
        result: _core.VioResult = vio.track(t_ns, [texture(step, 0), texture(step + 1, 0)])
        if result.status != _core.VioStatus.Tracking:
            continue
        snapshot: _core.VioSnapshot | None = vio.snapshot()
        frame: _core.FlowFrame | None = vio.flow_frame()
        assert snapshot is not None
        assert frame is not None
        logger.log(result, snapshot, frame, elapsed_ms=1.5)
        tracked.append(t_ns)
    rr.disconnect()
    return Logged(rows=read_rows(output), tracked=tracked, logger=logger)


def test_the_frustum_wireframe_sits_where_the_camera_does(camera: CameraFactory) -> None:
    """Apex at the camera's origin in rig coordinates, corners a fixed depth in front."""
    offset: float = 0.1
    strip: Float64[ndarray, "10 3"] = frustum_strip(camera(1, offset), depth_m=0.5)
    assert strip.shape == (10, 3)
    # The rig transform of the synthetic camera is a pure x translation, so the
    # apex — the only point at the camera origin — lands on it exactly.
    apex: Float64[ndarray, "n 3"] = strip[np.array([5, 8])]
    np.testing.assert_allclose(apex, np.tile([offset, 0.0, 0.0], (2, 1)), atol=1e-12)
    corners: Float64[ndarray, "n 3"] = np.delete(strip, [5, 8], axis=0)
    assert np.all(corners[:, 2] == 0.5), "every corner sits at the requested depth along +z"
    # A closed rectangle: the strip returns to the corner it started from.
    np.testing.assert_allclose(strip[0], strip[4])


def test_a_deeper_frustum_is_the_same_shape_scaled(camera: CameraFactory) -> None:
    """Depth is a scale on the rays, so the wireframe never changes direction."""
    near: Float64[ndarray, "10 3"] = frustum_strip(camera(0, 0.0), depth_m=0.05)
    far: Float64[ndarray, "10 3"] = frustum_strip(camera(0, 0.0), depth_m=0.5)
    np.testing.assert_allclose(10.0 * near, far, atol=1e-12)


def test_every_tracked_frameset_writes_the_rung(logged: Logged) -> None:
    """One row per tracked frameset on each of the rung's entities, at its own time."""
    assert len(logged.tracked) >= FRAMESETS - 2, "only the framesets before the first covered one may fail to track"
    for entity in (
        f"{RUN_ENTITY}/rig",
        f"{RUN_ENTITY}/trajectory",
        f"{RUN_ENTITY}/window",
        f"{RUN_ENTITY}/landmarks",
        f"{GT_ENTITY}/trajectory",
        f"{CPP_ENTITY}/trajectory",
        f"{STATS_ENTITY}/num_landmarks",
        f"{STATS_ENTITY}/lm_iterations",
        f"{STATS_ENTITY}/stage_ms/measure",
    ):
        assert entity in logged.rows, f"{entity} never reached the recording"
        assert [t_ns for t_ns, _ in logged.rows[entity]] == logged.tracked, entity


def test_the_estimated_path_grows_by_one_pose_a_frameset(logged: Logged) -> None:
    """The strip is the whole path so far, which is what makes the three lines comparable."""
    lengths: list[int] = [len(values["LineStrips3D:strips"][0]) for _, values in logged.rows[f"{RUN_ENTITY}/trajectory"]]
    assert lengths == list(range(1, len(logged.tracked) + 1))
    assert len(logged.logger.estimated()) == len(logged.tracked)


def test_a_reference_is_drawn_up_to_the_cursor_and_no_further(logged: Logged) -> None:
    """A reference known in advance still stops where the estimate has got to."""
    for entity in (f"{GT_ENTITY}/trajectory", f"{CPP_ENTITY}/trajectory"):
        for t_ns, values in logged.rows[entity]:
            drawn: list = values["LineStrips3D:strips"][0]
            # One reference pose every frame interval, from zero, inclusive.
            assert len(drawn) == t_ns // FRAME_INTERVAL_NS + 1, entity


def test_an_alignment_recovers_a_known_rigid_offset() -> None:
    """The transform a run carries is the one that takes it onto the ground truth."""
    t_ns: Int64[ndarray, " n"] = np.arange(0, 40 * FRAME_INTERVAL_NS, FRAME_INTERVAL_NS, dtype=np.int64)
    source: Trajectory = straight_line(t_ns)
    turn: Float64[ndarray, "3 3"] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    offset: Float64[ndarray, " 3"] = np.array([3.0, -2.0, 0.5])
    target: Trajectory = Trajectory(
        t_ns=t_ns,
        position_m=source.position_m @ turn.T + offset,
        quaternion_wxyz=source.quaternion_wxyz,
    )
    recovered = alignment_onto(source, target)
    np.testing.assert_allclose(recovered.dst_R_src, turn, atol=1e-9)
    np.testing.assert_allclose(recovered.dst_t_src, offset, atol=1e-9)
    np.testing.assert_allclose(recovered.apply(source.position_m), target.position_m, atol=1e-9)


def test_too_short_a_run_carries_no_alignment() -> None:
    """Below the association floor the identity is honest: nothing has been measured yet."""
    short: Trajectory = straight_line(np.arange(0, 3 * FRAME_INTERVAL_NS, FRAME_INTERVAL_NS, dtype=np.int64))
    long: Trajectory = straight_line(np.arange(0, 40 * FRAME_INTERVAL_NS, FRAME_INTERVAL_NS, dtype=np.int64))
    assert alignment_onto(short, long) is IDENTITY
    assert alignment_onto(long, empty_trajectory()) is IDENTITY


def test_the_cpp_reference_is_placed_once_at_the_first_tracked_frameset(logged: Logged) -> None:
    """Both it and the ground truth are known up front, so its alignment never changes."""
    rows: list[tuple[int, dict[str, list]]] = logged.rows[CPP_ENTITY]
    assert [t_ns for t_ns, _ in rows] == logged.tracked[:1]
    assert "Transform3D:translation" in rows[0][1]


def test_the_keypoints_land_on_the_camera_images(logged: Logged) -> None:
    """The rung draws the estimator's own frontend output, on the frontend rung's paths."""
    for index in range(2):
        entity: str = f"/world/rig_00/cam_{index:02d}/pinhole/keypoints"
        assert [t_ns for t_ns, _ in logged.rows[entity]] == logged.tracked
        assert all(len(values["Points2D:positions"]) > 0 for _, values in logged.rows[entity])


def test_a_run_without_references_still_logs_everything_else(pipeline: PipelineFactory, camera: CameraFactory) -> None:
    """A segment with no ground truth and no C++ trajectory draws neither line, and no ATE."""
    logger: VioLogger = VioLogger(cameras=(camera(0, 0.0), camera(1, 0.1)), ground_truth=empty_trajectory(), cpp=empty_trajectory())
    assert len(logger.estimated()) == 0
    assert logger.window_strip.shape == (10, 3)


def view_origins(node: rrb.Container | rrb.View) -> list[str]:
    """Every view origin under one blueprint node, in layout order."""
    if isinstance(node, rrb.api.Container):
        return [origin for child in node.contents for origin in view_origins(child)]
    return [str(node.origin)]


def test_the_blueprint_covers_the_world_the_cameras_and_the_counters(camera: CameraFactory) -> None:
    """One 3D view, one 2D view per camera, one time-series view, panels collapsed."""
    blueprint: rrb.Blueprint = vio_blueprint((camera(0, 0.0), camera(1, 0.1)))
    assert view_origins(blueprint.root_container) == [
        "/world",
        "/world/rig_00/cam_00/pinhole",
        "/world/rig_00/cam_01/pinhole",
        STATS_ENTITY,
    ]
    assert blueprint.collapse_panels
