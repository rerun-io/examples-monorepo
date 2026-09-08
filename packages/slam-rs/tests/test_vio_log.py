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

The rig, the pipeline and the recording reader are :mod:`conftest` fixtures,
which pytest injects; the aliases below are declared here rather than imported
from it, because ``tests`` is not on the typechecker's search path and every
module in this directory therefore stands alone.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, TypeAlias

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core, vio_log
from slam_rs.catalog_feed import TIMELINE, CameraCalib
from slam_rs.trajectory import AteResult, Trajectory, ate, empty_trajectory
from slam_rs.vio_log import (
    CPP_ENTITY,
    GT_ENTITY,
    IDENTITY,
    RUN_ENTITY,
    VIO_STATS_ENTITY,
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


class Row(NamedTuple):
    """One logged row of one entity, as :func:`conftest.read_rows` hands it back."""

    t_ns: int
    """Where on ``video_time`` the row sits, in nanoseconds."""
    values: dict[str, list]
    """The components this row set, by their short name."""


Rows: TypeAlias = dict[str, list[Row]]
"""Per entity path, its non-static rows in ``video_time`` order."""
RowsReader: TypeAlias = Callable[[Path], Rows]
"""The :mod:`conftest` fixture that reads a recording back."""


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


def drive(
    vio: _core.Vio,
    cameras: tuple[CameraCalib, ...],
    references: Trajectory,
    texture: TextureFactory,
    output: Path,
    read_rows: RowsReader,
) -> Logged:
    """Log ``FRAMESETS`` framesets of the synthetic rig and read the recording back.

    Args:
        vio: The pipeline the framesets go through.
        cameras: The rig the logger draws.
        references: Ground truth and C++ trajectory, the same line for both.
        texture: The scene each frameset is a shifted copy of.
        output: Where the ``.rrd`` is written.
        read_rows: Reads the recording back, from :mod:`conftest`.

    Returns:
        The recording's rows, the framesets that tracked, and the logger.
    """
    rr.init("slam-rs-vio-log-test", recording_id=f"vio-log-{output.parent.name}")
    rr.save(output)
    frame_t_ns: Int64[ndarray, " n_frames"] = np.arange(0, FRAMESETS * FRAME_INTERVAL_NS, FRAME_INTERVAL_NS, dtype=np.int64)
    logger: VioLogger = VioLogger(cameras=cameras, ground_truth=references, cpp=references, frame_t_ns=frame_t_ns)
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


@pytest.fixture
def logged(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> Logged:
    """Drive the whole pipeline over the synthetic rig and read back what was logged."""
    reference_t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_INTERVAL_NS, FRAME_INTERVAL_NS, dtype=np.int64)
    return drive(pipeline(2), (camera(0, 0.0), camera(1, 0.1)), straight_line(reference_t_ns), texture, tmp_path / "vio.rrd", read_rows)


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
        f"{RUN_ENTITY}/marginalized",
        f"{RUN_ENTITY}/landmarks",
        f"{GT_ENTITY}/trajectory",
        f"{CPP_ENTITY}/trajectory",
        f"{VIO_STATS_ENTITY}/num_landmarks",
        f"{VIO_STATS_ENTITY}/lm_iterations",
        f"{VIO_STATS_ENTITY}/lm_error_before",
        f"{VIO_STATS_ENTITY}/lm_error_after",
        f"{VIO_STATS_ENTITY}/stage_ms/measure",
    ):
        assert entity in logged.rows, f"{entity} never reached the recording"
        assert [t_ns for t_ns, _ in logged.rows[entity]] == logged.tracked, entity
    # The window this run marginalizes from the fifth frameset on, and a
    # marginalized frame is drawn from the poses of the window it just left: a
    # row that is always empty is the layer looking the removed frames up in the
    # window they are already gone from.
    faded: list[int] = [len(values["LineStrips3D:strips"]) for _, values in logged.rows[f"{RUN_ENTITY}/marginalized"]]
    assert max(faded) > 0, "the marginalized layer never drew a frame the last marginalization removed"


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


def test_the_plotted_ate_is_the_estimate_driven_one(
    pipeline: PipelineFactory,
    texture: TextureFactory,
    camera: CameraFactory,
    tmp_path: Path,
    read_rows: RowsReader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plotted ``ate_cm`` is the number the tool prints and the gate asserts.

    A 1 kHz ground truth is what tells the two directions apart: driving the
    association from the reference pairs about 33 truth poses to every frameset
    and scores a different metric (5.32 cm against this run's 5.74 cm), which is
    neither what :func:`slam_rs.apis.replay.main` prints nor what the V2 gate
    asserts. One ATE a frameset, so a 12-frameset run reaches the association
    floor and plots.
    """
    monkeypatch.setattr(vio_log, "ATE_EVERY", 1)
    dense_t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_INTERVAL_NS, IMU_PERIOD_NS, dtype=np.int64)
    truth: Trajectory = straight_line(dense_t_ns)
    logged: Logged = drive(pipeline(2), (camera(0, 0.0), camera(1, 0.1)), truth, texture, tmp_path / "dense.rrd", read_rows)
    estimate: Trajectory = logged.logger.estimated()
    estimate_driven: AteResult = ate(estimate, truth)
    reference_driven: AteResult = ate(truth, estimate)
    assert estimate_driven.n_associated == len(estimate), "every estimate pose takes the nearest truth pose"
    assert reference_driven.n_associated > len(estimate), "the reference-driven association is the denser one"
    plotted: float = logged.rows[f"{VIO_STATS_ENTITY}/ate_cm/gt"][-1][1]["Scalars:scalars"][0]
    assert plotted == pytest.approx(100.0 * estimate_driven.rmse_m)
    assert plotted != pytest.approx(100.0 * reference_driven.rmse_m)


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
    rows: list[Row] = logged.rows[CPP_ENTITY]
    assert [row.t_ns for row in rows] == logged.tracked[:1]
    assert "Transform3D:translation" in rows[0].values


def test_the_keypoints_land_on_the_camera_images(logged: Logged) -> None:
    """The rung draws the estimator's own frontend output, on the frontend rung's paths."""
    for index in range(2):
        entity: str = f"/world/rig_00/cam_{index:02d}/pinhole/keypoints"
        assert [t_ns for t_ns, _ in logged.rows[entity]] == logged.tracked
        assert all(len(values["Points2D:positions"]) > 0 for _, values in logged.rows[entity])


def test_a_run_without_references_still_logs_everything_else(pipeline: PipelineFactory, camera: CameraFactory) -> None:
    """A segment with no ground truth and no C++ trajectory draws neither line, and no ATE."""
    logger: VioLogger = VioLogger(
        cameras=(camera(0, 0.0), camera(1, 0.1)),
        ground_truth=empty_trajectory(),
        cpp=empty_trajectory(),
        frame_t_ns=np.arange(0, FRAMESETS * FRAME_INTERVAL_NS, FRAME_INTERVAL_NS, dtype=np.int64),
    )
    assert len(logger.estimated()) == 0
    assert logger.window_strip.shape == (10, 3)


def views_of(node: rrb.Container | rrb.View) -> list[rrb.View]:
    """Every view under one blueprint node, in layout order."""
    if isinstance(node, rrb.api.Container):
        return [view for child in node.contents for view in views_of(child)]
    return [node]


def test_the_blueprint_covers_the_world_the_cameras_and_the_counters(camera: CameraFactory) -> None:
    """One 3D view, one 2D view per camera, one time-series view per unit, panels collapsed."""
    blueprint: rrb.Blueprint = vio_blueprint((camera(0, 0.0), camera(1, 0.1)))
    views: list[rrb.View] = views_of(blueprint.root_container)
    assert [str(view.origin) for view in views] == [
        "/world",
        "/world/rig_00/cam_00/pinhole",
        "/world/rig_00/cam_01/pinhole",
        *[VIO_STATS_ENTITY] * 7,
    ]
    assert [str(view.name) for view in views[3:]] == [
        "counts",
        "keyframes & LM steps",
        "timing (ms)",
        "solve stages (ms)",
        "LM cost",
        "LM damping",
        "ATE (cm)",
    ]
    assert blueprint.collapse_panels


def test_every_logged_counter_sits_in_exactly_one_time_series_view(logged: Logged, camera: CameraFactory) -> None:
    """A scalar nobody put in a view is the bug this partition is checked to catch.

    One view per unit only stays readable while every scalar is in one: a new
    counter with no view of its own would otherwise fall back into whichever
    axis Rerun's own ``$origin/**`` default reached it through, which is how the
    LM cost came to flatten thirteen other series.

    The ATE is added by hand because it is logged every
    :data:`slam_rs.vio_log.ATE_EVERY` framesets and this drive is shorter than
    that, so the recording carries no row of it to read back.
    """
    counters: set[str] = {entity for entity in logged.rows if entity.startswith(VIO_STATS_ENTITY)}
    counters |= {f"{VIO_STATS_ENTITY}/ate_cm/gt", f"{VIO_STATS_ENTITY}/ate_cm/cpp"}
    blueprint: rrb.Blueprint = vio_blueprint((camera(0, 0.0), camera(1, 0.1)))
    plotted: list[str] = []
    for view in views_of(blueprint.root_container):
        if str(view.origin) != VIO_STATS_ENTITY:
            continue
        # ``contents`` is a query expression in general; the narrowing is what
        # says these views name their entities one by one rather than globbing.
        contents: object = view.contents
        assert isinstance(contents, list), f"{view.name} does not list its entities"
        plotted += [str(entity) for entity in contents]
    assert sorted(plotted) == sorted(set(plotted)), "a counter is plotted in two views, so it is drawn against two axes"
    assert set(plotted) == counters, "every logged counter is plotted exactly once, and nothing else is"
