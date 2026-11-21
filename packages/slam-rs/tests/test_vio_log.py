"""What the estimator's Rerun layer promises: the frustum geometry, and what ``log`` writes.

The logging contract is checked against a recording, not against a mock: the test
drives the whole pipeline over the synthetic rig, saves what
:class:`slam_rs.vio_log.VioLogger` logged into the global recording (D03: the
tools own no :class:`rerun.RecordingStream`) and reads the chunks back with
:class:`rerun.chunk.RrdReader` — the same rows a viewer would receive. An
entity path or a ``video_time`` value that never reached the file fails here.

The reference trajectories are synthetic straight lines rather than a segment's:
what is under test is that a reference is drawn up to the cursor and no further,
which a straight line says as well as a real one and in milliseconds.

The rig, the pipeline and the recording reader are :mod:`conftest` fixtures,
which pytest injects; the aliases below are declared here rather than imported
from it, because ``tests`` is not on the typechecker's search path and every
module in this directory therefore stands alone.
"""

import dataclasses
import itertools
import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from fixture_types import (
    FRAME,
    FRAME_PERIOD_NS,
    IMU_PERIOD_NS,
    CameraFactory,
    PipelineFactory,
    RigFactory,
    Row,
    Rows,
    RowsReader,
    TextureFactory,
    gravity_batch,
)
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform

from slam_rs import _core, vio_log
from slam_rs.catalog_feed import RIG_ENTITY, TIMELINE, CameraCalib
from slam_rs.follow_frame import FloorHeight, floor_entity, follow_entity
from slam_rs.landmark_colors import pack_rgba
from slam_rs.trajectory import AteResult, Trajectory, ate, empty_trajectory
from slam_rs.vio_log import (
    FRUSTUM_DEPTH_M,
    GT_ENTITY,
    IMAGE_PLANE_M,
    KEYFRAME_COLOR,
    LTKF_COLOR,
    MAP_KEYFRAMES_ENTITY,
    MAP_KEYFRAMES_EVERY_NS,
    MAP_POINT_BLOCKS_ENTITY,
    MAP_POINTS_ENTITY,
    MAP_POINTS_EVERY_NS,
    MARGINALIZED_COLOR,
    POSE_COLOR,
    RUN_ENTITY,
    VIO_RUN_ENTITY,
    VIO_STATS_ENTITY,
    VioLogger,
    dataset_videos,
    display_intrinsics,
    image_from_camera,
    inverted,
    log_rig,
    vio_blueprint,
)

FRAMESETS: int = 12
"""Enough framesets for the estimator to initialise, optimise and fill a window."""


class Logged(NamedTuple):
    """What one logged run gives back."""

    rows: Rows
    """Every non-static row of the recording, by entity path."""
    tracked: list[int]
    """``video_time`` of every frameset that tracked, in order."""
    logger: VioLogger
    """The logger that produced them, for its accumulated estimate."""
    recording: Path
    """The ``.rrd`` itself, for the rows :func:`conftest.read_rows` drops: the static ones."""
    newest_keyframes: list[int]
    """The window's newest keyframe at each tracked frameset (-1 with none), in order."""


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
    map_every_ns: tuple[int, int] = (MAP_KEYFRAMES_EVERY_NS, MAP_POINTS_EVERY_NS),
    finish: Callable[[VioLogger], None] | None = None,
    colors: list[np.ndarray] | None = None,
    animate_base_rig: bool = False,
    video_entities: tuple[str, ...] = (),
) -> Logged:
    """Log ``FRAMESETS`` framesets of the synthetic rig and read the recording back.

    Args:
        vio: The pipeline the framesets go through.
        cameras: The rig the logger draws.
        references: Synthetic ground-truth trajectory.
        texture: The scene each frameset is a shifted copy of.
        output: Where the ``.rrd`` is written.
        read_rows: Reads the recording back, from :mod:`conftest`.
        map_every_ns: The logger's map-keyframes and map-points cadences.
        finish: Called with the logger at the last cursor, after the complete paths (what a replay ends with).
        colors: Each frameset's RGB frames (``Frameset.colors``); None logs without them.
        animate_base_rig: The logger's :attr:`VioLogger.animate_base_rig`.
        video_entities: The logger's :attr:`VioLogger.video_entities`.

    Returns:
        The recording's rows, the framesets that tracked, and the logger.
    """
    rr.init("slam-rs-vio-log-test", recording_id=f"vio-log-{output.parent.name}")
    rr.save(output)
    frame_t_ns: Int64[ndarray, " n_frames"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    logger: VioLogger = VioLogger(
        cameras=cameras,
        ground_truth=references,
        frame_t_ns=frame_t_ns,
        tracked_resolution=tuple(vio.tracked_resolution),
        cam_time_offset_ns=0,
        animate_base_rig=animate_base_rig,
        video_entities=video_entities,
        map_keyframes_every_ns=map_every_ns[0],
        map_points_every_ns=map_every_ns[1],
    )
    tracked: list[int] = []
    newest_keyframes: list[int] = []
    for step in range(FRAMESETS):
        t_ns: int = step * FRAME_PERIOD_NS
        samples: Int64[ndarray, " n_samples"] = np.arange(t_ns, t_ns + FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
        vio.push_imu_batch(samples, *gravity_batch(samples))
        rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
        # Camera 1 is the scene shifted one pixel along the baseline, so the
        # stereo pass matches: a shift across it matches nothing and camera 1
        # would carry no keypoint at all.
        result: _core.VioResult = vio.track(t_ns, [texture(step, 0), texture(step - 2, 0)])
        if result.status != _core.VioStatus.Tracking:
            continue
        snapshot: _core.VioSnapshot | None = vio.snapshot()
        frame: _core.FlowFrame | None = vio.flow_frame()
        assert snapshot is not None
        assert frame is not None
        # What `VioStage` does: read the map only when it is due (None with the back end off).
        map_delta: _core.MapDelta | None = vio.map_delta(logger.map_revision) if logger.map_is_due(snapshot) else None
        logger.log(result, snapshot, frame, elapsed_ms=1.5, map_delta=map_delta, colors=colors)
        tracked.append(t_ns)
        newest_keyframes.append(int(snapshot.kf_ids[-1]) if len(snapshot.kf_ids) else -1)
    # What `VioStage.flush` does: flush (which makes the map current), then draw the final map.
    vio.flush()
    final_map: _core.MapSnapshot | None = vio.map_snapshot()
    if final_map is not None and tracked:
        logger.log_final_map(final_map, tracked[-1])
    # What a replay ends with, and the other half of the segment form: the rows
    # above are one segment each, these are the whole paths, static.
    logger.log_complete_paths()
    if finish is not None:
        finish(logger)
    rr.disconnect()
    return Logged(rows=read_rows(output), tracked=tracked, logger=logger, recording=output, newest_keyframes=newest_keyframes)


@pytest.fixture
def logged(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> Logged:
    """Drive the whole pipeline over the synthetic rig and read back what was logged."""
    reference_t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    return drive(pipeline(2), (camera(0, 0.0), camera(1, 0.1)), straight_line(reference_t_ns), texture, tmp_path / "vio.rrd", read_rows)


def image_planes(recording: Path) -> dict[str, float]:
    """Every ``Pinhole``'s image-plane distance in one recording, by entity path.

    Read off the static chunks, which is why this is not :func:`conftest.read_rows`:
    a rig's geometry is logged once and for all time, not at a cursor.
    """
    planes: dict[str, float] = {}
    for chunk in rrc.RrdReader(recording).stream().collect().stream():
        batch = chunk.to_record_batch()
        if "Pinhole:image_plane_distance" in batch.schema.names:
            planes[chunk.entity_path] = batch.column("Pinhole:image_plane_distance").to_pylist()[0][0]
    return planes


def static_path_lengths(recording: Path) -> dict[str, int]:
    """Every static line strip in one recording: how many points it has, by entity path.

    Static like the rig geometry above and read the same way, because
    :func:`conftest.read_rows` drops exactly the rows this asks about.
    """
    lengths: dict[str, int] = {}
    for chunk in rrc.RrdReader(recording).stream().collect().stream():
        batch = chunk.to_record_batch()
        if chunk.is_static and "LineStrips3D:strips" in batch.schema.names:
            lengths[chunk.entity_path] = len(batch.column("LineStrips3D:strips").to_pylist()[0][0])
    return lengths


def test_every_rig_camera_pins_its_frustum_to_one_size(camera: CameraFactory, tmp_path: Path) -> None:
    """Both drawn rigs name the image-plane distance, so a frustum keeps its size all replay.

    Rerun's own default is a heuristic on the scene bounds, so the frusta grew as
    the landmarks came in. Both rigs have the dataset's layout, camera then
    ``pinhole``: the dataset's own (:func:`slam_rs.vio_log.log_calibration`) and the
    estimated run's copy, whose frames hang under that ``pinhole`` too.
    """
    output: Path = tmp_path / "rig.rrd"
    rr.init("slam-rs-vio-log-test", recording_id=f"rig-{tmp_path.name}")
    rr.save(output)
    log_rig((camera(0, 0.0), camera(1, 0.1)), f"{VIO_RUN_ENTITY}/rig", display=True)
    log_rig((camera(0, 0.0), camera(1, 0.1)), RIG_ENTITY)
    rr.disconnect()
    # Float32 in the file, so approximately 0.1 is the most the component can say.
    assert image_planes(output) == pytest.approx(
        {f"{prefix}/cam_{index:02d}/pinhole": IMAGE_PLANE_M for prefix in (f"{VIO_RUN_ENTITY}/rig", RIG_ENTITY) for index in range(2)}
    )


def test_a_drawn_frustum_of_a_wide_lens_takes_its_paraxial_focal(camera: CameraFactory) -> None:
    """A unified-sphere lens projects ``gamma / (1 + xi)`` pixels per radian at its axis, not ``gamma``: a frustum drawn
    with the model's own focal is ``1 + xi`` times too narrow, and so is the picture inside it. The X4 Air's lens 0
    (gamma 1835.37 px, xi 2.31494) draws at 553.67 px; a kb4 focal is paraxial already."""
    mei: CameraCalib = dataclasses.replace(
        camera(0, 0.0), fx=1835.37, fy=1835.01, model="mei_v6", distortion=np.array([2.31494, 1.7453293, *np.zeros(13)], dtype=np.float64)
    )
    np.testing.assert_allclose(np.diag(display_intrinsics(mei))[:2], [553.67, 553.56], atol=0.01)
    np.testing.assert_allclose(np.diag(image_from_camera(mei))[:2], [1835.37, 1835.01])
    np.testing.assert_allclose(display_intrinsics(camera(0, 0.0)), image_from_camera(camera(0, 0.0)))


def video_references(rows: Rows, entity: str) -> list[tuple[int, int]]:
    """Each row of a frame-reference entity as (row time, referenced video time), nanoseconds."""
    return [(t_ns, int(values["VideoFrameReference:timestamp"][0])) for t_ns, values in rows.get(entity, []) if "VideoFrameReference:timestamp" in values]


def test_the_dataset_videos_are_the_fed_cameras_own_streams() -> None:
    """A replay references the videos of the rig cameras it was fed, at the dataset's paths (exoego:v2 section 3)."""
    assert dataset_videos((0, 1)) == ("/world/rig_00/cam_00/pinhole/video", "/world/rig_00/cam_01/pinhole/video")
    assert dataset_videos((2,)) == ("/world/rig_00/cam_02/pinhole/video",)


def test_the_live_rig_shows_each_lens_frame_in_its_frustum(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """Told where the dataset's videos are, the run's rig references the current frame of each lens under its
    ``pinhole`` (camera, pinhole, video: the dataset's own layout), a row a frameset. Told nothing, it logs none."""
    rig: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    videos: tuple[str, ...] = (f"{RIG_ENTITY}/cam_00/pinhole/video", f"{RIG_ENTITY}/cam_01/pinhole/video")
    shown: Logged = drive(pipeline(2), rig, empty_trajectory(), texture, tmp_path / "shown.rrd", read_rows, video_entities=videos)
    for index in range(2):
        entity: str = f"{VIO_RUN_ENTITY}/rig/cam_{index:02d}/pinhole/video"
        assert video_references(shown.rows, entity) == [(t_ns, t_ns) for t_ns in shown.tracked], entity
    plain: Logged = drive(pipeline(2), rig, empty_trajectory(), texture, tmp_path / "plain.rrd", read_rows)
    assert not any(entity.endswith("/video") for entity in plain.rows)


def test_a_window_keyframe_shows_both_lenses_at_its_own_time_and_keeps_its_slot(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """Every window frame is two frusta, one a lens. A keyframe's hold the lens frames of the keyframe's own time, and
    the keyframe keeps its slot while it is in the window: its frames are referenced once, so the viewer decodes them once."""
    rig: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    videos: tuple[str, ...] = (f"{RIG_ENTITY}/cam_00/pinhole/video", f"{RIG_ENTITY}/cam_01/pinhole/video")
    shown: Logged = drive(pipeline(2), rig, empty_trajectory(), texture, tmp_path / "shown.rrd", read_rows, video_entities=videos)
    slots: list[str] = slot_entities(shown.rows, f"{VIO_RUN_ENTITY}/window")
    keyframes: set[int] = set(shown.newest_keyframes) - {-1}
    assert keyframes, "the synthetic run never added a keyframe"
    held: dict[int, str] = {}
    for slot in slots:
        for index in range(2):
            assert any("Pinhole:color" in values for _, values in shown.rows[f"{slot}/cam_{index:02d}/pinhole"]), (slot, index)
        lens0: list[tuple[int, int]] = video_references(shown.rows, f"{slot}/cam_00/pinhole/video")
        assert lens0 == video_references(shown.rows, f"{slot}/cam_01/pinhole/video"), "both lenses show the same instant"
        for logged_at, frame_ns in lens0:
            assert frame_ns in keyframes and frame_ns <= logged_at, (slot, frame_ns)
            assert frame_ns not in held, (frame_ns, "one keyframe, one slot, referenced once")
            held[frame_ns] = slot
    assert set(held) == keyframes, "every keyframe the window held shows its frames"


def test_every_tracked_frameset_writes_the_rung(logged: Logged) -> None:
    """One row per tracked frameset on each of the rung's entities, at its own time.

    The three ``trajectory`` entities are not here: they are drawn one segment at
    a time and a line with no second pose yet writes nothing, which is what the
    two path tests below pin.
    """
    assert len(logged.tracked) >= FRAMESETS - 2, "only the framesets before the first covered one may fail to track"
    for entity in (
        f"{VIO_RUN_ENTITY}/rig",
        f"{VIO_RUN_ENTITY}/window/00",
        f"{VIO_RUN_ENTITY}/landmarks",
        f"{VIO_STATS_ENTITY}/num_landmarks",
        f"{VIO_STATS_ENTITY}/lm_iterations",
        f"{VIO_STATS_ENTITY}/lm_error_before",
        f"{VIO_STATS_ENTITY}/lm_error_after",
        f"{VIO_STATS_ENTITY}/stage_ms/measure",
    ):
        assert entity in logged.rows, f"{entity} never reached the recording"
        assert [t_ns for t_ns, _ in logged.rows[entity]] == logged.tracked, entity


def test_the_estimated_path_gains_one_segment_a_frameset(logged: Logged) -> None:
    """Each frameset writes the pair of poses it joined, not the path so far.

    The whole path is what a viewer shows, and it is the view's visible time
    range that assembles it (:func:`slam_rs.vio_log.vio_blueprint`); the rows
    themselves are two points each, which is what makes the rung linear. The
    first tracked frameset has one pose and no pair, so it writes no row.
    """
    rows: list[Row] = logged.rows[f"{VIO_RUN_ENTITY}/trajectory"]
    assert [t_ns for t_ns, _ in rows] == logged.tracked[1:]
    assert [len(values["LineStrips3D:strips"][0]) for _, values in rows] == [2] * len(rows)
    assert len(logged.logger.estimated()) == len(logged.tracked)


def test_a_reference_segment_ends_at_the_cursor_and_no_further(logged: Logged) -> None:
    """A reference known in advance still stops where the estimate has got to."""
    for entity in (f"{GT_ENTITY}/trajectory",):
        rows: list[Row] = logged.rows[entity]
        # One reference pose every frame interval, from zero, so a cursor before
        # the second one has no segment to draw.
        assert [t_ns for t_ns, _ in rows] == [t_ns for t_ns in logged.tracked if t_ns >= FRAME_PERIOD_NS], entity
        for t_ns, values in rows:
            drawn: list = values["LineStrips3D:strips"][0]
            assert len(drawn) == 2, entity
            # The reference runs along +x at 1 m/s from zero, so the pose the
            # segment ends on is the cursor's own time in seconds — never the
            # one after it.
            assert drawn[-1][0] == pytest.approx(t_ns / 1e9), entity


def test_the_whole_paths_are_logged_once_at_the_end(logged: Logged) -> None:
    """One static row per path, so a viewer opened at any cursor sees where each run went."""
    strips: dict[str, int] = static_path_lengths(logged.recording)
    assert sorted(strips) == sorted([f"{VIO_RUN_ENTITY}/path", f"{GT_ENTITY}/path"])
    assert strips[f"{VIO_RUN_ENTITY}/path"] == len(logged.tracked)
    assert strips[f"{GT_ENTITY}/path"] == FRAMESETS


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
    dense_t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
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
    t_ns: Int64[ndarray, " n"] = np.arange(0, 40 * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    source: Trajectory = straight_line(t_ns)
    turn: Float64[ndarray, "3 3"] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    offset: Float64[ndarray, " 3"] = np.array([3.0, -2.0, 0.5])
    target: Trajectory = Trajectory(
        t_ns=t_ns,
        position_m=source.position_m @ turn.T + offset,
        quaternion_wxyz=source.quaternion_wxyz,
    )
    recovered: SimilarityTransform = inverted(ate(source, target).alignment)
    np.testing.assert_allclose(recovered.dst_R_src, turn, atol=1e-9)
    np.testing.assert_allclose(recovered.dst_t_src, offset, atol=1e-9)
    np.testing.assert_allclose(recovered.apply(source.position_m), target.position_m, atol=1e-9)


def test_the_keypoints_land_on_the_camera_images(logged: Logged) -> None:
    """The rung draws the estimator's own frontend output, on the frontend rung's paths."""
    for index in range(2):
        entity: str = f"/world/rig_00/cam_{index:02d}/pinhole/keypoints"
        assert [t_ns for t_ns, _ in logged.rows[entity]] == logged.tracked
        assert all(len(values["Points2D:positions"]) > 0 for _, values in logged.rows[entity])


def test_the_rig_rides_the_estimate_only_when_asked(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """The dataset's rig belongs to the layers that own it (gt, or the catalog slam_rs layer): the replay leaves it alone,
    even without ground truth, unless a recording viewed on its own asks for it."""
    rig: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    layer: Logged = drive(pipeline(2), rig, empty_trajectory(), texture, tmp_path / "layer.rrd", read_rows)
    assert layer.tracked and RIG_ENTITY not in layer.rows
    alone: Logged = drive(pipeline(2), rig, empty_trajectory(), texture, tmp_path / "alone.rrd", read_rows, animate_base_rig=True)
    assert [t_ns for t_ns, _ in alone.rows[RIG_ENTITY]] == alone.tracked
    assert alone.rows[RIG_ENTITY] == alone.rows[f"{VIO_RUN_ENTITY}/rig"]


def test_each_run_gets_its_own_level_follow_frame_floor_and_path(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """The raw VIO run's follow frame rides its poses as they come (a turn about z only); once the final run is known,
    the final run gets its own follow frame, floor and whole path, at the same times. A follow view of the final run
    must not centre on the causal VIO: on clip 007 they part by 16 m at 65 s. Run-scoped, so with or without the base rig."""
    rig: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    # A floor 1.6 m under the origin, where the synthetic rig stays; the synthetic scene itself has none.
    floor: Float64[ndarray, "p 3"] = np.column_stack([np.random.default_rng(0).uniform(-2.0, 2.0, size=(600, 2)), np.full(600, -1.6)])
    shift: Float64[ndarray, " 3"] = np.array([5.0, 0.0, 0.0])

    def finish(logger: VioLogger) -> None:
        logger.floor_height = FloorHeight()
        logger.floor_height.add(np.zeros(3), floor)
        logger.log_complete_paths()
        estimate: Trajectory = logger.estimated()
        logger.log_final_run(dataclasses.replace(estimate, position_m=estimate.position_m + shift))

    logged: Logged = drive(pipeline(2), rig, empty_trajectory(), texture, tmp_path / "runs.rrd", read_rows, finish=finish)
    vio_follow: str = follow_entity(VIO_RUN_ENTITY)
    final_follow: str = follow_entity(RUN_ENTITY)
    assert vio_follow == "/world/runs/slam_rs_vio/follow" and final_follow == "/world/runs/slam_rs/follow"
    assert [t_ns for t_ns, _ in logged.rows[vio_follow]] == logged.tracked
    for follow, rig_row in zip(logged.rows[vio_follow], logged.rows[f"{VIO_RUN_ENTITY}/rig"], strict=True):
        assert follow.values["Transform3D:translation"] == rig_row.values["Transform3D:translation"]
        np.testing.assert_allclose(np.asarray(follow.values["Transform3D:quaternion"])[0, :2], 0.0, atol=1e-12)
    final_rows = logged.rows[final_follow]
    assert [t_ns for t_ns, _ in final_rows] == logged.tracked, "the final follow frame covers the whole walk, on its times"
    for final_row, vio_row in zip(final_rows, logged.rows[vio_follow], strict=True):
        np.testing.assert_allclose(np.asarray(final_row.values["Transform3D:translation"]) - np.asarray(vio_row.values["Transform3D:translation"]), [shift], atol=1e-9)
    rig_z: float = float(np.median([row.values["Transform3D:translation"][0][2] for row in logged.rows[f"{VIO_RUN_ENTITY}/rig"]]))
    for run in (VIO_RUN_ENTITY, RUN_ENTITY):
        floor_rows = logged.rows[floor_entity(run)]
        assert floor_rows and {t_ns for t_ns, _ in floor_rows} <= set(logged.tracked), run
        for _, values in floor_rows:
            np.testing.assert_allclose(np.asarray(values["LineStrips3D:strips"], dtype=np.float64)[:, :, 2], rig_z - 1.6, atol=0.05)
    final_x: Float64[ndarray, " s"] = np.concatenate([np.asarray(v["LineStrips3D:strips"], dtype=np.float64)[:, :, 0].ravel() for _, v in logged.rows[floor_entity(RUN_ENTITY)]])
    vio_x: Float64[ndarray, " s"] = np.concatenate([np.asarray(v["LineStrips3D:strips"], dtype=np.float64)[:, :, 0].ravel() for _, v in logged.rows[floor_entity(VIO_RUN_ENTITY)]])
    assert abs(float(np.median(final_x) - np.median(vio_x)) - 5.0) < 1.0, "each floor lies under its own run"
    assert static_strips(logged.recording, f"{RUN_ENTITY}/path") is not None, "the final run's whole path, static, for every cursor"


def static_strips(recording: Path, entity: str) -> Float64[ndarray, "s n 3"] | None:
    """One static ``LineStrips3D`` entity's strips, or None when the recording has no such entity."""
    for chunk in rrc.RrdReader(recording).stream().collect().stream():
        batch = chunk.to_record_batch()
        if chunk.entity_path == entity and chunk.is_static and "LineStrips3D:strips" in batch.schema.names:
            return np.asarray(batch.column("LineStrips3D:strips").to_pylist()[0], dtype=np.float64)
    return None


def test_landmarks_take_the_colour_of_the_image_under_their_keypoint(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """Each camera's copy has its own colour: a landmark takes the colour of the first camera that tracked it."""
    rig: tuple[CameraCalib, ...] = (camera(0, 0.0), camera(1, 0.1))
    tones: list[tuple[int, int, int]] = [(10, 20, 30), (200, 100, 50)]
    small: list[np.ndarray] = [np.full((cam.height // 4, cam.width // 4, 3), tone, dtype=np.uint8) for cam, tone in zip(rig, tones, strict=True)]
    reference_t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    colored: Logged = drive(pipeline(2), rig, straight_line(reference_t_ns), texture, tmp_path / "rgb.rrd", read_rows, colors=small)
    rows: list[Row] = [row for row in colored.rows[f"{VIO_RUN_ENTITY}/landmarks"] if len(row[1]["Points3D:positions"])]
    assert rows, "the window holds landmarks"
    seen: set[int] = set().union(*(values["Points3D:colors"] for _, values in rows))
    assert packed((*tones[0], 255)) in seen, "camera 0 hosts landmarks"
    assert seen <= {packed((*tone, 255)) for tone in tones}, "every landmark has an image colour"


def test_a_run_without_references_still_logs_everything_else(pipeline: PipelineFactory, camera: CameraFactory) -> None:
    """A segment with no ground truth draws neither line, and no ATE."""
    logger: VioLogger = VioLogger(
        cameras=(camera(0, 0.0), camera(1, 0.1)),
        ground_truth=empty_trajectory(),
        frame_t_ns=np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64),
        tracked_resolution=((FRAME, FRAME), (FRAME, FRAME)),
        cam_time_offset_ns=0,
    )
    assert len(logger.estimated()) == 0


def packed(color: tuple[int, int, int, int]) -> int:
    """A colour as the recording stores it: one RGBA integer."""
    return int(pack_rgba(np.array([color[:3]], dtype=np.uint8), color[3])[0])


def slot_entities(rows: Rows, prefix: str) -> list[str]:
    """The numbered slot entities under a prefix, in order: ``prefix/00``, ``prefix/01``, ..."""
    return sorted(entity for entity in rows if re.fullmatch(rf"{re.escape(prefix)}/\d\d", entity))


def test_window_frames_are_pinhole_cameras_on_numbered_slots(logged: Logged) -> None:
    """Every pose still in the window is drawn the way the live rig is: a rig node with camera 0's ``Pinhole`` under it.

    Same archetype as the rig, so the viewer treats them as cameras; the role
    (keyframe, long-term keyframe, plain frame) is the frustum's colour. No
    hand-drawn wireframes anywhere under the window.
    """
    slots: list[str] = slot_entities(logged.rows, f"{VIO_RUN_ENTITY}/window")
    assert slots and slots[0] == f"{VIO_RUN_ENTITY}/window/00"
    for entity, rows in logged.rows.items():
        if entity.startswith((f"{VIO_RUN_ENTITY}/window", f"{VIO_RUN_ENTITY}/marginalized")):
            assert not any("LineStrips3D:strips" in values for _, values in rows), entity
    for slot in slots:
        assert all("Transform3D:translation" in values for _, values in logged.rows[slot] if "Clear:is_recursive" not in values), slot
        colours: set[int] = {values["Pinhole:color"][0] for _, values in logged.rows[f"{slot}/cam_00/pinhole"] if "Pinhole:color" in values}
        assert colours and colours <= {packed(KEYFRAME_COLOR), packed(LTKF_COLOR), packed(POSE_COLOR)}, slot
    planes: dict[str, float] = image_planes(logged.recording)
    assert planes[f"{VIO_RUN_ENTITY}/window/00/cam_00/pinhole"] == pytest.approx(FRUSTUM_DEPTH_M)
    assert planes[f"{VIO_RUN_ENTITY}/rig/cam_00/pinhole"] == pytest.approx(IMAGE_PLANE_M)


def test_the_frames_a_marginalization_removed_are_faded_pinholes_at_the_poses_they_held(logged: Logged) -> None:
    """A frame the last marginalization removed is drawn once more, as the same camera, in the marginalized colour."""
    slots: list[str] = slot_entities(logged.rows, f"{VIO_RUN_ENTITY}/marginalized")
    assert slots, "no frameset drew a frame the last marginalization removed"
    drawn: list[int] = [t_ns for t_ns, values in logged.rows[slots[0]] if "Transform3D:translation" in values]
    assert drawn
    for index in range(2):
        colours: set[int] = {values["Pinhole:color"][0] for _, values in logged.rows[f"{slots[0]}/cam_{index:02d}/pinhole"] if "Pinhole:color" in values}
        assert colours == {packed(MARGINALIZED_COLOR)}, index


def test_a_slot_is_cleared_once_when_it_empties_and_not_again_until_it_is_reused(logged: Logged) -> None:
    """Rerun keeps the latest state, so one clear at the frameset a slot empties is exactly enough; repeating it is waste."""
    for prefix in (f"{VIO_RUN_ENTITY}/window", f"{VIO_RUN_ENTITY}/marginalized"):
        for slot in slot_entities(logged.rows, prefix):
            drawn: set[int] = {t_ns for t_ns, values in logged.rows[slot] if "Transform3D:translation" in values}
            cleared: list[int] = [t_ns for t_ns, values in logged.rows[slot] if "Clear:is_recursive" in values]
            assert not (drawn & set(cleared)), (slot, "drawn and cleared at the same frameset")
            previous_drawn: bool = False
            for t_ns in logged.tracked:
                if t_ns in cleared:
                    assert previous_drawn, (slot, t_ns, "cleared without having been drawn the frameset before")
                previous_drawn = t_ns in drawn


def translations(rows: Rows, entity: str) -> dict[int, tuple[float, ...]]:
    """Each frameset's logged translation on an entity, rounded so equal poses compare equal."""
    return {t_ns: tuple(round(x, 9) for x in values["Transform3D:translation"][0]) for t_ns, values in rows[entity] if "Transform3D:translation" in values}


def test_window_slots_hold_real_poses_and_marginalized_slots_hold_the_poses_they_held(logged: Logged) -> None:
    """The newest window pose is where the rig is; a marginalized frame is drawn where the previous window had it."""
    rig: dict[int, tuple[float, ...]] = translations(logged.rows, f"{VIO_RUN_ENTITY}/rig")
    window: list[dict[int, tuple[float, ...]]] = [translations(logged.rows, slot) for slot in slot_entities(logged.rows, f"{VIO_RUN_ENTITY}/window")]
    for t_ns in logged.tracked:
        drawn_now: set[tuple[float, ...]] = {slot[t_ns] for slot in window if t_ns in slot}
        assert rig[t_ns] in drawn_now, (t_ns, "the current rig pose is not among the window frames")
    marginalized: list[dict[int, tuple[float, ...]]] = [translations(logged.rows, slot) for slot in slot_entities(logged.rows, f"{VIO_RUN_ENTITY}/marginalized")]
    checked: int = 0
    for previous, t_ns in zip(logged.tracked, logged.tracked[1:], strict=False):
        drawn_before: set[tuple[float, ...]] = {slot[previous] for slot in window if previous in slot}
        for slot in marginalized:
            if t_ns in slot:
                assert slot[t_ns] in drawn_before, (t_ns, "a marginalized frame is drawn somewhere the previous window never was")
                checked += 1
    assert checked > 0, "no marginalized frame was ever checked"

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
        *[VIO_STATS_ENTITY] * 8,
    ]
    assert [str(view.name) for view in views[3:]] == [
        "counts",
        "keyframes & LM steps",
        "timing (ms)",
        "solve stages (ms)",
        "frontend stages (ms)",
        "LM cost",
        "LM damping",
        "ATE (cm)",
    ]
    assert blueprint.collapse_panels


def test_only_the_three_paths_reach_back_to_the_start_of_the_recording(camera: CameraFactory) -> None:
    """The segments become a path because the view says so — and only for the paths.

    Without the override Rerun's default for a 3D view is latest-at, and each
    two-point strip would render alone. With it on the view instead of on these
    three entities, every frameset's window frusta and landmarks would be drawn
    at once as well.
    """
    world: rrb.View = views_of(vio_blueprint((camera(0, 0.0), camera(1, 0.1))).root_container)[0]
    assert sorted(str(entity) for entity in world.visualizer_overrides) == [
        f"{GT_ENTITY}/trajectory",
        f"{RUN_ENTITY}/trajectory",
        f"{VIO_RUN_ENTITY}/trajectory",
    ]
    reaching_back: rrb.VisibleTimeRanges = rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(TIMELINE, start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.cursor_relative())
    )
    for override in world.visualizer_overrides.values():
        assert isinstance(override, rrb.VisibleTimeRanges)
        assert override.ranges is not None
        assert reaching_back.ranges is not None
        assert override.ranges.as_arrow_array() == reaching_back.ranges.as_arrow_array()


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
    counters |= {f"{VIO_STATS_ENTITY}/ate_cm/gt"}
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


def test_the_map_is_drawn_under_the_run_as_keyframes_join_and_ends_on_the_final_map(
    rig: RigFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """With the back end on, the map's keyframes and points land under the run (DESIGN §8) at their cadences, then once more after the flush.

    Keyframes with no cadence limit: one draw per keyframe join. Points at most once a second, longer than the run:
    their blocks are drawn at the first join, and after the flush the blocks are cleared and the whole flushed map
    lands on the points entity itself.
    """
    assert MAP_KEYFRAMES_ENTITY == "/world/runs/slam_rs/keyframes"
    assert MAP_POINTS_ENTITY == "/world/runs/slam_rs/map/points"
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig().with_backend("offline"), backend="cpu")
    t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    points_every_ns: int = 1_000_000_000
    assert points_every_ns > FRAMESETS * FRAME_PERIOD_NS
    logged: Logged = drive(
        vio, (camera(0, 0.0), camera(1, 0.1)), straight_line(t_ns), texture, tmp_path / "map.rrd", read_rows, map_every_ns=(0, points_every_ns)
    )
    # Every frameset where the newest keyframe changed, then the final draw at the last frameset.
    joins: list[int] = [
        t for t, (previous, newest) in zip(logged.tracked, itertools.pairwise([-1, *logged.newest_keyframes]), strict=True) if newest not in (-1, previous)
    ]
    assert joins, "the synthetic run never added a keyframe"
    keyframe_times: list[int] = [t for t, _ in logged.rows[MAP_KEYFRAMES_ENTITY]]
    assert keyframe_times == [*joins, logged.tracked[-1]]
    assert len(joins) >= 2, "the run must add keyframes after the first for the points' cadence to show"
    block_rows: dict[str, list[int]] = {e: [t for t, _ in rows] for e, rows in logged.rows.items() if e.startswith(f"{MAP_POINT_BLOCKS_ENTITY}/")}
    assert block_rows, "the run drew no map-point block"
    assert {t for times in block_rows.values() for t in times} == {joins[0]}, "blocks are drawn at the points' cadence, not at the final draw"
    assert [t for t, _ in logged.rows[MAP_POINTS_ENTITY]] == [logged.tracked[-1]], "the whole map is drawn on the entity itself only at the end"
    assert [t for t, _ in logged.rows[MAP_POINT_BLOCKS_ENTITY]] == [logged.tracked[-1]], "the final draw clears the blocks"

    final: _core.MapSnapshot | None = vio.map_snapshot()
    assert final is not None
    # The final draw shares the last frameset's time (which may have drawn too); one of the rows there is the flushed map.
    last_keyframes: list[Float64[ndarray, "k 3"]] = [
        np.asarray(values["Points3D:positions"], dtype=np.float64).reshape(-1, 3)
        for t, values in logged.rows[MAP_KEYFRAMES_ENTITY]
        if t == logged.tracked[-1]
    ]
    assert any(
        drawn.shape == final.keyframe_poses[:, 0:3].shape and np.allclose(drawn, final.keyframe_poses[:, 0:3], rtol=1e-6, atol=1e-6)
        for drawn in last_keyframes
    )
    last_points: list[Float32[ndarray, "p 3"]] = [
        np.asarray(values["Points3D:positions"], dtype=np.float32).reshape(-1, 3) for t, values in logged.rows[MAP_POINTS_ENTITY] if t == logged.tracked[-1]
    ]
    assert any(np.array_equal(drawn, final.points_world) for drawn in last_points)
    assert final.points_world.shape[0] >= 1
    assert np.isfinite(final.points_world).all()


def test_the_final_map_draws_the_pose_graph_and_the_keyframes_before_and_after_pgo(
    rig: RigFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """After the flush the run carries ``pgo/{before,after}`` (DESIGN §8, PR f1) and the pose graph; the final trajectory lands on the map keyframes."""
    from slam_rs.map_loops import PGO_AFTER_ENTITY, PGO_BEFORE_ENTITY, POSE_GRAPH_ENTITY
    from slam_rs.tracking import reanchored

    assert (PGO_BEFORE_ENTITY, PGO_AFTER_ENTITY, POSE_GRAPH_ENTITY) == (
        "/world/runs/slam_rs/pgo/before", "/world/runs/slam_rs/pgo/after", "/world/runs/slam_rs/pose_graph"
    )
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig().with_backend("offline"), backend="cpu")
    t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    logged: Logged = drive(vio, (camera(0, 0.0), camera(1, 0.1)), straight_line(t_ns), texture, tmp_path / "pgo.rrd", read_rows)
    final_map: _core.MapSnapshot | None = vio.map_snapshot()
    assert final_map is not None
    k: int = final_map.keyframe_t_ns.shape[0]
    assert k >= 2
    g: int = final_map.pose_graph_edges.shape[0]
    assert final_map.pose_graph_edges.dtype == np.int64
    assert final_map.pose_graph_edge_kinds.dtype == np.uint8
    assert final_map.pose_graph_edge_kinds.shape == (g,)
    assert g >= k - 1, "the spanning tree alone links every keyframe"
    assert np.isin(final_map.pose_graph_edges, final_map.keyframe_t_ns).all()
    last: int = logged.tracked[-1]
    before = [values["LineStrips3D:strips"] for t, values in logged.rows[PGO_BEFORE_ENTITY] if t == last]
    after = [values["LineStrips3D:strips"] for t, values in logged.rows[PGO_AFTER_ENTITY] if t == last]
    graph = [values["LineStrips3D:strips"] for t, values in logged.rows[POSE_GRAPH_ENTITY] if t == last]
    np.testing.assert_allclose(np.asarray(after[-1][0], dtype=np.float64), final_map.keyframe_poses[:, 0:3], atol=1e-6)
    # No loop closes on this run: before and after are the same path.
    np.testing.assert_allclose(np.asarray(before[-1][0], dtype=np.float64), final_map.keyframe_raw_poses[:, 0:3], atol=1e-6)
    assert len(graph[-1]) == g

    estimate: Trajectory = logged.logger.estimated()
    final: Trajectory = reanchored(estimate, final_map)
    np.testing.assert_array_equal(final.t_ns, estimate.t_ns)
    rows: Int64[ndarray, " k"] = np.searchsorted(estimate.t_ns, final_map.keyframe_t_ns)
    np.testing.assert_array_equal(estimate.t_ns[rows], final_map.keyframe_t_ns)
    np.testing.assert_allclose(final.position_m[rows], final_map.keyframe_poses[:, 0:3], atol=1e-12)
    assert reanchored(estimate, None) is estimate


def test_the_default_cadence_draws_a_short_run_once_and_then_the_final_map(
    rig: RigFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """At the default cadences a 0.4 s run draws the map at its first keyframe and after the flush, never per keyframe."""
    assert min(MAP_KEYFRAMES_EVERY_NS, MAP_POINTS_EVERY_NS) > FRAMESETS * FRAME_PERIOD_NS
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig().with_backend("realtime"), backend="cpu")
    t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    logged: Logged = drive(vio, (camera(0, 0.0), camera(1, 0.1)), straight_line(t_ns), texture, tmp_path / "rt.rrd", read_rows)
    first_join: int = next(t for t, newest in zip(logged.tracked, logged.newest_keyframes, strict=True) if newest != -1)
    # Realtime never waits for the back-end thread: a read it holds the map for comes back None, and the draw follows
    # on the next frameset that finds it free.
    keyframe_times: list[int] = [t for t, _ in logged.rows[MAP_KEYFRAMES_ENTITY]]
    assert len(keyframe_times) == 2 and first_join <= keyframe_times[0] < logged.tracked[-1] and keyframe_times[1] == logged.tracked[-1]
    block_times: set[int] = {t for e, rows in logged.rows.items() if e.startswith(f"{MAP_POINT_BLOCKS_ENTITY}/") for t, _ in rows}
    assert block_times <= {keyframe_times[0]}, "blocks are drawn only at the points' cadence (a map with no points yet draws none)"
    assert [t for t, _ in logged.rows[MAP_POINTS_ENTITY]] == [logged.tracked[-1]]


def test_no_map_is_drawn_with_the_back_end_off(
    rig: RigFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig().with_backend("off"), backend="cpu")
    t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    logged: Logged = drive(vio, (camera(0, 0.0), camera(1, 0.1)), straight_line(t_ns), texture, tmp_path / "off.rrd", read_rows)
    assert len(logged.tracked) >= FRAMESETS - 2
    assert MAP_KEYFRAMES_ENTITY not in logged.rows
    assert not [e for e in logged.rows if e.startswith(MAP_POINTS_ENTITY)]


def test_the_live_run_is_the_raw_vio_and_the_final_run_has_its_own_alignment(
    pipeline: PipelineFactory, texture: TextureFactory, camera: CameraFactory, tmp_path: Path, read_rows: RowsReader
) -> None:
    """The replay uses the layer's names (f0): the live raw VIO under ``slam_rs_vio``, the final run under ``slam_rs``.

    The final run is placed by an alignment fitted to the final run itself, so a loop the back end closed lands on
    the ground truth instead of on the raw run's drift.
    """
    assert (RUN_ENTITY, VIO_RUN_ENTITY) == ("/world/runs/slam_rs", "/world/runs/slam_rs_vio")
    t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    truth: Trajectory = straight_line(t_ns)
    finals: list[Trajectory] = []

    def finish(logger: VioLogger) -> None:
        estimate: Trajectory = logger.estimated()
        # Half a metre off the raw run and bent: its alignment cannot be the raw run's.
        bend: Float64[ndarray, "n 3"] = np.column_stack([np.zeros(len(estimate)), 0.5 + 0.1 * np.arange(len(estimate)) ** 2, np.zeros(len(estimate))])
        final: Trajectory = Trajectory(t_ns=estimate.t_ns, position_m=estimate.position_m + bend, quaternion_wxyz=estimate.quaternion_wxyz)
        logger.log_final_run(final)
        finals.append(final)

    logged: Logged = drive(pipeline(2), (camera(0, 0.0), camera(1, 0.1)), truth, texture, tmp_path / "final.rrd", read_rows, finish=finish)
    for part in ("rig", "window/00", "landmarks", "trajectory"):
        assert f"{VIO_RUN_ENTITY}/{part}" in logged.rows, part
    assert f"{RUN_ENTITY}/rig" not in logged.rows
    assert sorted(static_path_lengths(logged.recording)) == sorted([f"{VIO_RUN_ENTITY}/path", f"{GT_ENTITY}/path", f"{RUN_ENTITY}/path"])

    # The final run: its whole path once, at the last cursor, under its own alignment.
    (final,) = finals
    drawn: list[Row] = logged.rows[f"{RUN_ENTITY}/trajectory"]
    assert [row.t_ns for row in drawn] == [logged.tracked[-1]]
    np.testing.assert_allclose(np.asarray(drawn[0].values["LineStrips3D:strips"][0], dtype=np.float64), final.position_m, atol=1e-6)
    placed: list[Row] = [row for row in logged.rows[RUN_ENTITY] if "Transform3D:translation" in row.values]
    assert [row.t_ns for row in placed] == [logged.tracked[-1]]
    expected: SimilarityTransform = inverted(ate(final, truth).alignment)
    raw: SimilarityTransform = inverted(ate(logged.logger.estimated(), truth).alignment)
    np.testing.assert_allclose(placed[0].values["Transform3D:translation"][0], expected.dst_t_src, atol=1e-5)
    assert not np.allclose(expected.dst_t_src, raw.dst_t_src, atol=1e-2)


def test_until_the_final_run_the_map_sits_on_the_raw_runs_alignment(
    pipeline: PipelineFactory,
    texture: TextureFactory,
    camera: CameraFactory,
    tmp_path: Path,
    read_rows: RowsReader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The map is drawn under ``slam_rs`` while the run goes, so that entity carries the raw run's alignment until the final run replaces it."""
    monkeypatch.setattr(vio_log, "ATE_EVERY", 1)
    t_ns: Int64[ndarray, " n"] = np.arange(0, FRAMESETS * FRAME_PERIOD_NS, FRAME_PERIOD_NS, dtype=np.int64)
    logged: Logged = drive(pipeline(2), (camera(0, 0.0), camera(1, 0.1)), straight_line(t_ns), texture, tmp_path / "mirror.rrd", read_rows)

    def placements(entity: str) -> list[tuple[int, list[float]]]:
        return [(row.t_ns, row.values["Transform3D:translation"][0]) for row in logged.rows[entity] if "Transform3D:translation" in row.values]

    assert placements(VIO_RUN_ENTITY)
    assert placements(RUN_ENTITY) == placements(VIO_RUN_ENTITY)
