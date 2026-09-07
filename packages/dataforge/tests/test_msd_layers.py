"""The msd layer machinery: the world-up measurement, and what each layer actually wrote.

Every assertion reads a converted rrd back through the public reader, so it is
what a consumer sees rather than what the writer intended. The verbs that drive
these writers — discover, download, the skip and rebuild rules, the budget — are
next door in ``test_msd``; the sequence they all convert is ``msd_hub``'s.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from conftest import calibration_fixture, column_rows, read_back
from jaxtyping import Float64, Int64
from msd_hub import (
    FIXTURE_WORLD_R_RIG,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GT_DROPOUT_ROW,
    GT_NUM_POSES,
    GT_PERIOD_NS,
    FakeHub,
    build_hub,
    recording_properties,
)
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import paths, schema
from dataforge.basalt import BasaltPose, CalibratedCamera, load_calibration, rotate_camera_cw, upright_quarter_turns
from dataforge.datasets.msd import MSD_DEVICES, MsdDataset, MsdDeviceChoice
from dataforge.datasets.msd_layers import MEASURED_UP_WINDOW_NS, WORLD_UP_VIEW_COORDINATES, MeasuredUp, measured_world_up
from dataforge.euroc import GtTrajectory, TimestampedSamples, gt_trajectory
from dataforge.logging_toolkit import ImuChannel

VIEWER_AXIS_VECTORS: dict[int, tuple[float, float, float]] = {
    rr.encodings.ViewDir.Right.value: (1.0, 0.0, 0.0),
    rr.encodings.ViewDir.Left.value: (-1.0, 0.0, 0.0),
    rr.encodings.ViewDir.Up.value: (0.0, 1.0, 0.0),
    rr.encodings.ViewDir.Down.value: (0.0, -1.0, 0.0),
    rr.encodings.ViewDir.Back.value: (0.0, 0.0, 1.0),
    rr.encodings.ViewDir.Forward.value: (0.0, 0.0, -1.0),
}
"""Each ``ViewDir`` as a vector in one right-handed viewer basis (x right, y up, z back)."""


def test_each_world_up_axis_maps_to_a_right_handed_frame_with_that_axis_up() -> None:
    """The axis name is the whole decision; handedness then fixes the other two."""
    assert set(WORLD_UP_VIEW_COORDINATES) == {"+x", "-x", "+y", "-y", "+z", "-z"}
    for axis, coordinates in WORLD_UP_VIEW_COORDINATES.items():
        directions: list[int] = [int(direction.value) for direction in coordinates.coordinates]
        column: int = "xyz".index(axis[1])
        expected: int = (rr.encodings.ViewDir.Up if axis[0] == "+" else rr.encodings.ViewDir.Down).value
        assert directions[column] == expected, f"{axis} does not put that axis up"
        basis: Float64[ndarray, "3 3"] = np.array([VIEWER_AXIS_VECTORS[direction] for direction in directions], dtype=np.float64)
        assert np.linalg.det(basis) > 0.0, f"{axis} maps to a left-handed frame"


# ── the world up measurement ────────────────────────────────────


def constant_pose_gt(times_ns: Int64[ndarray, "n_poses"], quaternion_xyzw: Float64[ndarray, "4"]) -> TimestampedSamples:
    """A gt table holding one fixed orientation at the origin, in the file's wxyz order."""
    return TimestampedSamples(
        times_ns=times_ns,
        values=np.column_stack([np.zeros((times_ns.size, 3)), np.tile(quaternion_xyzw[[3, 0, 1, 2]], (times_ns.size, 1))]),
    )


def test_the_world_up_axis_is_measured_by_rotating_the_accelerometer_into_the_world() -> None:
    """An accelerometer at rest reads +g pointing *up*, so ``world_R_rig @ a_rig`` averages to the up axis."""
    # -90 deg about x maps the rig's +z onto the world's +y, so a headset held level
    # in a Y-up world reads gravity along its own +z.
    world_R_rig: Rotation = Rotation.from_euler("x", -90.0, degrees=True)
    times_ns: Int64[ndarray, "n_poses"] = np.arange(4_000, dtype=np.int64) * 1_000_000
    rig_accel_xyz: Float64[ndarray, "n_samples 3"] = np.tile([0.1, -0.2, 9.81], (times_ns.size, 1))
    # The second half of the capture points the other way; the 2 s window must ignore it.
    rig_accel_xyz[times_ns >= MEASURED_UP_WINDOW_NS] = [0.1, -0.2, -9.81]
    gt: GtTrajectory = gt_trajectory(constant_pose_gt(times_ns, np.asarray(world_R_rig.as_quat(), dtype=np.float64)))

    measured: MeasuredUp = measured_world_up(gt, ImuChannel(times_ns=times_ns, values_xyz=rig_accel_xyz))

    assert measured.axis == "+y"
    # At rest the whole of gravity lands on that one axis.
    assert measured.fraction == pytest.approx(1.0, abs=0.01)


@pytest.fixture(scope="module")
def converted_index(tmp_path_factory, nvenc_ffmpeg: Path) -> Iterator[tuple[FakeHub, Path, Path]]:
    """One real Index convert, shared by every test that only reads back what it wrote.

    A convert encodes a whole synthetic sequence through NVENC, so the tests that
    only inspect its two rrds share one. Anything whose *setup* differs — another
    device, a budget, a patched registry, a captured warning — still converts for
    itself. ``MonkeyPatch.context()`` because the module-scoped fixture outlives
    the function-scoped ``monkeypatch``.

    Yields:
        The hub the convert ran against, its base rrd, and its gt rrd.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        hub: FakeHub = build_hub(tmp_path_factory.mktemp("converted"), monkeypatch)
        dataset: MsdDataset = MsdDataset(hub.config)
        identity, source = dataset.discover()[0]
        base_target: Path = dataset.convert(identity, source, force=False)
        yield hub, base_target, paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)


def test_the_logged_camera_node_carries_rig_T_cam(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """``T_imu_cam`` is the camera's pose in the rig frame, and that is what lands on the node.

    ``log_pinhole`` stores the child-from-parent step, so the recording holds
    ``cam_T_rig``; inverting it must give back the calibration's pose.
    """
    hub, target, _ = converted_index

    cameras: tuple[CalibratedCamera, ...] = load_calibration(hub.remote / "M_monado_datasets/MI_valve_index/extras/calibration.json")
    store: rr.experimental.ChunkStore = read_back(target)
    for index in range(2):
        node: str = schema.cam_path(0, index)
        row: dict[str, list[object]] = store.reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
        assert row[f"{node}:Transform3D:relation"][0] == rr.components.TransformRelation.ChildFromParent.value
        cam_R_rig: Float64[ndarray, "3 3"] = np.asarray(row[f"{node}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T
        cam_t_rig: Float64[ndarray, "3"] = np.asarray(row[f"{node}:Transform3D:translation"][0], dtype=np.float64)

        pose: BasaltPose = cameras[index].rig_pose
        rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        rig_t_cam: Float64[ndarray, "3"] = np.array([pose.px, pose.py, pose.pz])
        # float32 on the wire, so a loose tolerance is the honest one.
        np.testing.assert_allclose(cam_R_rig.T, rig_R_cam, atol=1e-6)
        np.testing.assert_allclose(-cam_R_rig.T @ cam_t_rig, rig_t_cam, atol=1e-6)


def convert_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: MsdDeviceChoice) -> Path:
    """Convert the synthetic sequence for one device and return its base rrd."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch, device=device)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    return dataset.convert(identity, source, force=False)


def test_a_radtan8_camera_node_names_its_projection_and_carries_its_validity_radius(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """A consumer reads the projection off the node, plus where the rational model stops holding.

    The G2, whose four cameras state a real ~2.8 radius. The Odyssey+ is the next
    test: it states ``0.0``, which is not a radius.
    """
    target: Path = convert_device(tmp_path, monkeypatch, "g2")

    expected: float | None = load_calibration(calibration_fixture("g2"))[0].distortion_valid_radius
    assert expected is not None and expected > 2.7
    store: rr.experimental.ChunkStore = read_back(target)
    node: str = schema.cam_path(0, 0)
    row: dict[str, list[object]] = store.reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
    assert row[f"{node}:camera_model"][0] == "pinhole-radtan8"
    assert row[f"{node}:distortion_valid_radius"][0] == pytest.approx(expected)


def test_an_odyssey_camera_node_names_radtan8_and_states_no_validity_radius(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """Both Odyssey+ cameras ship ``rpmax: 0.0``, which basalt reads as no validity check.

    Logging that through as ``distortion_valid_radius = 0.0`` would tell a consumer
    the rational model holds nowhere on this device, so the key is left off — the
    same thing a kb4 camera does, for the same reason.
    """
    target: Path = convert_device(tmp_path, monkeypatch, "odyssey")

    store: rr.experimental.ChunkStore = read_back(target)
    for index in range(2):
        node: str = schema.cam_path(0, index)
        table: pa.Table = store.reader(index=None, contents=node).to_arrow_table()
        assert table.to_pylist()[0][f"{node}:camera_model"][0] == "pinhole-radtan8"
        # AnyValues only *omits* a None key while it is untyped: a g2 convert
        # earlier in this process types it, and later Nones then arrive as nulls.
        radius: str = f"{node}:distortion_valid_radius"
        assert radius not in table.column_names or table.column(radius).null_count == table.num_rows


def test_a_kb4_camera_node_names_its_projection_and_claims_no_validity_radius(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """kb4 is valid over the whole fisheye, so it declares no radius at all."""
    _, target, _ = converted_index

    store: rr.experimental.ChunkStore = read_back(target)
    node: str = schema.cam_path(0, 0)
    table: pa.Table = store.reader(index=None, contents=node).to_arrow_table()
    assert table.to_pylist()[0][f"{node}:camera_model"][0] == "kb4"
    # AnyValues only *omits* a None key while it is untyped: a radtan8 convert earlier in
    # this process types it, and later Nones then arrive as nulls. Assert on the value.
    radius: str = f"{node}:distortion_valid_radius"
    assert radius not in table.column_names or table.column(radius).null_count == table.num_rows


# ── the upright roll ──────────────────────────────────────────────────────


def test_the_g2_camera_nodes_state_the_quarter_turn_their_frames_were_encoded_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """All four G2 cameras are mounted rolled, so all four record a rotation on the node.

    The value is the turn ``upright_quarter_turns`` chose for that camera, in
    degrees, which is how a consumer knows the frames are not in the orientation
    the sensor read them out in. Which integer each camera answers is pinned in
    ``test_basalt``; what this asserts is that the number reaching the rrd is the
    one the encoder was given.
    """
    target: Path = convert_device(tmp_path, monkeypatch, "g2")

    up_rig: tuple[float, float, float] = MSD_DEVICES["g2"].follow.up
    store: rr.experimental.ChunkStore = read_back(target)
    logged: list[object] = []
    for camera in load_calibration(calibration_fixture("g2")):
        node: str = schema.cam_path(0, camera.index)
        row: dict[str, list[object]] = store.reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
        assert row[f"{node}:image_rotation_cw_deg"][0] == 90 * upright_quarter_turns(camera, up_rig)
        logged.append(row[f"{node}:image_rotation_cw_deg"][0])
    assert all(rotation in (90, 180, 270) for rotation in logged), f"the G2's sideways cameras were not turned: {logged}"


def test_a_rolled_camera_logs_the_calibration_of_the_pixels_it_encoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """The calibration follows the pixels: what lands is the rolled record, not the file's.

    Resolution, intrinsics and ``rig_T_cam`` are all asserted against
    ``rotate_camera_cw`` of the calibration, **and** asserted to differ from the
    unrolled one — because a wiring that forgot to roll the calibration would
    pass the first half on a device whose cameras happen to be square.
    """
    target: Path = convert_device(tmp_path, monkeypatch, "g2")

    up_rig: tuple[float, float, float] = MSD_DEVICES["g2"].follow.up
    store: rr.experimental.ChunkStore = read_back(target)
    for camera in load_calibration(calibration_fixture("g2")):
        rolled: CalibratedCamera = rotate_camera_cw(camera, upright_quarter_turns(camera, up_rig))
        assert rolled.resolution != camera.resolution, "this test needs a camera the roll actually changes"
        node: str = schema.cam_path(0, camera.index)
        row: dict[str, list[object]] = store.reader(index=None, contents=f"{node}/**").to_arrow_table().to_pylist()[0]

        resolution: Float64[ndarray, "2"] = np.asarray(row[f"{node}/pinhole:Pinhole:resolution"][0], dtype=np.float64)
        np.testing.assert_array_equal(resolution, rolled.resolution)

        # Rerun's mat3x3 is column-major, so the transpose is the textbook K.
        image_from_camera: Float64[ndarray, "3 3"] = (
            np.asarray(row[f"{node}/pinhole:Pinhole:image_from_camera"][0], dtype=np.float64).reshape(3, 3).T
        )
        np.testing.assert_allclose(
            [image_from_camera[0, 0], image_from_camera[1, 1], image_from_camera[0, 2], image_from_camera[1, 2]],
            [rolled.model.fx, rolled.model.fy, rolled.model.cx, rolled.model.cy],
            atol=1e-3,
        )
        assert image_from_camera[0, 2] != pytest.approx(camera.model.cx, abs=1e-3), "the principal point was not rolled"

        cam_R_rig: Float64[ndarray, "3 3"] = np.asarray(row[f"{node}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T
        rolled_pose: BasaltPose = rolled.rig_pose
        rolled_rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat(
            [rolled_pose.qx, rolled_pose.qy, rolled_pose.qz, rolled_pose.qw]
        ).as_matrix()
        # float32 on the wire, so a loose tolerance is the honest one.
        np.testing.assert_allclose(cam_R_rig.T, rolled_rig_R_cam, atol=1e-6)
        pose: BasaltPose = camera.rig_pose
        unrolled_rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        assert not np.allclose(cam_R_rig.T, unrolled_rig_R_cam, atol=1e-3), "the pose was not rolled with the pixels"


def test_the_g2_writes_frames_whose_dimensions_the_quarter_turn_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """The turn reaches the encoder, not only the metadata: the mp4 itself comes out H x W.

    ``--keep-raw`` leaves the per-camera clips behind, which is the only place the
    encoded frame size is readable without decoding an rrd's video samples.
    """
    hub: FakeHub = build_hub(tmp_path, monkeypatch, device="g2", keep_raw=True)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=False)

    up_rig: tuple[float, float, float] = MSD_DEVICES["g2"].follow.up
    for camera in load_calibration(calibration_fixture("g2")):
        clip: Path = hub.config.root / "work" / source.sequence / f"cam{camera.index}.mp4"
        with av.open(str(clip)) as container:
            stream: av.video.stream.VideoStream = container.streams.video[0]
            encoded: tuple[int, int] = (stream.codec_context.width, stream.codec_context.height)
        turns: int = upright_quarter_turns(camera, up_rig)
        assert turns % 2 == 1, "this test needs a camera whose turn swaps the frame"
        assert encoded == (FRAME_HEIGHT, FRAME_WIDTH), f"cam{camera.index} was encoded {encoded} after {turns} quarter turn(s)"


@pytest.mark.parametrize("device", ["index", "odyssey"])
def test_an_upright_headset_is_left_exactly_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path, device: MsdDeviceChoice
) -> None:
    """The Index's and the Odyssey+'s cameras are mounted upright, so the rule is a no-op there.

    Zero turns means no rotation key on the node — a rotation of ``0`` would tell
    a consumer a decision was made where none was needed — and the calibration
    that lands is the file's own, unrolled.
    """
    target: Path = convert_device(tmp_path, monkeypatch, device)

    up_rig: tuple[float, float, float] = MSD_DEVICES[device].follow.up
    store: rr.experimental.ChunkStore = read_back(target)
    for camera in load_calibration(calibration_fixture(device)):
        assert upright_quarter_turns(camera, up_rig) == 0
        node: str = schema.cam_path(0, camera.index)
        table: pa.Table = store.reader(index=None, contents=f"{node}/**").to_arrow_table()
        # AnyValues only *omits* a None key while it is untyped: a g2 convert
        # earlier in this process types it, and later Nones then arrive as nulls.
        rotation: str = f"{node}:image_rotation_cw_deg"
        assert rotation not in table.column_names or table.column(rotation).null_count == table.num_rows
        resolution: Float64[ndarray, "2"] = np.asarray(
            table.to_pylist()[0][f"{node}/pinhole:Pinhole:resolution"][0], dtype=np.float64
        )
        np.testing.assert_array_equal(resolution, camera.resolution)

# ── gt layer ──────────────────────────────────────────────────────────────


def test_the_gt_layer_is_a_sibling_rrd_of_the_same_recording(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """One convert writes both layers: same recording id, own layer directory."""
    _, base_target, gt_target = converted_index

    assert gt_target.is_file()
    assert gt_target.name == base_target.name
    assert (gt_target.parent.name, base_target.parent.name) == (paths.GT_LAYER, paths.BASE_LAYER)


def test_the_gt_layer_animates_the_rig_node_at_the_full_gt_rate(converted_index: tuple[FakeHub, Path, Path]) -> None:
    hub, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    poses: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:translation")
    assert poses.num_rows == GT_NUM_POSES, "gt is logged raw: no resampling, one row per csv row"
    times_ns: list[int] = poses.column(schema.TIMELINE).combine_chunks().cast(pa.int64()).to_pylist()
    # gt is the earliest stream in the fixture, so it owns t0 and starts at video_time 0.
    assert hub.clocks.firsts["gt"] == min(hub.clocks.firsts.values())
    assert times_ns[0] == 0
    assert times_ns[-1] == (GT_NUM_POSES - 1) * GT_PERIOD_NS


def test_the_rig_quaternion_is_the_file_quaternion_reordered_to_xyzw(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """The csv writes the scalar first; a viewer reading the rrd must see it last."""
    _, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    stored: list[list[list[float]]] = column_rows(store, f"{schema.rig_path(0)}:Transform3D:quaternion").column(1).to_pylist()
    expected_xyzw: Float64[ndarray, "4"] = np.asarray(FIXTURE_WORLD_R_RIG.as_quat(), dtype=np.float64)
    # float32 on the wire, so a loose tolerance is the honest one.
    np.testing.assert_allclose(np.asarray(stored[0][0], dtype=np.float64), expected_xyzw, atol=1e-6)
    # The dropout row keeps its translation but loses its rotation, per slam-evals' repair.
    np.testing.assert_allclose(np.asarray(stored[GT_DROPOUT_ROW][0], dtype=np.float64), [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_the_gt_layer_carries_a_full_path_and_a_per_pose_trail(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """The overview strip is static and whole; the trail is one point per pose, for the cursor window."""
    _, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    trajectory: str = schema.trajectory_path("gt")
    strips: list[list[list[float]]] = (
        store.reader(index=None, contents=trajectory).to_arrow_table().to_pylist()[0][f"{trajectory}:LineStrips3D:strips"]
    )
    assert len(strips) == 1, "the whole trajectory is one strip"
    assert len(strips[0]) == GT_NUM_POSES
    assert column_rows(store, f"{schema.trail_path('gt')}:Points3D:positions").num_rows == GT_NUM_POSES


def test_only_the_gt_layer_states_the_world_axes(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """The pose layer establishes a world frame, so it owns the root ViewCoordinates."""
    _, base_target, gt_target = converted_index

    gt_root: pa.Table = read_back(gt_target).reader(index=None, contents="/").to_arrow_table()
    assert "/:ViewCoordinates:xyz" in gt_root.column_names
    declared: list[int] = [int(direction.value) for direction in WORLD_UP_VIEW_COORDINATES[MSD_DEVICES["index"].world_up].coordinates]
    assert [int(value) for value in gt_root.to_pylist()[0]["/:ViewCoordinates:xyz"][0]] == declared
    assert "/:ViewCoordinates:xyz" not in read_back(base_target).reader(index=None, contents="/").to_arrow_table().column_names


def test_the_gt_properties_report_the_poses_the_repairs_and_the_measured_axis(converted_index: tuple[FakeHub, Path, Path]) -> None:
    _, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    gt: dict[str, object] = recording_properties(store, "gt")
    assert gt["num_poses"] == GT_NUM_POSES
    assert gt["duration_ns"] == (GT_NUM_POSES - 1) * GT_PERIOD_NS
    assert gt["num_sanitized"] == 1
    assert gt["source"] == MSD_DEVICES["index"].gt_source
    assert gt["world_up"] == MSD_DEVICES["index"].world_up
    assert gt["measured_up"] == "+y"
    measured_fraction: object = gt["measured_up_fraction"]
    assert isinstance(measured_fraction, float) and measured_fraction > 0.9


def test_a_derived_layer_carries_its_own_properties_and_no_recording_info(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """A derived layer is the same recording as its base, so it states nothing about the recording.

    Base owns the ``RecordingInfo`` — the name a viewer shows and the wall clock
    of the conversion. gt is written with ``send_properties=False`` and no
    recording name, so a rebuild cannot silently restate either (its own
    ``start_time`` would be whenever it was last rebuilt). Its ``property:gt:*``
    group still lands, which is the pair of behaviours this asserts together.
    """
    _, base_target, gt_target = converted_index

    gt: dict[str, object] = recording_properties(read_back(gt_target), "gt")
    assert gt["num_poses"] == GT_NUM_POSES, "the layer's own property group lands with send_properties=False"

    gt_columns: set[str] = set(read_back(gt_target).reader(index=None, contents="/__properties/**").to_arrow_table().column_names)
    base_columns: set[str] = set(read_back(base_target).reader(index=None, contents="/__properties/**").to_arrow_table().column_names)
    assert not [name for name in gt_columns if name.startswith("property:RecordingInfo:")], f"gt states a RecordingInfo: {sorted(gt_columns)}"
    assert "property:RecordingInfo:start_time" in base_columns, "base owns the recording's wall clock"
    assert "property:RecordingInfo:name" in base_columns, "and its name"

