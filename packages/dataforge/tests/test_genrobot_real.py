"""Gen-HumanWholeBody on two real 8 s episodes from the Hub (device V6 firmware 2.1.28, device V7 firmware 2.2.9).

The files are the pinned commit's own bytes, fetched once into the raw root at their Hub paths:

    hf download genrobot2025/Gen-HumanWholeBody --repo-type dataset --revision <REVISION> \\
        --local-dir packages/dataforge/data/raw/genrobot <path>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.chunk as rrc
from conftest import column_rows, raw_asset, read_back, recording_properties
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray

from dataforge import paths, schema, writing
from dataforge.datasets.genrobot_layers import BaseCensus, shipped_counts, write_base_layer, write_sidecar
from dataforge.datasets.genrobot_pose import (
    PoseTrack,
    check_mano,
    project,
    read_pose_track,
    read_rig_poses,
    world_T_cams,
    write_actions_layer,
    write_pose_layer,
)
from dataforge.datasets.genrobot_source import (
    NUM_CAMERAS,
    REVISION,
    RIG_FROM_CALIB,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    Episode,
    EpisodeFile,
    is_keyframe,
    parse_hub_path,
    read_episode,
    rig_T_cam,
)
from dataforge.identity import SequenceIdentity

pytestmark = pytest.mark.integration

BARBER: str = "data/business/beauty_personal_care/barber_shop/daily_work/d3/d3982934590a450ca7fe212e749ecf75.mcap"
"""Device V7 (HardwareVersion V6), firmware 2.2.9; camera4 starts a frame late."""
CLOTHES: str = "data/business/fashion/clothes_shop/daily_work/d9/d9b8ece34b9f4f1ead81c78bf725824c.mcap"
"""Device V6, firmware 2.1.28; cameras 1 and 4 start a frame late."""


def sample(path: str) -> Path:
    """A sample MCAP in the raw root, or a skip naming it."""
    return raw_asset("Gen-HumanWholeBody sample episode", paths.raw_root() / "genrobot" / path)


@pytest.mark.parametrize("path", [BARBER, CLOTHES])
def test_episode_records_read_from_the_mcap(path: str) -> None:
    episode: Episode = read_episode(EpisodeFile(sample(path)))
    assert len(episode.calibrations) == NUM_CAMERAS
    assert all((calibration.width, calibration.height) == (SOURCE_WIDTH, SOURCE_HEIGHT) for calibration in episode.calibrations)
    assert episode.annotation is not None and episode.system is not None, "both samples ship annotation and system info"
    assert episode.annotation.bold_mark and episode.annotation.sub_segments
    assert episode.annotation.video_id == episode.system.uuid
    assert path.removeprefix("data/").startswith(episode.annotation.sst), "sst is the Hub domain path"
    frames: int = episode.clock.frame_times_ns.size
    assert 240 <= frames <= 252 and np.all(np.diff(episode.clock.frame_times_ns) > 0)
    assert episode.clock.t0_ns <= episode.clock.frame_times_ns[0]


@pytest.mark.parametrize("path", [BARBER, CLOTHES])
def test_cameras_fan_out_left_to_right_with_a_front_stereo_pair(path: str) -> None:
    """The handoff's SIFT/essential-matrix check found the fan; ``T_b_c`` through ``RIG_FROM_CALIB`` must reproduce it.

    In the forward-left-up rig frame camera0 looks 42 degrees left and camera5 43 degrees right; camera2 and
    camera3 both look ahead, 6 cm apart. Every image is upright: its down axis points at the ground.
    """
    episode: Episode = read_episode(EpisodeFile(sample(path)))
    poses: Float64[ndarray, "6 4 4"] = np.stack([rig_T_cam(calibration.T_b_c) for calibration in episode.calibrations])
    axes: Float64[ndarray, "6 3"] = poses[:, :3, 2]
    yaw_deg: Float64[ndarray, "6"] = np.degrees(np.arctan2(axes[:, 1], axes[:, 0]))
    np.testing.assert_allclose(yaw_deg, [42.0, 23.0, 0.0, 0.0, -22.0, -43.0], atol=2.0)
    assert np.all(np.abs(np.degrees(np.arcsin(axes[:, 2]))) < 20.0), "every camera looks within 20 degrees of level"
    assert np.all(poses[:, 2, 1] < -0.75), "image down points at the ground"
    assert 0.055 < np.linalg.norm(poses[2, :3, 3] - poses[3, :3, 3]) < 0.065


@pytest.fixture(scope="module")
def barber_base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, BaseCensus, Episode]:
    """The barber-shop episode's base layer, written once for the module."""
    mcap: Path = sample(BARBER)
    target: Path = tmp_path_factory.mktemp("genrobot") / "base.rrd"
    episode: Episode = read_episode(EpisodeFile(mcap))
    domain, uuid = parse_hub_path(BARBER)
    census: BaseCensus = write_base_layer(
        SequenceIdentity("genrobot", (uuid,)), mcap, target, episode=episode, domain=domain, hf_revision=REVISION, default_blueprint=None
    )
    return target, census, episode


def test_base_layer_holds_every_shipped_camera_frame_as_h264_video(barber_base: tuple[Path, BaseCensus, Episode]) -> None:
    target, census, _ = barber_base
    store: rrc.ChunkStore = read_back(target)
    counts: dict[str, int] = shipped_counts(sample(BARBER))
    for cam in range(NUM_CAMERAS):
        samples: pa.Table = column_rows(store, f"{schema.video_path(0, cam)}:VideoStream:sample")
        skipped: int = static_row(store, schema.cam_path(0, cam))[f"{schema.cam_path(0, cam)}:skipped_leading_samples"][0]
        assert samples.num_rows == census.camera_frames[cam]
        assert samples.num_rows + skipped == counts[f"/robot0/sensor/camera{cam}/compressed"], "every shipped sample is written or counted as skipped"
        first: bytes = bytes(samples.column(1).to_pylist()[0][0])
        assert is_keyframe(first), "the video starts at a keyframe"
        sps: int = first.find(b"\x00\x00\x01\x67")
        assert sps >= 0 and first[sps + 5] & 0x10 == 0, "constraint_set3_flag is cleared"
        codec: list = static_row(store, schema.video_path(0, cam))[f"{schema.video_path(0, cam)}:VideoStream:codec"]
        assert codec == [rr.VideoCodec.H264.value]


def test_base_layer_rig_pose_rows_carry_translation_and_rotation_together(barber_base: tuple[Path, BaseCensus, Episode]) -> None:
    target, census, _ = barber_base
    table: pa.Table = read_back(target).reader(index=schema.TIMELINE, contents=schema.rig_path(0)).to_arrow_table()
    translation: pa.ChunkedArray = table.column(f"{schema.rig_path(0)}:Transform3D:translation")
    quaternion: pa.ChunkedArray = table.column(f"{schema.rig_path(0)}:Transform3D:quaternion")
    assert translation.null_count == quaternion.null_count == 0
    assert table.num_rows == census.rig_poses > 0


def test_base_layer_imu_is_in_m_s2_under_the_solved_mounting(barber_base: tuple[Path, BaseCensus, Episode]) -> None:
    target, census, _ = barber_base
    store: rrc.ChunkStore = read_back(target)
    accel: Float64[ndarray, "n 3"] = np.asarray(column_rows(store, f"{schema.accel_path(0, 0)}:Scalars:scalars").column(1).to_pylist(), dtype=np.float64).reshape(-1, 3)
    assert accel.shape[0] == census.imu_samples > 1000
    assert 9.3 < np.linalg.norm(accel, axis=1).mean() < 10.3
    node: dict[str, list] = static_row(store, schema.imu_path(0, 0))
    np.testing.assert_allclose(np.asarray(node[f"{schema.imu_path(0, 0)}:Transform3D:mat3x3"][0]).reshape(3, 3, order="F"), RIG_FROM_CALIB)


def test_base_layer_frames_and_properties(barber_base: tuple[Path, BaseCensus, Episode]) -> None:
    target, census, episode = barber_base
    store: rrc.ChunkStore = read_back(target)
    frames: pa.Table = column_rows(store, "/frames:source_frame_id")
    assert frames.num_rows == census.frames == episode.clock.frame_times_ns.size
    # Read from the chunks: a reader view leaves out a column whose rows are all empty, as they are here.
    missing: list[list[str]] = [
        row for chunk in rrc.RrdReader(target).stream().filter(content="/frames", components=["missing_cameras"]) for row in chunk.to_record_batch().column("missing_cameras").to_pylist()
    ]
    assert missing == [[]] * census.frames, "every camera has an image at every reference frame of this episode"
    capture: dict[str, object] = recording_properties(store, "capture")
    assert capture["num_cameras"] == NUM_CAMERAS and capture["hf_revision"] == REVISION
    episode_properties: dict[str, object] = recording_properties(store, "episode")
    assert episode_properties["domain_0"] == "business" and episode_properties["domain_3"] == "daily_work"
    assert episode_properties["firmware"] == "2.2.9" and episode_properties["device"] == "DAS Ego V7"
    assert episode_properties["is_success"] is True
    instruction: list = static_row(store, schema.instruction_path())[f"{schema.instruction_path()}:TextDocument:text"]
    assert episode.annotation is not None
    assert instruction == [episode.annotation.bold_mark]


def static_row(store: rrc.ChunkStore, entity_path: str) -> dict[str, list]:
    """The static components of one entity."""
    return store.reader(index=None, contents=entity_path).to_arrow_table().to_pylist()[0]


@pytest.fixture(scope="module")
def barber_sidecar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The barber-shop episode's sidecar: its non-camera topics, the only input of every derived layer."""
    target: Path = tmp_path_factory.mktemp("genrobot-sidecar") / "source.rrd"
    write_sidecar(sample(BARBER), target, recording_id="genrobot__d3982934590a450ca7fe212e749ecf75")
    return target


def test_the_sidecar_reproduces_the_episode_without_the_cameras(barber_sidecar: Path) -> None:
    from_mcap: Episode = read_episode(EpisodeFile(sample(BARBER)))
    from_sidecar: Episode = read_episode(EpisodeFile(barber_sidecar))
    assert from_sidecar.clock.t0_ns == from_mcap.clock.t0_ns, "the sidecar keeps the MCAP's first stamp, cameras included"
    np.testing.assert_array_equal(from_sidecar.clock.frame_times_ns, from_mcap.clock.frame_times_ns)
    assert from_sidecar.calibrations == from_mcap.calibrations and from_sidecar.annotation == from_mcap.annotation
    assert from_sidecar.system == from_mcap.system and from_sidecar.validity == from_mcap.validity, "convert's base layer reads them from the sidecar"
    assert barber_sidecar.stat().st_size < 0.2 * sample(BARBER).stat().st_size


def test_mano_reproduces_both_shipped_hands(barber_sidecar: Path) -> None:
    """The handoff measured 0.00 mm (right) and 0.12 mm (left, after the two left-hand fixes) on this episode."""
    episode: Episode = read_episode(EpisodeFile(barber_sidecar))
    track: PoseTrack = read_pose_track(EpisodeFile(barber_sidecar), episode.clock)
    left, right = check_mano(track, 0), check_mano(track, 1)
    assert left.passed and right.passed
    assert left.median_m < 0.0005 and right.median_m < 0.0005


def test_pose_layer_fills_coco133_from_body_and_hands(barber_sidecar: Path, tmp_path: Path) -> None:
    episode: Episode = read_episode(EpisodeFile(barber_sidecar))
    track: PoseTrack = read_pose_track(EpisodeFile(barber_sidecar), episode.clock)
    coco: Float32[ndarray, "f 133 3"] = track.coco133
    present: Bool[ndarray, "133"] = np.isfinite(coco).all(axis=-1).all(axis=0)
    assert set(np.flatnonzero(present)) == set(range(5, 17)) | set(range(91, 133)), "body slots 5-16 and both hands; no face or feet"
    np.testing.assert_allclose(coco[:, 91], track.bones[:, 0, 0], atol=1e-6, err_msg="the left hand root is the tracked wrist")
    head_offset_m: Float64[ndarray, "f"] = np.linalg.norm(track.body[:, 15] - read_rig_poses(EpisodeFile(barber_sidecar), episode.clock, track.frame_indices)[:, :3, 3], axis=1)
    assert np.nanmedian(head_offset_m) < 0.25, "the body's head follows the device, so both share one world"
    target: Path = tmp_path / "pose.rrd"
    assert track.has_pose
    with writing.atomic_recording(target, recording_id="genrobot__d3982934590a450ca7fe212e749ecf75", send_properties=False) as recording:
        write_pose_layer(recording, track)
    store: rrc.ChunkStore = read_back(target)
    assert column_rows(store, f"{schema.coco133_xyz_path()}:Points3D:positions").num_rows == episode.clock.frame_times_ns.size
    assert column_rows(store, f"{schema.body_path('smplx22')}:Points3D:positions").num_rows == episode.clock.frame_times_ns.size
    assert recording_properties(store, "pose")["coverage_left"] == 1.0


def test_projected_hands_land_inside_the_front_camera(barber_sidecar: Path) -> None:
    """Pixel evidence decides whether they land ON the hands; this checks the chain puts them in frame at all."""
    episode: Episode = read_episode(EpisodeFile(barber_sidecar))
    track: PoseTrack = read_pose_track(EpisodeFile(barber_sidecar), episode.clock)
    rig: Float64[ndarray, "f 4 4"] = read_rig_poses(EpisodeFile(barber_sidecar), episode.clock, track.frame_indices)
    pixels: Float64[ndarray, "f 133 2"] = project(world_T_cams(rig, episode.calibrations)[2], episode.calibrations[2], track.coco133.astype(np.float64))
    wrists_seen: Bool[ndarray, "f"] = np.isfinite(pixels[:, [91, 112]]).all(axis=(1, 2))
    assert wrists_seen.mean() > 0.8, "both wrists in the front camera on most frames of a cape-folding task"


def test_actions_layer_tracks_coarse_and_fine_labels(barber_sidecar: Path, tmp_path: Path) -> None:
    episode: Episode = read_episode(EpisodeFile(barber_sidecar))
    target: Path = tmp_path / "actions.rrd"
    assert episode.annotation is not None and episode.annotation.segments_info
    with writing.atomic_recording(target, recording_id="genrobot__d3982934590a450ca7fe212e749ecf75", send_properties=False) as recording:
        write_actions_layer(recording, episode.clock, episode.annotation)
    store: rrc.ChunkStore = read_back(target)
    fine: pa.Table = column_rows(store, f"{schema.actions_path('fine')}:StateChange:state")
    states: list[list[str]] = fine.column(1).to_pylist()
    assert all(len(row) == 1 for row in states), "the sample's sub-segments never overlap, so the fine track is one lane"
    shipped: list[str] = [sub.fine_label for sub in sorted(episode.annotation.sub_segments, key=lambda sub: sub.start_frame)]
    assert [row[0] for row in states if row[0]] == shipped
    assert recording_properties(store, "actions")["sub_segment_is_success"] == [sub.is_success for sub in episode.annotation.sub_segments]
