"""How each catalog dataset's recordings are read: the rig profile slam.toml states, and the tools that honour it.

EuRoC's ``video_time`` is the benchmark's own nanosecond clock, MSD's is relative to
``property:capture:start_time_ns``. A tool that read EuRoC as MSD would export every
pose 1.4e18 ns late, and RootSLAM's scorer would associate nothing.
"""

from pathlib import Path

import pytest
from rerun.catalog import DatasetEntry
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import tracking
from slam_rs.apis import catalog_layer, dump_clip, replay
from slam_rs.catalog_feed import MSD_RIG, CatalogSegment, RigProfile, dataset_rig, open_segment
from slam_rs.config import SlamConfig
from slam_rs.reference import Benchmarks

EUROC_SEGMENT: str = "vslam-lab-euroc__MH_01_easy"
"""RootSLAM's EMH01, the EuRoC sequence of the smoke set."""
EUROC_FIRST_CAM0_NS: int = 1403636579763555584
"""The first row of MH_01_easy's ``mav0/cam0/data.csv``."""
EUROC_RIG: RigProfile = RigProfile(video_time_is_absolute=True, frameset_tolerance_ns=1_000_000)
"""EuRoC as slam.toml states it: the benchmark clock on ``video_time``, framesets matched within 1 ms."""
TUMVI_SEGMENT: str = "tum-vi__room1"
"""RootSLAM's TR1."""
TUMVI_FIRST_PAIR_NS: int = 1520530308199447626
"""The first row of room1's ``mav0/cam0/data.csv`` (cam1's first row has the same stamp)."""
TUMVI_RIG: RigProfile = RigProfile(video_time_is_absolute=True)
"""TUM-VI as slam.toml states it: the dataset clock on ``video_time``; the conversion keeps only shared stamps, so no tolerance."""


class _Opened(Exception):
    """Raised by the stand-in feed so a tool stops once it has chosen how to read the segment."""


def _capture_profile(monkeypatch: pytest.MonkeyPatch, module: object) -> list[RigProfile]:
    seen: list[RigProfile] = []

    def opened(_source: object, *_args: object, profile: RigProfile = MSD_RIG, **_kwargs: object) -> None:
        seen.append(profile)
        raise _Opened

    monkeypatch.setattr(module, "open_segment", opened)
    return seen


def test_msd_datasets_read_as_msd_and_euroc_reads_the_device_clock(settings: SlamConfig) -> None:
    for name in ("msd-index", "msd-g2", "msd-odyssey"):
        assert dataset_rig(settings.dataset(name)) == MSD_RIG
    assert dataset_rig(settings.dataset("vslam-lab-euroc")) == EUROC_RIG
    assert dataset_rig(settings.dataset("tum-vi")) == TUMVI_RIG


@pytest.mark.parametrize(("segment", "absolute"), [(EUROC_SEGMENT, True), ("msd-g2__MGO_others__MGO07_mapping_easy", False)])
def test_replay_opens_a_segment_with_its_datasets_rig(monkeypatch: pytest.MonkeyPatch, segment: str, absolute: bool) -> None:
    seen: list[RigProfile] = _capture_profile(monkeypatch, replay)
    with pytest.raises(_Opened):
        replay.main(replay.Config(rr_config=RerunTyroConfig(headless=True), segment=segment, decode_device="cpu"))
    assert [profile.video_time_is_absolute for profile in seen] == [absolute]


def test_the_gate_runs_a_euroc_segment_on_the_device_clock(
    benchmarks: Benchmarks, settings: SlamConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[RigProfile] = _capture_profile(monkeypatch, tracking)
    with pytest.raises(_Opened):
        tracking.run_segment(settings, benchmarks.by_id(EUROC_SEGMENT), backend="cpu")
    assert seen == [EUROC_RIG]


@pytest.mark.parametrize(("segment", "expected"), [(EUROC_SEGMENT, EUROC_RIG), ("msd-g2__MGO_others__MGO07_mapping_easy", MSD_RIG)])
def test_the_catalog_layer_opens_a_segment_with_its_datasets_rig(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, segment: str, expected: RigProfile
) -> None:
    """Read as MSD, EuRoC's MH_04 and V2_03 were refused and the layer stored the capture start as its export offset."""

    def resolved(sources: tuple[CatalogSegment, ...], *_args: object, **_kwargs: object) -> tuple[CatalogSegment, ...]:
        # A bare handle: the stand-in feed stops the tool before it touches the dataset.
        return tuple(CatalogSegment(s.url, s.dataset_name, s.segment_id, dataset=object.__new__(DatasetEntry)) for s in sources)

    monkeypatch.setattr(catalog_layer, "resolve_catalog_segments", resolved)
    seen: list[RigProfile] = _capture_profile(monkeypatch, catalog_layer)
    with pytest.raises(_Opened):
        catalog_layer.main(catalog_layer.Config(segment=segment, output_dir=tmp_path, decode_device="cpu"))
    assert seen == [expected]


def test_dump_clip_opens_a_euroc_segment_on_the_device_clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[RigProfile] = _capture_profile(monkeypatch, dump_clip)
    with pytest.raises(_Opened):
        dump_clip.main(dump_clip.Config(segment=EUROC_SEGMENT, output=tmp_path))
    assert seen == [EUROC_RIG]


@pytest.mark.integration
@pytest.mark.usefixtures("live_catalog")
def test_euroc_opens_from_the_catalog_on_its_own_clock(settings: SlamConfig) -> None:
    """MH_01_easy as registered: two radtan cameras with no valid radius, Basalt's EuRoC IMU model, no clock shift, ground truth."""
    from rerun.catalog import CatalogClient

    if "vslam-lab-euroc" not in {entry.name for entry in CatalogClient(settings.catalog_url).datasets()}:
        pytest.skip(f"vslam-lab-euroc is not registered on {settings.catalog_url}; register it as packages/slam-rs/docs/reproduce.md describes")
    with open_segment(
        CatalogSegment(settings.catalog_url, "vslam-lab-euroc", EUROC_SEGMENT), profile=dataset_rig(settings.dataset("vslam-lab-euroc"))
    ) as feed:
        assert feed.export_offset_ns == 0
        assert feed.imu.cam_time_offset_ns == 0
        assert int(feed.frame_t_ns[0]) == EUROC_FIRST_CAM0_NS
        assert len(feed.frame_t_ns) == 3682
        assert [(camera.model, camera.width, camera.height, camera.distortion_valid_radius) for camera in feed.cameras] == [
            ("radtan8", 752, 480, None)
        ] * 2
        # The model RootSLAM and Basalt run EuRoC with (euroc_ds_calib.json), not VSLAM-LAB's OKVIS sigmas.
        assert (feed.imu.accel_noise_std, feed.imu.gyro_noise_std, feed.imu.accel_bias_std, feed.imu.gyro_bias_std) == (
            0.016, 0.000282, 0.001, 0.0001
        )
        assert feed.has_ground_truth
        truth = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
        # state_groundtruth_estimate0 starts 1.075 s after the first frame and runs at 200 Hz.
        assert int(truth.t_ns[0]) == 1403636580838555648


@pytest.mark.integration
@pytest.mark.usefixtures("live_catalog")
@pytest.mark.parametrize(("sequence", "framesets"), [("MH_04_difficult", 2031), ("V2_03_difficult", 1626)])
def test_only_frames_both_euroc_cameras_took_together_are_fed(settings: SlamConfig, sequence: str, framesets: int) -> None:
    """Each camera's frames keep their own stamps, but the conversion holds equal frame counts, not equal stamps.

    The raw ASL has 2032 (MH_04) and 1921 (V2_03) stamps in both cameras. VSLAM-LAB keeps
    as many cam1 frames as cam0 has, so the converted cam1 lists 1 and 296 frames cam0 never took
    (up to 20.6 s away): matching within 1 ms feeds 2031 and 1626 stereo framesets, and every pair it feeds
    holds two frames with the same stamp. V2_03's lost 295 pairs are a known data issue of the conversion.
    """
    from rerun.catalog import CatalogClient

    if "vslam-lab-euroc" not in {entry.name for entry in CatalogClient(settings.catalog_url).datasets()}:
        pytest.skip(f"vslam-lab-euroc is not registered on {settings.catalog_url}; register it as packages/slam-rs/docs/reproduce.md describes")
    with open_segment(
        CatalogSegment(settings.catalog_url, "vslam-lab-euroc", f"vslam-lab-euroc__{sequence}"),
        profile=dataset_rig(settings.dataset("vslam-lab-euroc")),
        include_ground_truth=False,
    ) as feed:
        assert len(feed.frame_t_ns) == framesets
        stamps = [stream.t_ns for stream in feed.video.streams]
        pairs = feed.index.frame_index
        assert (stamps[0][pairs[:, 0]] == stamps[1][pairs[:, 1]]).all()


@pytest.mark.integration
@pytest.mark.usefixtures("live_catalog")
def test_tum_vi_opens_from_the_catalog_as_kb4_stereo_on_its_own_clock(settings: SlamConfig) -> None:
    """room1 as registered: two 512x512 KB4 cameras, imu_config.yaml's noise model, no clock shift, every stereo pair, mocap GT."""
    from rerun.catalog import CatalogClient

    if "tum-vi" not in {entry.name for entry in CatalogClient(settings.catalog_url).datasets()}:
        pytest.skip(f"tum-vi is not registered on {settings.catalog_url}; register it as packages/slam-rs/docs/reproduce.md describes")
    with open_segment(CatalogSegment(settings.catalog_url, "tum-vi", TUMVI_SEGMENT), profile=dataset_rig(settings.dataset("tum-vi"))) as feed:
        assert feed.export_offset_ns == 0
        assert feed.imu.cam_time_offset_ns == 0
        assert int(feed.frame_t_ns[0]) == TUMVI_FIRST_PAIR_NS
        assert len(feed.frame_t_ns) == 2821
        assert [(camera.model, camera.width, camera.height) for camera in feed.cameras] == [("kb4", 512, 512)] * 2
        # TUM-VI's own Kalibr model (the inflated values), not Basalt's tumvi_512_ds_calib.json that RootSLAM ran.
        assert (feed.imu.accel_noise_std, feed.imu.gyro_noise_std, feed.imu.accel_bias_std, feed.imu.gyro_bias_std) == (
            0.0028, 0.00016, 0.00086, 0.000022
        )
        truth = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
        # mocap0 runs at ~120 Hz from 9.8 ms before the first frame: the first pose inside the frame span is its third row.
        assert int(truth.t_ns[0]) == 1520530308206345351
