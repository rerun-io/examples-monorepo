"""How each catalog dataset's recordings are read: the rig profile slam.toml states, and the tools that honour it.

EuRoC's ``video_time`` is the benchmark's own nanosecond clock, MSD's is relative to
``property:capture:start_time_ns``. A tool that read EuRoC as MSD would export every
pose 1.4e18 ns late, and RootSLAM's scorer would associate nothing.
"""

import pytest
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import tracking
from slam_rs.apis import replay
from slam_rs.catalog_feed import MSD_RIG, CatalogSegment, RigProfile, dataset_rig, open_segment
from slam_rs.config import SlamConfig
from slam_rs.reference import Benchmarks

EUROC_SEGMENT: str = "vslam-lab-euroc__MH_01_easy"
"""RootSLAM's EMH01, the EuRoC sequence of the smoke set."""
EUROC_FIRST_CAM0_NS: int = 1403636579763555584
"""The first row of MH_01_easy's ``mav0/cam0/data.csv``."""


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
    assert dataset_rig(settings.dataset("vslam-lab-euroc")) == RigProfile(video_time_is_absolute=True)


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
    assert seen == [RigProfile(video_time_is_absolute=True)]


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
