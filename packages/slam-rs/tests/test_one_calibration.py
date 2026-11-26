"""The replay, the gate's ``run_segment`` and the catalog layer build one estimator from one feed.

Each tool is run up to the point where it constructs ``_core.Vio``; a stand-in records the calibration and the config
it was handed. A feed that states a non-default gravity and a rolling-shutter readout must reach all three the same
way, and the run's hashed provenance must cover those recording-derived settings.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from fixture_types import CameraFactory

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, ImuCalib
from slam_rs.trajectory import Trajectory, empty_trajectory

GRAVITY_MS2: float = 9.8
READOUT_S: float = 0.0159


class Built(Exception):
    """Raised by the stand-in ``Vio``: the tool got as far as constructing the estimator."""

    def __init__(self, calibration: _core.Calibration, config: _core.VioConfig) -> None:
        super().__init__("built")
        self.calibration_json: str = calibration.to_json()
        self.config_json: str = config.to_json()
        self.gravity_ms2: float = calibration.gravity_ms2
        self.readout_s: list[float | None] = calibration.readout_s


def standin_vio(calibration: _core.Calibration, config: _core.VioConfig, backend: str = "auto", **_kwargs: object) -> _core.Vio:
    """Record what the tool built the estimator from, then stop the tool."""
    raise Built(calibration, config)


@dataclass(slots=True)
class FakeFeed:
    """What the three tools read from a feed before they build the estimator."""

    cameras: tuple[CameraCalib, ...]
    imu: ImuCalib
    segment_id: str = "msd-index__fake"
    capture_start_time_ns: int = 0
    export_offset_ns: int = 0
    camera_positions: tuple[int, ...] = (0, 1)
    has_ground_truth: bool = False

    @property
    def frame_t_ns(self) -> list[int]:
        return [0, 1]

    def ground_truth_between(self, start_ns: int, stop_ns: int) -> Trajectory:
        return empty_trajectory()


@pytest.fixture
def feed(camera: CameraFactory, imu: ImuCalib) -> FakeFeed:
    cameras: tuple[CameraCalib, ...] = (replace(camera(0, 0.0), readout_s=READOUT_S), replace(camera(1, 0.1), readout_s=READOUT_S))
    return FakeFeed(cameras=cameras, imu=replace(imu, gravity_ms2=GRAVITY_MS2))


def built_by(run: object) -> Built:
    """Run a tool and return what its estimator was built from."""
    with pytest.raises(Built) as caught:
        run()  # type: ignore[operator]
    return caught.value


def test_replay_gate_and_catalog_layer_build_the_same_estimator(feed: FakeFeed, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rerun.catalog import DatasetEntry
    from simplecv.rerun_log_utils import RerunTyroConfig

    import slam_rs.apis.catalog_layer as catalog_layer
    import slam_rs.apis.replay as replay
    import slam_rs.tracking as tracking
    from slam_rs.config import load_slam_config
    from slam_rs.reference import load_benchmarks

    @contextmanager
    def open_fake(*args: object, **kwargs: object) -> Iterator[FakeFeed]:
        yield feed

    monkeypatch.setattr(_core, "Vio", standin_vio)
    for module in (tracking, replay, catalog_layer):
        monkeypatch.setattr(module, "open_segment", open_fake)
    settings = load_slam_config()
    segment = next(s for s in load_benchmarks(settings).segments if s.dataset_name == "msd-index")

    gate: Built = built_by(lambda: tracking.run_segment(settings, segment, backend="cpu"))
    replayed: Built = built_by(
        lambda: replay.main(replay.Config(rr_config=RerunTyroConfig(headless=True), segment=segment.segment_id, backend="cpu", output_csv=tmp_path / "t.csv"))
    )
    handle: DatasetEntry = DatasetEntry.__new__(DatasetEntry)
    monkeypatch.setattr(catalog_layer, "resolve_catalog_segments", lambda segments: (replace(segments[0], dataset=handle),))
    layered: Built = built_by(
        lambda: catalog_layer.main(catalog_layer.Config(segment=segment.segment_id, output_dir=tmp_path, backend="cpu", decode_device="cpu"))
    )

    assert gate.gravity_ms2 == GRAVITY_MS2
    assert gate.readout_s == [READOUT_S, READOUT_S]
    for other in (replayed, layered):
        assert other.calibration_json == gate.calibration_json
        # The gate scores the raw VIO, so it builds the same config with the back end off (`tracking.raw_vio`).
        assert _core.VioConfig.from_json(other.config_json).with_backend("off").to_json() == gate.config_json


def test_the_run_hash_covers_the_recording_derived_calibration(feed: FakeFeed) -> None:
    from slam_rs.config import config_text_sha256

    stated: str = _core.Calibration.from_catalog(feed.cameras, feed.imu).to_json()
    default: str = _core.Calibration.from_catalog(feed.cameras, replace(feed.imu, gravity_ms2=None)).to_json()
    no_readout: str = _core.Calibration.from_catalog(tuple(replace(c, readout_s=None) for c in feed.cameras), feed.imu).to_json()
    assert len({config_text_sha256(text) for text in (stated, default, no_readout)}) == 3
    assert _core.Calibration.from_json(default).gravity_ms2 == 9.81
