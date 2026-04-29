"""pytest fixtures for slam-evals.

The synthetic-sequence builder lives in ``slam_evals.data.synthetic`` so
both the test suite and the ``slam-evals-smoke`` CLI tool can build the
same per-modality fixtures.

This module also hosts the ``segment_summary`` fixture, an FFI-safe
wrapper around ``Dataset.segment_table().to_pandas()`` used by the
ingest round-trip tests. The rerun-sdk 0.31 catalog bindings hit a
``TaskContextProvider went out of scope over FFI boundary`` DataFusion
error if you try to call ``select(*cols).to_pandas()`` *inline* in a
caller. Wrapping the same code in a closure below makes it work —
empirically the function-call frame keeps the right Rust-side context
alive long enough for the materialisation. Inlining the body, even
verbatim, fails. We've verified this against the
``test_ingest_modalities`` round-trips: the helper passes, the same body
copied into a test fails. When the SDK fixes the underlying lifetime
bug, this can shrink to a one-liner or be deleted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import rerun as rr

from slam_evals.data.synthetic import Fixture, build_fixture
from slam_evals.data.types import Modality

# Property columns the test summary tries to surface. The select() filter in
# the ``segment_summary`` closure drops any that aren't present in the segment_table
# schema, so this list can include columns from optional layers (video_1,
# depth_<i>, imu_0) — they come back missing for sequences whose modality
# doesn't have those layers.
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "rerun_segment_id",
    "property:RecordingInfo:name",
    # info — cross-cutting (calibration.rrd)
    "property:info:dataset",
    "property:info:sequence",
    "property:info:slug",
    "property:info:modality",
    "property:info:has_imu",
    "property:info:has_depth",
    "property:info:has_stereo",
    "property:info:has_calibration",
    # groundtruth — trajectory shape (groundtruth.rrd)
    "property:groundtruth:num_poses",
    "property:groundtruth:trajectory_len_m",
    "property:groundtruth:duration_s",
    "property:groundtruth:has_rotation",
    # video streams (video_<i>.rrd)
    "property:video_0:codec",
    "property:video_0:fps",
    "property:video_0:num_frames",
    "property:video_0:width",
    "property:video_0:height",
    "property:video_1:codec",
    "property:video_1:fps",
    "property:video_1:num_frames",
    "property:video_1:width",
    "property:video_1:height",
    # depth cameras (depth_<i>.rrd)
    "property:depth_0:depth_factor",
    "property:depth_0:num_frames",
    "property:depth_0:width",
    "property:depth_0:height",
    "property:depth_1:depth_factor",
    "property:depth_1:num_frames",
    "property:depth_1:width",
    "property:depth_1:height",
    # imu (imu_0.rrd)
    "property:imu_0:num_samples",
    "property:imu_0:rate_hz",
    # calibration summary (calibration.rrd)
    "property:calibration:num_cameras",
    "property:calibration:cam0_name",
    "property:calibration:cam0_width",
    "property:calibration:cam0_height",
    "property:calibration:cam0_fx",
    "property:calibration:cam0_fy",
    "property:calibration:cam0_cx",
    "property:calibration:cam0_cy",
    "property:calibration:cam0_distortion_type",
    "property:calibration:depth_factor",
    "property:calibration:has_imu_params",
)


@pytest.fixture
def segment_summary():
    """Factory fixture: ``segment_summary(server, source="EUROC")`` → ``pd.DataFrame``.

    One row per segment in ``source``'s catalog Dataset, with the standard
    property columns. ``source`` is the original (uppercase) source-benchmark
    name; the catalog Dataset is stored under its lowercased form. Columns
    absent for a given segment (e.g. ``property:video_1:codec`` on a mono
    sequence) come back as nulls — that's the catalog server filling in
    missing values across heterogeneous-modality rows.

    Implemented as a closure to keep the FFI-safe function-call frame; see
    the module docstring.
    """

    def _summary(server: rr.server.Server, *, source: str) -> pd.DataFrame:
        dataset = server.client().get_dataset(source.lower())
        table = dataset.segment_table()
        existing = {c.name for c in table.schema()}
        keep = [c for c in _SUMMARY_COLUMNS if c in existing]
        return table.select(*keep).to_pandas()

    return _summary


@pytest.fixture
def fixture_factory(tmp_path: Path):
    """Factory fixture: ``fixture_factory(modality, n_frames=5)`` → ``Fixture``."""

    def _make(modality: Modality, *, n_frames: int = 5) -> Fixture:
        return build_fixture(tmp_path / "bench", modality=modality, n_frames=n_frames)

    return _make
