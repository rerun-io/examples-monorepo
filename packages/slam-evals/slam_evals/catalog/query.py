"""Tested FFI-safe wrapper around ``Dataset.segment_table().to_pandas()``.

This module exists for one reason: the rerun-sdk 0.31 catalog bindings
hit a ``TaskContextProvider went out of scope over FFI boundary``
DataFusion error if you try to call ``select(*cols).to_pandas()``
*inline* in a caller. Wrapping the same code in ``segment_summary``
below makes it work — empirically the function-call frame keeps the
right Rust-side context alive long enough for the materialisation.
Inlining the body, even verbatim, fails. We've verified this against
the ``test_ingest_modalities`` round-trips: ``segment_summary`` passes,
the same body copied into a test helper fails.

So treat this file as load-bearing infrastructure, not a "summary
helper" you can rewrite at the call site. When the SDK fixes the
underlying lifetime bug, this can shrink to a one-liner or be deleted.

Property columns follow the per-layer schema documented in
``docs/schema.md``. Cross-cutting metadata lives on the calibration layer
under ``info``; per-stream metadata lives on each stream's own layer.
"""

from __future__ import annotations

import pandas as pd
import rerun as rr

# Columns the summary view tries to surface. The select() filter below drops
# any that aren't actually present in the segment_table schema, so this list
# can include columns from optional layers (video_1, depth_<i>, imu_0) — they
# come back missing for sequences whose modality doesn't have those layers.
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


def segment_summary(server: rr.server.Server, *, dataset_name: str = "vslam") -> pd.DataFrame:
    """One row per segment with the standard properties as columns.

    Columns absent for a given segment (e.g. ``property:video_1:codec`` on a
    mono sequence) come back as nulls — that's the catalog server filling
    in missing values across heterogeneous-modality rows.
    """
    dataset = server.client().get_dataset(dataset_name)
    table = dataset.segment_table()
    existing = {c.name for c in table.schema()}
    keep = [c for c in _SUMMARY_COLUMNS if c in existing]
    return table.select(*keep).to_pandas()
