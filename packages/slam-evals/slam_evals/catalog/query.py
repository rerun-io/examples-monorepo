"""Eager queries over the mounted slam-evals catalog.

We materialise to ``pandas.DataFrame`` inside the helper rather than
returning a lazy DataFusion ``DataFrame``: the rerun-sdk 0.31 catalog
bindings can drop the underlying ``TaskContextProvider`` when a lazy
frame escapes the original ``rr.server.Server`` context, and several
``Expr`` methods the docs imply (``Expr.like`` etc.) aren't actually
exposed in this datafusion-py build. Push filters into pandas — at
catalog scale (109 rows of metadata here) the difference doesn't matter,
and the API is reliable.
"""

from __future__ import annotations

import pandas as pd
import rerun as rr

# Column names in the segment table follow the pattern
# ``property:<property_name>:<archetype_or_custom>:<field>``. For AnyValues
# properties the archetype segment is collapsed, so we see ``property:info:<field>``
# directly. Keep this list in sync with ``send_sequence_properties``.
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "rerun_segment_id",
    "property:RecordingInfo:name",
    "property:info:dataset",
    "property:info:sequence",
    "property:info:slug",
    "property:info:modality",
    "property:info:has_imu",
    "property:info:has_depth",
    "property:info:has_stereo",
    "property:info:num_rgb_frames",
    "property:info:num_gt_poses",
    "property:info:num_imu_samples",
    "property:info:trajectory_len_m",
    "property:info:duration_s",
    "property:info:fps_rgb",
    "property:info:has_calibration",
)


def segment_summary(server: rr.server.Server, *, dataset_name: str = "vslam") -> pd.DataFrame:
    """One row per recording with the standard ``info`` properties as columns.

    Columns absent in a given recording layer come back as nulls — that's
    fine; it just means the producer of that RRD didn't emit them.
    """
    dataset = server.client().get_dataset(dataset_name)
    table = dataset.segment_table()
    existing = {c.name for c in table.schema()}
    keep = [c for c in _SUMMARY_COLUMNS if c in existing]
    return table.select(*keep).to_pandas()
