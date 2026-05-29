"""Tests for Robocap RRD semantic comparison helpers."""

import pyarrow as pa
import pytest

from robocap_slam.rrd_compare import (
    RrdComparisonError,
    comparable_schema_columns,
    compare_arrow_tables,
)


def test_comparable_schema_columns_skip_media_and_recording_metadata() -> None:
    schema: pa.Schema = pa.schema(
        [
            pa.field("video_time", pa.duration("ns")),
            pa.field("/world/rig/cam0/video:VideoStream:sample", pa.binary()),
            pa.field("/world/rig/cam0/image:Image:buffer", pa.binary()),
            pa.field("property:RecordingInfo:start_time", pa.timestamp("ns")),
            pa.field("/world/trajectory:LineStrips3D:columns", pa.list_(pa.list_(pa.float32()))),
            pa.field("/world/rig/landmarks:Points3D:keypoint_ids", pa.list_(pa.uint64())),
        ]
    )

    comparable_columns: tuple[str, ...] = comparable_schema_columns(schema, index="video_time")

    assert comparable_columns == (
        "/world/rig/landmarks:Points3D:keypoint_ids",
        "/world/trajectory:LineStrips3D:columns",
    )


def test_comparable_schema_columns_can_filter_to_semantic_prefixes() -> None:
    schema: pa.Schema = pa.schema(
        [
            pa.field("video_time", pa.duration("ns")),
            pa.field("/world/rig:Transform3D:translation", pa.list_(pa.float32())),
            pa.field("/world/rig/landmarks:Points3D:positions", pa.list_(pa.list_(pa.float32()))),
        ]
    )

    comparable_columns: tuple[str, ...] = comparable_schema_columns(
        schema,
        index="video_time",
        include_column_prefixes=("/world/rig:Transform3D",),
    )

    assert comparable_columns == ("/world/rig:Transform3D:translation",)


def test_compare_arrow_tables_accepts_float_drift_within_tolerance() -> None:
    reference_table: pa.Table = pa.table(
        {
            "video_time": pa.array([0, 33_333_333], type=pa.duration("ns")),
            "/world/trajectory:LineStrips3D:columns": [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]],
            ],
            "/world/rig/landmarks:Points3D:keypoint_ids": [
                [0, 1, 2],
                [0, 1, 2],
            ],
        }
    )
    candidate_table: pa.Table = pa.table(
        {
            "video_time": pa.array([0, 33_333_333], type=pa.duration("ns")),
            "/world/trajectory:LineStrips3D:columns": [
                [[1.0005, 2.0, 3.0]],
                [[4.0, 5.0, 6.0005]],
            ],
            "/world/rig/landmarks:Points3D:keypoint_ids": [
                [0, 1, 2],
                [0, 1, 2],
            ],
        }
    )

    compared_columns: tuple[str, ...] = compare_arrow_tables(
        reference_table=reference_table,
        candidate_table=candidate_table,
        index="video_time",
        columns=(
            "/world/trajectory:LineStrips3D:columns",
            "/world/rig/landmarks:Points3D:keypoint_ids",
        ),
        rtol=1e-3,
        atol=1e-3,
    )

    assert compared_columns == (
        "/world/rig/landmarks:Points3D:keypoint_ids",
        "/world/trajectory:LineStrips3D:columns",
    )


def test_compare_arrow_tables_rejects_changed_null_mask() -> None:
    reference_table: pa.Table = pa.table(
        {
            "video_time": pa.array([0, 33_333_333], type=pa.duration("ns")),
            "/world/trajectory:LineStrips3D:columns": [
                [[1.0, 2.0, 3.0]],
                None,
            ],
        }
    )
    candidate_table: pa.Table = pa.table(
        {
            "video_time": pa.array([0, 33_333_333], type=pa.duration("ns")),
            "/world/trajectory:LineStrips3D:columns": [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]],
            ],
        }
    )

    with pytest.raises(RrdComparisonError, match="Null mask differs"):
        compare_arrow_tables(
            reference_table=reference_table,
            candidate_table=candidate_table,
            index="video_time",
            columns=("/world/trajectory:LineStrips3D:columns",),
            rtol=1e-3,
            atol=1e-3,
        )
