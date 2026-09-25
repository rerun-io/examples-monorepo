"""Layer comparison ignores chunk boundaries but detects changed data."""

from pathlib import Path

import pytest
import rerun as rr

from dataforge import writing
from dataforge.apis.compare_layers import compare_layers, component_rows, main


def test_comparison_handles_batches_and_rejects_changed_values(tmp_path: Path, capsys) -> None:
    for name, batched, values in [("a", True, [1.0, 2.0]), ("b", False, [1.0, 2.0]), ("c", True, [1.0, 3.0])]:
        with writing.atomic_recording(tmp_path / f"{name}.rrd", recording_id="test", send_properties=False) as recording:
            if batched:
                rr.send_columns(
                    "/value", indexes=[rr.TimeColumn("frame_index", sequence=[0, 1])], columns=rr.Scalars.columns(scalars=values), recording=recording
                )
            else:
                for index, value in enumerate(values):
                    rr.send_columns(
                        "/value",
                        indexes=[rr.TimeColumn("frame_index", sequence=[index])],
                        columns=rr.Scalars.columns(scalars=[value]),
                        recording=recording,
                    )
    assert compare_layers(tmp_path / "a.rrd", tmp_path / "b.rrd").mismatches == []
    assert "/value" in compare_layers(tmp_path / "a.rrd", tmp_path / "c.rrd").mismatches[0]
    assert capsys.readouterr().out == ""
    main(tmp_path / "a.rrd", tmp_path / "b.rrd", atol=0.0)
    assert capsys.readouterr().out == f"equal: {tmp_path / 'a.rrd'} == {tmp_path / 'b.rrd'} (1 component tracks, atol=0.0)\n"


def test_reports_all_differences_and_supports_extra_ignores(tmp_path: Path, capsys) -> None:
    for name, values in [("a", [1.0, 2.0]), ("b", [1.0, 5.0])]:
        with writing.atomic_recording(tmp_path / f"{name}.rrd", recording_id="test", send_properties=False) as recording:
            for entity in ("/first", "/second"):
                rr.send_columns(entity, indexes=[rr.TimeColumn("frame_index", sequence=[0, 1])],
                                columns=rr.Scalars.columns(scalars=values), recording=recording)
    with pytest.raises(SystemExit, match=r"2 mismatch\(es\)"):
        main(tmp_path / "a.rrd", tmp_path / "b.rrd")
    output = capsys.readouterr().out
    assert "/first" in output and "/second" in output
    assert "row 1" in output and "frame_index" in output
    assert "max abs float difference=3.0" in output
    assert "equal:" not in output
    assert compare_layers(tmp_path / "a.rrd", tmp_path / "b.rrd",
                   ignore=[("/first", "Scalars:scalars"), ("/second", "Scalars:scalars")]).mismatches == []


def test_start_time_is_ignored(tmp_path: Path) -> None:
    for name in ("a", "b"):
        with writing.atomic_recording(tmp_path / f"{name}.rrd", recording_id="test"):
            pass
    key = ("/__properties", "RecordingInfo:start_time", ())
    assert not component_rows(tmp_path / "a.rrd")[key].equals(component_rows(tmp_path / "b.rrd")[key])
    assert compare_layers(tmp_path / "a.rrd", tmp_path / "b.rrd").mismatches == []


def test_reports_missing_extra_tracks_row_counts_and_clock_values(tmp_path: Path, capsys) -> None:
    for name, count in (("a", 2), ("b", 3)):
        with writing.atomic_recording(tmp_path / f"{name}.rrd", recording_id="test", send_properties=False) as recording:
            for entity, clock in ((f"/{name}", "frame_index"), ("/clock", f"clock_{name}"), ("/rows", "frame_index")):
                rr.send_columns(entity, indexes=[rr.TimeColumn(clock, sequence=list(range(count)))],
                                columns=rr.Scalars.columns(scalars=[1.0] * count), recording=recording)
            rr.send_columns("/time", indexes=[rr.TimeColumn("frame_index", sequence=[count])],
                            columns=rr.Scalars.columns(scalars=[1.0]), recording=recording)
    with pytest.raises(SystemExit, match="6 mismatch"):
        main(tmp_path / "a.rrd", tmp_path / "b.rrd")
    output = capsys.readouterr().out
    assert output.count("missing in b") == 2
    assert output.count("extra in b") == 2
    assert "row counts differ: 2 != 3" in output
    assert "first differing row 0, time a={'frame_index': 2}, b={'frame_index': 3}" in output
    assert "equal:" not in output
