"""Every streamed camera has an upright turn count, and the stream applies it only when asked."""
from handtrack.data.catalog import UPRIGHT_TURNS
from handtrack.data.rerun_source import DATASET_SPECS, source_spec


def test_every_dataset_names_one_turn_count_per_camera() -> None:
    for dataset, (cameras, *_rest) in DATASET_SPECS.items():
        assert len(UPRIGHT_TURNS[dataset]) == len(cameras), dataset
        assert all(turns in (0, 1, 2, 3) for turns in UPRIGHT_TURNS[dataset])


def test_umetrack_follows_its_release_camera_angles() -> None:
    # The release's source_camera_angle_deg is (0, 90, 90, 180): cameras 1-2 are mounted a quarter turn, camera 3 upside down.
    assert UPRIGHT_TURNS["dataforge-umetrack"] == (0, 3, 3, 2)


def test_the_spec_carries_the_turns_only_when_upright() -> None:
    assert source_spec("dataforge-show3d", 1).quarter_turns is None
    assert source_spec("dataforge-show3d", 1, upright=True).quarter_turns == (0, 0)
    assert source_spec("gsi-six-view", 2, upright=True).quarter_turns == (0, 0, 2, 2, 0, 0)


CAMERAS: tuple[str, ...] = tuple(f"/world/rig_00/cam_0{i}" for i in range(4))


def _statics(segments: list[list[int] | None]) -> "pa.Table":
    """One static row per segment; None: a segment converted before the property existed."""
    import pyarrow as pa

    columns = {
        f"{camera}/pinhole:upright_quarter_turns_cw": pa.array([None if turns is None else [turns[c]] for turns in segments], pa.list_(pa.int64()))
        for c, camera in enumerate(CAMERAS)
    }
    return pa.table({"rerun_segment_id": [f"s{i}" for i in range(len(segments))], **columns})


def test_a_catalog_without_the_property_was_converted_as_recorded() -> None:
    import pyarrow as pa

    from handtrack.data.catalog import stored_upright_turns

    assert stored_upright_turns(pa.table({"rerun_segment_id": ["s0"]}), CAMERAS) == (0, 0, 0, 0)


def test_the_stream_skips_the_turns_the_conversion_already_applied() -> None:
    from handtrack.data.catalog import stored_upright_turns, turns_after_conversion

    stored = stored_upright_turns(_statics([[0, 3, 3, 2], [0, 3, 3, 2]]), CAMERAS)
    assert stored == (0, 3, 3, 2)
    assert turns_after_conversion(UPRIGHT_TURNS["dataforge-umetrack"], stored) == (0, 0, 0, 0)
    assert turns_after_conversion((0, 0, 2, 2, 0, 0), (0,) * 6) == (0, 0, 2, 2, 0, 0)


def test_segments_converted_two_ways_are_refused() -> None:
    import pytest

    from handtrack.data.catalog import CatalogDataError, stored_upright_turns

    with pytest.raises(CatalogDataError, match="cam_01"):
        stored_upright_turns(_statics([[0, 3, 3, 2], None]), CAMERAS)
