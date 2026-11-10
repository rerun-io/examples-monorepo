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
