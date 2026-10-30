from pathlib import Path

import pytest

from selfcap_soma.batch import resolve_quest_directory, selected_segments


def test_segment_list_is_ordered_deduplicated_and_rejects_path_injection(tmp_path: Path) -> None:
    path = tmp_path / "list.txt"
    path.write_text("selfcap__2025-12-03__d31bf1da__ep001\n# comment\n")
    assert selected_segments(["selfcap__2025-12-03__d31bf1da__ep001"], path) == ["selfcap__2025-12-03__d31bf1da__ep001"]
    with pytest.raises(ValueError, match="segment"):
        selected_segments(["../../other"], None)


def test_raw_orientation_lookup_requires_exactly_one_matching_session(tmp_path: Path) -> None:
    quest = tmp_path / "2025-12-03" / "d31bf1da-full-uuid" / "episodes" / "episode-001" / "quest"
    quest.mkdir(parents=True)
    for side in ("left", "right"):
        (quest / f"{side}_hand_poses.csv").write_text("placeholder")
    assert resolve_quest_directory(tmp_path, "selfcap__2025-12-03__d31bf1da__ep001") == quest
    (quest / "right_hand_poses.csv").unlink()
    with pytest.raises(FileNotFoundError, match="right_hand_poses"):
        resolve_quest_directory(tmp_path, "selfcap__2025-12-03__d31bf1da__ep001")


def test_protected_dataset_fails_before_any_catalog_or_transport(tmp_path: Path) -> None:
    from selfcap_soma.batch import Config, run_segment

    config = Config(dataset='dataforge-selfcap', spark_dir=Path('/tmp/soma'), output=tmp_path)
    with pytest.raises(ValueError, match='protected dataset'):
        run_segment(config, 'selfcap__2025-12-03__d31bf1da__ep001')
    assert not list(tmp_path.iterdir())
