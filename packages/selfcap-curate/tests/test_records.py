"""Frozen sync records remain readable and strict."""

from dataclasses import replace
from pathlib import Path

import pytest
from serde import SerdeError
from serde.json import from_json

from selfcap_curate import records
from selfcap_curate.records import TimeSync, read_frozen_sync, read_sync, write_sync

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("source", sorted(FIXTURES.glob("*.sync.json")), ids=lambda path: path.stem)
def test_sync_roundtrip(tmp_path: Path, source: Path) -> None:
    record = read_sync(source)
    target = tmp_path / "sync.json"
    write_sync(target, record)
    assert read_sync(target) == record


def test_unknown_sync_fields_fail() -> None:

    text = (FIXTURES / "2025-11-24__027d8215__ep002.sync.json").read_text()
    with pytest.raises(SerdeError):
        from_json(TimeSync, text.replace('"method":', '"unexpected": true, "method":'))


def test_legacy_nan_links_become_null_without_relaxing_the_owned_json_reader(tmp_path: Path) -> None:

    source = (FIXTURES / "2025-11-24__027d8215__ep002.sync.json").read_text()
    source = source.replace('"lag_s": -0.14944', '"lag_s": NaN')
    target = tmp_path / "legacy.json"
    target.write_text(source)
    record = read_frozen_sync(target)
    assert record.links.lag_left_to_pose.lag_s is None
    with pytest.raises(ValueError, match="legacy.json"):
        read_sync(target)
    write_sync(tmp_path / "owned.json", record)
    assert "NaN" not in (tmp_path / "owned.json").read_text()


@pytest.mark.parametrize(
    "flags,anchor,relative",
    [
        ([], True, True),
        (["quest_unsynced_kept_as_cut"], False, False),
        (["quest_pose_unsynced"], False, False),
        (["oak_imu_unsynced"], True, False),
        (["quest_pose_unsynced", "oak_imu_via_pose"], False, True),
        (["quest_pose_unsynced", "oak_imu_via_pose", "oak_imu_unsynced"], False, False),
    ],
)
def test_oak_anchor_and_relative_gyro_clock_evidence(flags, anchor, relative, fixture_sync):
    sync = fixture_sync(offsets_ms={"quest_pose": 0.0, "oak_imu": 0.0}, flags=flags)
    assert records.oak_quest_offset_ms(sync) == (0.0 if anchor else None)
    pair = records.gyro_head_offsets(sync)
    assert (pair is not None) == relative
    if pair is not None:
        assert pair.quest_pose_ms == 0.0 and pair.oak_imu_ms == 0.0


@pytest.mark.parametrize("missing", ["quest_pose", "oak_imu"])
def test_relative_gyro_clock_requires_both_original_offsets(missing, fixture_sync):
    sync = fixture_sync(offsets_ms={"quest_pose": 100.0, "oak_imu": -25.0}, flags=[])
    pair = records.gyro_head_offsets(sync)
    assert pair.quest_pose_ms == 100.0 and pair.oak_imu_ms == -25.0
    assert records.gyro_head_offsets(replace(sync, offsets_ms={**sync.offsets_ms, missing: None})) is None


def test_episode_identity_selects_in_order_and_names_one_sync_file(tmp_path):
    first = records.EpisodeKey.parse("selfcap__2025-11-24__027d8215__ep002")
    assert (first.date, first.sid8, first.episode) == ("2025-11-24", "027d8215", "ep002")
    assert first.key == "2025-11-24__027d8215__ep002"
    assert first.segment == "selfcap__2025-11-24__027d8215__ep002"
    assert first.session == "2025-11-24__027d8215"
    assert first.sync_name == first.segment + ".json"
    assert first.rrd_name == first.segment + ".rrd"
    key_file = tmp_path / "keys.txt"
    key_file.write_text(first.key + "\n2025-11-24__027d8215__ep003\n")
    keys = records.episode_keys((first.segment,), key_file)
    assert keys == (first, records.EpisodeKey.parse("2025-11-24__027d8215__ep003"))
    assert records.one_session(keys) == first.session
    with pytest.raises(ValueError, match="exactly one session"):
        records.one_session((*keys, records.EpisodeKey.parse("2025-12-30__65d00748__ep001")))


@pytest.mark.parametrize("text", ["", "dataforge-selfcap", "../2025-11-24__027d8215__ep002", "2025-11-24__027d8215__ep002__extra"])
def test_episode_identity_refuses_malformed_keys(text):
    with pytest.raises(ValueError, match="episode key"):
        records.EpisodeKey.parse(text)


@pytest.mark.parametrize(
    "field,value",
    [("date", "2025-12-01"), ("episode", "ep001"), ("session", "ffffffff-0000-0000-0000-000000000000"), ("key", "2025-11-24__027d8215__ep001")],
)
def test_sync_identity_fields_must_agree_when_decoded(tmp_path, field, value):
    import json

    source = FIXTURES / "2025-11-24__027d8215__ep002.sync.json"
    document = json.loads(source.read_text())
    document[field] = value
    target = tmp_path / "inconsistent.json"
    target.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="identity"):
        from_json(TimeSync, target.read_text())
    with pytest.raises(ValueError, match="inconsistent.json"):
        read_frozen_sync(target)
