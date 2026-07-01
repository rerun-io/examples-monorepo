from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from simplecv.apis.exoego_forge_catalog import (
    CATALOG_CAMERA_NAMES,
    DEFAULT_CATALOG_DATASETS,
    build_exoego_catalog_blueprint,
    discover_rrd_paths,
    discover_rrd_uris,
)
from simplecv.data.exoego.aria_gen2_pilot import AriaGen2PilotConfig, AriaGen2PilotSequence
from simplecv.data.exoego.assembly101 import Assembly101Config, Assembly101Sequence
from simplecv.data.exoego.ego_dex import EgoDexConfig, EgoDexSequence
from simplecv.data.exoego.epfl_smart_kitchen import EpflSmartKitchenConfig, EpflSmartKitchenSequence
from simplecv.data.exoego.hocap import HocapConfig, HocapSequence
from simplecv.data.exoego.hot3d import Hot3dConfig, Hot3dSequence
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.exoego.umetrack import UmeTrackConfig, UmeTrackSequence

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MONOREPO_ROOT: Path = PROJECT_ROOT.parents[1]


def test_sequence_identity_paths_and_recording_id() -> None:
    identity = SequenceIdentity(
        dataset="umetrack",
        parts=("real", "hand_hand", "testing", "user_05", "recording_09"),
    )

    assert identity.sequence_key == "real/hand_hand/testing/user_05/recording_09"
    assert identity.recording_id == "umetrack__real__hand_hand__testing__user_05__recording_09"
    assert identity.rrd_path(Path("data/exoego-forge-catalog")) == Path(
        "data/exoego-forge-catalog/umetrack/real/hand_hand/testing/user_05/recording_09.rrd"
    )


@pytest.mark.parametrize(
    ("dataset", "parts"),
    [
        ("bad__dataset", ("sequence",)),
        ("hocap", ("subject__1", "recording")),
        ("hocap", ("subject_1/bad__recording",)),
    ],
)
def test_sequence_identity_rejects_recording_id_separator(
    dataset: str,
    parts: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="cannot contain '__'"):
        SequenceIdentity(dataset=dataset, parts=parts)


@pytest.mark.parametrize(
    ("identity", "dataset", "sequence_key", "recording_id"),
    [
        (
            AriaGen2PilotSequence.sequence_identity_for_config(AriaGen2PilotConfig(sequence_name="cook_0")),
            "aria-gen2",
            "cook_0",
            "aria-gen2__cook_0",
        ),
        (
            Assembly101Sequence.sequence_identity_for_config(Assembly101Config(split=None, sequence_name="seq_01")),
            "assembly101",
            "all/seq_01",
            "assembly101__all__seq_01",
        ),
        (
            HocapSequence.sequence_identity_for_config(HocapConfig(subject_id="8", sequence_name="20231024_180733")),
            "hocap",
            "subject_8/20231024_180733",
            "hocap__subject_8__20231024_180733",
        ),
        (
            Hot3dSequence.sequence_identity_for_config(Hot3dConfig(headset="quest3", sequence_name="P0001_10a27bf7")),
            "hot3d-quest3",
            "P0001_10a27bf7",
            "hot3d-quest3__P0001_10a27bf7",
        ),
        (
            UmeTrackSequence.sequence_identity_for_config(
                UmeTrackConfig(
                    data_type="real",
                    hand_interaction="hand_hand",
                    split="testing",
                    user=5,
                    recording_id=9,
                )
            ),
            "umetrack",
            "real/hand_hand/testing/user_05/recording_09",
            "umetrack__real__hand_hand__testing__user_05__recording_09",
        ),
        (
            EgoDexSequence.sequence_identity_for_config(EgoDexConfig(split="test", sequence_name="add_remove_lid", episode=3)),
            "ego-dex",
            "test/add_remove_lid/episode_0003",
            "ego-dex__test__add_remove_lid__episode_0003",
        ),
        (
            EpflSmartKitchenSequence.sequence_identity_for_config(
                EpflSmartKitchenConfig(
                    split="train",
                    participant_id="YH2002",
                    session_name="2023_12_04_10_15_23",
                )
            ),
            "epfl-smart-kitchen",
            "train/YH2002/2023_12_04_10_15_23",
            "epfl-smart-kitchen__train__YH2002__2023_12_04_10_15_23",
        ),
    ],
)
def test_dataset_sequence_identity_for_config(
    identity: SequenceIdentity,
    dataset: str,
    sequence_key: str,
    recording_id: str,
) -> None:
    assert identity.dataset == dataset
    assert identity.sequence_key == sequence_key
    assert identity.recording_id == recording_id


def test_discover_rrd_paths_and_uris_group_by_dataset(tmp_path: Path) -> None:
    aria_rrd: Path = tmp_path / "aria-gen2" / "cook_0.rrd"
    hocap_rrd: Path = tmp_path / "hocap" / "subject_8" / "20231024_180733.rrd"
    hot3d_aria_rrd: Path = tmp_path / "hot3d-aria" / "P0001_4bf4e21a.rrd"
    skipped_rrd: Path = tmp_path / "ego100k" / "video.rrd"
    legacy_hot3d_rrd: Path = tmp_path / "hot3d" / "aria" / "P0001_4bf4e21a.rrd"
    for path in (aria_rrd, hocap_rrd, hot3d_aria_rrd, skipped_rrd, legacy_hot3d_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    paths: dict[str, list[Path]] = discover_rrd_paths(
        tmp_path,
        datasets=("aria-gen2", "hocap", "hot3d-aria", "missing"),
    )
    uris: dict[str, list[str]] = discover_rrd_uris(
        tmp_path,
        datasets=("aria-gen2", "hocap", "hot3d-aria", "missing"),
    )

    assert set(paths) == {"aria-gen2", "hocap", "hot3d-aria"}
    assert paths["aria-gen2"] == [aria_rrd.resolve()]
    assert paths["hocap"] == [hocap_rrd.resolve()]
    assert paths["hot3d-aria"] == [hot3d_aria_rrd.resolve()]
    assert set(uris) == {"aria-gen2", "hocap", "hot3d-aria"}
    assert uris["aria-gen2"] == [aria_rrd.resolve().as_uri()]
    assert uris["hocap"] == [hocap_rrd.resolve().as_uri()]
    assert uris["hot3d-aria"] == [hot3d_aria_rrd.resolve().as_uri()]


def test_discover_rrd_paths_raises_when_no_requested_dataset_has_rrds(tmp_path: Path) -> None:
    skipped_rrd: Path = tmp_path / "ego100k" / "video.rrd"
    skipped_rrd.parent.mkdir(parents=True)
    skipped_rrd.write_bytes(b"not a real rrd")

    with pytest.raises(FileNotFoundError):
        discover_rrd_paths(tmp_path, datasets=("aria-gen2", "hocap"))


def test_hot3d_has_hand_labels_respects_no_gt_metadata(tmp_path: Path) -> None:
    seq_dir: Path = tmp_path / "P0016_0ca96b7b"
    seq_dir.mkdir()
    (seq_dir / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": False}))

    assert Hot3dSequence._has_hand_labels(seq_dir) is False


def test_hot3d_has_hand_labels_rejects_empty_files_when_gt_available(tmp_path: Path) -> None:
    seq_dir: Path = tmp_path / "P0015_e7458eb3"
    seq_dir.mkdir()
    (seq_dir / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": True}))
    (seq_dir / "umetrack_hand_user_profile.json").touch()
    (seq_dir / "umetrack_hand_pose_trajectory.jsonl").touch()

    with pytest.raises(AssertionError, match="Hand GT is marked available"):
        Hot3dSequence._has_hand_labels(seq_dir)


def test_hot3d_has_hand_labels_accepts_nonempty_gt_files(tmp_path: Path) -> None:
    seq_dir: Path = tmp_path / "P0015_e7458eb3"
    seq_dir.mkdir()
    (seq_dir / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": True}))
    (seq_dir / "umetrack_hand_user_profile.json").write_text("{}")
    (seq_dir / "umetrack_hand_pose_trajectory.jsonl").write_text("{}\n")

    assert Hot3dSequence._has_hand_labels(seq_dir) is True


def test_epfl_smart_kitchen_catalog_uses_hololens_and_all_nine_exo_cameras() -> None:
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES["epfl-smart-kitchen"]

    assert camera_names["ego"] == ("hololens",)
    assert camera_names["exo"] == (
        "output0",
        "Aoutput0",
        "Aoutput1",
        "Aoutput2",
        "Aoutput3",
        "Boutput0",
        "Boutput1",
        "Boutput2",
        "Boutput3",
    )


def test_catalog_blueprints_exist_for_default_datasets() -> None:
    for dataset_name in DEFAULT_CATALOG_DATASETS:
        blueprint = build_exoego_catalog_blueprint(dataset_name)
        root_container = blueprint.root_container
        root_contents: list[Any] = list(root_container.contents)
        column_shares: list[float] | None = getattr(root_container, "column_shares", None)
        row_shares: list[float] | None = getattr(root_container, "row_shares", None)

        assert blueprint is not None
        if column_shares is not None:
            assert len(column_shares) == len(root_contents)
        if row_shares is not None:
            assert len(row_shares) == len(root_contents)
