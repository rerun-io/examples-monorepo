from pathlib import Path

from sapiens2_pose.sapiens_lite import MODEL_SPECS, parse_pose_metainfo


def test_model_specs_include_expected_sizes() -> None:
    assert set(MODEL_SPECS) == {"0.4B", "0.8B", "1B", "5B"}


def test_pose_metadata_file_parses() -> None:
    meta = parse_pose_metainfo(
        {
            "from_file": str(
                Path(__file__).resolve().parents[1]
                / "src"
                / "sapiens2_pose"
                / "assets"
                / "configs"
                / "_base_"
                / "keypoints308.py"
            )
        }
    )
    assert meta["num_keypoints"] == 308
    assert meta["dataset_name"] == "goliath"
    assert len(meta["skeleton_links"]) > 0
