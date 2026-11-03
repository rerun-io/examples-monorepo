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


def test_coco133_projection_fills_both_hands_from_goliath() -> None:
    # Goliath names the hand root "wrist" and each finger's base "*_third_joint"; matching by name alone left them
    # out, so every finger hung loose from its second joint.
    from sapiens2_pose.api.metadata import get_sapiens_metainfo

    mapping = {int(coco): int(goliath) for coco, goliath in get_sapiens_metainfo()["coco_wholebody_to_goliath_mapping"].items()}
    aliases = {91: 62, 92: 45, 96: 49, 100: 53, 104: 57, 108: 61, 112: 41, 113: 24, 117: 28, 121: 32, 125: 36, 129: 40}
    assert aliases.items() <= mapping.items()
    assert set(range(91, 133)) <= set(mapping)  # all 42 hand joints
    assert len([coco for coco in mapping if coco < 133]) == 65  # 17 body + 6 feet + 42 hands; the face stays unmapped
