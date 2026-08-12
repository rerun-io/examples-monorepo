"""Published ARKitScenes layer taxonomy checks."""

from arkitscenes_download.ingest.layers import ALL_LAYER_NAMES, LAYER_NAMES, OPTIONAL_LAYER_NAMES


def test_required_and_optional_layer_taxonomy() -> None:
    """Required ingest layers stay ordered while CA-1M laser-GT layers remain optional."""
    assert LAYER_NAMES == (
        "base",
        "calibration",
        "video_wide",
        "video_ultrawide",
        "arkit_depth",
        "imu",
        "arkit_mesh",
        "gt_boxes",
    )
    assert len(LAYER_NAMES) == 8
    assert LAYER_NAMES[0] == "base"
    assert OPTIONAL_LAYER_NAMES == ("gt_poses", "gt_depth")
    assert (*LAYER_NAMES, *OPTIONAL_LAYER_NAMES) == ALL_LAYER_NAMES
