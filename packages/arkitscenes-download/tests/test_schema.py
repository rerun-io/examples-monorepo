"""Published ARKitScenes layer taxonomy checks."""

from arkitscenes_download.schema import ALL_LAYER_NAMES, OPTIONAL_LAYER_NAMES, REQUIRED_LAYER_NAMES


def test_required_and_optional_layer_taxonomy() -> None:
    """Required ingest layers stay ordered while CA-1M laser-GT layers remain optional."""
    assert REQUIRED_LAYER_NAMES == (
        "base",
        "calibration",
        "video_wide",
        "video_ultrawide",
        "arkit_depth",
        "imu",
        "arkit_mesh",
        "gt_boxes",
    )
    assert REQUIRED_LAYER_NAMES[0] == "base"
    assert OPTIONAL_LAYER_NAMES == ("gt_poses", "gt_depth")
    assert (*REQUIRED_LAYER_NAMES, *OPTIONAL_LAYER_NAMES) == ALL_LAYER_NAMES
