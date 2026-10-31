import pytest

from selfcap_soma.catalog import exo_columns


def test_sapiens_selection_uses_registered_confidence_and_ignores_legacy_predictions() -> None:
    names = [
        "/world/pred/sapiens2/coco133_xyz:Points3D:positions",
        "/world/gt/sapiens2/coco133_xyz:Points3D:positions",
        "/world/gt/sapiens2/coco133_xyz:simplecv.KeypointConfidence3D:confidences",
    ]
    position, confidence = exo_columns(names, "sapiens2")
    assert position == "/world/gt/sapiens2/coco133_xyz:Points3D:positions"
    assert confidence.endswith("KeypointConfidence3D:confidences")
    with pytest.raises(ValueError, match="unique"):
        exo_columns([], "sapiens2")
