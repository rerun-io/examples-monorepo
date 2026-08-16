"""Canonical ARKitScenes entity-path checks."""

from arkitscenes_download.ingest.paths import (
    ARKIT_MESH,
    GT,
    GT_BOXES,
    GT_CAM_WIDE,
    GT_DEPTH,
    GT_PINHOLE_WIDE,
    GT_RIG,
    NORMALS_SPLAT_WIDE,
    PINHOLE_WIDE,
    gt_box,
)


def test_arkit_and_laser_ground_truth_paths_are_separate() -> None:
    """ARKit geometry stays outside the optional laser-GT subtree."""
    assert ARKIT_MESH == "world/arkit_mesh"
    assert GT_BOXES == "world/gt_boxes"
    assert gt_box("box_00") == "world/gt_boxes/box_00"
    assert GT == "world/gt"
    assert GT_RIG == "world/gt/rig_00"
    assert GT_CAM_WIDE == "world/gt/rig_00/cam_00"
    assert GT_PINHOLE_WIDE == "world/gt/rig_00/cam_00/pinhole"
    assert GT_DEPTH == "world/gt/rig_00/cam_00/pinhole/depth"


def test_wide_splat_normals_share_the_wide_pinhole() -> None:
    assert f"{PINHOLE_WIDE}/normals_splat" == NORMALS_SPLAT_WIDE
