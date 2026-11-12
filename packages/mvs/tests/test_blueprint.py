from typing import get_args

import pytest
import rerun.blueprint as rrb
from simplecv.rerun_rig_logger import REFERENCE_CAM_COLOR

from mvs.blueprint import CURRENT_CAM_COLOR, SOURCE_CAMS, EntityPaths, Variant, create_blueprint
from mvs.pose_stream import NUM_SOURCE_VIEWS


def test_entity_paths_keep_the_catalog_camera_and_the_original_names() -> None:
    paths: EntityPaths = EntityPaths()

    # The camera, its pinholes and its video are the ARKitScenes catalog's (world/rig_00/cam_00/...).
    assert paths.current_cam == "world/rig_00/cam_00"
    assert paths.current_pinhole == "world/rig_00/cam_00/pinhole"
    assert paths.video == "world/rig_00/cam_00/pinhole/video"
    assert paths.depth_pinhole == "world/rig_00/cam_00/pinhole_lowres"
    assert paths.depth == "world/rig_00/cam_00/pinhole_lowres/depth"
    # What the catalog does not carry keeps the original demo's names.
    assert paths.normal == "world/rig_00/cam_00/pinhole_lowres/normal"
    assert paths.lowest_cost == "world/rig_00/cam_00/pinhole_lowres/lowest_cost"
    assert paths.source_cam(3) == "world/source_cam/3"
    assert paths.source_pinhole(3) == "world/source_cam/3/pinhole"
    assert paths.source_rgb(2) == "world/source_cam/2/pinhole/rgb"
    assert paths.mesh == "world/mesh"
    assert paths.timing("tsdf_fuse") == "timings/tsdf_fuse"
    assert EntityPaths("scan").timing("model_forward") == "scan/timings/model_forward"


def _views(container: rrb.Container | rrb.View) -> list[rrb.View]:
    """Every view below ``container``, depth first."""
    if isinstance(container, rrb.View):
        return [container]
    return [view for child in container.contents for view in _views(child)]


def _names(container: rrb.Container | rrb.View) -> list[str]:
    return [str(view.name) for view in _views(container)]


def test_native_layout_matches_the_original_demo() -> None:
    """3D left; right: current view over its 7-view filmstrip, depth | normals | lowest cost, stage timings."""
    paths: EntityPaths = EntityPaths()
    blueprint: rrb.Blueprint = create_blueprint(paths, variant="native")
    root: rrb.Container = blueprint.root_container
    assert _names(root) == [
        "Reconstruction",
        "Current view",
        *[f"src {index}" for index in range(SOURCE_CAMS)],
        "Depth",
        "Normals",
        "Lowest cost",
        "Stage timings (ms)",
    ]
    assert isinstance(root, rrb.Horizontal) and root.column_shares == [3.0, 2.0]
    right = list(root.contents)[1]
    assert isinstance(right, rrb.Vertical) and right.row_shares == [4.0, 2.0, 1.8]
    camera = list(right.contents)[0]
    assert isinstance(camera, rrb.Vertical) and camera.row_shares == [4.0, 1.0]
    assert blueprint.collapse_panels
    assert blueprint.time_panel is not None and blueprint.time_panel.state == "collapsed"
    # The relayed catalog video lives on video_time only.
    assert blueprint.time_panel.timeline == "video_time"

    views: dict[str, rrb.View] = {str(view.name): view for view in _views(root)}
    world: rrb.View = views["Reconstruction"]
    assert isinstance(world, rrb.Spatial3DView) and world.origin == "world"
    # The depth pinhole stays out: the viewer would back-project the DepthImage into points over the mesh.
    assert world.contents == ["+ world/**", "- world/rig_00/cam_00/pinhole_lowres/**"]
    assert views["Current view"].origin == paths.current_pinhole and views["Current view"].contents == [paths.video]
    assert views["Depth"].origin == paths.depth_pinhole and views["Depth"].contents == paths.depth
    assert views["Normals"].origin == paths.depth_pinhole and views["Normals"].contents == paths.normal
    assert views["Lowest cost"].origin == paths.lowest_cost
    assert views["src 6"].origin == paths.source_pinhole(6) and views["src 6"].contents == paths.source_rgb(6)
    assert isinstance(views["Stage timings (ms)"], rrb.TimeSeriesView) and views["Stage timings (ms)"].origin == "timings"


def test_no_source_cams_leaves_the_filmstrip_out() -> None:
    names: list[str] = _names(create_blueprint(EntityPaths(), variant="native", source_cams=0).root_container)
    assert not [name for name in names if name.startswith("src ")]
    assert "Current view" in names


def test_no_timings_leaves_the_stage_timings_view_out() -> None:
    names: list[str] = _names(create_blueprint(EntityPaths(), variant="native", timings=False).root_container)
    assert "Stage timings (ms)" not in names
    assert {"Depth", "Normals", "Lowest cost"} <= set(names)


def test_source_cams_is_the_model_tuple_minus_the_current_view() -> None:
    assert SOURCE_CAMS == NUM_SOURCE_VIEWS


def test_current_camera_is_the_examples_monorepo_green() -> None:
    assert CURRENT_CAM_COLOR == REFERENCE_CAM_COLOR == (40, 200, 80, 255)


def test_variants_are_native_and_the_two_phone_screens() -> None:
    assert set(get_args(Variant)) == {"native", "phone", "phone-landscape"}


@pytest.mark.parametrize("variant", ["phone", "phone-landscape"])
def test_phone_layouts_show_only_the_3d_view_the_camera_and_the_depth(variant: Variant) -> None:
    landscape: bool = variant == "phone-landscape"
    blueprint: rrb.Blueprint = create_blueprint(EntityPaths(), variant=variant)
    root: rrb.Container = blueprint.root_container
    assert _names(root) == ["Reconstruction", "Current view", "Depth"]
    world, rest = root.contents
    assert isinstance(world, rrb.Spatial3DView)
    # Portrait: Reconstruction above Current view | Depth. Landscape: Reconstruction left of Current view above Depth.
    assert (isinstance(root, rrb.Horizontal), isinstance(rest, rrb.Vertical)) == (landscape, landscape)
    assert blueprint.time_panel is not None and blueprint.time_panel.state == "hidden"
    assert blueprint.time_panel.timeline == "video_time"
