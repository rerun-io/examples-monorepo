from simplecv.catalog_rig_layout import CatalogCamera, CatalogRigLayout

from exo_calib.catalog_io import RigLayout, select_rig_layout
from exo_calib.layer_io import PIPELINE_CALIBRATION_MARKER


def _camera(
    rig: str,
    camera: str = "cam_00",
    *,
    kind: str | None = None,
    moving: bool = False,
    calibrated: bool = False,
    markers: tuple[str, ...] = (),
) -> CatalogCamera:
    return CatalogCamera(
        rig=rig,
        camera=camera,
        rig_kind=kind,
        rig_is_moving=moving,
        has_calibration=calibrated,
        camera_node_components=frozenset(("Transform3D:translation", *markers) if calibrated else markers),
    )


def test_static_calibrated_rigs_are_exo_and_the_moving_rig_is_ego() -> None:
    layout: CatalogRigLayout = CatalogRigLayout(
        cameras=(
            _camera("rig_00", calibrated=True),
            _camera("rig_01", calibrated=True),
            _camera("rig_02", calibrated=True),
            _camera("rig_03", moving=True),
            _camera("rig_03", "cam_01", moving=True),
        )
    )

    selected: RigLayout = select_rig_layout(layout)

    assert selected.exo_camera_names == ("rig_00/cam_00", "rig_01/cam_00", "rig_02/cam_00")
    assert selected.ego_camera_names == ("rig_03/cam_00", "rig_03/cam_01")
    assert selected.calibrated_camera_names == selected.exo_camera_names


def test_rig_kinds_select_exo_rigs_without_calibration() -> None:
    layout: CatalogRigLayout = CatalogRigLayout(
        cameras=(_camera("rig_00", kind="exo"), _camera("rig_01", kind="exo"), _camera("rig_02", kind="ego"))
    )

    selected: RigLayout = select_rig_layout(layout)

    assert selected.exo_camera_names == ("rig_00/cam_00", "rig_01/cam_00")
    assert selected.ego_camera_names == ("rig_02/cam_00",)
    assert selected.calibrated_camera_names == ()


def test_tagged_ego_and_quest_rigs_are_never_exo_even_when_calibrated() -> None:
    layout: CatalogRigLayout = CatalogRigLayout(
        cameras=(_camera("rig_00", kind="exo", calibrated=True), _camera("rig_01", kind="quest", calibrated=True), _camera("rig_02", kind="ego", calibrated=True))
    )

    selected: RigLayout = select_rig_layout(layout)

    assert selected.exo_camera_names == ("rig_00/cam_00",)
    assert selected.ego_camera_names == ("rig_01/cam_00", "rig_02/cam_00")


def test_exo_rigs_override_is_the_escape_hatch() -> None:
    layout: CatalogRigLayout = CatalogRigLayout(cameras=tuple(_camera(f"rig_{idx:02d}") for idx in range(3)))

    selected: RigLayout = select_rig_layout(layout, exo_rigs=("rig_00", "rig_02"))

    assert selected.exo_camera_names == ("rig_00/cam_00", "rig_02/cam_00")
    assert selected.ego_camera_names == ("rig_01/cam_00",)


def test_calibrated_cameras_are_reported_beside_kind_tagged_rigs() -> None:
    layout: CatalogRigLayout = CatalogRigLayout(
        cameras=(_camera("rig_00", calibrated=True), _camera("rig_01", kind="exo"), _camera("rig_02", kind="ego"))
    )

    selected: RigLayout = select_rig_layout(layout)

    assert selected.exo_camera_names == ("rig_00/cam_00", "rig_01/cam_00")
    assert selected.ego_camera_names == ("rig_02/cam_00",)
    assert selected.calibrated_camera_names == ("rig_00/cam_00",)


def test_pipeline_written_base_calibration_is_not_ground_truth() -> None:
    layout: CatalogRigLayout = CatalogRigLayout(
        cameras=(_camera("rig_00", kind="exo", calibrated=True, markers=(PIPELINE_CALIBRATION_MARKER,)),)
    )

    selected: RigLayout = select_rig_layout(layout)

    assert selected.exo_camera_names == ("rig_00/cam_00",)
    assert selected.calibrated_camera_names == ()
