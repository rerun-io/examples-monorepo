from simplecv.catalog_rig_layout import CatalogComponent, CatalogRigLayout, catalog_components, parse_rig_layout


def _components(rows: list[tuple[str, str, bool]]) -> list[CatalogComponent]:
    return [CatalogComponent.parse(*row) for row in rows]


def test_parse_rig_layout_orders_cameras_and_flags_moving_rigs() -> None:
    rows: list[tuple[str, str, bool]] = []
    for rig_idx in (2, 0, 1):
        camera_path: str = f"/world/rig_{rig_idx:02d}/cam_00"
        rows.extend(
            (
                (camera_path, "Transform3D:translation", True),
                (f"{camera_path}/pinhole", "Pinhole:image_from_camera", True),
                (f"{camera_path}/pinhole/video", "VideoStream:sample", False),
            )
        )
    rows.extend(
        (
            ("/world/rig_03", "Transform3D:translation", False),
            ("/world/rig_03/cam_01/pinhole/video", "VideoStream:sample", False),
            ("/world/rig_03/cam_00/pinhole/video", "VideoStream:sample", False),
            ("/world/rig_04/cam_00/pinhole", "Pinhole:image_from_camera", True),  # no video: not a camera
            ("/world/rig_05/cam_00/pinhole/video", "VideoStream:codec", True),  # codec column alone: not a video
        )
    )

    layout: CatalogRigLayout = parse_rig_layout(_components(rows))

    assert [camera.name for camera in layout.cameras] == [
        "rig_00/cam_00",
        "rig_01/cam_00",
        "rig_02/cam_00",
        "rig_03/cam_00",
        "rig_03/cam_01",
    ]
    assert [camera.rig_is_moving for camera in layout.cameras] == [False, False, False, True, True]
    assert [camera.has_calibration for camera in layout.cameras] == [True, True, True, False, False]
    assert layout.cameras[0].video_entity == "/world/rig_00/cam_00/pinhole/video"
    assert layout.cameras[0].pinhole_entity == "/world/rig_00/cam_00/pinhole"
    assert layout.cameras[0].transform_entity == "/world/rig_00/cam_00"
    assert layout.rigs == ("rig_00", "rig_01", "rig_02", "rig_03")


def test_parse_rig_layout_attaches_rig_kinds_and_camera_node_markers() -> None:
    rows: list[tuple[str, str, bool]] = [
        ("/world/rig_00", "kind", True),
        ("/world/rig_00/cam_00", "Transform3D:translation", True),
        ("/world/rig_00/cam_00", "exocalib_written", True),
        ("/world/rig_00/cam_00/pinhole", "Pinhole:image_from_camera", True),
        ("/world/rig_00/cam_00/pinhole/video", "VideoStream:sample", False),
        ("/world/rig_01", "kind", True),
        ("/world/rig_01/cam_00/pinhole/video", "VideoStream:sample", False),
    ]

    layout: CatalogRigLayout = parse_rig_layout(_components(rows), rig_kinds={"rig_00": "exo", "rig_01": "ego"})

    exo, ego = layout.cameras
    assert (exo.rig_kind, ego.rig_kind) == ("exo", "ego")
    assert exo.has_calibration and not ego.has_calibration
    assert exo.camera_node_components == frozenset({"Transform3D:translation", "exocalib_written"})
    assert ego.camera_node_components == frozenset()


def test_parse_rig_layout_requires_both_transform_and_pinhole_for_calibration() -> None:
    rows: list[tuple[str, str, bool]] = [
        ("/world/rig_00/cam_00", "Transform3D:translation", True),
        ("/world/rig_00/cam_00/pinhole/video", "VideoStream:sample", False),
        ("/world/rig_01/cam_00/pinhole", "Pinhole:image_from_camera", True),
        ("/world/rig_01/cam_00/pinhole/video", "VideoStream:sample", False),
        ("/world/rig_02/cam_00", "Transform3D:translation", False),  # temporal transform is not calibration
        ("/world/rig_02/cam_00/pinhole", "Pinhole:image_from_camera", True),
        ("/world/rig_02/cam_00/pinhole/video", "VideoStream:sample", False),
    ]

    layout: CatalogRigLayout = parse_rig_layout(_components(rows))

    assert [camera.has_calibration for camera in layout.cameras] == [False, False, False]


class _Column:
    def __init__(self, entity_path: str, component: str, is_static: bool = True) -> None:
        self.entity_path: str = entity_path
        self.component: str = component
        self.is_static: bool = is_static


class _Schema:
    def __init__(self, columns: list[_Column]) -> None:
        self._columns: list[_Column] = columns

    def component_columns(self) -> list[_Column]:
        return self._columns


def test_catalog_components_parses_rerun_column_identifiers() -> None:
    schema: _Schema = _Schema(
        [
            _Column("world/rig_00/cam_00", "Transform3D:translation"),
            _Column("/world/rig_00/cam_00", "exocalib_written"),
            _Column("/world/rig_00/cam_00/pinhole/video", "VideoStream:sample", is_static=False),
        ]
    )

    components: list[CatalogComponent] = catalog_components(schema)

    assert components == [
        CatalogComponent("/world/rig_00/cam_00", "Transform3D", "translation", True),
        CatalogComponent("/world/rig_00/cam_00", None, "exocalib_written", True),
        CatalogComponent("/world/rig_00/cam_00/pinhole/video", "VideoStream", "sample", False),
    ]
    assert components[0].component == "Transform3D:translation"
    assert components[1].component == "exocalib_written"


def test_catalog_components_reduces_namespaced_archetypes() -> None:
    component: CatalogComponent = CatalogComponent.parse("/world/exo/C10095", "rerun.archetypes.Transform3D:mat3x3", True)

    assert (component.archetype, component.field, component.component) == ("Transform3D", "mat3x3", "Transform3D:mat3x3")
