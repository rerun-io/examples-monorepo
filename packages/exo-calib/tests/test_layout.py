import pytest

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
from simplecv.catalog_rig_layout import CatalogCamera, CatalogRigLayout  # noqa: E402

from exo_calib.sources.layout import lenses_from_table, segment_cameras, select_rig_layout  # noqa: E402
from exo_calib.types import (  # noqa: E402
    FISHEYE,
    PINHOLE,
    Lens,
    RigLayout,  # noqa: E402
)


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
        cameras=(_camera("rig_00", kind="exo", calibrated=True, markers=("exocalib_written",)),)
    )

    selected: RigLayout = select_rig_layout(layout)

    assert selected.exo_camera_names == ("rig_00/cam_00",)
    assert selected.calibrated_camera_names == ()


def _lens_table(lenses: dict[str, tuple[str, list[float]]]) -> "pa.Table":
    """The static columns ``read_lenses`` queries: each camera's pinhole node with its stored lens model and terms."""
    import pyarrow as pa

    columns: dict[str, pa.Array] = {}
    for camera, (model, coefficients) in lenses.items():
        columns[f"/world/{camera}/pinhole:simplecv.components.DistortionModel"] = pa.array([[model]], type=pa.list_(pa.string()))
        columns[f"/world/{camera}/pinhole:simplecv.components.DistortionCoefficients"] = pa.array([[coefficients]], type=pa.list_(pa.list_(pa.float32())))
    return pa.table(columns)


def test_a_camera_takes_its_stored_lens_and_one_without_takes_the_fallback() -> None:
    gopro: list[float] = [0.0369, 0.05713, -0.05715, 0.01846, 0.0, 0.0, 0.0, 0.0]
    table = _lens_table({"rig_00/cam_00": ("kannala_brandt", gopro), "rig_01/cam_00": ("brown_conrady", [-0.13, 0.12] + [0.0] * 12)})

    lenses: dict[str, Lens] = lenses_from_table(table, ("rig_00/cam_00", "rig_01/cam_00", "rig_02/cam_00"), fallback=FISHEYE)

    assert lenses["rig_00/cam_00"].model == "kannala_brandt"
    np.testing.assert_allclose(lenses["rig_00/cam_00"].coefficients, gopro, rtol=1e-6)  # stored as float32
    assert lenses["rig_01/cam_00"].model == "brown_conrady" and len(lenses["rig_01/cam_00"].coefficients) == 14
    assert lenses["rig_02/cam_00"] == FISHEYE


def test_a_lens_model_refine_cannot_undistort_is_refused_by_camera() -> None:
    table = _lens_table({"rig_00/cam_00": ("fisheye624", [0.1] * 12)})

    with pytest.raises(ValueError, match="rig_00/cam_00.*fisheye624"):
        lenses_from_table(table, ("rig_00/cam_00",), fallback=PINHOLE)


def test_a_segment_s_cameras_and_their_calibration_come_from_its_own_statics_not_the_dataset_union() -> None:
    """The union says all four phones are calibrated and rig_02 carries the pipeline's marker (another episode's). This
    episode: rig_00 has GT, rig_01 a base calibration the pipeline wrote, rig_02 no calibration, rig_03 no video."""
    union = CatalogRigLayout(cameras=tuple(_camera(f"rig_0{i}", calibrated=True, markers=("exocalib_written",) if i == 2 else ()) for i in range(4)))

    def stored(*values: object) -> pa.Array:
        return pa.array([None, *values], type=pa.list_(pa.string()))

    def cell(present: bool) -> pa.Array:
        return stored(["x"] if present else None)

    columns: dict[str, pa.Array] = {"/world/rig_00:kind": stored(["exo"])}
    for i, (video, transform, pinhole, marker) in enumerate(((1, 1, 1, 0), (1, 1, 1, 1), (1, 0, 1, 0), (0, 1, 1, 0))):
        camera = f"/world/rig_0{i}/cam_00"
        columns |= {f"{camera}/pinhole/video:VideoStream:codec": cell(video), f"{camera}:Transform3D:translation": cell(transform)}
        columns |= {f"{camera}/pinhole:Pinhole:image_from_camera": cell(pinhole), f"{camera}:exocalib_written": cell(marker)}

    cameras = segment_cameras(union, pa.table(columns)).cameras

    assert [(c.name, c.rig_kind, c.has_calibration) for c in cameras] == [("rig_00/cam_00", "exo", True), ("rig_01/cam_00", None, True), ("rig_02/cam_00", None, False)]
    assert [c.camera_node_components for c in cameras] == [{"Transform3D:translation"}, {"Transform3D:translation", "exocalib_written"}, frozenset()]
    assert select_rig_layout(CatalogRigLayout(cameras), exo_rigs=("rig_00", "rig_01", "rig_02")).calibrated_camera_names == ("rig_00/cam_00",)
