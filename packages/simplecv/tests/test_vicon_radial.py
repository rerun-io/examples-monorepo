"""MAMMA's Vicon radial lens: one formula for numpy and torch, and the reader of its catalog cells."""

import numpy as np
import pyarrow as pa
import pytest
import torch

from simplecv.catalog_calibration import read_lens
from simplecv.sensors.camera.vicon_radial import ViconRadialLens

# A lens of MAMMA's magnitude (IOI_01 of the breakdance segment, native 2056x1504).
LENS: ViconRadialLens = ViconRadialLens(centre=np.array([1033.76, 719.77]), radial=np.array([1.98e-8, -1.2e-14, -3.0e-22]), native_wh=(2056, 1504))
POINTS: np.ndarray = np.array([[0.0, 0.0], [2055.0, 1503.0], [1033.76, 719.77], [150.0, 1380.0], [1900.25, 120.5], [700.0, 300.0]])
# dataforge's ViconLens (mamma_source.py before the move) on POINTS; float repr round-trips, so these are its exact bits.
OLD_UNDISTORT: list[list[float]] = [
    [-0.006470359734294107, -0.004505079347268293],
    [2053.478197805479, 1501.8328687352482],
    [1033.76, 719.77],
    [144.8887914316997, 1383.8184272121944],
    [1906.12746354228, 116.43510879873713],
    [698.4330596953716, 298.0292589535179],
]
OLD_DISTORT: list[list[float]] = [
    [0.006921640653445138, 0.004819290060709136],
    [2056.647309950151, 1504.2633882067453],
    [1033.76, 719.77],
    [155.2137717810998, 1376.1049509561017],
    [1894.3117016518338, 124.60696494028275],
    [701.5483016688792, 301.94729923161975],
]


def test_numpy_answers_are_the_old_dataforge_bits() -> None:
    np.testing.assert_array_equal(LENS.undistort(POINTS), np.array(OLD_UNDISTORT))
    np.testing.assert_array_equal(LENS.distort(POINTS), np.array(OLD_DISTORT))


def test_a_point_that_does_not_converge_is_nan_and_the_others_are_not() -> None:
    # dataforge's ViconLens on the same strong barrel lens: the far corner has no preimage.
    strong = ViconRadialLens(centre=np.array([100.0, 80.0]), radial=np.array([-8e-6, 0.0, 0.0]), native_wh=(200, 160))
    points = np.array([[0.0, 0.0], [100.0, 80.0], [300.0, 260.0]])
    expected = np.array([[-26.65869104504256, -21.326952836034053], [100.0, 80.0], [np.nan, np.nan]])
    np.testing.assert_array_equal(strong.distort(points), expected)
    np.testing.assert_array_equal(strong.distort(torch.from_numpy(points)).numpy(), expected)


def test_torch_agrees_with_numpy_and_keeps_the_type() -> None:
    points = np.random.default_rng(0).uniform([0.0, 0.0], [2056.0, 1504.0], size=(500, 2))
    undistorted = LENS.undistort(torch.from_numpy(points))
    distorted = LENS.distort(torch.from_numpy(points))
    assert isinstance(undistorted, torch.Tensor) and isinstance(distorted, torch.Tensor)
    np.testing.assert_allclose(undistorted.numpy(), LENS.undistort(points), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(distorted.numpy(), LENS.distort(points), rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_distort_inverts_undistort_on_a_grid(backend: str) -> None:
    ys, xs = np.meshgrid(np.linspace(0, 1503, 40), np.linspace(0, 2055, 50), indexing="ij")
    distorted = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)
    undistorted = LENS.undistort(distorted)
    assert np.abs(undistorted - distorted).max() > 5.0  # the lens is not the identity at the border
    back = LENS.distort(undistorted) if backend == "numpy" else LENS.distort(torch.from_numpy(undistorted)).numpy()
    np.testing.assert_allclose(back, distorted, rtol=0.0, atol=1e-6)


def test_the_lens_follows_the_points_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    points = torch.from_numpy(POINTS).cuda()
    assert LENS.distort(points).device == points.device
    np.testing.assert_allclose(LENS.undistort(points).cpu().numpy(), OLD_UNDISTORT, rtol=0.0, atol=1e-9)


def _static_table(cells: dict[str, object]) -> pa.Table:
    """A one-row static table as ``reader(index=None)`` returns it: every cell a one-element list."""
    return pa.table({name: [[value]] for name, value in cells.items()})


def test_read_lens_decodes_the_mamma_cells() -> None:
    entity = "/world/rig_01/cam_00"
    table = _static_table(
        {f"{entity}:camera_model": "vicon_radial_2", f"{entity}:source_width": 2056, f"{entity}:source_height": 1504}
    ).append_column(f"{entity}:vicon_radial_2", pa.array([[1033.76, 719.77, 1.98e-8, -1.2e-14, -3.0e-22]]))
    lens = read_lens(table, entity)
    assert lens is not None
    np.testing.assert_array_equal(lens.centre, LENS.centre)
    np.testing.assert_array_equal(lens.radial, LENS.radial)
    assert lens.native_wh == (2056, 1504)


def test_read_lens_is_none_for_a_pinhole_camera_and_raises_for_an_unknown_model() -> None:
    entity = "/world/rig_01/cam_00"
    assert read_lens(_static_table({f"{entity}:camera_model": "pinhole"}), entity) is None
    assert read_lens(_static_table({"/world/rig_02/cam_00:camera_model": "vicon_radial_2"}), entity) is None
    with pytest.raises(ValueError, match="kannala"):
        read_lens(_static_table({f"{entity}:camera_model": "kannala"}), entity)
    with pytest.raises(ValueError, match="source_width"):
        read_lens(_static_table({f"{entity}:camera_model": "vicon_radial_2"}), entity)
