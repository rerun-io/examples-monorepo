"""Native PLY conversion and portable blueprint contracts."""
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from plyfile import PlyData, PlyElement

from gsplat_rust_renderer.gaussians3d import compute_visualizer, splats_from_ply


def _write_synthetic_ply(path: Path) -> None:
    """Write a 2-splat INRIA 3DGS PLY with degree-1 SH (K=3 -> 9 f_rest values).

    Chosen so exp/sigmoid/DC/quat-reorder/SH-transpose all have easy hand values.
    """
    fields: list[str] = (
        ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(9)]
    )
    dtype: list[tuple[str, str]] = [(name, "f4") for name in fields]
    data: np.ndarray = np.zeros(2, dtype=dtype)

    # Splat 0
    data["x"], data["y"], data["z"] = 1.0, 2.0, 3.0
    data["scale_0"], data["scale_1"], data["scale_2"] = 0.0, np.log(2.0), np.log(4.0)  # exp -> 1, 2, 4
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = 1.0, 0.0, 0.0, 0.0  # wxyz identity
    data["opacity"] = 0.0  # sigmoid(0) = 0.5
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = 0.0, 1.0, -1.0
    # f_rest channel-major: [R_c0,R_c1,R_c2, G_c0,G_c1,G_c2, B_c0,B_c1,B_c2]
    for i in range(9):
        data[f"f_rest_{i}"][0] = float(i)

    # Splat 1
    data["x"][1], data["y"][1], data["z"][1] = -1.0, -2.0, -3.0
    data["scale_0"][1] = data["scale_1"][1] = data["scale_2"][1] = 0.0
    data["rot_0"][1], data["rot_1"][1], data["rot_2"][1], data["rot_3"][1] = 0.0, 0.0, 0.0, 2.0  # wxyz -> xyzw (0,0,2,0) -> (0,0,1,0)
    data["opacity"][1] = 10.0  # sigmoid(10) ~ 1
    data["f_dc_0"][1] = data["f_dc_1"][1] = data["f_dc_2"][1] = 0.0

    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def test_from_ply_conversion(tmp_path: Path) -> None:
    path = tmp_path / "tiny.ply"
    _write_synthetic_ply(path)
    splats = splats_from_ply(path)
    assert isinstance(splats, rr.GaussianSplats3D)
    assert splats.centers is not None and splats.scales is not None
    assert splats.quaternions is not None and splats.colors is not None
    assert splats.sh_coefficients is not None and splats.spherical_harmonics_degree is not None
    np.testing.assert_array_equal(splats.centers.as_arrow_array().to_pylist(), [[1, 2, 3], [-1, -2, -3]])
    np.testing.assert_allclose(splats.scales.as_arrow_array().to_pylist()[0], [1, 2, 4], rtol=1e-6)
    np.testing.assert_array_equal(splats.quaternions.as_arrow_array().to_pylist(), [[0, 0, 0, 1], [0, 0, 1, 0]])
    assert splats.colors.as_arrow_array().to_pylist()[0] == int.from_bytes([128, 199, 56, 128], "big")
    np.testing.assert_array_equal(splats.sh_coefficients.as_arrow_array().to_pylist()[0][:3], [[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    assert splats.spherical_harmonics_degree.as_arrow_array().to_pylist() == [1]


def test_native_schema_and_blueprint_property(tmp_path: Path) -> None:
    path = tmp_path / "tiny.ply"
    _write_synthetic_ply(path)
    splats = splats_from_ply(path)
    batches = list(splats.as_component_batches())
    assert all(batch.component_descriptor().archetype == "rerun.archetypes.GaussianSplats3D" for batch in batches)
    assert splats.sh_coefficients is not None
    assert str(splats.sh_coefficients.as_arrow_array().type.value_type.value_type) == "halffloat"
    assert compute_visualizer("mip") is not None


def test_logging_blueprint_defaults_and_mode_validation() -> None:
    import rerun.blueprint as rrb

    from gsplat_rust_renderer.apis.log_gaussian_ply import splat_blueprint
    from gsplat_rust_renderer.apis.log_splats_with_cameras import scene_blueprint
    splats = rr.GaussianSplats3D(centers=[[0, 0, 0]])
    default = next(iter(splat_blueprint(splats).root_container.contents))
    assert isinstance(default, rrb.Spatial3DView)
    assert not default.visualizer_overrides
    explicit = next(iter(splat_blueprint(splats, compute=True, render_mode="mip").root_container.contents))
    assert isinstance(explicit, rrb.Spatial3DView)
    assert explicit.visualizer_overrides
    with pytest.raises(ValueError, match="--compute"):
        splat_blueprint(splats, render_mode="mip")
    with pytest.raises(ValueError, match="--compute"):
        scene_blueprint({}, 1, render_mode="default")


@pytest.mark.integration
def test_python_recording_is_decoded_by_shared_core(tmp_path: Path) -> None:
    """Exercise Python -> RRD -> native Arrow -> core, including f16 SH descriptors."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    subprocess.run(["cargo", "build", "--locked", "-p", "gsplat-bench"], cwd=root, check=True, timeout=1200)
    ply = tmp_path / "tiny.ply"
    _write_synthetic_ply(ply)
    recording = rr.RecordingStream("native-roundtrip")
    rrd = tmp_path / "native.rrd"
    recording.save(rrd)
    recording.log("splats", splats_from_ply(ply), static=True)
    recording.flush(timeout_sec=30.0)
    recording.disconnect()
    result = subprocess.run(
        [str(root / "target/debug/gsplat-bench"), "parity", "--impl", "ours-archetype", "--oracle", "ours", "--ply", str(ply),
         "--archetype-rrd", str(rrd), "--path", "orbit:1", "--res", "32x32", "--out", str(tmp_path / "parity.json")],
        check=False, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "view 0:" in result.stderr + result.stdout
