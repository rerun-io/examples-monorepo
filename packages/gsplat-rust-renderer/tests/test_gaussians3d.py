"""Native PLY conversion and portable blueprint contracts."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32
from plyfile import PlyData, PlyElement

from gsplat_rust_renderer.gaussians3d import compute_visualizer


def _write_synthetic_ply(path: Path) -> None:
    """Write a 2-splat INRIA 3DGS PLY with degree-1 SH (K=3 -> 9 f_rest values).

    Chosen so exp/sigmoid/DC/quat-reorder/SH-transpose all have easy hand values.
    """
    fields: list[str] = [
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    ] + [f"f_rest_{i}" for i in range(9)]
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


@pytest.mark.integration
def test_native_ply_import_preserves_values_path_and_framing(tmp_path: Path) -> None:
    """The native importer owns decoding; the logger preserves its public path and framing."""
    import pyarrow as pa
    from rerun.chunk import Chunk, RrdReader

    from gsplat_rust_renderer.gaussians3d import SPLATS_ENTITY, log_ply

    path: Path = tmp_path / "tiny.ply"
    _write_synthetic_ply(path)
    recording: rr.RecordingStream = rr.RecordingStream("native-ply-contract")
    output: Path = tmp_path / "native.rrd"
    recording.save(output)
    bounds: Float32[np.ndarray, "2 3"] = log_ply(path, recording=recording)
    recording.flush(timeout_sec=30.0)
    recording.disconnect()
    np.testing.assert_allclose(bounds, [[-0.96, -1.92, -2.88], [0.96, 1.92, 2.88]], rtol=1e-6)
    chunks: list[Chunk] = list(RrdReader(output).stream().filter(content=SPLATS_ENTITY).to_chunks())
    native: list[Chunk] = [chunk for chunk in chunks if "GaussianSplats3D:centers" in chunk.to_record_batch().schema.names]
    assert len(native) == 1
    batch: pa.RecordBatch = native[0].to_record_batch()
    centers: pa.Array = batch.column("GaussianSplats3D:centers").flatten()
    scales: pa.Array = batch.column("GaussianSplats3D:scales").flatten()
    rotations: pa.Array = batch.column("GaussianSplats3D:quaternions").flatten()
    colors: pa.Array = batch.column("GaussianSplats3D:colors").flatten()
    sh: pa.Array = batch.column("GaussianSplats3D:sh_coefficients").flatten()
    np.testing.assert_array_equal(centers.to_pylist(), [[1, 2, 3], [-1, -2, -3]])
    np.testing.assert_allclose(scales.to_pylist()[0], [1, 2, 4], rtol=1e-6)
    np.testing.assert_array_equal(rotations.to_pylist(), [[0, 0, 0, 1], [0, 0, 1, 0]])
    assert colors.to_pylist()[0] == int.from_bytes([128, 199, 56, 128], "big")
    np.testing.assert_array_equal(sh.to_pylist()[0][:3], [[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    assert str(sh.type.value_type.value_type) == "halffloat"


def test_logging_blueprint_defaults_and_mode_validation() -> None:
    import rerun.blueprint as rrb

    from gsplat_rust_renderer.apis.log_gaussian_ply import splat_blueprint
    from gsplat_rust_renderer.apis.log_splats_with_cameras import scene_blueprint

    bounds: Float32[np.ndarray, "2 3"] = np.zeros((2, 3), dtype=np.float32)
    with pytest.warns(UserWarning, match="custom viewer"):
        assert compute_visualizer("mip") is not None
    default = next(iter(splat_blueprint(bounds).root_container.contents))
    assert isinstance(default, rrb.Spatial3DView)
    assert not default.visualizer_overrides
    explicit = next(iter(splat_blueprint(bounds, compute=True, render_mode="mip").root_container.contents))
    assert isinstance(explicit, rrb.Spatial3DView)
    assert explicit.visualizer_overrides
    with pytest.raises(ValueError, match="--compute"):
        splat_blueprint(bounds, render_mode="mip")
    with pytest.raises(ValueError, match="--compute"):
        scene_blueprint({}, 1, render_mode="default")


@pytest.mark.integration
def test_python_recording_is_decoded_by_shared_core(tmp_path: Path) -> None:
    """Exercise Python -> RRD -> native Arrow -> core, including f16 SH descriptors."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    subprocess.run(["cargo", "build", "--locked", "-p", "gsplat-cli"], cwd=root, check=True, timeout=1200)
    ply = tmp_path / "tiny.ply"
    _write_synthetic_ply(ply)
    recording = rr.RecordingStream("native-roundtrip")
    rrd = tmp_path / "native.rrd"
    recording.save(rrd)
    from gsplat_rust_renderer.gaussians3d import log_ply

    log_ply(ply, recording=recording)
    recording.flush(timeout_sec=30.0)
    recording.disconnect()
    result = subprocess.run(
        [
            str(root / "target/debug/gsplat"),
            "parity",
            "--impl",
            "ours",
            "--oracle",
            "ours",
            "--ply",
            str(ply),
            "--archetype",
            str(rrd),
            "--path",
            "orbit:1",
            "--res",
            "32x32",
            "--out",
            str(tmp_path / "parity.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "view 0:" in result.stderr + result.stdout
