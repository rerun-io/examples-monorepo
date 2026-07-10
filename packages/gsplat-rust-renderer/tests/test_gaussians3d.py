"""Tests for the Gaussians3D wire schema: PLY conversion + Arrow round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import Float32
from plyfile import PlyData, PlyElement

from gsplat_rust_renderer.gaussians3d import SH_C0, SH_REST_VALUES, Gaussians3D


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


def test_from_ply_rejects_malformed_sh(tmp_path: Path) -> None:
    """Incomplete SH bands are rejected loudly, never silently compacted/truncated."""
    fields: list[str] = (
        ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(6)]  # 6 values = 2 coeffs/channel: not a complete degree band
    )
    data: np.ndarray = np.zeros(1, dtype=[(name, "f4") for name in fields])
    ply_path: Path = tmp_path / "malformed.ply"
    PlyData([PlyElement.describe(data, "vertex")]).write(str(ply_path))

    with pytest.raises(ValueError, match="unsupported f_rest_"):
        Gaussians3D.from_ply(ply_path)


def test_from_ply_conversion(tmp_path: Path) -> None:
    """PLY conversion matches hand-computed exp/sigmoid/DC/quat-reorder/SH values."""
    ply_path: Path = tmp_path / "tiny.ply"
    _write_synthetic_ply(ply_path)
    gaussians: Gaussians3D = Gaussians3D.from_ply(ply_path)

    # centers straight through
    np.testing.assert_array_equal(gaussians.centers, np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32))

    # scales = exp(scale_*)
    assert gaussians.scales is not None
    np.testing.assert_allclose(gaussians.scales[0], [1.0, 2.0, 4.0], rtol=1e-6)

    # quaternions: wxyz -> xyzw, normalized; identity stays identity, (0,0,2,0)->(0,0,1,0)
    assert gaussians.quaternions_xyzw is not None
    np.testing.assert_allclose(gaussians.quaternions_xyzw[0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(gaussians.quaternions_xyzw[1], [0.0, 0.0, 1.0, 0.0], atol=1e-6)

    # color: rgb = clip(0.5 + SH_C0 * f_dc, 0, 1); alpha = sigmoid(opacity); u8 RGBA
    assert gaussians.colors_rgba is not None
    expected_rgb: Float32[np.ndarray, "3"] = np.clip(0.5 + SH_C0 * np.array([0.0, 1.0, -1.0]), 0.0, 1.0)
    expected_rgba0: np.ndarray = np.round(np.append(expected_rgb, 0.5) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(gaussians.colors_rgba[0], expected_rgba0)
    assert gaussians.colors_rgba[1, 3] == 255  # sigmoid(10) rounds to full alpha

    # SH: channel-major [R_c0..R_c2, G_c0..G_c2, B_c0..B_c2] = 0..8 ->
    # coefficient-major [c0.rgb, c1.rgb, c2.rgb] = [R_c0,G_c0,B_c0, R_c1,G_c1,B_c1, R_c2,G_c2,B_c2]
    assert gaussians.sh_coefficients is not None
    assert gaussians.sh_coefficients.shape == (2, SH_REST_VALUES)
    expected_sh_head: list[float] = [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]
    np.testing.assert_array_equal(gaussians.sh_coefficients[0, :9], np.array(expected_sh_head, dtype=np.float32))
    # remainder zero-padded to 45
    np.testing.assert_array_equal(gaussians.sh_coefficients[0, 9:], np.zeros(SH_REST_VALUES - 9, dtype=np.float32))
    assert gaussians.show_spherical_harmonics is True


def test_construction_validates_lengths() -> None:
    """Mismatched leading dimensions raise ValueError."""
    with pytest.raises(ValueError):
        Gaussians3D(
            centers=np.zeros((10, 3), dtype=np.float32),
            scales=np.ones((5, 3), dtype=np.float32),
        )


def test_bad_center_shape_raises() -> None:
    """Wrong centers width raises ValueError or a beartype violation (dev)."""
    with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
        Gaussians3D(centers=np.zeros((10, 2), dtype=np.float32))


def _read_data_fields(rrd_path: Path, entity: str) -> dict[str, pa.Field]:
    """Read an rrd and return ``{component_name: arrow Field}`` for one entity's data columns."""
    import rerun_bindings as rb

    reader = rb.RrdReaderInternal(str(rrd_path))
    fields: dict[str, pa.Field] = {}
    for chunk in reader.stream():
        if chunk.entity_path.lstrip("/") != entity.lstrip("/"):
            continue
        schema: pa.Schema = chunk.to_record_batch().schema
        for field in schema:
            if field.metadata and field.metadata.get(b"rerun:kind") == b"data":
                fields[field.name] = field
    return fields


def test_schema_round_trip(tmp_path: Path) -> None:
    """Log to an rrd and assert the exact descriptor strings + Arrow datatypes."""
    rng: np.random.Generator = np.random.default_rng(0)
    n: int = 4
    gaussians: Gaussians3D = Gaussians3D(
        centers=rng.standard_normal((n, 3)).astype(np.float32),
        scales=np.abs(rng.standard_normal((n, 3))).astype(np.float32),
        quaternions_xyzw=rng.standard_normal((n, 4)).astype(np.float32),
        colors_rgba=rng.integers(0, 256, size=(n, 4)).astype(np.uint8),
        sh_coefficients=rng.standard_normal((n, SH_REST_VALUES)).astype(np.float32),
        show_spherical_harmonics=True,
    )

    rrd_path: Path = tmp_path / "gaussians.rrd"
    rr.init("gaussians3d-test", recording_id="roundtrip")
    rr.save(str(rrd_path))
    rr.set_time("frame", sequence=0)
    rr.log("/world/splats", gaussians)
    rr.disconnect()

    fields: dict[str, pa.Field] = _read_data_fields(rrd_path, "/world/splats")

    expected: dict[str, tuple[str, str]] = {
        "Gaussians3D:centers": ("Gaussians3D:centers", "rerun.components.Position3D"),
        "Gaussians3D:scales": ("Gaussians3D:scales", "rerun.components.Scale3D"),
        "Gaussians3D:quaternions": ("Gaussians3D:quaternions", "rerun.components.RotationQuat"),
        "Gaussians3D:colors": ("Gaussians3D:colors", "rerun.components.Color"),
        "Gaussians3D:sh_coefficients": ("Gaussians3D:sh_coefficients", "rerun.components.SphericalHarmonics3"),
        "Gaussians3D:show_spherical_harmonics": ("Gaussians3D:show_spherical_harmonics", "rerun.components.ShowSphericalHarmonics"),
    }
    assert set(fields) == set(expected)

    for name, (component, component_type) in expected.items():
        meta = fields[name].metadata
        assert meta[b"rerun:archetype"] == b"rerun.archetypes.Gaussians3D"
        assert meta[b"rerun:component"] == component.encode()
        assert meta[b"rerun:component_type"] == component_type.encode()

    def _inner(name: str) -> pa.DataType:
        # Rerun wraps each per-row component batch in an outer list<>.
        return fields[name].type.value_type

    def _assert_fsl(name: str, dtype: pa.DataType, size: int) -> None:
        vt: pa.DataType = _inner(name)
        assert pa.types.is_fixed_size_list(vt), f"{name} not a fixed_size_list: {vt}"
        assert vt.list_size == size
        child: pa.Field = vt.field(0)
        assert child.name == "item"
        assert child.nullable is False
        assert child.type == dtype

    _assert_fsl("Gaussians3D:centers", pa.float32(), 3)
    _assert_fsl("Gaussians3D:scales", pa.float32(), 3)
    _assert_fsl("Gaussians3D:quaternions", pa.float32(), 4)
    _assert_fsl("Gaussians3D:sh_coefficients", pa.float16(), SH_REST_VALUES)
    assert _inner("Gaussians3D:colors") == pa.uint32()
    assert _inner("Gaussians3D:show_spherical_harmonics") == pa.bool_()


def test_omits_optional_batches_when_absent() -> None:
    """A centers-only Gaussians3D emits exactly one component batch."""
    gaussians: Gaussians3D = Gaussians3D(centers=np.zeros((3, 3), dtype=np.float32))
    assert len(gaussians.as_component_batches()) == 1
