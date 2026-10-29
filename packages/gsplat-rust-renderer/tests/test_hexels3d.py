"""Hexels3D archetype contracts: hexel PLY -> typed arrays -> rrd with the Points3D fallback."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from jaxtyping import Float32
from plyfile import PlyData, PlyElement
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation

from gsplat_rust_renderer.hexels3d import HEXELS_ENTITY, Hexels3D, HexelVariant, log_hexels, property_names, read_hexel_ply, write_hexel_ply

HALF_SQRT2: float = float(np.sqrt(0.5))


def write_rows(path: Path, rows: Float32[np.ndarray, "n k"], variant: HexelVariant, comments: list[str] | None = None) -> None:
    """Write a contract hexel PLY (hexel-ply.md) whose float rows follow `property_names(variant)`."""
    names: list[str] = property_names(variant)
    assert rows.shape[1] == len(names)
    data: np.ndarray = np.zeros(len(rows), dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(names):
        data[name] = rows[:, index]
    header: list[str] = comments if comments is not None else ["hexels_format 1", f"hexels_variant {variant}", "hexels_source test", "hexels_step 7"]
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<", comments=header).write(str(path))


def sample_rows(variant: HexelVariant) -> Float32[np.ndarray, "2 k"]:
    """Two hexels: a 90-degree turn about x with an out-of-range c0, and an identity one."""
    count: int = len(property_names(variant))
    rows: Float32[np.ndarray, "2 k"] = np.arange(2 * count, dtype=np.float32).reshape(2, count) / 10.0
    rows[0, 3:7] = [HALF_SQRT2, HALF_SQRT2, 0.0, 0.0]  # qw qx qy qz
    rows[1, 3:7] = [1.0, 0.0, 0.0, 0.0]
    rows[0, 7:9] = [0.2, 0.4]
    rows[0, 11:14] = [-0.5, 0.25, 1.5]  # c0 leaves [0, 1]: the archetype keeps f32, the fallback clamps
    return rows


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
def test_ply_reader_splits_contract_columns(tmp_path: Path, variant: HexelVariant) -> None:
    """Each contract column lands in its archetype field; the variant picks the optional view term."""
    rows: Float32[np.ndarray, "2 k"] = sample_rows(variant)
    path: Path = tmp_path / f"{variant}.ply"
    write_rows(path, rows, variant)
    hexels: Hexels3D = read_hexel_ply(path)
    np.testing.assert_array_equal(hexels.centers, rows[:, 0:3])
    np.testing.assert_array_equal(hexels.scales, rows[:, 7:9])
    np.testing.assert_array_equal(hexels.opacities, rows[:, 9])
    np.testing.assert_array_equal(hexels.sharpness, rows[:, 10])
    np.testing.assert_array_equal(hexels.color_field, rows[:, 11:20])
    assert (hexels.view_linear is not None) == (variant == "v2")
    assert (hexels.sh_coefficients is not None) == (variant == "v3")
    if hexels.view_linear is not None:
        np.testing.assert_array_equal(hexels.view_linear, rows[:, 20:29])
    if hexels.sh_coefficients is not None:
        np.testing.assert_array_equal(hexels.sh_coefficients, rows[:, 20:65])


def test_quaternion_is_reordered_to_rerun_xyzw(tmp_path: Path) -> None:
    """The PLY stores wxyz; Rerun's RotationQuat is xyzw. A 90-degree turn about x maps t_v = R e_y to +z."""
    path: Path = tmp_path / "turn.ply"
    write_rows(path, sample_rows("v1"), "v1")
    hexels: Hexels3D = read_hexel_ply(path)
    np.testing.assert_allclose(hexels.quaternions_xyzw, [[HALF_SQRT2, 0.0, 0.0, HALF_SQRT2], [0.0, 0.0, 0.0, 1.0]])
    tangent_v: Float32[np.ndarray, "3"] = Rotation.from_quat(hexels.quaternions_xyzw[0]).apply([0.0, 1.0, 0.0]).astype(np.float32)
    np.testing.assert_allclose(tangent_v, [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    ("comments", "message"),
    [
        (["hexels_variant v1"], "hexels_format 1"),
        (["hexels_format 1"], "hexels_variant"),
        (["hexels_format 1", "hexels_variant 2d"], "3D"),
        (["hexels_format 1", "hexels_variant v2"], "properties"),
    ],
)
def test_ply_reader_rejects_files_outside_the_contract(tmp_path: Path, comments: list[str], message: str) -> None:
    path: Path = tmp_path / "bad.ply"
    write_rows(path, sample_rows("v1"), "v1", comments=comments)
    with pytest.raises(ValueError, match=message):
        read_hexel_ply(path)


@pytest.mark.parametrize("variant", ["v1", "v3"])
def test_archetype_round_trip_keeps_f32_fields_and_writes_the_points_fallback(tmp_path: Path, variant: HexelVariant) -> None:
    """PLY -> rrd keeps archetype, component names and types; stock viewers get clamped centres."""
    path: Path = tmp_path / "model.ply"
    rows: Float32[np.ndarray, "2 k"] = sample_rows(variant)
    write_rows(path, rows, variant)
    recording: rr.RecordingStream = rr.RecordingStream("hexels3d-contract")
    output: Path = tmp_path / "hexels.rrd"
    recording.save(output)
    hexels: Hexels3D = read_hexel_ply(path)
    log_hexels(hexels, recording=recording)
    recording.flush(timeout_sec=30.0)
    recording.disconnect()

    batches: list[pa.RecordBatch] = [chunk.to_record_batch() for chunk in RrdReader(output).stream().filter(content=HEXELS_ENTITY).to_chunks()]
    columns: dict[str, pa.Array] = {field.name: batch.column(field.name).flatten() for batch in batches for field in batch.schema if ":" in field.name}
    fields: dict[str, pa.Field] = {field.name: field for batch in batches for field in batch.schema if ":" in field.name}
    expected_types: dict[str, str | None] = {
        "Hexels3D:centers": "rerun.components.Position3D",
        "Hexels3D:quaternions": "rerun.components.RotationQuat",
        "Hexels3D:scales": "rerun.components.Vector2D",
        "Hexels3D:opacities": "rerun.components.Opacity",
        "Hexels3D:sharpness": None,
        "Hexels3D:color_field": None,
    }
    if variant == "v3":
        expected_types["Hexels3D:sh_coefficients"] = None
    for name, component_type in expected_types.items():
        metadata: dict[bytes, bytes] = fields[name].metadata or {}
        assert metadata[b"rerun:archetype"] == b"Hexels3D"
        assert metadata[b"rerun:component"] == name.encode()
        assert metadata.get(b"rerun:component_type") == (component_type.encode() if component_type is not None else None)
    assert "Hexels3D:view_linear" not in fields
    np.testing.assert_array_equal(np.asarray(columns["Hexels3D:color_field"].to_pylist(), dtype=np.float32), rows[:, 11:20])
    assert columns["Hexels3D:color_field"].type.value_type == pa.float32()
    np.testing.assert_array_equal(np.asarray(columns["Hexels3D:quaternions"].to_pylist(), dtype=np.float32), rows[:, [4, 5, 6, 3]])
    if variant == "v3":
        np.testing.assert_array_equal(np.asarray(columns["Hexels3D:sh_coefficients"].to_pylist(), dtype=np.float32), rows[:, 20:65])

    np.testing.assert_array_equal(np.asarray(columns["Points3D:positions"].to_pylist(), dtype=np.float32), rows[:, 0:3])
    first_color: bytes = int(columns["Points3D:colors"].to_pylist()[0]).to_bytes(4, "big")
    assert list(first_color) == [0, 64, 255, 255]
    assert columns["Points3D:radii"].to_pylist() == [-1.5], "one radius in UI points (negative) for every centre"


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
def test_writer_is_the_reader_inverse(tmp_path: Path, variant: HexelVariant) -> None:
    source: Path = tmp_path / "source.ply"
    write_rows(source, sample_rows(variant), variant)
    hexels: Hexels3D = read_hexel_ply(source)
    copy: Path = tmp_path / "copy.ply"
    write_hexel_ply(copy, hexels)
    assert copy.read_bytes().split(b"end_header")[1] == source.read_bytes().split(b"end_header")[1]
    again: Hexels3D = read_hexel_ply(copy)
    assert again.variant == variant
    np.testing.assert_array_equal(again.quaternions_xyzw, hexels.quaternions_xyzw)
