"""The shared SOMA-X layer writer: XLO mesh, white-linked green joints, SOMA-X and source parameters, read back as a consumer sees them."""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun.chunk as rrc
from conftest import column_rows, read_back, static_values
from jaxtyping import Bool, Float32, Int64, UInt32
from numpy import ndarray

from dataforge import schema, soma_x
from dataforge.writing import atomic_recording

ROOT: str = schema.hands_path("left")
FRAMES: int = 4
VERTICES: int = 5
JOINTS: int = 3


def topology() -> soma_x.SomaXTopology:
    faces: UInt32[ndarray, "f 3"] = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.uint32)
    parents: Int64[ndarray, "j"] = np.array([0, 0, 1], dtype=np.int64)
    return soma_x.SomaXTopology(faces=faces, parents=parents, joint_names=("Wrist", "Index1", "Index2"))


def fit(trusted: Bool[ndarray, "t"]) -> soma_x.SomaXFit:
    count: int = int(trusted.sum())
    rotations: Float32[ndarray, "t j 3"] = np.full((FRAMES, JOINTS, 3), np.nan, dtype=np.float32)
    rotations[trusted] = 0.1
    return soma_x.SomaXFit(
        times_ns=np.arange(FRAMES, dtype=np.int64) * 33_000_000,
        frame_indices=np.arange(FRAMES, dtype=np.int64),
        trusted=trusted,
        vertices=np.arange(count * VERTICES * 3, dtype=np.float32).reshape(count, VERTICES, 3),
        joints=np.ones((count, JOINTS, 3), dtype=np.float32),
        rotations=rotations,
        root_translation=np.where(trusted[:, None], np.float32(0.5), np.float32(np.nan)).repeat(3, axis=1).astype(np.float32),
        identity_coeffs=np.where(trusted[:, None], np.float32(0.25), np.float32(np.nan)).repeat(10, axis=1).astype(np.float32),
        fit_error_mm=np.array([1.0, np.nan, 3.0, 1.5], dtype=np.float32),
    )


def model() -> soma_x.SomaXModel:
    return soma_x.SomaXModel(kind="hand", identity_model_type="mano", max_median_error_mm=2.0)


def source() -> soma_x.SourceParameters:
    return soma_x.SourceParameters(
        model="MANO",
        encoding={"hand_pose": "45 PCA coefficients, hands_mean added"},
        static={"betas": np.zeros(10, dtype=np.float32)},
        temporal={"hand_pose": np.zeros((FRAMES, 45), dtype=np.float32), "transl": np.zeros((FRAMES, 3), dtype=np.float32)},
    )


def write(tmp_path: Path, trusted: Bool[ndarray, "t"]) -> rrc.ChunkStore:
    target: Path = tmp_path / "seg.rrd"
    with atomic_recording(target, recording_id="seg", send_properties=False) as recording:
        soma_x.log_soma_x(recording, ROOT, fit=fit(trusted), topology=topology(), model=model(), source=source(), albedo=soma_x.HAND_ALBEDO["left"])
    return read_back(target)


def temporal_mesh_columns(store: rrc.ChunkStore) -> set[str]:
    """Component columns of the mesh's temporal chunks."""
    return {name for chunk in store.stream().filter(content=soma_x.mesh_path(ROOT), is_static=False).to_chunks() for name in chunk.to_record_batch().schema.names}


def clear_times(store: rrc.ChunkStore) -> list[int]:
    """Times of the mesh's Clear rows, read from the chunks (the dataframe reader applies a Clear and hides its column)."""
    times: list[int] = []
    for chunk in store.stream().filter(content=soma_x.mesh_path(ROOT), is_static=False).to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "Clear:is_recursive" in batch.schema.names:
            stamps: list[int] = batch.column(schema.TIMELINE).cast("int64").to_pylist()
            times += [stamp for stamp, row in zip(stamps, batch.column("Clear:is_recursive").to_pylist(), strict=True) if row is not None]
    return sorted(times)


def test_mesh_rows_are_xlo_vertices_on_trusted_rows_with_static_triangles_and_a_clear_elsewhere(tmp_path: Path) -> None:
    store: rrc.ChunkStore = write(tmp_path, np.array([True, False, True, True]))
    mesh: str = soma_x.mesh_path(ROOT)
    positions: pa.Table = column_rows(store, f"{mesh}:Mesh3D:vertex_positions")
    # The file holds a null at the untrusted row; the reader applies its Clear and returns an empty list there.
    assert [len(row) for row in positions.column(1).to_pylist()] == [VERTICES, 0, VERTICES, VERTICES]
    assert clear_times(store) == [33_000_000]
    statics: dict[str, object] = static_values(store, mesh)
    assert store.reader(index=None, contents=mesh).to_arrow_table().column(f"{mesh}:Mesh3D:triangle_indices").to_pylist() == [topology().faces.tolist()]
    assert statics["albedo_factor"] == soma_x.rgba_u32(soma_x.HAND_ALBEDO["left"])
    assert "Mesh3D:triangle_indices" not in temporal_mesh_columns(store)


def test_joints_are_green_and_their_context_links_each_joint_to_its_parent_in_white(tmp_path: Path) -> None:
    store: rrc.ChunkStore = write(tmp_path, np.array([True, True, False, True]))
    joints: str = soma_x.joints_path(ROOT)
    assert column_rows(store, f"{joints}:Points3D:positions").num_rows == FRAMES
    joint_statics: dict[str, object] = static_values(store, joints)
    assert joint_statics["colors"] == soma_x.rgba_u32((*soma_x.JOINT_COLOR, 255))
    assert joint_statics["show_labels"] is False
    context: dict[str, Any] = static_values(store, soma_x.soma_x_path(ROOT))
    joint_class: dict[str, Any] = context["context"][0]["class_description"]
    assert joint_class["info"]["color"] == soma_x.rgba_u32((*soma_x.LINK_COLOR, 255))
    assert [(link["keypoint0"], link["keypoint1"]) for link in joint_class["keypoint_connections"]] == [(0, 1), (1, 2)]


def test_soma_x_and_source_parameters_are_in_the_layer(tmp_path: Path) -> None:
    store: rrc.ChunkStore = write(tmp_path, np.array([True, False, True, True]))
    params: str = soma_x.params_path(ROOT)
    statics: dict[str, object] = static_values(store, params)
    assert statics["identity_model_type"] == "mano"
    assert statics["lod"] == "xlo"
    assert statics["assets_revision"] == soma_x.SOMA_X_REVISION
    identities: pa.Table = column_rows(store, f"{params}:identity_coeffs")
    assert identities.num_rows == FRAMES and len(identities.column(1)[0].as_py()[0]) == 10
    rotations: pa.Table = column_rows(store, f"{params}:rotations")
    assert rotations.num_rows == FRAMES
    errors: list[float | None] = [row[0] for row in column_rows(store, f"{params}:fit_error_mm").column(1).to_pylist()]
    assert errors[0] == 1.0 and errors[2] == 3.0
    origin: str = soma_x.source_path(ROOT)
    assert static_values(store, origin)["model"] == "MANO"
    assert column_rows(store, f"{origin}:hand_pose").num_rows == FRAMES


def test_mismatched_trusted_rows_are_refused(tmp_path: Path) -> None:
    broken: soma_x.SomaXFit = fit(np.array([True, True, True, True]))
    try:
        soma_x.SomaXFit(**{**{name: getattr(broken, name) for name in soma_x.SomaXFit.__slots__}, "trusted": np.array([True, False, True, True])})
    except ValueError:
        return
    raise AssertionError("a fit whose vertex rows disagree with its trusted mask must be refused")


def test_untrusted_rows_carry_a_clear_but_keep_their_error(tmp_path: Path) -> None:
    trusted: Bool[ndarray, "t"] = np.array([False, False, False, False])
    store: rrc.ChunkStore = write(tmp_path, trusted)
    assert f"{soma_x.mesh_path(ROOT)}:Mesh3D:vertex_positions" not in store.reader(index=schema.TIMELINE).to_arrow_table().schema.names
    assert len(clear_times(store)) == FRAMES
    errors: Float32[ndarray, "t"] = np.array(
        [row[0] if row[0] is not None else np.nan for row in column_rows(store, f"{soma_x.params_path(ROOT)}:fit_error_mm").column(1).to_pylist()],
        dtype=np.float32,
    )
    assert np.isfinite(errors).sum() == 3


def test_rows_written_in_windows_and_small_mesh_chunks_read_back_whole(tmp_path: Path) -> None:
    target: Path = tmp_path / "windows.rrd"
    trusted: Bool[ndarray, "t"] = np.array([True, False, True, True])
    whole: soma_x.SomaXFit = fit(trusted)
    with atomic_recording(target, recording_id="seg", send_properties=False) as recording:
        soma_x.log_soma_x_static(recording, ROOT, topology=topology(), model=model(), source=source(), albedo=soma_x.HAND_ALBEDO["left"])
        for rows in (slice(0, 2), slice(2, 4)):
            part: soma_x.SomaXFit = soma_x.window(whole, rows)
            soma_x.log_soma_x_rows(recording, ROOT, fit=part, topology=topology(), source_temporal={}, mesh_rows=1)
    store: rrc.ChunkStore = read_back(target)
    lengths: list[int] = [len(row) for row in column_rows(store, f"{soma_x.mesh_path(ROOT)}:Mesh3D:vertex_positions").column(1).to_pylist()]
    assert lengths == [VERTICES, 0, VERTICES, VERTICES]
    assert clear_times(store) == [33_000_000]
    assert "Mesh3D:triangle_indices" not in temporal_mesh_columns(store)


def test_selfcap_params_path_is_the_shared_params_path() -> None:
    from dataforge.selfcap_schema import SOMA_MESH, SOMA_PARAMS

    assert soma_x.params_path("/world/pred") == SOMA_PARAMS
    assert soma_x.mesh_path("/world/pred") == SOMA_MESH
