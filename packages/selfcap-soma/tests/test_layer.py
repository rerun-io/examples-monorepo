"""The soma layer comes straight from the fit; soma-overlay gives the same layer from a legacy fit.rrd."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rerun as rr
import trimesh
from jaxtyping import Float32, Int32
from numpy import ndarray
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from rerun.server import Server
from test_quality import small_fit

from selfcap_soma import overlay, xlo
from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import FitResult
from selfcap_soma.output import save, write_report
from selfcap_soma.overlay import SOMA_CROP, Config, angle_weighted_normals, layer_fit, read_fit, read_layer_crop, write_layer
from selfcap_soma.quality import read_quality

FACES: Int32[ndarray, "6 3"] = np.array([[0, 1, 2], [0, 2, 3], [1, 3, 2], [3, 1, 0], [0, 0, 1], [1, 2, 4]], dtype=np.int32)


def test_normals_match_trimesh_on_every_frame_of_a_small_mesh() -> None:
    # Face 4 repeats a vertex (degenerate), vertex 4 makes a needle triangle, and vertex 5 belongs to no face.
    vertices: Float32[ndarray, "3 6 3"] = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [0.5, 1.0, 1e-3], [9, 9, 9]],
            [[0.2, 0, 0], [1, 0.4, 0], [0, 2, 0.5], [0.1, 0, 1], [0.5, 1.2, 0.1], [9, 9, 9]],
            [[0, 0, 0], [0.01, 0, 0], [0, 5, 0], [0, 0, 0.3], [0.2, 2.0, 0.1], [0, 0, 0]],
        ],
        dtype=np.float32,
    )
    expected = np.stack([trimesh.Trimesh(v, FACES, process=False).vertex_normals for v in vertices])
    normals: Float32[ndarray, "3 6 3"] = angle_weighted_normals(vertices, FACES)
    assert normals.dtype == np.float32
    np.testing.assert_allclose(normals, expected, atol=1e-6)
    assert not np.allclose(normals, xlo.vertex_normals(vertices, FACES), atol=1e-3)


def moving_fit() -> tuple[Observations, FitResult]:
    data, result = small_fit()
    vertices: Float32[ndarray, "2 4 3"] = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], [[0, 0, 1], [1, 0, 1], [0, 2, 1], [1, 1, 2]]], dtype=np.float32
    )
    return data, replace(result, vertices=vertices, parents=np.arange(-1, 77, dtype=np.int64))


@pytest.fixture
def no_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(overlay, "CatalogClient", lambda _url: SimpleNamespace(get_dataset=lambda _name: None))
    monkeypatch.setattr(overlay, "read_views", lambda *_args: ([], None))


def contents(path: Path) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Every component of a recording, in time order, comparable across files."""
    found: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for chunk in RrdReader(path).stream().to_chunks():
        batch = chunk.to_record_batch()
        stamps = batch.column("video_time").cast("int64").to_pylist() if "video_time" in batch.schema.names else [-1] * batch.num_rows
        for name in batch.schema.names:
            if name != "video_time" and not name.startswith("rerun") and name != "RecordingInfo:start_time":
                found.setdefault((str(chunk.entity_path), name), []).extend(zip(stamps, map(repr, batch.column(name).to_pylist()), strict=True))
    return {key: sorted(rows) for key, rows in found.items()}


def test_layer_from_memory_holds_the_crop_hidden_and_needs_no_fit_recording(tmp_path: Path, no_catalog: None) -> None:
    data, result = moving_fit()
    layer = tmp_path / "layer.rrd"
    write_layer(Config(output=layer, dataset="test", segment="segment"), layer_fit(data, result))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["layer.rrd"]
    np.testing.assert_array_equal(read_layer_crop(RrdReader(layer)), [0, 1, 2])
    found = contents(layer)
    assert not SOMA_CROP.startswith("/world")
    assert {name for entity, name in found if entity == SOMA_CROP} == {"vertex_ids"}  # no visualizer, so no view spawns for it
    assert read_quality(layer).body_median_mm == pytest.approx(10.0)
    assert [stamp for stamp, _ in found[(overlay.SOMA_MESH, "Mesh3D:vertex_normals")]] == [0, 40_000_000]


def test_legacy_fit_recording_gives_the_same_layer(tmp_path: Path, no_catalog: None) -> None:
    data, result = moving_fit()
    save(tmp_path, data, result, 0.0)
    direct = tmp_path / "direct.rrd"
    legacy = tmp_path / "legacy.rrd"
    write_layer(Config(output=direct, dataset="test", segment="segment"), layer_fit(data, result))
    overlay.main(Config(source=tmp_path / "fit.rrd", output=legacy, dataset="test", segment="segment"))
    assert contents(legacy) == contents(direct)


def test_full_body_layer_keeps_every_vertex(tmp_path: Path, no_catalog: None) -> None:
    data, result = moving_fit()
    save(tmp_path, data, result, 0.0)
    layer = tmp_path / "full.rrd"
    overlay.main(Config(source=tmp_path / "fit.rrd", output=layer, dataset="test", segment="segment", upper_body_only=False))
    chunks = RrdReader(layer).stream().filter(content=overlay.SOMA_MESH, is_static=False).to_chunks()
    assert overlay.mesh_vertex_rows(chunks[0].to_record_batch().column("Mesh3D:vertex_positions")).shape == (2, 4, 3)
    assert not RrdReader(layer).stream().filter(content=SOMA_CROP).to_chunks()


def test_read_fit_keeps_float32_points_from_the_recording(tmp_path: Path) -> None:
    data, result = moving_fit()
    save(tmp_path, data, result, 0.0)
    fit = read_fit(tmp_path / "fit.rrd")
    assert fit.names == result.names
    assert fit.edges[:2] == [(0, 1), (1, 2)]
    np.testing.assert_array_equal(fit.vertices, result.vertices)
    np.testing.assert_array_equal(fit.upper_body_vertices, [0, 1, 2])


def test_report_alone_writes_no_recording(tmp_path: Path) -> None:
    data, result = moving_fit()
    write_report(tmp_path, data, result, 0.0, upper_body_only=True)
    assert [path.name for path in tmp_path.iterdir()] == ["report.json"]


def test_crop_reads_back_through_the_catalog_and_stays_out_of_world(tmp_path: Path, no_catalog: None) -> None:
    data, result = moving_fit()
    layer = tmp_path / "segment.rrd"
    write_layer(Config(output=layer, dataset="test", segment="segment"), layer_fit(data, result))
    with Server(port=0) as server:
        client: CatalogClient = server.client()
        dataset = client.create_dataset("layer-test")
        dataset.register([layer.as_uri()], layer_name="soma").wait()
        table = dataset.filter_segments("segment").filter_contents([SOMA_CROP]).reader(index=None).to_arrow_table()
        assert table[f"{SOMA_CROP}:vertex_ids"][0].as_py() == [0, 1, 2]
        assert read_layer_crop(RrdReader(layer)).tolist() == [0, 1, 2]
        default_view = dataset.filter_segments("segment").filter_contents(["/world/**"]).reader(index=None).to_arrow_table()
        assert not [name for name in default_view.column_names if SOMA_CROP in name]


def test_missing_crop_in_a_layer_says_to_refit(tmp_path: Path) -> None:
    path = tmp_path / "old.rrd"
    recording = rr.RecordingStream("dataforge", recording_id="segment")
    recording.save(path)
    recording.log("/world/other", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    recording.flush()
    recording.disconnect()
    with pytest.raises(ValueError, match="upper-body vertex selection"):
        read_layer_crop(RrdReader(path))
