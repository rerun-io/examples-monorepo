"""The xlo layer maps stored low-LOD crops onto the xlo topology, from the soma layer on disk alone."""

import os
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32, Int64
from numpy import ndarray
from rerun.chunk import RrdReader

from selfcap_soma import xlo
from selfcap_soma.observations import video_time_column
from selfcap_soma.overlay import SOMA_CROP
from selfcap_soma.xlo import Config, RestMeshes, write_segment

DATASET = "test-dataset"
GOOD = "selfcap__2025-12-03__d31bf1da__ep001"
OTHER = "selfcap__2025-12-04__5cb1e14c__ep001"

# Low vertices 1..4 survive the crop; low vertex 0 sits below the waist.
REST = RestMeshes(
    low_vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 2, 0]], dtype=np.float32),
    # Nearest low vertex, in xlo order: 3, 1, 4, 0, 2. Xlo vertex 3 therefore leaves with low vertex 0.
    xlo_vertices=np.array([[0.9, 1.1, 0], [1.1, 0, 0], [0.1, 2, 0], [0, 0.1, 0], [0, 0.9, 0]], dtype=np.float32),
    xlo_faces=np.array([[0, 1, 4], [1, 3, 2], [0, 4, 2]], dtype=np.int32),
)
CROP: Int64[ndarray, "4"] = np.array([1, 2, 3, 4], dtype=np.int64)
# Rows hold the stored crop, in crop order: low vertices 1, 2, 3, 4.
CROP_ROWS: Float32[ndarray, "2 4 3"] = np.array(
    [
        [[10, 0, 0], [0, 10, 0], [10, 10, 0], [0, 20, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 2]],
    ],
    dtype=np.float32,
)


def write_layer(path: Path, segment: str, crop: Int64[ndarray, "v"] | None, rows: Float32[ndarray, "n v 3"]) -> None:
    recording = rr.RecordingStream("dataforge", recording_id=segment)
    recording.save(path)
    if crop is not None:
        recording.log(SOMA_CROP, rr.AnyValues(vertex_ids=crop), static=True)
    recording.log(
        "/world/pred/soma_x/mesh", rr.Mesh3D.from_fields(triangle_indices=np.array([[0, 1, 2]]), albedo_factor=[1.0, 0.0, 0.0, 1.0]), static=True
    )
    recording.send_columns(
        "/world/pred/soma_x/mesh",
        indexes=[video_time_column(np.array([0.0, 0.04], dtype=np.float64))],
        columns=rr.Mesh3D.columns(vertex_positions=rows.reshape(-1, 3), vertex_normals=rows.reshape(-1, 3)).partition([rows.shape[1]] * len(rows)),
    )
    recording.log("/world/pred/soma_x/ears", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    recording.flush()
    recording.disconnect()


def write_batch_segment(root: Path, segment: str, crop: Int64[ndarray, "v"] | None = CROP, rows: Float32[ndarray, "n v 3"] = CROP_ROWS) -> None:
    directory = root / DATASET / segment
    directory.mkdir(parents=True)
    write_layer(directory / f"{segment}.rrd", segment, crop, rows)


def config(root: Path, segments: list[str]) -> Config:
    return Config(soma_dir=root, dataset=DATASET, assets=root / "assets", output_dir=root / "out", segment=segments)


def component(path: Path, name: str, static: bool) -> list:
    """Every row of one mesh component, in time order, as Python lists."""
    rows: list[tuple[int, list]] = []
    for chunk in RrdReader(path).stream().filter(content=xlo.SOMA_XLO).to_chunks():
        batch = chunk.to_record_batch()
        if name in batch.schema.names and chunk.is_static == static:
            stamps = [0] * chunk.num_rows if static else batch.column("video_time").cast("int64").to_pylist()
            rows.extend(zip(stamps, batch.column(name).to_pylist(), strict=True))
    return [row for _, row in sorted(rows, key=lambda row: row[0])]


def test_positions_follow_the_nearest_low_vertex_through_the_crop(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD)
    target = write_segment(config(tmp_path, [GOOD]), REST, GOOD)
    assert target == tmp_path / "out" / f"{GOOD}.rrd"
    positions = component(target, "Mesh3D:vertex_positions", static=False)
    assert positions == [
        [[10, 10, 0], [10, 0, 0], [0, 20, 0], [0, 10, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 2], [0, 1, 0]],
    ]


def test_normals_are_area_weighted_on_the_kept_faces(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD)
    target = write_segment(config(tmp_path, [GOOD]), REST, GOOD)
    # Kept faces [0, 1, 3] and [0, 3, 2]. First frame: both face -z. Second frame: areas 1/2 toward +z and 1 toward +x, so 2:1 on the shared edge.
    normals = component(target, "Mesh3D:vertex_normals", static=False)
    np.testing.assert_allclose(normals[0], [[0, 0, -1]] * 4, atol=1e-6)
    shared = [2 / 5**0.5, 0, 1 / 5**0.5]
    np.testing.assert_allclose(normals[1], [shared, [0, 0, 1], [1, 0, 0], shared], atol=1e-6)


def test_faces_and_albedo_are_static_and_the_layer_carries_no_properties(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD)
    target = write_segment(config(tmp_path, [GOOD]), REST, GOOD)
    assert component(target, "Mesh3D:triangle_indices", static=True) == [[[0, 1, 3], [0, 3, 2]]]
    # 0.25, 0.65, 0.8, 0.5 in 8-bit RGBA.
    assert component(target, "Mesh3D:albedo_factor", static=True) == [[0x40A6CC80]]
    reader = RrdReader(target)
    assert [(entry.application_id, entry.recording_id) for entry in reader.recordings()] == [("selfcap-soma-xlo", GOOD)]
    assert {chunk.entity_path for chunk in reader.stream().to_chunks()} == {xlo.SOMA_XLO}
    times = [
        chunk.to_record_batch().column("video_time").cast("int64").to_pylist()
        for chunk in reader.stream().filter(has_timeline="video_time").to_chunks()
    ]
    assert sorted(sum(times, [])) == [0, 40_000_000]


def test_existing_target_is_refused_and_left_alone(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD)
    target = tmp_path / "out" / f"{GOOD}.rrd"
    target.parent.mkdir()
    target.write_bytes(b"registered")
    with pytest.raises(FileExistsError, match=GOOD):
        write_segment(config(tmp_path, [GOOD]), REST, GOOD)
    assert target.read_bytes() == b"registered"
    assert [path.name for path in target.parent.iterdir()] == [target.name]


def test_new_layer_is_world_readable_and_leaves_a_concurrent_runs_files_alone(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    concurrent = output_dir / f"{GOOD}.pending.rrd"  # the name a fixed stage would take
    concurrent.write_bytes(b"another run")
    previous_umask = os.umask(0o077)  # the catalog server must read the layer whatever the writer's umask
    try:
        target = write_segment(config(tmp_path, [GOOD]), REST, GOOD)
    finally:
        os.umask(previous_umask)
    assert target.stat().st_mode & 0o777 == 0o644
    assert concurrent.read_bytes() == b"another run"
    assert sorted(path.name for path in output_dir.iterdir()) == sorted([target.name, concurrent.name])


def test_no_fit_recording_is_read(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD)
    assert not (tmp_path / DATASET / GOOD / "fit.rrd").exists()
    assert write_segment(config(tmp_path, [GOOD]), REST, GOOD).is_file()


def test_crop_that_does_not_fit_the_stored_mesh_leaves_no_file(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD, crop=np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="stored mesh has 4 vertices, the layer's crop has 3"):
        write_segment(config(tmp_path, [GOOD]), REST, GOOD)
    assert list((tmp_path / "out").iterdir()) == []


def test_layer_without_a_crop_is_an_error_not_a_recovery(tmp_path: Path) -> None:
    write_batch_segment(tmp_path, GOOD, crop=None)
    with pytest.raises(ValueError, match="upper-body vertex selection"):
        write_segment(config(tmp_path, [GOOD]), REST, GOOD)


def test_failed_segment_exits_nonzero_after_the_others_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(xlo, "rest_meshes", lambda _assets: REST)
    missing = "selfcap__2025-12-01__01d9f1bf__ep001"
    write_batch_segment(tmp_path, OTHER, crop=None)
    write_batch_segment(tmp_path, GOOD)
    with pytest.raises(SystemExit) as raised:
        xlo.main(config(tmp_path, [missing, OTHER, GOOD]))
    assert raised.value.code == 1
    captured = capsys.readouterr()
    failures = captured.err.splitlines()
    assert [line.split(":")[0] for line in failures] == [f"soma_xlo FAIL {missing}", f"soma_xlo FAIL {OTHER}"]
    assert f"soma_xlo ok {GOOD}: {tmp_path / 'out' / GOOD}.rrd" in captured.out
    assert "1/3 segments written" in captured.out
    assert [path.name for path in (tmp_path / "out").iterdir()] == [f"{GOOD}.rrd"]


def test_one_process_loads_the_rest_meshes_once_for_several_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[Path] = []
    monkeypatch.setattr(xlo, "rest_meshes", lambda assets: loads.append(assets) or REST)
    write_batch_segment(tmp_path, GOOD)
    write_batch_segment(tmp_path, OTHER)
    listing = tmp_path / "segments.txt"
    listing.write_text(f"# one per line\n{OTHER}\n")
    xlo.main(Config(soma_dir=tmp_path, dataset=DATASET, assets=tmp_path, output_dir=tmp_path / "out", segment=[GOOD], segments_file=listing))
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted([f"{GOOD}.rrd", f"{OTHER}.rrd"])
    assert loads == [tmp_path]


def test_unsafe_segment_id_is_rejected_before_the_model_loads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="segment"):
        xlo.main(config(tmp_path, ["../../other"]))


@pytest.mark.integration
def test_rest_meshes_from_the_soma_assets() -> None:
    assets = Path("/tmp/soma-fit/assets")
    if not (assets / "SOMA_neutral.npz").exists():
        pytest.skip("requires /tmp/soma-fit/assets/SOMA_neutral.npz")
    rest = xlo.rest_meshes(assets)
    assert rest.low_vertices.shape == (4505, 3)
    assert rest.xlo_vertices.shape == (612, 3)
    assert rest.xlo_faces.shape == (1220, 3)
    assert rest.xlo_faces.max() == 611
    # Bounds from the arrays that built the first registered soma_xlo layers.
    np.testing.assert_allclose([rest.low_vertices.min(), rest.low_vertices.max()], [-0.9531919, 0.8687375], atol=1e-5)
    np.testing.assert_allclose([rest.xlo_vertices.min(), rest.xlo_vertices.max()], [-0.952439, 0.86810285], atol=1e-5)


def test_layer_without_mesh_frames_fails_and_leaves_no_file(tmp_path: Path) -> None:
    directory = tmp_path / DATASET / GOOD
    directory.mkdir(parents=True)
    recording = rr.RecordingStream("dataforge", recording_id=GOOD)
    recording.save(directory / f"{GOOD}.rrd")
    recording.log(SOMA_CROP, rr.AnyValues(vertex_ids=CROP), static=True)
    recording.log("/world/pred/soma_x/mesh", rr.Mesh3D.from_fields(triangle_indices=np.array([[0, 1, 2]])), static=True)
    recording.flush()
    recording.disconnect()
    with pytest.raises(ValueError, match="no mesh frames"):
        write_segment(config(tmp_path, [GOOD]), REST, GOOD)
    assert list((tmp_path / "out").iterdir()) == []
