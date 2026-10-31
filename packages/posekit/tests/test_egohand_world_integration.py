"""The generated layer keeps its identity, empty observations and typed fields."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rerun as rr
from rerun.chunk import RrdReader

pytest.importorskip("egohand_inference", reason="requires the egohand Pixi environment")

from posekit.apis.egohand_world import MeshLod, write_world_layer

pytestmark = pytest.mark.integration


def test_layer_round_trip_keeps_empty_frames_and_float32_normals(tmp_path: Path) -> None:
    vector = pa.list_(pa.float32(), 3)
    pixel = pa.list_(pa.float32(), 2)
    types = {"vertices": vector, "normals": vector, "joints": vector, "rgb": pixel, "mono_left": pixel, "mono_right": pixel}
    schema = pa.schema([("video_time", pa.int64()), *[(f"hand_{name}", pa.list_(kind)) for name, kind in types.items()]])
    rows = [
        {"video_time": 100, **{f"hand_{name}": np.ones((25, 3 if kind == vector else 2), np.float32).tolist() for name, kind in types.items()}},
        {"video_time": 200, **{f"hand_{name}": [] for name in types}},
    ]
    parts = {side: tmp_path / f"{side}.parquet" for side in ("left", "right")}
    for path in parts.values():
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    lod = MeshLod(np.arange(25), np.array([[0, 1, 2]], np.uint32))
    output = tmp_path / "layer.rrd"
    write_world_layer(parts, {side: lod for side in parts}, output, "dataforge", "segment")
    reader = RrdReader(output)
    assert reader.recordings()[0].application_id == "dataforge"
    assert reader.recordings()[0].recording_id == "segment"
    found = False
    annotation_found = False
    for chunk in reader.stream().to_chunks():
        if str(chunk.entity_path) == "/world/pred/egohand/left":
            context = chunk.to_record_batch()["AnnotationContext:context"].to_pylist()[0][0][0]
            assert len(context["class_description"]["keypoint_connections"]) == 24
            annotation_found = True
        if str(chunk.entity_path) == "/world/pred/egohand/left/mesh":
            batch = chunk.to_record_batch()
            if "Mesh3D:vertex_normals" in batch.schema.names:
                assert batch["Mesh3D:vertex_normals"].type.value_type == rr.components.Vector3DBatch([]).as_arrow_array().type
                assert batch["Mesh3D:vertex_positions"].to_pylist()[-1] == []
                assert batch["video_time"].cast(pa.int64()).to_pylist() == [100, 200]
                found = True
    assert found and annotation_found
