"""Content-addressed ONNX rewrites preserve their source artifacts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper
from trtkit.trt_builder import onnx_content_hash

from posekit.models.yolox import dynamic_yolox


@pytest.mark.parametrize("workers", [1, 2])
def test_dynamic_export_preserves_posekit_source_in_default_cache(tmp_path, monkeypatch, workers):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("POSEKIT_ONNX_CACHE", raising=False)
    monkeypatch.setattr("posekit.models.yolox.DEFAULT_ONNX_CACHE_DIR", tmp_path / ".cache/posekit/onnx")
    shared = tmp_path / "posekit"
    shared.mkdir()
    nodes, constants = [], []
    for i in range(5):
        constants.append(numpy_helper.from_array(np.array([1, 2], dtype=np.int64), name=f"shape{i}"))
        nodes.append(helper.make_node("Reshape", ["input" if i == 0 else f"x{i - 1}", f"shape{i}"], [f"x{i}"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "five-reshapes",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("x4", onnx.TensorProto.FLOAT, [1, 2])],
            constants,
        )
    )
    source = shared / "model.onnx"
    onnx.save(model, source)
    before = source.read_bytes()
    destination = tmp_path / f".cache/posekit/onnx/model_{onnx_content_hash(source, cache_dir=tmp_path / 'cache')[:16]}_dynamic_batch_v1.onnx"
    save = onnx.save
    barrier = Barrier(workers)
    saved_paths = []

    def save_before_publication(model, path):
        saved_paths.append(Path(path))
        assert Path(path) != destination
        assert not destination.exists()
        save(model, path)
        barrier.wait(timeout=10)

    monkeypatch.setattr(onnx, "save", save_before_publication)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(dynamic_yolox, [source] * workers))
    result = results[0]
    assert all(path == destination for path in results)
    assert len(set(saved_paths)) == workers
    assert all(not path.exists() for path in saved_paths)
    onnx.checker.check_model(onnx.load(result))
    assert result.parent == tmp_path / ".cache/posekit/onnx"
    assert source.read_bytes() == before
    assert list(shared.iterdir()) == [source]

