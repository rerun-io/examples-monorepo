"""Content-addressed ONNX rewrites preserve their source artifacts."""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from trtkit.onnx_precision import fp16_trunk
from trtkit.trt_builder import onnx_content_hash


def test_fp16_trunk_is_cached_beside_the_other_rewrites_and_leaves_the_source(tmp_path, monkeypatch):
    monkeypatch.setenv("POSEKIT_ONNX_CACHE", str(tmp_path / "cache"))
    weight = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="w")
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Conv", ["input", "w"], ["c"]), helper.make_node("Sigmoid", ["c"], ["output"])],
            "conv-then-tail",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 2, 2])],
            [weight],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    source = tmp_path / "rtmw.onnx"
    onnx.save(model, source)
    before = source.read_bytes()
    result = fp16_trunk(source, tmp_path / "cache")
    assert result == tmp_path / f"cache/rtmw_{onnx_content_hash(source, cache_dir=tmp_path / 'cache')[:16]}_fp16trunk_v1.onnx"
    assert source.read_bytes() == before
    assert [t.data_type for t in onnx.load(result).graph.initializer] == [onnx.TensorProto.FLOAT16]
    written = result.stat().st_mtime_ns
    assert fp16_trunk(source, tmp_path / "cache") == result and result.stat().st_mtime_ns == written
    assert sorted(path.name for path in result.parent.iterdir()) == sorted([result.name, f"{result.name}.lock", "onnx-sha256"])



def test_fp16_cache_separates_sources_with_the_same_name(tmp_path: Path) -> None:
    paths = []
    for index in (1, 2):
        source = tmp_path / str(index) / "model.onnx"
        source.parent.mkdir()
        model = helper.make_model(helper.make_graph(
            [helper.make_node("Conv", ["input", "w"], ["c"]), helper.make_node("Sigmoid", ["c"], ["output"])], "conv",
            [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 2, 2])],
            [numpy_helper.from_array(np.full((1, 1, 1, 1), float(index), np.float32), "w")],
        ), opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, source)
        paths.append(fp16_trunk(source, tmp_path / "cache"))
    assert paths[0] != paths[1]
    for index, path in enumerate(paths, 1):
        np.testing.assert_array_equal(numpy_helper.to_array(onnx.load(path).graph.initializer[0]), np.full((1, 1, 1, 1), index, np.float16))
