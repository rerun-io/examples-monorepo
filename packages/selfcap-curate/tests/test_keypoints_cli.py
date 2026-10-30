"""Session CLI rejects unsafe selections/outputs before catalog or GPU work."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from selfcap_curate.apis.keypoints import Config, main
from selfcap_curate.keypoints_gpu import dynamic_yolox, fp16_trunk


def test_keypoints_cli_rejects_invalid_session_and_existing_targets(tmp_path: Path):
    calib = tmp_path / "calib"
    calib.mkdir()
    key = "2025-12-30__65d00748__ep001"
    with pytest.raises(ValueError, match="explicit session episode keys"):
        main(Config(calib, tmp_path / "out", rate=3.0))
    with pytest.raises(ValueError, match="exactly one session"):
        main(Config(calib, tmp_path / "out", rate=3.0, segment_ids=(key, "2025-12-20__25aeb66a__ep001")))
    with pytest.raises(FileNotFoundError):
        main(Config(calib, tmp_path / "out", rate=3.0, segment_ids=(key,)))
    (calib / f"selfcap__{key}.rrd").touch()
    output = tmp_path / "out"
    output.mkdir()
    target = output / f"selfcap__{key}.rrd"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        main(Config(calib, output, rate=3.0, segment_ids=(key,)))
    assert target.read_bytes() == b"keep"
    with pytest.raises(ValueError):
        main(Config(calib, calib / "nested", rate=3.0, segment_ids=(key,)))
    with pytest.raises(ValueError):
        main(Config(calib, Path("/mnt/nas/forbidden-keypoints"), rate=3.0, segment_ids=(key,)))


def test_dynamic_export_preserves_posekit_source_in_default_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("POSEKIT_ONNX_CACHE", raising=False)
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
    result = dynamic_yolox(source)
    assert result.parent == tmp_path / ".cache/selfcap-curate/onnx"
    assert source.read_bytes() == before
    assert list(shared.iterdir()) == [source]


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
    result = fp16_trunk(source)
    assert result == tmp_path / "cache/rtmw_fp16trunk_v1.onnx"
    assert source.read_bytes() == before
    assert [t.data_type for t in onnx.load(result).graph.initializer] == [onnx.TensorProto.FLOAT16]
    written = result.stat().st_mtime_ns
    assert fp16_trunk(source) == result and result.stat().st_mtime_ns == written
    assert sorted(path.name for path in result.parent.iterdir()) == [result.name]


def test_rate_is_required(tmp_path):
    with pytest.raises(TypeError, match="rate"):
        Config(tmp_path / "calib", tmp_path / "out")
