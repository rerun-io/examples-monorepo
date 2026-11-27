"""The flexglue_ort backend's COLMAP options (FlexGlue type, local graphs, image root, covisibility gate) and the graph pre-flight."""

from __future__ import annotations

from pathlib import Path

import pycolmap
import pytest

from colsfm import model_assets
from colsfm.matching import FLEXGLUE_MODEL_FILES, FlexGlueOptions, MatchingOptions, _matching_options, gpu_has_fp8
from colsfm.model_assets import required_raco_graphs


def test_flexglue_options_point_colmap_at_local_graphs_and_the_image_root(tmp_path: Path) -> None:
    """Every graph path is a local file under `flexglue_model_dir`, so COLMAP never downloads one."""
    options = MatchingOptions(
        backend="flexglue_ort", image_dir=tmp_path / "images", flexglue_model_dir=tmp_path / "models", flexglue_min_covisibility=0.6
    )
    built: pycolmap.FeatureMatchingOptions = _matching_options(options, "cuda")
    assert built.type == pycolmap.FeatureMatcherType.FLEXGLUE
    assert built.flexglue.image_path == str(tmp_path / "images")
    assert built.flexglue.min_covisibility == pytest.approx(0.6)
    for field_name, file_name in FLEXGLUE_MODEL_FILES.items():
        assert getattr(built.flexglue, field_name) == str(tmp_path / "models" / file_name)


def test_flexglue_without_an_image_root_is_refused() -> None:
    """FlexGlue reads pixels; without `image_dir` it would fail inside a COLMAP worker thread instead."""
    with pytest.raises(ValueError, match="image_dir"):
        _matching_options(MatchingOptions(backend="flexglue_ort"), "cuda")


def test_a_flexglue_run_requires_its_three_graphs_and_the_fp8_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-flight names the FlexGlue graphs and the calibrated FP8 matcher (TensorRT backend on an FP8 GPU only),
    next to the RaCo extractor; COLMAP's own ONNX Runtime FlexGlue needs no FP8 graph."""
    monkeypatch.setattr(model_assets, "gpu_has_fp8", lambda: True)
    names = [path.name for path in required_raco_graphs("raco", "flexglue", "tensorrt")]
    assert names[1:] == [*FLEXGLUE_MODEL_FILES.values(), FlexGlueOptions().fp8_file()]
    assert FlexGlueOptions().fp8_file() not in [path.name for path in required_raco_graphs("raco", "flexglue_ort", "tensorrt")]


def test_a_gpu_without_fp8_needs_no_fp8_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below sm_89 (an RTX 3060) TensorRT refuses FP8 Q/DQ graphs: the matcher runs its prepared graph, so the pre-flight
    asks for the three FlexGlue graphs alone."""
    monkeypatch.setattr(model_assets, "gpu_has_fp8", lambda: False)
    names = [path.name for path in required_raco_graphs("raco", "flexglue", "tensorrt")]
    assert names[1:] == list(FLEXGLUE_MODEL_FILES.values())


@pytest.mark.parametrize(("capability", "fp8"), [((8, 6), False), ((8, 9), True), ((12, 0), True)])
def test_fp8_starts_at_compute_capability_8_9(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], fp8: bool) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert gpu_has_fp8() is fp8
