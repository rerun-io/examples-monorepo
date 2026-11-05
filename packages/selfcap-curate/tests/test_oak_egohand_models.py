"""The OAK fit uses the default Hugging Face files and the trtkit cache unless an asset root overrides them."""

from pathlib import Path

from posekit.models.egohand import CACHED_BACKBONE_ONNX, EgoHandConfig, EgoHandDetectorConfig
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR

from selfcap_curate.oak_egohand import egohand_configs


def test_no_asset_root_uses_the_downloads_and_the_trtkit_cache() -> None:
    detector, predictor = egohand_configs(None)
    assert detector.weights is None and predictor.weights is None and predictor.soma_assets is None
    assert detector.backend.cache_dir == DEFAULT_TRT_CACHE_DIR
    assert predictor.backbone_backend.cache_dir == DEFAULT_TRT_CACHE_DIR
    assert predictor.backbone_onnx == CACHED_BACKBONE_ONNX == DEFAULT_TRT_CACHE_DIR / "egohand-dinov3-bf16-24a325fd.onnx"
    assert isinstance(detector, EgoHandDetectorConfig) and isinstance(predictor, EgoHandConfig)


def test_an_asset_root_overrides_every_file(tmp_path: Path) -> None:
    detector, predictor = egohand_configs(tmp_path)
    assert detector.weights == tmp_path / "weights/detector/hand_detector.onnx"
    assert (predictor.weights, predictor.soma_assets) == (tmp_path / "weights/egohand/egohand_s060000.safetensors", tmp_path / "weights/soma")
    assert detector.backend.cache_dir == predictor.backbone_backend.cache_dir == tmp_path / "engines"
    assert predictor.backbone_onnx == tmp_path / "dinov3-bf16.onnx"


def test_fast_mount_models_use_posekit_fast_roles_without_changing_default() -> None:
    detector, predictor = egohand_configs(None, fast=True)
    assert detector.gpu_preprocessing and detector.fp16_trunk
    assert predictor.nvfp4_backbone and predictor.cuda_execution
    assert predictor.head_onnx is not None
    default_detector, default_predictor = egohand_configs(None)
    assert not default_detector.gpu_preprocessing
    assert not default_predictor.nvfp4_backbone and not default_predictor.cuda_execution
