"""The bf16-1024 ONNX export must not rewrite the cached torch model the torch backend runs."""

from pathlib import Path

import pytest

# ruff: noqa: E402

pytest.importorskip("torch")
import torch


@pytest.mark.integration
def test_export_leaves_cached_torch_model_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache("facebook/sapiens2-pose-0.4b", "sapiens2_0.4b_pose.safetensors"), str):
        pytest.skip("Missing cached sapiens2_0.4b_pose.safetensors checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required to load the Sapiens2 checkpoint")
    import trtkit

    from posekit.models import sapiens
    from posekit.models.sapiens_net.checkpoints import load_pose_model

    exported: list[torch.nn.Module] = []
    monkeypatch.setattr(sapiens, "DEFAULT_ONNX_CACHE_DIR", tmp_path)
    monkeypatch.setattr(trtkit, "export_onnx", lambda model, *args, **kwargs: exported.append(model))
    sapiens._ensure_sapiens_onnx("0.4B", 1, bf16=True)

    cached: torch.nn.Module = load_pose_model("0.4B")
    assert len(exported) == 1 and exported[0] is not cached
    assert any(isinstance(module, torch.nn.RMSNorm) for module in cached.modules())
    assert not any(isinstance(module, torch.nn.RMSNorm) for module in exported[0].modules())
