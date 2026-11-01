"""GPU proof of the production Sapiens2 engine and its dynamic batches."""

import json
from dataclasses import replace
from typing import Any

import pytest

# ruff: noqa: E402

pytest.importorskip("torch")
import torch

from posekit.models.sapiens import SapiensPoseConfig
from posekit.runtimes import TensorRtBackendConfig, run_chunked


@pytest.mark.integration
def test_fast_sapiens_engine_kernels_and_batches() -> None:
    from huggingface_hub import try_to_load_from_cache

    checkpoint = try_to_load_from_cache("facebook/sapiens2-pose-0.4b", "sapiens2_0.4b_pose.safetensors")
    if not isinstance(checkpoint, str):
        pytest.skip("Missing cached sapiens2_0.4b_pose.safetensors checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for TensorRT Sapiens2 engine")
    import tensorrt

    trt: Any = tensorrt

    # The production default caps calls at 8; also prove an explicitly requested
    # 16-crop dynamic profile, using the same export and build path.
    for maximum in (8, 16):
        config = replace(SapiensPoseConfig(), backend=TensorRtBackendConfig(max_batch_size=maximum, opt_batch_size=4))
        model = config.setup()
        runtime: Any = model.runtime
        inspector = runtime._engine.create_engine_inspector()
        description = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        layers = json.loads(description)["Layers"]
        fp4 = [layer for layer in layers if "block_scaled_ue4m3xe2m1" in layer.get("TacticName", "")]
        mha = [layer for layer in layers if "_gemm_mha_v2" in layer.get("TacticName", "")]
        head = [layer for layer in layers if "e4m3" in layer.get("TacticName", "") and "conv" in layer["Name"].lower()]
        assert len(fp4) == 144
        assert len(mha) == 24
        assert all(all("FP8" in tensor["Datatype"] for tensor in layer["Inputs"][:3]) for layer in mha)
        assert len(head) >= 5
        for batch in (1, 3, 8, 16):
            inputs = torch.zeros((batch, 3, 768, 576), device="cuda")
            output = run_chunked(runtime, {"inputs": inputs})["heatmaps"]
            assert output.shape == (batch, 65, 192, 144)
            assert torch.isfinite(output).all()
