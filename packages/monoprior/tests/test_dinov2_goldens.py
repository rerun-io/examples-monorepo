from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from monopriors.third_party.dinov2 import DINOv2, dinov2_vits14
from monopriors.third_party.dinov2.vision_transformer import DinoVisionTransformer


@pytest.mark.parametrize(
    ("golden_prefix", "model_factory"),
    (
        ("dav2", partial(DINOv2, "vits")),
        ("moge", partial(dinov2_vits14, pretrained=False)),
    ),
)
def test_vits_intermediate_layers_match_permanent_goldens(
    golden_prefix: str,
    model_factory: Callable[[], DinoVisionTransformer],
) -> None:
    torch.manual_seed(1234)
    model: DinoVisionTransformer = model_factory()
    model.eval()

    torch.manual_seed(7)
    image_bchw: torch.Tensor = torch.randn(1, 3, 154, 210)
    with torch.inference_mode():
        outputs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = model.get_intermediate_layers(
            image_bchw,
            4,
            return_class_token=True,
        )

    golden_path: Path = Path(__file__).parent / "reference_data" / "dinov2_vits_forward_goldens.npz"
    with np.load(golden_path) as goldens:
        for layer_index, (patch_tokens, class_token) in enumerate(outputs):
            np.testing.assert_allclose(
                patch_tokens.cpu().numpy(),
                goldens[f"{golden_prefix}/layer{layer_index}/patch"],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                class_token.cpu().numpy(),
                goldens[f"{golden_prefix}/layer{layer_index}/cls"],
                rtol=0.0,
                atol=0.0,
            )
