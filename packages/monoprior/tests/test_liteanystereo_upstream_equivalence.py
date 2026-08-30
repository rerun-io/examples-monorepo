"""Prove that the vendored LiteAnyStereo V2 models stay numerically identical to upstream."""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch
from jaxtyping import Float
from torch import nn

from monopriors.third_party.liteanystereo.liteanystereov2 import build_liteanystereo
from monopriors.third_party.liteanystereo.liteanystereov2_H import LiteAnyStereoH

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "liteanystereo"
UPSTREAM_PACKAGE: str = "las_upstream"


def _load_upstream_module(name: str) -> ModuleType:
    """Load one pristine upstream source fixture as a synthetic package module.

    Args:
        name: Upstream module basename without the ``upstream_`` fixture prefix.

    Returns:
        The loaded upstream module.
    """
    path: Path = REFERENCE_DIR / f"upstream_{name}.py"
    qualified_name: str = f"{UPSTREAM_PACKAGE}.{name}"
    spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load upstream fixture {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_package() -> dict[str, ModuleType]:
    """Load the mutually importing upstream fixtures into a synthetic package.

    Returns:
        The loaded upstream modules keyed by their source basenames.
    """
    package: ModuleType = ModuleType(UPSTREAM_PACKAGE)
    package.__path__ = [str(REFERENCE_DIR)]
    sys.modules[UPSTREAM_PACKAGE] = package
    return {name: _load_upstream_module(name) for name in ("submodule", "aggregation_fasternet", "fnet", "liteanystereov2", "liteanystereov2_H")}


def _assert_models_equal(upstream_model: nn.Module, ours_model: nn.Module, seed: int, max_disp: int) -> None:
    """Assert state-dict and exact inference parity for two stereo models.

    Args:
        upstream_model: Model built from the pristine upstream fixtures.
        ours_model: Model built from the vendored package.
        seed: Random seed used to create the stereo pair.
        max_disp: Maximum disparity accepted by the model architecture.
    """
    upstream_state_dict: dict[str, torch.Tensor] = dict(upstream_model.state_dict())
    ours_state_dict: dict[str, torch.Tensor] = dict(ours_model.state_dict())
    assert list(upstream_state_dict) == list(ours_state_dict)
    ours_model.load_state_dict(upstream_state_dict, strict=True)

    upstream_model.eval()
    ours_model.eval()
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    image0_b3hw: Float[torch.Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96, generator=generator) * 255.0
    image1_b3hw: Float[torch.Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96, generator=generator) * 255.0
    with torch.no_grad():
        upstream_output_b1hw: Float[torch.Tensor, "1 1 64 96"] = upstream_model(
            image0_b3hw, image1_b3hw, max_disp=max_disp, test_mode=True
        )
        ours_output_b1hw: Float[torch.Tensor, "1 1 64 96"] = ours_model(image0_b3hw, image1_b3hw, max_disp=max_disp, test_mode=True)
    assert torch.equal(upstream_output_b1hw, ours_output_b1hw)


@pytest.mark.parametrize("model_size", ("s", "m", "l"))
def test_feed_forward_model_matches_upstream(model_size: str) -> None:
    upstream_modules: dict[str, ModuleType] = _load_upstream_package()
    upstream_build: Callable[..., nn.Module] = upstream_modules["liteanystereov2"].build_liteanystereo

    torch.manual_seed(41)
    upstream_model: nn.Module = upstream_build(model_size=model_size, fnet_pretrained=False, max_disp=64)
    torch.manual_seed(41)
    ours_model: nn.Module = build_liteanystereo(model_size=model_size, fnet_pretrained=False, max_disp=64)
    # The feed-forward release aggregation is fixed at 48 disparity channels (192 / 4); the builder's max_disp only configures H.
    _assert_models_equal(upstream_model, ours_model, seed=42, max_disp=192)


def test_h_model_matches_upstream() -> None:
    upstream_modules: dict[str, ModuleType] = _load_upstream_package()
    upstream_h_factory: Callable[..., nn.Module] = upstream_modules["liteanystereov2_H"].LiteAnyStereoH

    torch.manual_seed(43)
    upstream_model: nn.Module = upstream_h_factory(fnet_pretrained=False, max_disp=64)
    torch.manual_seed(43)
    ours_model: nn.Module = LiteAnyStereoH(fnet_pretrained=False, max_disp=64)
    _assert_models_equal(upstream_model, ours_model, seed=44, max_disp=64)
