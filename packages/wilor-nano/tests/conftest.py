"""Skip model tests when their local pretrained assets are absent."""

from pathlib import Path

import pytest

PRETRAINED_MODELS_DIR: Path = Path(__file__).parents[1] / "src/wilor_nano/pretrained_models"


@pytest.fixture
def mano_assets() -> None:
    for name in ("mano_mean_params.npz", "mano_clean/MANO_RIGHT.pkl"):
        path: Path = PRETRAINED_MODELS_DIR / name
        if not path.is_file():
            pytest.skip(f"WiLoR asset not downloaded: {path}")


@pytest.fixture
def wilor_assets(mano_assets: None) -> None:
    for name in ("wilor_final.ckpt", "detector.pt"):
        path: Path = PRETRAINED_MODELS_DIR / name
        if not path.is_file():
            pytest.skip(f"WiLoR asset not downloaded: {path}")
