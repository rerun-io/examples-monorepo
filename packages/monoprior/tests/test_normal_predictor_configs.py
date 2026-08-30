from pathlib import Path

import pytest

from monopriors.models.surface_normal import OmniNormalConfig, normal_predictor_defaults


def test_normal_predictor_defaults() -> None:
    assert set(normal_predictor_defaults) == {"dsine-normal", "moge-v2-normal", "omni-normal"}


def test_omni_normal_missing_weights_explains_how_to_fetch(tmp_path: Path) -> None:
    expected_checkpoint: Path = tmp_path / "omnidata_dpt_normal_v2.ckpt"

    with pytest.raises(FileNotFoundError, match="download_surface_normal_models.sh") as exc_info:
        OmniNormalConfig(omnidata_pretrained_weights_path=tmp_path).setup(device="cpu")

    assert str(expected_checkpoint) in str(exc_info.value)
    assert "zenodo.org/records/10447888" in str(exc_info.value)
