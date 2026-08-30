from pathlib import Path
from typing import Literal, Never

import pytest

from monopriors.models.surface_normal import DSineNormalPredictor, OmniNormalConfig, normal_predictor_defaults
from monopriors.models.surface_normal import dsine_model as dsine_model_module


def test_normal_predictor_defaults() -> None:
    assert set(normal_predictor_defaults) == {"dsine-normal", "moge-v2-normal", "omni-normal"}


def test_omni_normal_missing_weights_explains_how_to_fetch(tmp_path: Path) -> None:
    expected_checkpoint: Path = tmp_path / "omnidata_dpt_normal_v2.ckpt"

    with pytest.raises(FileNotFoundError, match="download_surface_normal_models.sh") as exc_info:
        OmniNormalConfig(omnidata_pretrained_weights_path=tmp_path).setup(device="cpu")

    assert str(expected_checkpoint) in str(exc_info.value)
    assert "zenodo.org/records/10447888" in str(exc_info.value)


@pytest.mark.parametrize(
    ("model_type", "expected_checkpoint"),
    [("dsine", "checkpoints/dsine.pt"), ("dsine_kappa", "checkpoints/dsine_kappa.pt")],
)
def test_dsine_model_type_selects_matching_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    model_type: Literal["dsine", "dsine_kappa"],
    expected_checkpoint: str,
) -> None:
    selected_checkpoints: list[str] = []

    def stop_after_selection(local_file_path: str, **_kwargs: object) -> Never:
        selected_checkpoints.append(local_file_path)
        raise RuntimeError("checkpoint selected")

    monkeypatch.setattr(dsine_model_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(dsine_model_module.torch, "load", stop_after_selection)
    predictor: DSineNormalPredictor = object.__new__(DSineNormalPredictor)

    with pytest.raises(RuntimeError, match="checkpoint selected"):
        predictor.load_model(model_type)

    assert selected_checkpoints == [expected_checkpoint]
