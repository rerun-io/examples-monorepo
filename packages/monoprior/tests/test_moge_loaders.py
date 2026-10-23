import pytest

from monopriors.third_party.moge.model.v1 import MoGeV1Config
from monopriors.third_party.moge.model.v2 import MoGeV2Config


def _v1_checkpoint_config() -> dict[str, object]:
    return {
        "encoder": "dinov2_vitl14",
        "remap_output": "exp",
        "output_mask": True,
        "split_head": True,
        "intermediate_layers": 4,
        "dim_upsample": [256, 128, 64],
        "dim_times_res_block_hidden": 2,
        "num_res_blocks": 2,
        "trained_area_range": [250000, 500000],
        "last_conv_channels": 32,
        "last_conv_size": 1,
    }


def test_v1_checkpoint_config_is_translated_strictly() -> None:
    config: dict[str, object] = _v1_checkpoint_config()

    spec: MoGeV1Config = MoGeV1Config.from_checkpoint_config(config)

    assert spec.remap_output == "exp"
    assert spec.num_tokens_range == (1275, 2551)


def test_v1_checkpoint_config_rejects_unknown_keys() -> None:
    config: dict[str, object] = _v1_checkpoint_config()
    config["unexpected"] = True

    with pytest.raises(ValueError, match="unexpected"):
        MoGeV1Config.from_checkpoint_config(config)


def _v2_conv_stack_config(dim_in: int, dim_out: int) -> dict[str, object]:
    return {
        "dim_in": [dim_in, 256, 128, 64, 32],
        "dim_out": [None, None, None, None, dim_out],
        "dim_res_blocks": [dim_in, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    }


def _v2_checkpoint_config(*, include_normals: bool) -> dict[str, object]:
    config: dict[str, object] = {
        "encoder": {
            "backbone": "dinov2_vits14",
            "intermediate_layers": [5, 11],
            "dim_out": 384,
        },
        "neck": {
            "dim_in": [386, 2, 2, 2, 2],
            "dim_out": None,
            "dim_res_blocks": [384, 256, 128, 64, 32],
            "num_res_blocks": [0, 1, 1, 1, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        },
        "points_head": _v2_conv_stack_config(384, 3),
        "mask_head": _v2_conv_stack_config(384, 1),
        "scale_head": {"dims": [384, 384, 384, 1]},
        "remap_output": "exp",
        "num_tokens_range": [1200, 3600],
    }
    if include_normals:
        config["normal_head"] = _v2_conv_stack_config(384, 3)
    return config


def test_v2_checkpoint_config_requires_invariant_heads_and_allows_optional_normals() -> None:
    config: dict[str, object] = _v2_checkpoint_config(include_normals=False)

    spec: MoGeV2Config = MoGeV2Config.from_checkpoint_config(config)

    assert spec.normal_head is None
    assert spec.num_tokens_range == (1200, 3600)

    del config["scale_head"]
    with pytest.raises(ValueError, match="scale_head"):
        MoGeV2Config.from_checkpoint_config(config)


def test_v2_checkpoint_config_rejects_unknown_nested_keys() -> None:
    config: dict[str, object] = _v2_checkpoint_config(include_normals=True)
    raw_encoder: object = config["encoder"]
    assert isinstance(raw_encoder, dict)
    raw_encoder["unexpected"] = True

    with pytest.raises(ValueError, match="unexpected"):
        MoGeV2Config.from_checkpoint_config(config)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_moge_builds_on_its_device(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    """``from_pretrained`` runs under the target device: built on the CPU, MoGe-2 spent ~2 s on a random init that the
    checkpoint then overwrites (0.3 s when built on the GPU)."""
    import torch

    from monopriors.models.moge_v2 import export
    from monopriors.models.relative_depth import moge_v1

    module = moge_v1 if version == "v1" else export
    devices: list[torch.device] = []

    class Empty(module.MoGeModel):  # the class without its layers
        onnx_compatible_mode = False  # shadows the property that reaches into the encoder
        num_tokens_range = export.MOGE_V2_NUM_TOKENS_RANGE

        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

    def from_pretrained(*args: object, **kwargs: object) -> torch.nn.Module:
        devices.append(torch.get_default_device())
        return Empty()

    monkeypatch.setattr(module.MoGeModel, "from_pretrained", from_pretrained)
    if version == "v1":
        moge_v1.MoGeV1Predictor(device="cuda")
    else:
        export.load_pinned_moge_v2("vitl", 9)
    assert [d.type for d in devices] == ["cuda"]
