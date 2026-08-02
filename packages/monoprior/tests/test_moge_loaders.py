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
