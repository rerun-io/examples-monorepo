# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint loading for the public LAMP SMPL model."""

# pyright: reportPrivateImportUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import torch
from lamp.models.model import LampNet
from lamp.models.model_utils import GRAVITY_DIRECTION_VIO, R_CG_CGZ

_UNUSED_CHECKPOINT_PREFIXES: tuple[str, ...] = ("dec_readout.",)


def build_lampnet_from_checkpoint(
    checkpoint_path: str | Path,
    smpl_model_path: str | Path,
    device: torch.device,
) -> LampNet:
    """Construct `LampNet`, load weights, and return an eval-mode model."""
    checkpoint_state: dict[str, torch.Tensor] = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True
    )
    checkpoint_state = _drop_unused_checkpoint_keys(checkpoint_state)

    model = LampNet(
        dim_in=7,
        dim_feat=256,
        depth=3,
        num_heads=8,
        mlp_ratio=4,
        num_joints=17,
        maxlen=20,
        smpl_model_path=str(smpl_model_path),
    )
    model = model.to(device).eval()

    result = model.load_state_dict(checkpoint_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "Loading the LAMP checkpoint produced unexpected keys: "
            f"{sorted(result.unexpected_keys)}."
        )

    expected_missing = {"_r_cg_cgz", "_gravity_w"}
    unexpected_missing = set(result.missing_keys) - expected_missing
    if unexpected_missing:
        raise RuntimeError(
            "Loading the LAMP checkpoint left unexpected missing keys: "
            f"{sorted(unexpected_missing)}."
        )

    _validate_runtime_constants(model, device)
    return model


def _drop_unused_checkpoint_keys(
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith(_UNUSED_CHECKPOINT_PREFIXES)
    }


def _validate_runtime_constants(model: LampNet, device: torch.device) -> None:
    expected_r_cg = torch.tensor(list(R_CG_CGZ), dtype=torch.float32, device=device)
    expected_g_w = torch.tensor(
        list(GRAVITY_DIRECTION_VIO), dtype=torch.float32, device=device
    )
    assert torch.equal(model._r_cg_cgz, expected_r_cg), (  # pyright: ignore[reportPrivateUsage]
        "`_r_cg_cgz` buffer was modified during checkpoint load."
    )
    assert torch.equal(model._gravity_w, expected_g_w), (  # pyright: ignore[reportPrivateUsage]
        "`_gravity_w` buffer was modified during checkpoint load."
    )
