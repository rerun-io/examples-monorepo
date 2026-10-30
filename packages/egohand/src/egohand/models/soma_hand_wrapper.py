# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""SOMA hand model wrapper for EgoHand.

Thin wrapper around SOMAHandLayer (left + right) with frozen weights
excluded from checkpoints via NonPersistentModuleWrapper.
"""

import os
import time

import torch
import torch.nn as nn

from egohand.models.soma_wrapper import NonPersistentModuleWrapper


def _resolve_hand_data_root():
    """Find assets directory that contains SOMAHand.npz.

    Checks third_party/soma-hand/assets/ first (local repo), then falls
    back to the HuggingFace-cached soma assets directory.
    """
    # Local third_party path (relative to repo root)
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "third_party",
        "soma-hand",
        "assets",
    )
    if os.path.isfile(os.path.join(local_path, "SOMAHand.npz")):
        return local_path

    import soma

    return str(soma.get_assets_dir())


class SomaHandWrapper(nn.Module):
    """Wrapper around SOMAHandLayer for left and right hands.

    SOMA hand weights are excluded from checkpoints via NonPersistentModuleWrapper.
    """

    def __init__(
        self, data_root=None, device="cuda", mode="warp", identity_model_type="soma"
    ):
        super().__init__()
        from soma.hand import SOMAHandLayer
        from soma.units import Unit

        if data_root is None:
            data_root = _resolve_hand_data_root()

        self.profile_ms = None
        self.identity_model_type = identity_model_type
        self._hands = {}
        for hand_type in ("left", "right"):
            hand = SOMAHandLayer(
                data_root=data_root,
                hand_type=hand_type,
                device=device,
                identity_model_type=identity_model_type,
                mode=mode,
                output_unit=Unit.METERS,
                load_correctives_model=False,
            )
            wrapped = NonPersistentModuleWrapper(hand)
            wrapped.requires_grad_(False)
            setattr(self, f"hand_{hand_type}", wrapped)
            self._hands[hand_type] = hand

        self.left = self._hands["left"]
        self.right = self._hands["right"]

    @property
    def num_shape_components(self):
        return self.left.num_shape_components

    @property
    def num_scale_params(self):
        n = getattr(self.left.identity_model, "num_scale_params", None)
        return n if n is not None else 0

    def faces(self, hand_type):
        """Get face indices for the given hand type."""
        hand = self._hands[hand_type]
        return (
            hand.faces.cpu().numpy()
            if isinstance(hand.faces, torch.Tensor)
            else hand.faces
        )

    def wrist_global_id(self, hand_type):
        """Wrist joint index in 78-joint system."""
        return self._hands[hand_type].wrist_global_id

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        poses,
        identity_coeffs,
        hand_type,
        transl=None,
        pose2rot=True,
        global_scale=1.0,
        scale_params=None,
    ):
        """Run hand FK.

        Args:
            poses: (B, 25, 3) axis-angle or (B, 25, 3, 3) rot matrices.
                Joint 0 = wrist, joints 1-24 = fingers.
            identity_coeffs: (B, K) identity coefficients (K=20 SOMA, K=5 MHR)
            hand_type: 'left' or 'right'
            transl: optional (B, 3) wrist translation in meters
            pose2rot: convert axis-angle to rot matrices if True
            global_scale: uniform scale scalar or (B,) tensor. Default 1.0.
            scale_params: optional (B, S) per-identity scale params (MHR: S=26).
                None for SOMA.
        Returns:
            dict with 'vertices' (B, Vh, 3) and 'joints' (B, 25, 3) in meters
        """
        hand = self._hands[hand_type]
        if self.profile_ms is not None:
            torch.cuda.synchronize()
            started = time.perf_counter()
        result = hand(
            poses.float(),
            identity_coeffs.float(),
            pose2rot=pose2rot,
            global_translation=transl,
            global_scale=global_scale,
            scale_params=scale_params,
        )
        if self.profile_ms is not None:
            torch.cuda.synchronize()
            self.profile_ms.append((time.perf_counter() - started) * 1000.0)
        return result
