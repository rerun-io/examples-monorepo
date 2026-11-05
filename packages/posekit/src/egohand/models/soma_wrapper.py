# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""SOMA body model wrapper for EgoHand.

Handles rotation matrix -> axis-angle conversion before calling SOMA FK.
"""

import torch
import torch.nn as nn



class NonPersistentModuleWrapper(nn.Module):
    """Wrap a module but drop all of its state_dict entries (SOMA weights excluded from checkpoints)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def load_state_dict(self, state_dict, strict=True):
        return torch.nn.modules.module._IncompatibleKeys([], [])
