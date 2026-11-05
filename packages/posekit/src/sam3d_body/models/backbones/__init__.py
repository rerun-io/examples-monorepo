# Copyright (c) Meta Platforms, Inc. and affiliates.


def create_backbone(name, cfg=None):
    if name == "dinov3_vith16plus":
        from .dinov3 import Dinov3Backbone

        backbone = Dinov3Backbone(name, cfg=cfg)
    else:
        raise NotImplementedError("Backbone type is not implemented")

    return backbone
