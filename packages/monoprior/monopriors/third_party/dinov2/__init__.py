"""Consolidated from facebookresearch/dinov2 via the MoGe and Depth Anything V2 vendored forks.

This package contains the inference-only subset used by monopriors.
"""

from monopriors.third_party.dinov2.vision_transformer import DINOv2, dinov2_vitb14, dinov2_vitl14, dinov2_vits14

__all__: tuple[str, ...] = ("DINOv2", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vits14")
