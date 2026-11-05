"""Sapiens2 pose checkpoints on Hugging Face and the cached model loader."""

from functools import cache

from huggingface_hub import hf_hub_download
from torch import nn

from posekit.models.sapiens import SapiensModelSize
from sapiens2_lite.pose import init_pose_model

POSE_CHECKPOINTS: dict[SapiensModelSize, tuple[str, str]] = {
    "0.4B": ("facebook/sapiens2-pose-0.4b", "sapiens2_0.4b_pose.safetensors"),
    "0.8B": ("facebook/sapiens2-pose-0.8b", "sapiens2_0.8b_pose.safetensors"),
    "1B": ("facebook/sapiens2-pose-1b", "sapiens2_1b_pose.safetensors"),
}
"""Hugging Face repo and safetensors file of each checkpoint size."""


@cache
def load_pose_model(size: SapiensModelSize) -> nn.Module:
    """Download (once) and load a Sapiens2 pose network on CUDA, cached per size.

    The cache hands out one shared module per size: the ONNX export rewrites it in place,
    and the fast export deep-copies it first.

    Args:
        size: Checkpoint size.

    Returns:
        The eval-mode ``TopDownPoseModel``: normalized RGB crops to 308 Goliath heatmaps.
    """
    repo_id: str = POSE_CHECKPOINTS[size][0]
    filename: str = POSE_CHECKPOINTS[size][1]
    return init_pose_model(size, hf_hub_download(repo_id=repo_id, filename=filename), device="cuda")
