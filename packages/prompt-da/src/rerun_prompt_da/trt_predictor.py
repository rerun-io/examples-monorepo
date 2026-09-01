"""Backward-compatible re-exports of PromptDA completion predictors."""

from monopriors.models.depth_completion.prompt_da import (
    PromptDATorchPredictor,
    PromptDATrtPredictor,
    network_image_hw,
    postprocess_depth,
    preprocess_batch,
)

__all__ = (
    "PromptDATorchPredictor",
    "PromptDATrtPredictor",
    "network_image_hw",
    "postprocess_depth",
    "preprocess_batch",
)
