"""Depth-completion model registry."""

from monopriors.models.depth_completion.base_completion_depth import BaseCompletionPredictor
from monopriors.models.depth_completion.prompt_da import PromptDAConfig, PromptDAPredictor

__all__ = ("BaseCompletionPredictor", "PromptDAConfig", "PromptDAPredictor")
