"""Depth-completion model registry: tyro subcommands over model configs."""

from pathlib import Path
from typing import TYPE_CHECKING

import tyro

from monopriors.models.depth_completion.base_completion_depth import BaseCompletionPredictor
from monopriors.models.depth_completion.prompt_da import PromptDAConfig, PromptDAPredictor
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPromptConfig, ZipDepthPromptPredictor

if TYPE_CHECKING:
    CompletionConfig = PromptDAConfig | ZipDepthPromptConfig
else:
    CompletionConfig = tyro.extras.subcommand_type_from_defaults(
        {
            "prompt-da": PromptDAConfig(),
            "zipdepth-promptda": ZipDepthPromptConfig(checkpoint=Path()),
        },
        prefix_names=False,
    )

AnnotatedCompletionConfig = tyro.conf.OmitSubcommandPrefixes[CompletionConfig]

__all__ = (
    "AnnotatedCompletionConfig",
    "BaseCompletionPredictor",
    "CompletionConfig",
    "PromptDAConfig",
    "PromptDAPredictor",
    "ZipDepthPromptConfig",
    "ZipDepthPromptPredictor",
)
