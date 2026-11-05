"""Eager execution in the development environment, ``torch.compile`` everywhere else."""

import os
from collections.abc import Callable
from typing import Any, cast

import torch


def compile_unless_dev[CompiledCallable: Callable[..., Any]](function: CompiledCallable) -> CompiledCallable:
    """Compile ``function`` as one dynamic-shape graph, except under ``PIXI_DEV_MODE=1``.

    The development environment keeps the eager function so that beartype checks
    every call and annotated local; production runs the same math fused.
    """
    if os.environ.get("PIXI_DEV_MODE") == "1":
        return function
    return cast(CompiledCallable, torch.compile(function, fullgraph=True, dynamic=True))
