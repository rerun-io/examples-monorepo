"""Eager execution in the development environment or on request, ``torch.compile`` everywhere else."""

import os
from collections.abc import Callable
from typing import Any, cast

import torch


def compile_unless_dev[CompiledCallable: Callable[..., Any]](function: CompiledCallable) -> CompiledCallable:
    """Compile ``function`` as one dynamic-shape graph, except under ``PIXI_DEV_MODE=1`` or ``POSEKIT_COMPILE=0``.

    The development environment keeps the eager function so that beartype checks
    every call and annotated local; production runs the same math fused.
    ``POSEKIT_COMPILE=0`` keeps it eager in production too, for a host whose processes are short-lived:
    each new process traces for seconds on its first call (EgoHand fast: 6-13 s, for about 5 % more frames a second).
    """
    if os.environ.get("PIXI_DEV_MODE") == "1" or os.environ.get("POSEKIT_COMPILE") == "0":
        return function
    return cast(CompiledCallable, torch.compile(function, fullgraph=True, dynamic=True))
