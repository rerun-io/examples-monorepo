"""The slam-rs VIO core (``slam_rs._core``) and its Python plumbing."""

from typing import Literal, TypeAlias

BackendRequest: TypeAlias = Literal["auto", "cpu", "gpu"]
"""The frontend a caller asks for; ``auto``, the default of every slam-rs tool, is resolved by ``_core.select_backend``."""

GbaRequest: TypeAlias = Literal["follow", "cpu", "gpu"]
"""Where the global BA's PCG runs (``--ba-backend``); ``follow``, the default, takes the frontend's lane."""

Lane: TypeAlias = Literal["cpu", "gpu"]
"""The lane a pipeline stage runs on."""
