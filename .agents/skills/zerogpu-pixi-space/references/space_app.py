"""My app on ZeroGPU: the monorepo app, its runs in a @spaces.GPU worker. Keep this file short; settings go in as values."""

import spaces  # first: ZeroGPU patches torch so nothing initialises CUDA in the main process  # noqa: I001

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

# Heavy imports here, not in the worker: each worker is a fork of this process and inherits them.
from my_package.app_ui import layout  # placeholder: the app's build()/launch() module

EXAMPLES: Path = Path(__file__).parent / "examples"  # re-encoded with -bf 0: no transcode on a click
RUN_S: int = 30
"""GPU seconds a click reserves: the measured fresh-worker time plus margin (charged x1.5, aborted ~2 s past it)."""
MAX_FRAMES: int = 600
"""Frames a click processes at most, so an upload fits in RUN_S."""

# On the CPU, before any GPU time: prebuilt engines into the engine cache, so a worker only loads them.
snapshot_download("my-user/my-engines-repo", local_dir="/home/user/.cache/trtkit/trt")  # placeholders


@spaces.GPU(duration=RUN_S)
def run(*inputs: Any) -> Iterator[Any]:
    """One click: the app's own run generator, its frame count capped to fit the reservation."""
    yield from layout.run(*inputs, max_frames=MAX_FRAMES)


layout.build(run=run, examples=EXAMPLES).queue(default_concurrency_limit=1).launch(
    server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), ssr_mode=False,
    allowed_paths=[str(EXAMPLES)],
)
