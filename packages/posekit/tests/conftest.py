"""Test-process settings: pytest-beartype checks posekit here (see pyproject), and this flags it for the code below."""

import os

# beartype runs under pytest only; code that cannot run under it (torch.compile, ONNX export) reads this and stays eager.
os.environ.setdefault("PIXI_DEV_MODE", "1")
