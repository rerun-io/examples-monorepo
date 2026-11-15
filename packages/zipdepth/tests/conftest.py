"""Keep synthetic transform tests independent of the network."""

import os

# Albumentations otherwise contacts PyPI during test collection.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# beartype runs under pytest only; code that cannot run under it (torch.compile, ONNX export) reads this and stays eager.
os.environ.setdefault("PIXI_DEV_MODE", "1")
