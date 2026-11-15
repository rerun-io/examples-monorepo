"""Beartype runs only under pytest: instrument hexels_bench before any test module imports it.

Apps, CLIs and benchmarks never import beartype, in the dev env or the prod env.
"""

import os

from beartype.claw import beartype_package

# Read by code that cannot run under beartype (torch.compile, ONNX export): it stays eager while the tests run.
os.environ["PIXI_DEV_MODE"] = "1"
beartype_package("hexels_bench")
