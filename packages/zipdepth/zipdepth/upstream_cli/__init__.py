"""Upstream ZipDepth argparse CLIs (fork 5a80354 ``scripts/``), relocated here so beartype instruments them.

Bodies are unmodified apart from model imports (``monopriors.third_party.zipdepth``), ``eval.py``
running inference through ``monopriors`` ``ZipDepthPredictor``, and ``train.py``'s ``__main__`` block
wrapped as ``cli()``. The shared trainer has local catalog-lane support for sparse and prompted
metric targets, fixed-step checkpointing and resume skipping, and optional BatchNorm pinning. It
also disables compilation under beartype dev instrumentation. Its defaults preserve upstream SSI
training behavior.
Upstream ``infer.py`` was dropped — ``ZipDepthPredictor`` / ``apis/infer_rerun``
cover it. Converting them to tyro ``apis/`` is phase-2 debt.
"""
