"""Upstream ZipDepth argparse CLIs (fork 5a80354 ``scripts/``), relocated here so beartype instruments them.

Bodies are unmodified apart from model imports (``monopriors.third_party.zipdepth``), ``eval.py``
running inference through ``monopriors`` ``ZipDepthPredictor``, and ``train.py``'s ``__main__`` block
wrapped as ``cli()``. Upstream ``infer.py`` was dropped — ``ZipDepthPredictor`` / ``apis/infer_rerun``
cover it. Converting them to tyro ``apis/`` is phase-2 debt.
"""
