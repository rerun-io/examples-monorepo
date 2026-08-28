"""Upstream ZipDepth argparse CLIs (fork 5a80354 ``scripts/``), relocated here so beartype instruments them.

Bodies are unmodified apart from model imports (``monopriors.third_party.zipdepth``), ``eval.py``
running inference through ``monopriors`` ``ZipDepthPredictor``, and ``train.py``'s ``__main__`` block
wrapped as ``cli()``. The shared trainer also forwards an optional sparse-target mask to the loss.
Its ``train`` method accepts a local ``skip_batches`` override so unordered catalog resumes do not decode and discard prior batches; the default preserves upstream behavior.
Upstream ``infer.py`` was dropped — ``ZipDepthPredictor`` / ``apis/infer_rerun``
cover it. Converting them to tyro ``apis/`` is phase-2 debt.
"""
