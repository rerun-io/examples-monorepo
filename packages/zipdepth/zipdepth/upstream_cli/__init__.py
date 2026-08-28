"""Upstream ZipDepth argparse CLIs (fork 5a80354 ``scripts/``), relocated here so beartype instruments them.

Bodies are unmodified apart from model imports (``monopriors.third_party.zipdepth``) and, in
``train.py``, the ``__main__`` block wrapped as ``cli()``. Converting them to tyro ``apis/`` is phase-2 debt.
"""
