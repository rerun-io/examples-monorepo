# [vipe-src] Replaces ViPE's vendored DPVO altcorr (see ../__init__.py).
"""Re-exports :mod:`dpvo.altcorr`: ``corr`` and ``patchify`` (``dino_fusion.py``'s DINO patchifier)."""

from dpvo.altcorr import corr as corr
from dpvo.altcorr import patchify as patchify
