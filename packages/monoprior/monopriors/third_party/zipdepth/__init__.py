"""Vendored ZipDepth network (https://github.com/fabiotosi92/ZipDepth, ECCV 2026, MIT license).

Copied byte-for-byte from the pablovela5620/ZipDepth fork at revision 5a80354 (code-identical to
upstream for these files):

- ``architecture.py``  <- ``zipdepth/model/architecture.py`` (stays training-capable — the
  ``packages/zipdepth`` training package imports it from here)
- ``model_utils.py``   <- ``zipdepth/utils/model_utils.py``

This is the single copy of the architecture in the monorepo. Inference (resize, normalize,
upsample) lives in ``monopriors.models.relative_depth.zipdepth.ZipDepthPredictor``.
"""
