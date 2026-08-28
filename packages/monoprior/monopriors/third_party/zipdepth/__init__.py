"""Vendored ZipDepth (https://github.com/fabiotosi92/ZipDepth, ECCV 2026, MIT license).

Copied from the pablovela5620/ZipDepth fork at revision 5a80354 (code-identical to
upstream for these files):

- ``architecture.py``  <- ``zipdepth/model/architecture.py`` (unmodified; stays training-capable —
  the ``packages/zipdepth`` training package imports it from here)
- ``model_utils.py``   <- ``zipdepth/utils/model_utils.py`` (unmodified)
- ``colormap.py``      <- ``zipdepth/utils/colormap.py`` (unmodified)
- ``predictor.py``     <- ``zipdepth/inference/predictor.py`` (only the three ``zipdepth.*`` imports
  rewritten to relative imports)

This is the single copy of the architecture in the monorepo.
"""
