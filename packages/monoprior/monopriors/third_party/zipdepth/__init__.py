"""Owned fork of the ZipDepth network (https://github.com/fabiotosi92/ZipDepth, ECCV 2026, MIT license).

Taken from the pablovela5620/ZipDepth fork at revision 5a80354 (code-identical to upstream):

- ``architecture.py``  <- ``zipdepth/model/architecture.py`` (stays training-capable — the
  ``packages/zipdepth`` training package imports it from here)
- ``model_utils.py``   <- ``zipdepth/utils/model_utils.py``

Local changes vs. upstream: full type annotations (jaxtyping shapes, TypedDict configs, TypeAlias
tuples) and Google-style docstrings per the repo's Python conventions; behaviour, module/parameter
names and state_dict keys are unchanged (verified: released-weight disparity range and a bf16
training step are identical). Held to the normal ruff / pyrefly / vulture gates.

This is the single copy of the architecture in the monorepo. Inference (resize, normalize,
upsample) lives in ``monopriors.models.relative_depth.zipdepth.ZipDepthPredictor``.
"""
