"""Owned fork of the ZipDepth network (https://github.com/fabiotosi92/ZipDepth, ECCV 2026, MIT license).

Taken from the pablovela5620/ZipDepth fork at revision 5a80354 (code-identical to upstream):

- ``architecture.py``  <- ``zipdepth/model/architecture.py`` (stays training-capable — the
  ``packages/zipdepth`` training package imports it from here)
- ``model_utils.py``   <- ``zipdepth/utils/model_utils.py``

Local changes vs. upstream include full type annotations (jaxtyping shapes, TypedDict configs,
and TypeAlias tuples), runtime beartype checks in the dev environment, Google-style docstrings,
removal of the unused ``edge_ch`` upsampler parameter and decoder ``size`` parameter, and an
explicit error for unknown variants. The model math, module and parameter names, and state_dict
keys remain numerically identical to upstream.

Re-syncing with upstream:

1. Copy upstream ``architecture.py`` and ``model_utils.py`` over
   ``tests/reference_data/zipdepth/upstream_architecture.py`` and ``upstream_model_utils.py``.
2. Re-apply the local annotations, documentation, and owned-interface cleanup to this package.
3. Run ``tests/test_zipdepth_upstream_equivalence.py``.

This is the single copy of the architecture in the monorepo. Inference (resize, normalize,
upsample) lives in ``monopriors.models.relative_depth.zipdepth.ZipDepthPredictor``.
"""
