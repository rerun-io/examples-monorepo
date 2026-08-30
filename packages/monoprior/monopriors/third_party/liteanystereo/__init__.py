"""Owned fork of LiteAnyStereo V2 (https://github.com/TomTomTommi/LiteAnyStereo, MIT license).

Taken from the pablovela5620/LiteAnyStereo fork at ``main``, which is code-identical to
upstream revision 8c97bd4:

- ``liteanystereov2.py``   <- ``core/liteanystereov2.py``
- ``liteanystereov2_H.py`` <- ``core/liteanystereov2_H.py``
- ``aggregation_fasternet.py`` <- ``core/aggregation_fasternet.py``
- ``fnet.py``              <- ``core/fnet.py`` (V2 FasterNet feature network only)
- ``submodule.py``         <- ``core/submodule.py``
- ``padding.py``           <- ``core/utils.py`` (``InputPadder`` only)

Local changes vs. upstream include full type annotations (jaxtyping float32 shapes, TypedDict
configs, and TypeAlias containers), runtime beartype checks in the dev environment, Google-style
docstrings, absolute package imports, and removal of V1-only code, unused correlation/warping
helpers, and training-only ``kd_mode``/multi-output paths. ``padding.py`` retains only
``InputPadder`` from ``core/utils.py``. The model math, module and parameter names, state_dict keys,
and inference outputs remain numerically identical to upstream.

Re-syncing with upstream:

1. Copy the upstream sources to
   ``tests/reference_data/liteanystereo/upstream_*.py`` using the file mapping above.
2. Re-apply the local annotations, documentation, absolute imports, and owned-interface cleanup.
3. Run ``tests/test_liteanystereo_upstream_equivalence.py``.
"""
