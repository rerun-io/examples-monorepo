"""Vendored LiteAnyStereo V2 models (https://github.com/TomTomTommi/LiteAnyStereo, MIT license).

Taken from the pablovela5620/LiteAnyStereo fork at ``main``, which is code-identical to
upstream revision 8c97bd4:

- ``liteanystereov2.py``   <- ``core/liteanystereov2.py``
- ``liteanystereov2_H.py`` <- ``core/liteanystereov2_H.py``
- ``aggregation_fasternet.py`` <- ``core/aggregation_fasternet.py``
- ``fnet.py``              <- ``core/fnet.py`` (V2 FasterNet feature network only)
- ``submodule.py``         <- ``core/submodule.py``
- ``padding.py``           <- ``core/utils.py`` (``InputPadder`` only)

Local changes are limited to package-relative imports, replacing the wildcard import in
``fnet.py`` with explicit imports, dropping V1-only feature-network code, and retaining only
``InputPadder`` from ``core/utils.py``. V1 (MobileNetV2 LiteAnyStereo) is intentionally not vendored.
The V2 model math, module and parameter names, and state_dict keys remain identical to upstream.

Re-syncing with upstream:

1. Copy the upstream sources to
   ``tests/reference_data/liteanystereo/upstream_*.py`` using the file mapping above.
2. Re-apply the relative-import changes and the V1/utilities reductions to this package.
3. Run ``tests/test_liteanystereo_upstream_equivalence.py``.
"""
