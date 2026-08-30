"""Owned Fast-FoundationStereo inference fork (https://github.com/NVlabs/Fast-FoundationStereo).

Taken from upstream revision ``a290ba04c1b3ad1ec41a33974a157b2917b624d4`` under the
NVIDIA Source Code License. The license permits research and other non-commercial use;
see ``LICENSE.txt`` for the full terms.

- ``foundation_stereo.py`` <- ``core/foundation_stereo.py``
- ``extractor.py`` <- ``core/extractor.py``
- ``update.py`` <- ``core/update.py``
- ``geometry.py`` <- ``core/geometry.py``
- ``submodule.py`` <- ``core/submodule.py``
- ``distill_block.py`` <- ``core/distill_block.py``
- ``utils.py`` <- ``core/utils/utils.py`` (``InputPadder`` and stereo samplers)

Local changes vs upstream include full type annotations (jaxtyping tensor dtypes and shapes,
TypedDict configuration fields, and TypeAlias containers), runtime beartype checks in the dev
environment, Google-style docstrings, and absolute package imports. The fork retains only the
released inference architecture: training outputs, alternate feature-backbone and unused 3D
resize branches, the unused low-memory sampler, the duplicate concat-volume builder, profiling
scopes, and TensorRT/ONNX/export helpers are removed. The Triton GWC autotuner is created lazily
inside a closure so beartype never inspects the Autotuner object at module import. Model math,
module and class names required by the released pickle, parameter names, state_dict keys, and
inference outputs remain unchanged.

Re-syncing with upstream:

1. Copy the upstream sources to
   ``tests/reference_data/fast_foundationstereo/upstream_*.py`` using the file mapping above.
2. Re-apply the local annotations, documentation, absolute imports, inference-only cleanup, and
   lazy Triton-kernel construction without renaming serialized modules or classes.
3. Run ``tests/test_fast_foundationstereo_upstream_equivalence.py`` on CPU and in the slow CUDA band.
"""
