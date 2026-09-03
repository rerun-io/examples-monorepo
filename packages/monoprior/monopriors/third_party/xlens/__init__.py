"""Owned X-Lens inference fork.

Upstream: https://github.com/zhouhengamerica/XLens
Revision: ``e6bdf2f66b26a9e4ef1663feaaf7e4618e1e8f7d``
Code license: Apache-2.0; see ``LICENSE``.
Released weights: Hugging Face ``henryzhou998/X-Lens`` at revision
``1d0c96353b69464addad12389fadbb816e3978ae``. The gated weights are licensed
CC-BY-NC-4.0 and are downloaded only with the user's own Hugging Face login.

File mapping:

- ``inference/{pipeline,preprocess,geometry}.py`` <- ``xlens/inference/``
- ``models/{net,dpt_head,ray_map_encoder}.py`` <- ``xlens/models/``
- ``models/utils/head_utils.py`` <- ``xlens/models/utils/head_utils.py``
- ``models/dinov2/**`` <- ``xlens/models/dinov2/**``
- ``xlens_vits.yaml`` <- ``configs/xlens_vits.yaml``
- ``LICENSE`` <- ``LICENSE``

Local changes vs upstream include absolute package imports, complete Python and
jaxtyping annotations, runtime beartype checks in the dev environment,
Google-style docstrings, and a validated ``XLensArchitectureConfig`` at the
checkpoint/YAML boundary. The fork retains only released inference paths:
optimizer groups, selective-training freezes, stochastic-depth training,
gradient checkpointing, optional xFormers kernels, feature-export hooks, the
unused Gram-distillation output, and the separate partial-DINO bootstrap loader
are removed. The RoPE position cache includes its device to prevent cross-device
reuse. Released module, class, parameter and state-dict names and all outputs
used by the rig-depth predictor remain unchanged.

Re-syncing with upstream:

1. Copy the mapped frozen-revision sources into
   ``tests/reference_data/xlens/upstream_*.py`` without edits.
2. Re-apply the absolute imports, types, documentation, typed configuration,
   inference-only cleanup, and device-aware cache without renaming checkpoint
   modules or parameters.
3. Run ``tests/test_xlens_upstream_equivalence.py``; all four seeded CPU
   scenarios must remain bit-identical.
"""

from monopriors.third_party.xlens.models import XLensNet

__version__: str = "0.1.0"

__all__: list[str] = ["XLensNet"]
