"""Vendored X-Lens inference subset.

Upstream: https://github.com/henry123-boy/X-Lens
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

Local changes: import paths only. Pristine source fixtures live under
``tests/reference_data/xlens`` for equivalence testing.
"""

from monopriors.third_party.xlens.models import XLensNet

__version__: str = "0.1.0"

__all__: list[str] = ["XLensNet"]
