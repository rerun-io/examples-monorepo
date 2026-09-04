"""Unowned Fast-FoundationStereo inference vendor (https://github.com/NVlabs/Fast-FoundationStereo).

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

Local changes: absolute imports, tensorrt/onnx-only classes removed.
"""
